"""Activation patching as an Intervention, the mechanics behind ``PatchGrid``.

Patching splices a component's activation from a source forward into a target forward and reads the
change in reward. In v1 this lived in ``patching.py`` as a bespoke hook loop that could not compose
with capture. Here a patch is an :class:`~reward_lens.interventions.base.Intervention`: it compiles
against a signal into concrete site-addressed mount hooks, so the same object that names "replace the
attention output at layer 12 with this source activation" carries its own fingerprint into the
Evidence subject, and a patched-run number can never be mistaken for a clean-run number.

Two granularities are implemented. :class:`ComponentPatch` replaces a whole sublayer output
(``attn_out`` or ``mlp_out``) or the residual stream (``resid_post``). :class:`HeadPatch` replaces a
single attention head's contribution by editing the per-head slice of the ``o_proj`` input, which is
the head-level resolution E15 needs. Both follow v1's sequence-alignment rule exactly (replace the
first ``min(T_src, T_tgt)`` positions, keep the target beyond that), so a v3 patch reproduces the v1
patch effect on the same model.

The frozen runtime exposes a single mounting path, but its intervention contract and the frozen
``CompiledIntervention`` shape were specified for the full M6 interventions subsystem and do not yet
line up. Rather than touch either frozen file, :func:`run_patched_scores` installs a compiled
intervention's ``mounts`` directly through the same ``resolve_module`` machinery the capture path
uses. When the M6 contract lands, this runner is the one place that changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from reward_lens.core.types import Site, content_hash
from reward_lens.interventions.base import CompiledIntervention, MountHook

if TYPE_CHECKING:
    import torch

    from reward_lens.signals.classifier import ClassifierRM


def _seq_aligned_replace(hidden: "torch.Tensor", source: "torch.Tensor") -> "torch.Tensor":
    """Replace ``hidden`` with ``source`` over the shared leading positions (the v1 rule).

    ``hidden`` is the target activation ``(B, T_tgt, d)`` and ``source`` the source activation
    ``(B, T_src, d)``. When the lengths match, ``source`` replaces ``hidden`` outright; otherwise the
    first ``min(T_tgt, T_src)`` positions are taken from ``source`` and the remaining target
    positions are kept, which is exactly how v1's ``_patch_component`` handled a length mismatch. The
    source is moved to the target's device and dtype first.
    """
    src = source.to(device=hidden.device, dtype=hidden.dtype)
    if src.shape[1] == hidden.shape[1]:
        return src
    min_len = min(src.shape[1], hidden.shape[1])
    out = hidden.clone()
    out[:, :min_len, :] = src[:, :min_len, :]
    return out


@dataclass
class ComponentPatch:
    """Replace a sublayer or residual activation with a source (or zero); an ``Intervention``.

    ``site`` names where to patch (``attn_out``, ``mlp_out``, or ``resid_post`` at a layer).
    ``source`` is the full-sequence source activation ``(1, T_src, d)`` to splice in; when ``mode`` is
    ``"zero"`` the activation is replaced with zeros and ``source`` is ignored (the crude ablation
    baseline). ``label`` is a short human tag used in the component name.
    """

    site: Site
    source: "torch.Tensor | None" = None
    mode: str = "replace"
    label: str = ""
    id: str = "component_patch"

    def fingerprint(self) -> str:
        """A stable cache/provenance key for this patch (shape and site, not the raw payload)."""
        shape = tuple(self.source.shape) if self.source is not None else None
        return content_hash(
            {"kind": "component_patch", "site": str(self.site), "mode": self.mode, "src": shape},
            "iv",
        )

    def _hook(self) -> MountHook:
        mode = self.mode
        source = self.source

        def apply(hidden: "torch.Tensor", _ctx: dict) -> "torch.Tensor":
            import torch

            if mode == "zero":
                return torch.zeros_like(hidden)
            if source is None:
                return hidden
            return _seq_aligned_replace(hidden, source)

        return apply

    def compile(self, signal: "ClassifierRM") -> CompiledIntervention:
        """Resolve into a concrete mount hook at ``site`` (the ``Intervention`` protocol)."""
        del signal  # the site is architecture-resolved by the runner's SiteMap at mount time
        return CompiledIntervention(
            fingerprint=self.fingerprint(),
            mounts={self.site: self._hook()},
            meta={"kind": "component_patch", "site": str(self.site), "mode": self.mode},
        )


@dataclass
class HeadPatch:
    """Replace a single attention head's contribution by editing the ``o_proj`` input slice.

    The attention output is ``o_proj(concat_h head_h)``, so patching head ``h`` means overwriting its
    ``d_head`` slice of the ``o_proj`` input with the source side's slice while leaving every other
    head untouched. ``site`` is ``Site(layer, "head_out", head)``; ``source_head`` is the source
    per-head input ``(1, T_src, d_head)`` for that head.
    """

    site: Site
    source_head: "torch.Tensor"
    n_heads: int
    id: str = "head_patch"

    def fingerprint(self) -> str:
        return content_hash(
            {
                "kind": "head_patch",
                "site": str(self.site),
                "src": tuple(self.source_head.shape),
            },
            "iv",
        )

    def _hook(self) -> MountHook:
        head = self.site.head or 0
        n_heads = self.n_heads
        source_head = self.source_head

        def apply(x: "torch.Tensor", _ctx: dict) -> "torch.Tensor":
            # x is the o_proj input (B, T, n_heads * d_head); slice out the heads, replace one.
            b, t, feat = x.shape
            d_head = feat // n_heads
            view = x.view(b, t, n_heads, d_head).clone()
            src = source_head.to(device=x.device, dtype=x.dtype)
            min_len = min(t, src.shape[1])
            view[:, :min_len, head, :] = src[:, :min_len, :]
            return view.view(b, t, feat)

        return apply

    def compile(self, signal: "ClassifierRM") -> CompiledIntervention:
        del signal
        return CompiledIntervention(
            fingerprint=self.fingerprint(),
            mounts={self.site: self._hook()},
            meta={"kind": "head_patch", "site": str(self.site)},
        )


@dataclass
class ResidualAddPatch:
    """Add a precomputed delta to a site's activation; the path-patching primitive.

    Path patching perturbs only the sender to receiver path by adding the sender's source-minus-target
    residual contribution at the receiver's input, holding every other path clean. ``site`` is the
    receiver read point (typically ``Site(layer, "resid_pre")``) and ``delta`` is the
    ``(1, T, d_model)`` residual adjustment to add there. This is the exact mechanic v1's
    ``PathPatcher`` used, expressed as an Intervention.
    """

    site: Site
    delta: "torch.Tensor"
    id: str = "residual_add_patch"

    def fingerprint(self) -> str:
        return content_hash(
            {
                "kind": "residual_add_patch",
                "site": str(self.site),
                "delta": tuple(self.delta.shape),
            },
            "iv",
        )

    def _hook(self) -> MountHook:
        delta = self.delta

        def apply(hidden: "torch.Tensor", _ctx: dict) -> "torch.Tensor":
            d = delta.to(device=hidden.device, dtype=hidden.dtype)
            min_len = min(hidden.shape[1], d.shape[1])
            out = hidden.clone()
            out[:, :min_len, :] = out[:, :min_len, :] + d[:, :min_len, :]
            return out

        return apply

    def compile(self, signal: "ClassifierRM") -> CompiledIntervention:
        del signal
        return CompiledIntervention(
            fingerprint=self.fingerprint(),
            mounts={self.site: self._hook()},
            meta={"kind": "residual_add_patch", "site": str(self.site)},
        )


# ---------------------------------------------------------------------------
# Running a compiled patch (the direct mount path)
# ---------------------------------------------------------------------------


def _extract(adapter: Any, site: Site, output: Any) -> "torch.Tensor":
    """Pull the hidden tensor a mount edits out of a module's forward output for ``site``."""
    if site.point == "attn_out":
        return adapter.extract_attn_output(output)
    if site.point == "mlp_out":
        return adapter.extract_mlp_output(output)
    if site.point == "embed":
        return output[0] if isinstance(output, tuple) else output
    return adapter.extract_layer_output(output)


def _rewrap(output: Any, new_hidden: "torch.Tensor") -> Any:
    """Put an edited hidden tensor back into the container shape the module returned."""
    if isinstance(output, tuple):
        return (new_hidden,) + tuple(output[1:])
    return new_hidden


def run_patched_scores(
    signal: "ClassifierRM",
    compiled: CompiledIntervention,
    view: Any,
    readout: str = "reward",
) -> np.ndarray:
    """Score ``view`` under a compiled patch, returning the fp32 reward per item.

    The compiled intervention's ``mounts`` are installed on the resolved modules through the same
    ``resolve_module`` path captures use: a ``head_out`` mount is a forward pre-hook on ``o_proj``
    (it edits the head input), everything else is a forward hook that edits the module output. The
    reward is the fp32 projection of the head-input hidden state onto the readout vector, matching
    ``ClassifierRM.score`` exactly, so a patched score and a clean score are computed the same way.
    Handles are always removed.
    """
    import torch

    from reward_lens.runtime.hooks import resolve_module

    runtime = signal.runtime
    model = runtime.model
    site_map = runtime.site_map
    read = signal.readout(readout)
    bias = float(read.meta.get("bias", 0.0))

    tokenized = [signal.tokenize(it) for it in view]
    batch = runtime.collate(tokenized)

    handles: list[Any] = []

    def make_forward_hook(site: Site, hook: MountHook) -> Any:
        def _hook(_module: Any, _inputs: Any, output: Any) -> Any:
            hidden = _extract(runtime.adapter, site, output)
            return _rewrap(output, hook(hidden, {"site": site}))

        return _hook

    def make_pre_hook(site: Site, hook: MountHook) -> Any:
        def _hook(_module: Any, args: Any) -> Any:
            x = args[0] if isinstance(args, tuple) else args
            new_x = hook(x, {"site": site})
            rest = tuple(args[1:]) if isinstance(args, tuple) else ()
            return (new_x,) + rest

        return _hook

    try:
        for site, hook in compiled.mounts.items():
            # Head sites all live on the one head-agnostic o_proj module; the site map keys it with
            # head=None, so resolve through that key while the mount edits the requested head's slice.
            resolve_site = Site(site.layer, "head_out", None) if site.point == "head_out" else site
            module = resolve_module(model, site_map.resolve(resolve_site))
            if site.point in ("head_out", "resid_pre"):
                handles.append(module.register_forward_pre_hook(make_pre_hook(site, hook)))
            else:
                handles.append(module.register_forward_hook(make_forward_hook(site, hook)))
        raw = runtime.forward(batch)
    finally:
        for handle in handles:
            handle.remove()

    head_input = raw.extra["head_input"]
    final_pos = raw.extra["final_pos"]
    idx = torch.arange(head_input.shape[0], device=head_input.device)
    pooled = head_input[idx, final_pos]
    values = signal.policy.head_project(pooled, read.vector, bias)
    return values.detach().to("cpu", dtype=torch.float32).numpy()


__all__ = ["ComponentPatch", "HeadPatch", "ResidualAddPatch", "run_patched_scores"]

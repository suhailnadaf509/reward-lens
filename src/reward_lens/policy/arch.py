"""Where a decoder stack keeps its blocks, resolved structurally rather than by family.

This is the replacement for the eleven-method ``ModelAdapter`` abstract base class and its six
family subclasses. The v1 design asked "which architecture is this?" and dispatched to a class that
knew the answer, which meant a new family was a new subclass and an unrecognised one was a silent
`GenericAdapter` fallback nobody could see. This asks a different question: "where in this module
tree are the blocks, and inside a block, where are the attention and the MLP?" That question has a
structural answer for every decoder stack `transformers` ships, and the answer is checkable at load
rather than assumed.

Three facts make the structural walk reliable, and each of them is why a family table is not needed:

  - a decoder stack is the largest `nn.ModuleList` whose members each contain both an attention-like
    submodule and an MLP-like one, and there is exactly one such list in every model here;
  - the token embedding is `model.get_input_embeddings()`, which is `transformers`' own public API
    and is correct for every architecture by contract rather than by inspection;
  - the attention output projection is the one `nn.Linear` inside the attention submodule whose
    input width is a multiple of the head count and whose output width is the model width. Qwen3
    makes that concrete: its `o_proj` is `(8, 512)` on an 8-wide model with four 128-wide heads, so
    "the projection back into the residual stream" identifies it and "the last linear" would not.

What this does not do, stated because it is the reason `model_adapters/` cannot be deleted on the
strength of this module alone. It does not read a reward head, it does not know that ArmoRM's
nineteen objectives are gated, it does not disable Gemma-2's soft cap, and it does not carry
InternLM2's `v_head` convention. Those are grader-side concerns that live in `signals/`, and moving
them is a separate migration with its own subjects to test against. This module covers navigation,
which is the part both a grader and a policy need and the part `runtime/hooks.py` actually calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from reward_lens.core.types import Site
from reward_lens.runtime.backend import SiteMap

if TYPE_CHECKING:  # torch is referenced in annotations only
    import torch
    import torch.nn as nn


#: Names an attention submodule goes by, in the order they are tried. The order matters only when a
#: block has two of them, which no shipped architecture does.
_ATTENTION_NAMES = ("self_attn", "attn", "attention", "self_attention")

#: Names an MLP submodule goes by.
_MLP_NAMES = ("mlp", "feed_forward", "ffn", "feedforward")

#: Names an attention output projection goes by. Checked against the shape test below rather than
#: trusted, so a family that calls it something else is found by shape and a family that reuses one
#: of these names for something else is rejected.
_OUT_PROJ_NAMES = ("o_proj", "out_proj", "dense", "c_proj", "wo")


class ArchitectureError(RuntimeError):
    """The module tree does not look like a decoder stack, with what was looked for.

    Raised rather than returned because this fires at load, before any measurement exists to refuse
    on behalf of. A caller holding a model this cannot navigate has a model the white-box path
    cannot read at all, and that is a programming error rather than an anticipated condition.
    """


def _named_children(module: "nn.Module") -> list[tuple[str, "nn.Module"]]:
    return list(module.named_children())


def _first_named(module: "nn.Module", names: tuple[str, ...]) -> "nn.Module | None":
    for name in names:
        child = getattr(module, name, None)
        if child is not None:
            return child  # type: ignore[no-any-return]
    return None


def _looks_like_block(module: "nn.Module") -> bool:
    """A decoder block has both an attention submodule and an MLP submodule."""
    return (
        _first_named(module, _ATTENTION_NAMES) is not None
        and _first_named(module, _MLP_NAMES) is not None
    )


def _find_block_list(model: "nn.Module") -> tuple[str, Any]:
    """The dotted path and the `nn.ModuleList` holding the decoder blocks.

    Takes the longest qualifying list. A model with a second, shorter stack (a vision tower, a
    draft head) has one stack that is the language model and it is the long one; taking the longest
    is the rule that makes that choice visible instead of depending on traversal order.
    """
    import torch.nn as nn

    best: tuple[str, Any] | None = None
    for name, module in model.named_modules():
        if not isinstance(module, nn.ModuleList) or len(module) == 0:
            continue
        if not all(_looks_like_block(block) for block in module):
            continue
        if best is None or len(module) > len(best[1]):
            best = (name, module)
    if best is None:
        raise ArchitectureError(
            "no decoder stack found. This looks for an nn.ModuleList whose every member has both "
            "an attention submodule (one of "
            f"{list(_ATTENTION_NAMES)}) and an MLP submodule (one of {list(_MLP_NAMES)}), and this "
            "model has none. If the model is a decoder stack under different names, add them to "
            "the two tables at the top of reward_lens/policy/arch.py; if it is not a decoder "
            "stack, it has no layer sites to read and the white-box path does not apply to it."
        )
    return best


def _find_out_projection(attention: "nn.Module", d_model: int, n_heads: int) -> "nn.Module | None":
    """The linear that maps concatenated head outputs back into the residual stream.

    Identified by shape rather than by name: ``out_features == d_model`` and ``in_features`` a
    multiple of the head count. Qwen3 is the case that makes the shape test necessary rather than
    decorative, because its head width (128) is sixteen times its model width (8), so the o_proj is
    `(8, 512)` and any rule phrased as "the widest" or "the last" picks the wrong module.
    """
    import torch.nn as nn

    def qualifies(module: Any) -> bool:
        return (
            isinstance(module, nn.Linear)
            and int(module.out_features) == d_model
            and n_heads > 0
            and int(module.in_features) % n_heads == 0
        )

    named = _first_named(attention, _OUT_PROJ_NAMES)
    if named is not None and qualifies(named):
        return named  # type: ignore[no-any-return]
    for _name, child in _named_children(attention):
        if qualifies(child):
            return child
    return None


@dataclass(frozen=True)
class ArchitectureView:
    """A navigated decoder stack: the sites it exposes and the weights behind them.

    Built once per model by `describe`. Holds no torch modules, only the dotted paths that address
    them, for the same reason `SiteMap` does: a path survives being carried into a provenance record
    and a module object does not.

    ``extract_layer_output`` and its two siblings are the three methods `runtime/hooks.py` calls on
    what it still names an ``adapter``. They are static because none of them needs any per-model
    knowledge: every decoder block, attention submodule and MLP in `transformers` returns either the
    hidden state or a tuple whose first element is the hidden state, and which one it is has changed
    twice between library versions, which is exactly why the unwrap belongs in one place.
    """

    n_layers: int
    d_model: int
    n_heads: int
    #: `Site -> dotted module path`, the same mapping `SiteMap` carries.
    module_paths: dict[Site, str] = field(default_factory=dict)
    #: Dotted path of the block list, kept for error messages that name where the walk landed.
    block_path: str = ""
    #: Sites for which no module was found, named rather than silently absent. A head site missing
    #: here means per-head capture and path patching are unavailable on this model, and an
    #: instrument that needs one should say so rather than fail deep inside a hook.
    unresolved: tuple[Site, ...] = ()

    # -- the SiteMap -------------------------------------------------------

    def site_map(self) -> SiteMap:
        """The `SiteMap` the runtime and every hook mount consult."""
        return SiteMap(
            module_paths=dict(self.module_paths),
            n_layers=self.n_layers,
            d_model=self.d_model,
            n_heads=self.n_heads,
        )

    # -- weight lookup, which is what replaces the reach-through -----------

    def module_at(self, model: "nn.Module", site: Site) -> "nn.Module":
        """The module addressed by a site, resolved through the path table.

        The head index is dropped when resolving ``head_out``: every head in a layer shares one
        output projection, and the per-head slice is a slice of its weight rather than a separate
        module.
        """
        key = Site(site.layer, site.point, None) if site.point == "head_out" else site
        path = self.module_paths.get(key)
        if path is None:
            raise ArchitectureError(
                f"this architecture exposes no module at {key}. Resolved sites are "
                f"{sorted(str(s) for s in self.module_paths)}."
            )
        module: Any = model
        for part in path.split("."):
            module = module[int(part)] if part.isdigit() else getattr(module, part)
        return module  # type: ignore[no-any-return]

    def weight_at(self, model: "nn.Module", site: Site) -> "torch.Tensor":
        """The fp32 weight of the module at a site, sliced to one head where the site names one.

        For ``head_out`` this is the attention output projection restricted to the columns that
        head writes through, `(d_model, d_head)`, which is the object a head-level path patch
        multiplies a captured head output by. That slice used to be computed inside the instrument
        by walking `signal.runtime.adapter.get_layers(signal.runtime.model)[layer]`, four attribute
        hops past the last protocol call, and doing it here is the whole point of this method.
        """
        import torch

        module = self.module_at(model, site)
        weight = getattr(module, "weight", None)
        if weight is None:
            raise ArchitectureError(
                f"the module at {site} has no `weight`. This site addresses "
                f"{type(module).__name__}, which is not a parameterised projection."
            )
        weight = weight.detach().to(torch.float32)
        if site.point != "head_out" or site.head is None:
            return weight  # type: ignore[no-any-return]
        d_head = int(weight.shape[1]) // max(self.n_heads, 1)
        lo = site.head * d_head
        return weight[:, lo : lo + d_head]  # type: ignore[no-any-return]

    # -- the three unwraps `runtime/hooks.py` calls -------------------------

    @staticmethod
    def extract_layer_output(output: Any) -> "torch.Tensor":
        return output[0] if isinstance(output, tuple) else output  # type: ignore[no-any-return]

    @staticmethod
    def extract_attn_output(output: Any) -> "torch.Tensor":
        return output[0] if isinstance(output, tuple) else output  # type: ignore[no-any-return]

    @staticmethod
    def extract_mlp_output(output: Any) -> "torch.Tensor":
        return output[0] if isinstance(output, tuple) else output  # type: ignore[no-any-return]

    # -- what `HFRuntime.forward` asks an adapter for ------------------------

    @staticmethod
    def extract_reward_batch(output: Any, inputs: dict[str, Any]) -> "torch.Tensor | None":
        """A policy has no reward head, so there is no native scalar to extract.

        Returning None rather than raising, because the caller's contract is that a runtime with no
        scalar head reports one and the readout supplies the number. A policy's scalar comes from
        its readout applied to the final residual, which is `PolicySubject.score_under`.
        """
        return None


def describe(model: "nn.Module", *, n_heads: int | None = None) -> ArchitectureView:
    """Navigate a decoder stack and record where everything is.

    ``n_heads`` overrides the config value, which is what a model whose config lies about its head
    count needs. Grouped-query models are the case to watch: ``num_key_value_heads`` is not the
    query head count and slicing an output projection by the wrong one silently mixes two heads
    together, so this reads ``num_attention_heads`` and nothing else.
    """
    config = getattr(model, "config", None)
    block_path, blocks = _find_block_list(model)
    n_layers = len(blocks)

    if n_heads is None:
        n_heads = int(getattr(config, "num_attention_heads", 0) or 0)
    d_model = int(getattr(config, "hidden_size", 0) or getattr(config, "d_model", 0) or 0)

    # `get_input_embeddings` is a `PreTrainedModel` method; the torch stubs resolve it through
    # `nn.Module.__getattr__` and type the result as a tensor, so the Any binding ends there.
    hf_model: Any = model
    embedding = hf_model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    if d_model <= 0 and embedding is not None:
        d_model = int(embedding.weight.shape[-1])

    path_by_id = {id(module): name for name, module in model.named_modules()}
    paths: dict[Site, str] = {}
    unresolved: list[Site] = []

    def record(site: Site, module: Any) -> None:
        if module is None:
            unresolved.append(site)
            return
        path = path_by_id.get(id(module))
        if path is None:
            unresolved.append(site)
            return
        paths[site] = path

    for index, block in enumerate(blocks):
        # A decoder block's own output is the post-block residual; its input is the pre-block one.
        # Both address the same module and the hook direction is what distinguishes them, which is
        # `CaptureMount`'s business and not this table's.
        record(Site(index, "resid_post"), block)
        record(Site(index, "resid_pre"), block)
        attention = _first_named(block, _ATTENTION_NAMES)
        record(Site(index, "attn_out"), attention)
        record(Site(index, "mlp_out"), _first_named(block, _MLP_NAMES))
        out_proj = (
            _find_out_projection(attention, d_model, n_heads) if attention is not None else None
        )
        record(Site(index, "head_out", None), out_proj)

    record(Site(-1, "embed"), embedding)

    return ArchitectureView(
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        module_paths=paths,
        block_path=block_path,
        unresolved=tuple(unresolved),
    )


__all__ = ["ArchitectureError", "ArchitectureView", "describe"]

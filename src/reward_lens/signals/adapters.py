"""The grader-side half of model adaptation: reward heads, gating, capabilities.

The navigation half lives in `policy/arch.py`, which resolves the block list, the token embedding
and the attention output projection **structurally**, by asking where in a module tree the blocks
are rather than which family this is, and that is what retired the eleven-method ``ModelAdapter``
dispatch and its six family subclasses. Its own docstring names what it deliberately left behind,
and this module is that: a reward head is not a navigation question.

Four grader-side facts have no structural answer and cannot be read off a decoder stack:

  - **Where the reward head is, and what it is called.** Five conventions are in the wild and they
    are not interchangeable. ``score`` is the ``AutoModelForSequenceClassification`` default;
    ``regression_layer`` is what a multi-objective model ships; ``v_head`` is the OpenRLHF and
    InternLM2 convention; ``reward_head`` and ``classifier`` appear on ad-hoc heads. Verified on
    real checkpoints: ``Skywork/Skywork-Reward-Llama-3.1-8B`` carries ``score.weight`` (1, 4096),
    ``internlm/internlm2-1_8b-reward`` carries ``v_head.weight`` (1, 2048) and no ``score`` at all,
    and ``RLHFlow/ArmoRM-Llama3-8B-v0.1`` carries neither: it has ``regression_layer.weight``
    (19, 4096).
  - **That ArmoRM's nineteen objectives are gated.** The same checkpoint carries
    ``reward_transform_matrix`` (19, 19), a four-layer ``gating.layers.*`` stack and
    ``gating.logit_scale``. The scalar the model reports is the gate applied to the transformed
    objective vector, and the gate is a function of the prompt. So the row mean of the nineteen
    directions is **not** the model's reward: it is one particular fixed gate, the uniform one, and
    treating it as the model's own is how a multi-objective grader gets silently collapsed to a
    scalar. `is_gated_multi_objective` is what lets a caller tell the difference.
  - **How a scalar comes out of a forward.** Three conventions, and the third is where it bites.
    A sequence classifier returns ``logits`` of shape (B, num_labels) and the reward is column 0.
    ArmoRM returns a custom output with ``.score``. InternLM2's reward model returns ``logits`` of
    shape (B, T, 1), a reward **per token**, and the sequence reward is the last non-pad position.
    Reading column 0 of that is reading the reward of the first token, which for a left-padded batch
    is the reward of a pad.
  - **That Gemma-2 soft-caps its logits.** That one is already handled, and not here:
    `runtime.precision.NumericsPolicy.apply_to_config` nulls ``attn_logit_softcapping`` and
    ``final_logit_softcapping`` on the reward path at load and records what it disabled into
    ``SignalMeta.soft_cap``. `soft_cap_fields` below reports what a config carries so a caller can
    see the cap before the policy touches it; the disabling itself stays in one place.

Navigation is not reimplemented here. `GraderAdapter` holds an `ArchitectureView` and forwards to
it, using `policy.arch`'s own name tables and shape test, so there is exactly one architecture walk
in this library and this is not it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from reward_lens.core.types import Capability, Site
from reward_lens.policy.arch import (
    _ATTENTION_NAMES,
    _MLP_NAMES,
    ArchitectureError,
    ArchitectureView,
    _find_out_projection,
    _first_named,
    describe,
)
from reward_lens.runtime.backend import SiteMap

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

#: Reward head attribute names, in the order they are tried. Order is load-bearing on exactly one
#: pair: a model with both ``score`` and ``regression_layer`` is a multi-objective model whose
#: ``score`` is the gated composite, and the objective rows are the thing an instrument wants, so
#: ``score`` still wins here (it is the head the runtime pre-hooks to capture the head input) and
#: `per_objective_directions` reads the rows separately.
_HEAD_NAMES = ("score", "regression_layer", "v_head", "reward_head", "classifier")

#: Where a learned input-dependent gate over multiple objectives goes by, on the checkpoints that
#: have one. ArmoRM is the only shipped family with a gate; the name is checked rather than the
#: family so that a model that copies the layout is recognised without a new subclass.
_GATING_NAMES = ("gating", "gate", "router")

#: Config fields naming a logit soft cap. Gemma-2 is the only family that sets them, at 50.0 for
#: attention and 30.0 for the final logits on ``Ray2333/GRM-Gemma2-2B-rewardmodel-ft``. Kept in
#: step with ``runtime.precision``'s gemma2 policy, which is what actually nulls them.
_SOFT_CAP_FIELDS = ("attn_logit_softcapping", "final_logit_softcapping")

# The capability every classifier-family grader has: scalar scores, prefix scores, activation
# capture, autograd and its second order, a linear readout. GENERATIVE/PAIRED_MODELS/SPAN_TYPES are
# the judge/implicit/trajectory adapters' business and are not claimed here.
_CLASSIFIER_CAPS = (
    Capability.SCORES
    | Capability.PREFIX_SCORES
    | Capability.ACTIVATIONS
    | Capability.GRADIENTS
    | Capability.HVP
    | Capability.LINEAR_READOUT
)


# ---------------------------------------------------------------------------
# Reading a reward head off a checkpoint
# ---------------------------------------------------------------------------


def reward_head_module(adapter: Any, model: "nn.Module") -> "nn.Module | None":
    """The ``nn.Module`` mapping the final hidden state to the reward, or None.

    ``adapter`` is accepted and ignored. It was a v1 ``ModelAdapter`` and the search never consulted
    it; four call sites in this package already pass ``None``. The parameter stays because those
    call sites are not this module's to edit, and dropping it would be a signature change for no
    behaviour change.

    Searched by name over `_HEAD_NAMES`, and an ``nn.Sequential`` head resolves to its last
    ``nn.Linear``, which is the layer whose weight is the reward direction. Returns None when
    nothing linear is found: a generative judge has an ``lm_head``, not a reward head, and the
    caller falls back to reading the model's native logits. The runtime pre-hooks whatever comes
    back to capture the exact tensor the head consumes, for fp32 scoring, grad and hvp.
    """
    import torch.nn as nn

    for name in _HEAD_NAMES:
        head = getattr(model, name, None)
        if isinstance(head, nn.Linear):
            return head
        if isinstance(head, nn.Sequential):
            for sub in reversed(list(head.modules())):
                if isinstance(sub, nn.Linear):
                    return sub
    return None


def reward_head_name(model: "nn.Module") -> str | None:
    """Which of the five conventions this checkpoint uses, by name.

    Reported rather than inferred because it is the single most useful thing to put in a provenance
    record when a load goes wrong: ``v_head`` says the custom InternLM2 modeling code imported,
    ``regression_layer`` says the head is multi-objective, and None says ``transformers`` fell back
    to a bare backbone and there is no head at all.
    """
    import torch.nn as nn

    for name in _HEAD_NAMES:
        head = getattr(model, name, None)
        if isinstance(head, (nn.Linear, nn.Sequential)):
            return name
    return None


def _reward_head_weight(adapter: Any, model: "nn.Module") -> "torch.Tensor | None":
    head = reward_head_module(adapter, model)
    if head is None:
        return None
    return head.weight.data


def reward_head_params(model: "nn.Module") -> tuple["torch.Tensor", float]:
    """The reward direction ``(d_model,)`` and its bias, in fp32.

    A single-row head gives its row. A multi-row head gives the **row mean**, which is what v1's
    ``get_reward_head_params`` returned and what the v1-shaped callers still expect, and which is
    not the model's own scalar on a gated model. `is_gated_multi_objective` is the check that says
    so, and `per_objective_directions` is what an instrument should read instead.

    Raises ``ArchitectureError`` when no head is found, naming the five conventions searched,
    because a caller asking for a reward direction from a model that has no reward head is holding
    the wrong object rather than hitting an anticipated condition.
    """
    import torch

    head = reward_head_module(None, model)
    if head is None:
        raise ArchitectureError(
            f"{type(model).__name__}: no reward head found. Looked for "
            f"{list(_HEAD_NAMES)} as an nn.Linear or an nn.Sequential ending in one. This usually "
            f"means AutoModelForSequenceClassification fell back to a bare backbone, which happens "
            f"when a checkpoint's custom modeling code did not import; check the load for an "
            f"upstream warning. If this is a generative signal, it has an lm_head rather than a "
            f"reward head and signals.generative is the adapter for it."
        )
    weight = head.weight.data.detach().to(torch.float32)
    if weight.ndim > 1 and weight.shape[0] > 1:
        weight = weight.mean(dim=0)
    weight = weight.reshape(-1)
    bias = getattr(head, "bias", None)
    if bias is None:
        return weight, 0.0
    b = bias.data.detach().to(torch.float32).reshape(-1)
    return weight, float(b.mean().item()) if b.numel() else 0.0


def per_objective_directions(model: "nn.Module") -> "torch.Tensor | None":
    """The rows of a multi-objective head, ``(n_objectives, d_model)`` in fp32, or None.

    On ``RLHFlow/ArmoRM-Llama3-8B-v0.1`` this is ``regression_layer.weight``, nineteen rows of 4096
    covering helpsteer, ultrafeedback, beavertails and the rest. On QRM it is the same shape holding
    quantile rows. Returns None for a single-row head, which is the honest answer to "what are this
    model's objectives" when it has one.
    """
    import torch

    head = reward_head_module(None, model)
    if head is None:
        return None
    weight = head.weight.data
    if weight.ndim < 2 or weight.shape[0] < 2:
        return None
    return weight.detach().to(torch.float32)


def gating_module(model: "nn.Module") -> "nn.Module | None":
    """The learned gate over objectives, when the checkpoint has one.

    ArmoRM's is a four-layer MLP over the prompt's final hidden state, whose output weights the
    nineteen transformed objectives. Its presence is what makes the scalar input-dependent.
    """
    for name in _GATING_NAMES:
        module = getattr(model, name, None)
        if module is not None and hasattr(module, "forward"):
            return module  # type: ignore[no-any-return]
    return None


def is_gated_multi_objective(model: "nn.Module") -> bool:
    """Whether this model's scalar is a learned, input-dependent mix of several objectives.

    True needs both halves: more than one head row, and a gate. Either alone is a different object.
    Nineteen rows with no gate is a model that reports nineteen numbers and leaves the aggregation
    to you (QRM). A gate with one row is a mixture-of-experts backbone with an ordinary head.

    The consequence for a caller is specific: on a True, no fixed vector reproduces the model's own
    score, so a linear readout is an approximation whose error varies by prompt, and a study that
    needs the model's actual scalar has to read it from the forward rather than from the head
    weight. `reward_head_params`' row mean is the uniform gate, which is one point in that family
    and not the model's.
    """
    directions = per_objective_directions(model)
    return directions is not None and gating_module(model) is not None


def soft_cap_fields(model: "nn.Module") -> dict[str, float]:
    """Soft-cap config fields this model sets, and their values, before anything nulls them.

    Gemma-2 is the family that sets them: ``Ray2333/GRM-Gemma2-2B-rewardmodel-ft`` carries
    ``attn_logit_softcapping`` 50.0 and ``final_logit_softcapping`` 30.0. Reporting only.
    `runtime.precision.NumericsPolicy.apply_to_config` is what disables them on the reward path, and
    it records the originals into ``SignalMeta.soft_cap``, so this is here to let a caller see the
    cap rather than to give it a second place to be turned off.
    """
    config = getattr(model, "config", None)
    if config is None:
        return {}
    found: dict[str, float] = {}
    for field_name in _SOFT_CAP_FIELDS:
        value = getattr(config, field_name, None)
        if value is not None:
            found[field_name] = float(value)
    return found


def final_positions(attention_mask: "torch.Tensor") -> "torch.Tensor":
    """The last valid (non-pad) token index per row, for left- or right-padding alike.

    ``(arange * mask).argmax(dim=1)``, which is the largest index carrying a 1. The obvious
    alternative, ``mask.sum(dim=1) - 1``, counts valid tokens and is only the last one's index when
    the padding is on the right. v1's ``InternLM2Adapter.extract_reward`` used that form while both
    v1's own batch path (``model.py``, which sets ``padding_side = "left"``) and ``HFRuntime.collate``
    left-pad, so on any row shorter than the batch maximum it read a position inside the pad region
    and returned the reward of a pad token. This is the same rule ``HFRuntime._final_positions``
    uses, and it lives here so there is one of it.
    """
    import torch

    seq_len = attention_mask.shape[1]
    idx = torch.arange(seq_len, device=attention_mask.device)
    return (idx.unsqueeze(0) * attention_mask.to(torch.long)).argmax(dim=1)


def extract_reward_batch(
    output: Any, inputs: dict[str, Any] | None = None
) -> "torch.Tensor | None":
    """One reward per batch row from a forward output, or None if there is no scalar head.

    The three conventions, tried in the order that makes each unambiguous:

      1. ``output.score``: ArmoRM's custom output. Checked first because such a model also has
         ``logits``, and its logits are the objective rows rather than the reward.
      2. ``output.logits`` with three dimensions: a per-token reward, (B, T, 1). This is the
         InternLM2 reward convention, and the sequence reward is the last non-pad position, resolved
         from ``inputs['attention_mask']`` by `final_positions`. Without a mask this falls back to
         the final column, which is correct for left-padded and unpadded batches (the runtime
         left-pads, so the response end sits at ``T-1`` for every row) and wrong for right-padded
         ones. Pass the mask.
      3. ``output.logits`` with two dimensions: (B, num_labels), reward in column 0. The
         ``AutoModelForSequenceClassification`` default.

    Returns None rather than raising when none of the three applies, because a policy has no reward
    head and "this runtime reports no native scalar" is a fact its caller acts on: the readout
    supplies the number instead. Detached and cast to fp32, because the trunk may be bf16 and this
    value is used for parity checks against the fp32 readout.
    """
    import torch

    score = getattr(output, "score", None)
    if score is not None and isinstance(score, torch.Tensor):
        score = score.detach().float()
        return score.squeeze(-1) if score.ndim > 1 else score

    logits = getattr(output, "logits", None)
    if logits is None or not isinstance(logits, torch.Tensor):
        return None
    if logits.ndim == 3:
        mask = None if inputs is None else inputs.get("attention_mask")
        if mask is not None:
            pos = final_positions(mask).to(logits.device)
            rows = torch.arange(logits.shape[0], device=logits.device)
            return logits[rows, pos, 0].detach().float()
        return logits[:, -1, 0].detach().float()
    if logits.ndim == 2:
        return logits[:, 0].detach().float()
    return logits.reshape(logits.shape[0], -1)[:, 0].detach().float()


# ---------------------------------------------------------------------------
# The adapter object the runtime holds
# ---------------------------------------------------------------------------


class GraderAdapter:
    """What a grader's runtime holds: a navigated architecture plus its reward head.

    Constructed by `resolve_adapter`, which walks the module tree once through
    `policy.arch.describe` and keeps the resulting `ArchitectureView`. The navigation methods below
    forward to that view and to `policy.arch`'s own name tables; none of them re-derives anything.

    The method names are v1's, deliberately. ``runtime/hooks.py``, ``runtime/hf.py``,
    ``interventions/patch.py``, ``signals/_common.py`` and two acceptance tests call an object they
    still name ``adapter`` through those names, and changing the vocabulary and the implementation
    in one step would make a regression impossible to localise. What changed is that behind the
    names there is one structural walk instead of six family subclasses, and that the head-reading
    methods answer from the checkpoint rather than from a class the dispatch guessed.
    """

    def __init__(
        self,
        view: ArchitectureView,
        model_name: str = "",
        head_name: str | None = None,
        head_rows: int = 0,
        gated: bool = False,
    ):
        self.view = view
        self.model_name = model_name
        #: Which of the five conventions the head goes by, or None when there is no head.
        self.head_name = head_name
        #: Rows in the reward head. 1 for a scalar grader, 19 on ArmoRM and QRM, 0 for no head.
        #: Read once at construction so `capabilities_for` has one source for the fact and does not
        #: have to be handed the model again at every call site.
        self.head_rows = head_rows
        #: Whether those rows are combined by a learned input-dependent gate (ArmoRM).
        self.gated = gated

    def __repr__(self) -> str:
        return (
            f"GraderAdapter(n_layers={self.view.n_layers}, d_model={self.view.d_model}, "
            f"n_heads={self.view.n_heads}, head={self.head_name!r}, rows={self.head_rows}, "
            f"gated={self.gated}, model={self.model_name!r})"
        )

    # -- navigation, forwarded to the one architecture walk ------------------

    def get_layers(self, model: "nn.Module") -> Any:
        module: Any = model
        for part in self.view.block_path.split("."):
            module = module[int(part)] if part.isdigit() else getattr(module, part)
        return module

    def n_layers(self, model: "nn.Module") -> int:
        return self.view.n_layers

    def n_heads(self, model: "nn.Module") -> int:
        return self.view.n_heads

    def get_attn_module(self, layer: "nn.Module") -> "nn.Module | None":
        return _first_named(layer, _ATTENTION_NAMES)

    def get_mlp_module(self, layer: "nn.Module") -> "nn.Module | None":
        return _first_named(layer, _MLP_NAMES)

    def get_attn_o_proj(self, layer: "nn.Module") -> "nn.Module | None":
        attention = _first_named(layer, _ATTENTION_NAMES)
        if attention is None:
            return None
        return _find_out_projection(attention, self.view.d_model, self.view.n_heads)

    def get_embedding(self, model: "nn.Module") -> "nn.Module":
        embedding = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
        if embedding is None:
            raise ArchitectureError(
                f"{type(model).__name__} exposes no get_input_embeddings(), which is the "
                f"transformers API this reads the token embedding through. A model that does not "
                f"implement it is not a PreTrainedModel and has no embed site."
            )
        return embedding  # type: ignore[no-any-return]

    extract_layer_output = staticmethod(ArchitectureView.extract_layer_output)
    extract_attn_output = staticmethod(ArchitectureView.extract_attn_output)
    extract_mlp_output = staticmethod(ArchitectureView.extract_mlp_output)

    # -- the grader-side half ------------------------------------------------

    def reward_head_module(self, model: "nn.Module") -> "nn.Module | None":
        return reward_head_module(None, model)

    def reward_head_name(self, model: "nn.Module") -> str | None:
        return reward_head_name(model)

    def get_reward_head_params(self, model: "nn.Module") -> tuple["torch.Tensor", float]:
        return reward_head_params(model)

    def per_objective_directions(self, model: "nn.Module") -> "torch.Tensor | None":
        return per_objective_directions(model)

    def is_gated_multi_objective(self, model: "nn.Module") -> bool:
        return is_gated_multi_objective(model)

    def soft_cap_fields(self, model: "nn.Module") -> dict[str, float]:
        return soft_cap_fields(model)

    def extract_reward(self, output: Any, inputs: dict[str, Any]) -> "torch.Tensor | None":
        """The scalar reward of the first row, for the single-input v1-shaped callers."""
        batch = extract_reward_batch(output, inputs)
        return None if batch is None else batch.reshape(-1)[0]

    def extract_reward_batch(
        self, output: Any, inputs: dict[str, Any] | None = None
    ) -> "torch.Tensor | None":
        return extract_reward_batch(output, inputs)


def resolve_adapter(model: "nn.Module", model_name: str = "") -> GraderAdapter:
    """Navigate a model once and return the adapter its runtime holds.

    There is no family dispatch. v1 asked "does this name contain 'armorm'?" and picked a subclass,
    which meant an unrecognised family silently became `GenericAdapter` and a family whose name did
    not appear in the string got the wrong one. This walks the tree and reads the head, so what
    comes back describes the checkpoint in hand. ``model_name`` is kept for provenance only.
    """
    head = reward_head_module(None, model)
    weight = None if head is None else head.weight
    rows = 0 if weight is None else (1 if weight.ndim < 2 else int(weight.shape[0]))
    return GraderAdapter(
        describe(model),
        model_name=model_name,
        head_name=reward_head_name(model),
        head_rows=rows,
        gated=rows > 1 and gating_module(model) is not None,
    )


# ---------------------------------------------------------------------------
# Capabilities and the site map
# ---------------------------------------------------------------------------


def capabilities_for(adapter: Any, model: "nn.Module | None" = None) -> Capability:
    """The declared ``Capability`` set for a grader.

    ``MULTI_READOUT`` is declared when the head has more than one row, read off the checkpoint. v1
    declared it for `ArmoRMAdapter` by ``isinstance`` and nothing else, which left QRM inconsistent
    with itself: `get_adapter` routed ``"qrm"`` to `LlamaAdapter`, so ``capabilities_for`` withheld
    ``MULTI_READOUT`` while `is_multi_readout` read the same checkpoint's nineteen rows and said
    True. An Observable declaring ``MULTI_READOUT`` was then refused on a signal that had nineteen
    readouts. Reading the row count fixes that by not having two sources for one fact.

    ``model`` is optional. A `GraderAdapter` read its row count when it navigated the checkpoint, so
    the one-argument call that ``signals/classifier.py`` makes is answered from the adapter. Passing
    a model overrides that, which is what a caller holding a v1 adapter needs.
    """
    if model is not None:
        rows = 0 if per_objective_directions(model) is None else 2
    else:
        rows = int(getattr(adapter, "head_rows", 1) or 1)
    return _CLASSIFIER_CAPS | Capability.MULTI_READOUT if rows > 1 else _CLASSIFIER_CAPS


def is_multi_readout(adapter: Any, model: "nn.Module") -> bool:
    """Whether this signal exposes multiple readout rows (a non-scalar reward head).

    True for ArmoRM's nineteen objectives and for QRM's nineteen quantile rows alike. The row count
    is read off the checkpoint, so a multi-objective model is never silently collapsed to a row
    mean.
    """
    return per_objective_directions(model) is not None


def build_site_map(adapter: Any, model: "nn.Module") -> SiteMap:
    """Resolve every logical ``Site`` this architecture exposes to a module path.

    A `GraderAdapter` already carries the walk, so this returns its view's site map. Anything else
    is a v1 adapter, and the walk runs through `policy.arch.describe` on the model instead: there is
    no second traversal in this library and this function is not going to add one.

    The mapping is: ``Site(L, "resid_post")`` and ``Site(L, "resid_pre")`` to the decoder block at
    layer L (a forward hook reads the block output as resid_post, a pre-hook reads its input as
    resid_pre); ``Site(L, "attn_out")`` to the attention sublayer; ``Site(L, "mlp_out")`` to the
    MLP; ``Site(L, "head_out")`` to the attention output projection, whose input is the concatenated
    per-head outputs the runtime slices by head; and ``Site(-1, "embed")`` to the token embedding.

    One deliberate difference from the v1 builder it replaces: ``d_model`` comes from the config and
    the embedding rather than from the reward head's weight. Reading it off the head made the site
    map unbuildable for any model without one, which is every generative judge, and is why
    ``signals/_common.py`` had to carry a near-copy of this function that took ``d_model``
    explicitly. It no longer needs to.
    """
    view = getattr(adapter, "view", None)
    if isinstance(view, ArchitectureView):
        return view.site_map()
    return describe(model).site_map()


__all__ = [
    "ArchitectureError",
    "GraderAdapter",
    "Site",
    "build_site_map",
    "capabilities_for",
    "extract_reward_batch",
    "final_positions",
    "gating_module",
    "is_gated_multi_objective",
    "is_multi_readout",
    "per_objective_directions",
    "resolve_adapter",
    "reward_head_module",
    "reward_head_name",
    "reward_head_params",
    "soft_cap_fields",
]

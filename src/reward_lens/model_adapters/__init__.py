"""
Model Adapters — architecture-specific logic for different reward model families.

Design decision: Rather than maintaining a model zoo (like TransformerLens), we use
a lightweight adapter pattern. Each adapter knows how to navigate a specific model
architecture:
    - Where are the transformer layers?
    - Where is the attention sublayer? The MLP sublayer?
    - How is the reward head structured?
    - How are outputs formatted?

This means supporting a new model family requires writing ~50 lines of adapter code,
not modifying the core library. And any model that uses AutoModelForSequenceClassification
with a standard architecture gets auto-detected.

Supported families:
    - LlamaForSequenceClassification (Skywork, FsfairX, etc.)
    - ArmoRM (multi-objective with MoE gating — custom trust_remote_code model)
    - MistralForSequenceClassification
    - Gemma2ForSequenceClassification
    - Generic fallback for other AutoModelForSequenceClassification models
"""

from __future__ import annotations

from reward_lens.core.extras import require_extra

require_extra("white-box", subsystem="reward_lens.model_adapters")

from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
import torch.nn as nn


class ModelAdapter(ABC):
    """Abstract base for model architecture adapters.

    An adapter tells reward-lens how to navigate a specific model's module tree.
    Every adapter must implement:
        - get_reward_head_params: extract reward direction and bias
        - get_layers: return the list of transformer layers
        - n_layers, n_heads: architecture parameters
        - get_attn_module, get_mlp_module: sublayer access
        - extract_layer_output: parse layer forward output into hidden states
        - extract_attn_output, extract_mlp_output: parse sublayer outputs
        - get_embedding: the embedding module
        - extract_reward: parse model output into a scalar reward
    """

    @abstractmethod
    def get_reward_head_params(self, model: nn.Module) -> tuple[torch.Tensor, float]:
        """Extract the reward head weight vector and bias.

        Returns:
            Tuple of (weight_vector, bias) where weight_vector has shape (d_model,)
            and bias is a scalar float.
        """
        ...

    @abstractmethod
    def get_layers(self, model: nn.Module) -> nn.ModuleList:
        """Return the list of transformer layers (decoder blocks)."""
        ...

    @abstractmethod
    def n_layers(self, model: nn.Module) -> int:
        """Number of transformer layers."""
        ...

    @abstractmethod
    def n_heads(self, model: nn.Module) -> int:
        """Number of attention heads."""
        ...

    @abstractmethod
    def get_attn_module(self, layer: nn.Module) -> Optional[nn.Module]:
        """Return the attention sublayer module from a transformer layer."""
        ...

    @abstractmethod
    def get_mlp_module(self, layer: nn.Module) -> Optional[nn.Module]:
        """Return the MLP sublayer module from a transformer layer."""
        ...

    @abstractmethod
    def extract_layer_output(self, output: Any) -> torch.Tensor:
        """Extract hidden states from a layer's forward output.

        Different architectures return different formats:
        - Some return a tuple (hidden_states, ...)
        - Some return a named tuple or dataclass
        - Some return just the tensor

        Returns:
            Tensor of shape (batch, seq_len, d_model).
        """
        ...

    @abstractmethod
    def extract_attn_output(self, output: Any) -> torch.Tensor:
        """Extract hidden states from attention sublayer output."""
        ...

    @abstractmethod
    def extract_mlp_output(self, output: Any) -> torch.Tensor:
        """Extract hidden states from MLP sublayer output."""
        ...

    @abstractmethod
    def get_embedding(self, model: nn.Module) -> nn.Module:
        """Return the token embedding module."""
        ...

    @abstractmethod
    def extract_reward(self, output: Any, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract the scalar reward from the model's forward output.

        Returns:
            Scalar tensor with the reward value.
        """
        ...

    # --- Optional capabilities ---------------------------------------------
    # Adapters MAY override these to enable head-level analysis and batched
    # reward extraction. The defaults are conservative and let the caller
    # detect (via hasattr / try-fallback) whether the adapter supports the
    # given operation.

    def get_attn_o_proj(self, layer: nn.Module) -> Optional[nn.Module]:
        """Return the attention output-projection (o_proj) module.

        Hooking this module's *input* is the cleanest way to access per-head
        attention outputs: the input has shape (B, T, n_heads * d_head),
        which reshapes to (B, T, n_heads, d_head). Works uniformly for
        standard MHA and grouped-query attention because the per-head reshape
        happens before o_proj.

        Returns None if the architecture doesn't expose o_proj cleanly
        (e.g. fused QKVO modules); the caller falls back to sublayer-level
        analysis in that case.
        """
        return None

    def extract_reward_batch(self, output: Any, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Vectorised reward extraction for batched forward passes.

        Default implementation tries `output.logits[:, 0]` and `output.score`.
        Override for adapters whose output schema differs.

        Returns:
            1-D tensor of shape (B,) with one reward per batch row.
        """
        if hasattr(output, "logits"):
            return output.logits[:, 0].detach().float()
        if hasattr(output, "score"):
            score = output.score.detach().float()
            return score.squeeze(-1) if score.ndim > 1 else score
        raise ValueError(f"{type(self).__name__}: cannot extract batched reward from output")


class LlamaAdapter(ModelAdapter):
    """Adapter for Llama-based reward models (Skywork, FsfairX, etc.).

    Architecture:
        model.model.embed_tokens -> model.model.layers[i] -> model.model.norm -> model.score
        Each layer: self_attn, mlp, input_layernorm, post_attention_layernorm
    """

    def get_reward_head_params(self, model: nn.Module) -> tuple[torch.Tensor, float]:
        # Bug fix (deep_analysisv1): models like QRM-Llama3.1-8B ship a
        # ``regression_layer`` instead of ``score`` for their reward head.
        # When we land in a Llama branch with no ``model.score`` attribute
        # (e.g. because AutoModelForSequenceClassification fell back to a
        # bare LlamaModel and the regression_layer was loaded onto the
        # wrapping object), look for it before raising.
        score_module = getattr(model, "score", None)
        if score_module is None:
            score_module = getattr(model, "regression_layer", None)
        if score_module is None:
            v_head = getattr(model, "v_head", None)
            if v_head is not None and isinstance(v_head, nn.Linear):
                score_module = v_head
        if score_module is None:
            raise AttributeError(
                f"{type(model).__name__}: cannot find reward head — looked for "
                f"'score', 'regression_layer', 'v_head' on the top-level module. "
                f"This usually means AutoModelForSequenceClassification fell back "
                f"to a bare backbone; check that any custom modeling code from "
                f"the Hub imported successfully."
            )
        weight = score_module.weight.data.squeeze().float()
        if weight.ndim > 1:
            # Multi-objective head (e.g. QRM has 19 objectives) — collapse to
            # the row-mean as the "aggregate" reward direction. Fine-grained
            # per-objective analysis should use the architecture-specific
            # adapter (e.g. ArmoRMAdapter) instead.
            weight = weight.mean(dim=0).float()
        bias_val = 0.0
        if getattr(score_module, "bias", None) is not None:
            b = score_module.bias.data.float()
            bias_val = float(b.mean().item()) if b.numel() > 1 else float(b.item())
        return weight, bias_val

    def get_layers(self, model: nn.Module) -> nn.ModuleList:
        # Some custom-loaded models nest the layers as model.model.model.layers
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers
        if (
            hasattr(model, "model")
            and hasattr(model.model, "model")
            and hasattr(model.model.model, "layers")
        ):
            return model.model.model.layers
        if hasattr(model, "layers"):
            return model.layers
        raise AttributeError(
            f"{type(model).__name__}: cannot locate decoder layers — looked for "
            f"model.model.layers, model.model.model.layers, model.layers."
        )

    def n_layers(self, model: nn.Module) -> int:
        return len(self.get_layers(model))

    def n_heads(self, model: nn.Module) -> int:
        return model.config.num_attention_heads

    def get_attn_module(self, layer: nn.Module) -> Optional[nn.Module]:
        return getattr(layer, "self_attn", None)

    def get_mlp_module(self, layer: nn.Module) -> Optional[nn.Module]:
        return getattr(layer, "mlp", None)

    def extract_layer_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, tuple):
            return output[0]
        return output

    def extract_attn_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, tuple):
            return output[0]
        return output

    def extract_mlp_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, tuple):
            return output[0]
        return output

    def get_embedding(self, model: nn.Module) -> nn.Module:
        return model.model.embed_tokens

    def extract_reward(self, output: Any, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        # AutoModelForSequenceClassification stores logits in output.logits.
        # Shape is (batch, num_labels) for standard models, but some return
        # (batch, seq, num_labels) — squeeze to a scalar in both cases.
        logits = output.logits
        return logits[0, 0].squeeze()

    def get_attn_o_proj(self, layer: nn.Module) -> Optional[nn.Module]:
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            return None
        # Llama (incl. GQA Llama-3.x) uses self_attn.o_proj as the output
        # projection. Its input is the concatenated per-head attention
        # outputs, shape (B, T, n_heads * d_head).
        return getattr(attn, "o_proj", None)


class MistralAdapter(LlamaAdapter):
    """Adapter for Mistral-based reward models.

    Architecturally identical to Llama for our purposes.
    """

    pass


class Gemma2Adapter(ModelAdapter):
    """Adapter for Gemma-2-based reward models (Skywork-Reward-Gemma-2-27B).

    Architecture:
        model.model.embed_tokens -> model.model.layers[i] -> model.model.norm -> model.score
        Very similar to Llama but with sliding window attention and different
        layer normalization patterns.
    """

    def get_reward_head_params(self, model: nn.Module) -> tuple[torch.Tensor, float]:
        score_module = model.score
        weight = score_module.weight.data.squeeze().float()
        bias = score_module.bias.data.item() if score_module.bias is not None else 0.0
        return weight, bias

    def get_layers(self, model: nn.Module) -> nn.ModuleList:
        return model.model.layers

    def n_layers(self, model: nn.Module) -> int:
        return len(model.model.layers)

    def n_heads(self, model: nn.Module) -> int:
        return model.config.num_attention_heads

    def get_attn_module(self, layer: nn.Module) -> Optional[nn.Module]:
        return getattr(layer, "self_attn", None)

    def get_mlp_module(self, layer: nn.Module) -> Optional[nn.Module]:
        return getattr(layer, "mlp", None)

    def extract_layer_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, tuple):
            return output[0]
        return output

    def extract_attn_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, tuple):
            return output[0]
        return output

    def extract_mlp_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, tuple):
            return output[0]
        return output

    def get_embedding(self, model: nn.Module) -> nn.Module:
        return model.model.embed_tokens

    def extract_reward(self, output: Any, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        logits = output.logits
        return logits[0, 0]

    def get_attn_o_proj(self, layer: nn.Module) -> Optional[nn.Module]:
        # Gemma2 self_attn also exposes o_proj.
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            return None
        return getattr(attn, "o_proj", None)


class ArmoRMAdapter(ModelAdapter):
    """Adapter for ArmoRM-Llama3-8B-v0.1 (multi-objective reward model).

    Architecture:
        This model uses trust_remote_code=True and has a custom architecture:
        - Base: Llama-3-8B backbone
        - 19 reward objectives (helpsteer, ultrafeedback, beavertails, etc.)
        - MoE gating layer conditioned on the prompt
        - reward_transform_matrix for debiasing
        - Final score = sum(gating_weights @ transform_matrix * multi_obj_rewards)

    For interpretability, we treat the multi-objective head differently than
    single-scalar models. The "reward direction" in the simple sense doesn't
    exist — instead, there are 19 objective directions, combined via learned
    gating weights that depend on the input.

    We provide two modes:
        1. Aggregate mode: treat the final scalar score as the target
        2. Per-objective mode: analyze each objective independently
    """

    def get_reward_head_params(self, model: nn.Module) -> tuple[torch.Tensor, float]:
        # ArmoRM doesn't have a simple single reward head.
        # We approximate by using the regression layer weights averaged by
        # typical gating coefficients. This is an approximation — for precise
        # analysis, use per_objective_directions().
        if hasattr(model, "regression_layer"):
            # regression_layer maps hidden states to 19 objective scores
            weight = model.regression_layer.weight.data.float()  # (19, d_model)
            # Use equal weighting as default approximation
            avg_direction = weight.mean(dim=0)  # (d_model,)
            bias = (
                model.regression_layer.bias.data.float().mean().item()
                if model.regression_layer.bias is not None
                else 0.0
            )
            return avg_direction, bias
        # Fallback: try standard score head
        score_module = getattr(model, "score", None)
        if score_module is not None:
            weight = score_module.weight.data.squeeze().float()
            bias = score_module.bias.data.item() if score_module.bias is not None else 0.0
            return weight, bias
        raise ValueError("Cannot find reward head in ArmoRM model")

    def per_objective_directions(self, model: nn.Module) -> torch.Tensor:
        """Get the 19 per-objective reward directions.

        Returns:
            Tensor of shape (19, d_model) — one direction per objective.
        """
        if hasattr(model, "regression_layer"):
            return model.regression_layer.weight.data.float()
        raise ValueError("No regression_layer found in model")

    def get_layers(self, model: nn.Module) -> nn.ModuleList:
        # ArmoRM wraps a Llama model
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers
        if hasattr(model, "model") and hasattr(model.model, "model"):
            return model.model.model.layers
        raise ValueError("Cannot find layers in ArmoRM model")

    def n_layers(self, model: nn.Module) -> int:
        return len(self.get_layers(model))

    def n_heads(self, model: nn.Module) -> int:
        return model.config.num_attention_heads

    def get_attn_module(self, layer: nn.Module) -> Optional[nn.Module]:
        return getattr(layer, "self_attn", None)

    def get_mlp_module(self, layer: nn.Module) -> Optional[nn.Module]:
        return getattr(layer, "mlp", None)

    def extract_layer_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, tuple):
            return output[0]
        return output

    def extract_attn_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, tuple):
            return output[0]
        return output

    def extract_mlp_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, tuple):
            return output[0]
        return output

    def get_embedding(self, model: nn.Module) -> nn.Module:
        if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            return model.model.embed_tokens
        if hasattr(model, "model") and hasattr(model.model, "model"):
            return model.model.model.embed_tokens
        raise ValueError("Cannot find embedding in ArmoRM model")

    def extract_reward(self, output: Any, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        # ArmoRM outputs have a .score attribute
        if hasattr(output, "score"):
            return output.score.float().squeeze()
        if hasattr(output, "logits"):
            return output.logits[0, 0]
        raise ValueError("Cannot extract reward from ArmoRM output")

    def get_attn_o_proj(self, layer: nn.Module) -> Optional[nn.Module]:
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            return None
        return getattr(attn, "o_proj", None)

    def extract_reward_batch(self, output: Any, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        if hasattr(output, "score"):
            score = output.score.detach().float()
            # ArmoRM .score may be (B, 1) or (B,)
            return score.squeeze(-1) if score.ndim > 1 else score
        if hasattr(output, "logits"):
            return output.logits[:, 0].detach().float()
        raise ValueError("Cannot extract batched ArmoRM reward")


class InternLM2Adapter(ModelAdapter):
    """Adapter for internlm/internlm2-*-reward.

    The InternLM2 reward model ships a custom modeling class
    (``InternLM2ForRewardModel``) loaded via ``trust_remote_code=True``.
    Its config uses ``model_type='internlm2'`` which is not registered with
    AutoModelForSequenceClassification, so the standard load path either
    returns the bare ``InternLM2Model`` (no reward head) or — if the
    custom modeling code imported successfully — an ``InternLM2ForRewardModel``
    with a ``v_head`` (linear, d_model->1) on top of the backbone.

    This adapter handles both layouts.
    """

    def get_reward_head_params(self, model: nn.Module) -> tuple[torch.Tensor, float]:
        for name in ("v_head", "score", "reward_head", "regression_layer"):
            head = getattr(model, name, None)
            if isinstance(head, nn.Linear):
                w = head.weight.data.squeeze().float()
                if w.ndim > 1:
                    w = w.mean(dim=0)
                b = head.bias.data.float() if head.bias is not None else None
                bias_val = float(b.mean().item()) if b is not None and b.numel() > 0 else 0.0
                return w, bias_val
        raise AttributeError(
            f"{type(model).__name__}: no v_head/score/reward_head linear found. "
            f"This usually means the custom InternLM2 modeling code failed to "
            f"import (look for an upstream warning), causing transformers to "
            f"fall back to AutoModel which loads only the backbone."
        )

    def get_layers(self, model: nn.Module) -> nn.ModuleList:
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers
        if hasattr(model, "layers"):
            return model.layers
        raise AttributeError("InternLM2: cannot find layers")

    def n_layers(self, model: nn.Module) -> int:
        return len(self.get_layers(model))

    def n_heads(self, model: nn.Module) -> int:
        return model.config.num_attention_heads

    def get_attn_module(self, layer: nn.Module) -> Optional[nn.Module]:
        return getattr(layer, "attention", None) or getattr(layer, "self_attn", None)

    def get_mlp_module(self, layer: nn.Module) -> Optional[nn.Module]:
        return getattr(layer, "feed_forward", None) or getattr(layer, "mlp", None)

    def extract_layer_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, tuple):
            return output[0]
        return output

    def extract_attn_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, tuple):
            return output[0]
        return output

    def extract_mlp_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, tuple):
            return output[0]
        return output

    def get_embedding(self, model: nn.Module) -> nn.Module:
        if hasattr(model, "model") and hasattr(model.model, "tok_embeddings"):
            return model.model.tok_embeddings
        if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            return model.model.embed_tokens
        raise AttributeError("InternLM2: cannot find embedding")

    def extract_reward(self, output: Any, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        # InternLM2ForRewardModel returns a CausalLMOutputWithPast where
        # .logits has shape (B, T, 1) — the per-token reward. The standard
        # convention is to take the last non-pad position.
        if hasattr(output, "logits"):
            logits = output.logits
            if logits.ndim == 3:
                attn_mask = inputs.get("attention_mask")
                if attn_mask is not None:
                    pos = attn_mask.sum(dim=1) - 1  # (B,)
                    return logits[
                        torch.arange(logits.shape[0], device=logits.device), pos, 0
                    ].squeeze()
                return logits[:, -1, 0].squeeze()
            return logits[0, 0]
        if hasattr(output, "score"):
            return output.score.float().squeeze()
        raise ValueError("InternLM2: cannot extract reward from output")


class GenericAdapter(ModelAdapter):
    """Fallback adapter for any AutoModelForSequenceClassification model.

    This tries to detect the architecture by inspecting the module tree.
    It handles the most common patterns but may not work for highly
    unusual architectures.
    """

    def __init__(self, model: nn.Module):
        self._backbone_path, self._layers_path = self._detect_architecture(model)

    def _detect_architecture(self, model: nn.Module) -> tuple[str, str]:
        """Walk the module tree to find the backbone and layers."""
        # Common patterns:
        # model.model.layers (Llama, Mistral, Gemma)
        # model.transformer.h (GPT-2, GPT-Neo)
        # model.roberta.encoder.layer (RoBERTa)
        for backbone_name in ["model", "transformer", "roberta", "bert", "gpt_neox"]:
            backbone = getattr(model, backbone_name, None)
            if backbone is None:
                continue
            for layers_name in ["layers", "h", "layer"]:
                layers = getattr(backbone, layers_name, None)
                if layers is not None and isinstance(layers, nn.ModuleList):
                    return backbone_name, layers_name
            # Check one level deeper (e.g., model.model.layers)
            if hasattr(backbone, "model"):
                inner = backbone.model
                for layers_name in ["layers", "h", "layer"]:
                    layers = getattr(inner, layers_name, None)
                    if layers is not None and isinstance(layers, nn.ModuleList):
                        return f"{backbone_name}.model", layers_name
            if hasattr(backbone, "encoder"):
                encoder = backbone.encoder
                for layers_name in ["layer", "layers"]:
                    layers = getattr(encoder, layers_name, None)
                    if layers is not None and isinstance(layers, nn.ModuleList):
                        return f"{backbone_name}.encoder", layers_name
        raise ValueError(
            f"Cannot auto-detect architecture for {type(model).__name__}. "
            "Please implement a custom adapter."
        )

    def _get_backbone(self, model: nn.Module) -> nn.Module:
        module = model
        for part in self._backbone_path.split("."):
            module = getattr(module, part)
        return module

    def get_reward_head_params(self, model: nn.Module) -> tuple[torch.Tensor, float]:
        # Try common reward head names
        for name in ["score", "classifier", "v_head", "reward_head"]:
            head = getattr(model, name, None)
            if head is not None:
                if isinstance(head, nn.Linear):
                    weight = head.weight.data.squeeze().float()
                    bias = head.bias.data.item() if head.bias is not None else 0.0
                    return weight, bias
                elif isinstance(head, nn.Sequential):
                    # Last linear layer
                    for m in reversed(list(head.modules())):
                        if isinstance(m, nn.Linear):
                            weight = m.weight.data.squeeze().float()
                            bias = m.bias.data.item() if m.bias is not None else 0.0
                            return weight, bias
        raise ValueError(
            "Cannot find reward head. Expected one of: score, classifier, v_head, reward_head. "
            "Please implement a custom adapter."
        )

    def get_layers(self, model: nn.Module) -> nn.ModuleList:
        backbone = self._get_backbone(model)
        return getattr(backbone, self._layers_path)

    def n_layers(self, model: nn.Module) -> int:
        return len(self.get_layers(model))

    def n_heads(self, model: nn.Module) -> int:
        config = model.config
        for attr in ["num_attention_heads", "n_head", "num_heads"]:
            if hasattr(config, attr):
                return getattr(config, attr)
        raise ValueError("Cannot determine number of attention heads from config")

    def get_attn_module(self, layer: nn.Module) -> Optional[nn.Module]:
        for name in ["self_attn", "attn", "attention", "self_attention"]:
            module = getattr(layer, name, None)
            if module is not None:
                return module
        return None

    def get_mlp_module(self, layer: nn.Module) -> Optional[nn.Module]:
        for name in ["mlp", "feed_forward", "ffn", "ff"]:
            module = getattr(layer, name, None)
            if module is not None:
                return module
        return None

    def extract_layer_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, tuple):
            return output[0]
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        return output[0]

    def extract_attn_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, tuple):
            return output[0]
        return output

    def extract_mlp_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, tuple):
            return output[0]
        return output

    def get_embedding(self, model: nn.Module) -> nn.Module:
        backbone = self._get_backbone(model)
        for name in ["embed_tokens", "wte", "word_embeddings", "embed_in"]:
            emb = getattr(backbone, name, None)
            if emb is not None:
                return emb
        raise ValueError("Cannot find embedding module")

    def extract_reward(self, output: Any, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        if hasattr(output, "logits"):
            return output.logits[0, 0]
        if hasattr(output, "score"):
            return output.score.float().squeeze()
        raise ValueError("Cannot extract reward from model output")

    def get_attn_o_proj(self, layer: nn.Module) -> Optional[nn.Module]:
        attn = self.get_attn_module(layer)
        if attn is None:
            return None
        # Common output-projection names across HF families.
        for name in ("o_proj", "out_proj", "dense", "wo", "c_proj"):
            mod = getattr(attn, name, None)
            if isinstance(mod, nn.Linear):
                return mod
        return None


def get_adapter(model: nn.Module, model_name: str = "") -> ModelAdapter:
    """Auto-detect and return the appropriate adapter for a model.

    The detection logic:
    1. Check if the model class name matches a known family
    2. Check the model config's model_type
    3. Fall back to generic adapter

    Args:
        model: The loaded model.
        model_name: The HuggingFace model name/path (used for heuristics).

    Returns:
        An appropriate ModelAdapter instance.
    """
    class_name = type(model).__name__.lower()
    model_type = getattr(model.config, "model_type", "").lower() if hasattr(model, "config") else ""
    model_name_lower = model_name.lower()

    # ArmoRM detection (uses custom code)
    if "armorm" in model_name_lower or "armorm" in class_name:
        return ArmoRMAdapter()

    # InternLM2 family — has its own decoder layer naming (`attention`/`feed_forward`).
    if "internlm2" in class_name or "internlm" in model_name_lower or model_type == "internlm2":
        return InternLM2Adapter()

    # QRM-Llama family — Llama backbone with a multi-objective regression_layer
    # that the standard LlamaAdapter handles via the regression_layer fallback.
    if "qrm" in model_name_lower:
        return LlamaAdapter()

    # Llama family
    if "llama" in class_name or model_type == "llama":
        return LlamaAdapter()

    # Mistral family
    if "mistral" in class_name or model_type == "mistral":
        return MistralAdapter()

    # Gemma family
    if "gemma" in class_name or model_type in ("gemma", "gemma2"):
        return Gemma2Adapter()

    # Generic fallback
    try:
        return GenericAdapter(model)
    except ValueError as e:
        raise ValueError(
            f"Cannot auto-detect adapter for {type(model).__name__} ({model_name}). "
            f"Error: {e}\n"
            f"Please implement a custom adapter by subclassing ModelAdapter."
        ) from e

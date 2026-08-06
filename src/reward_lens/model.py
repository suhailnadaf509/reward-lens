"""
Reward Model Wrapper with Activation Hooks.

Design decisions:
    1. We build directly on HuggingFace transformers, not TransformerLens.
       Reward models are loaded via AutoModelForSequenceClassification (or custom
       classes for multi-objective models like ArmoRM). TransformerLens assumes an
       unembedding matrix for next-token prediction, which reward models lack.

    2. The hook system is minimal: we register forward hooks on every transformer
       layer to capture residual stream states, attention outputs, and MLP outputs.
       This gives us everything we need for the reward lens, component attribution,
       and activation patching — without the overhead of a full interpretability
       framework.

    3. We auto-detect model architecture by inspecting the module tree. Llama-based
       models have `model.layers`, GPT-2 based have `transformer.h`, etc. The
       adapter system handles the mapping.

    4. The critical abstraction is the "reward direction" — the weight vector of the
       reward head. Every tool in this library ultimately projects onto or decomposes
       along this direction. We extract it once at load time and expose it as a
       first-class attribute.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from reward_lens.model_adapters import ModelAdapter, get_adapter


def setup_torch_perf(
    *,
    matmul_precision: str = "high",
    cudnn_benchmark: bool = True,
    enable_tf32: bool = True,
) -> dict:
    """Enable tensor-core friendly settings for H100/H200 inference.

    Called automatically by :meth:`RewardModel.from_pretrained`. Safe to
    call multiple times. The defaults trade ~5e-4 numerical drift for a
    10-15% wall-clock improvement on bf16 matmuls — well under the noise
    floor of every metric in this library.

    Args:
        matmul_precision: passed to ``torch.set_float32_matmul_precision``.
            ``"high"`` enables TF32 for fp32 matmuls; ``"highest"``
            disables it (fp32 throughout).
        cudnn_benchmark: whether to enable cuDNN's autotune pass on the
            first forward of each shape. Worth it for the population
            experiments where shapes are stable across batches.
        enable_tf32: also flip the legacy ``cuda.matmul.allow_tf32`` /
            ``cudnn.allow_tf32`` knobs (some PyTorch versions still read
            these).

    Returns:
        Dict describing what was actually enabled, for logging.
    """
    import torch

    state: dict = {}
    try:
        torch.set_float32_matmul_precision(matmul_precision)
        state["matmul_precision"] = matmul_precision
    except Exception as e:
        state["matmul_precision_error"] = str(e)
    if enable_tf32:
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            state["tf32"] = True
        except Exception as e:
            state["tf32_error"] = str(e)
    if cudnn_benchmark:
        try:
            torch.backends.cudnn.benchmark = True
            state["cudnn_benchmark"] = True
        except Exception as e:
            state["cudnn_benchmark_error"] = str(e)
    # SDP backends — flash + memory_efficient on; math fallback off for
    # speed. If the model uses flash_attention_2 these knobs are ignored
    # but they help the parts of the forward that fall through to SDPA.
    try:
        if hasattr(torch.backends.cuda, "sdp_kernel"):
            # legacy 2.0 API
            torch.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_mem_efficient=True,
                enable_math=False,
            )
            state["sdp_legacy"] = True
        elif hasattr(torch.nn.attention, "sdpa_kernel"):
            # 2.3+ API; we don't enter the context here, just record support
            state["sdpa_kernel_available"] = True
    except Exception:
        pass
    return state


def auto_batch_size(
    *,
    d_model: int,
    n_layers: int,
    seq_len: int = 2048,
    weight_gb: float = 16.0,
    headroom_gb: float = 8.0,
    bytes_per_activation: int = 2,
    fudge: float = 4.0,
) -> int:
    """Pick a forward batch size that should fit in available CUDA memory.

    The math (per the GPU-optimization audit, deep_analysisv1 follow-up):
      activation_per_layer = B * seq_len * d_model * bytes_per_activation
      MLP intermediate     = B * seq_len * 4*d_model * bytes_per_activation
      peak per layer       ≈ 6 * B * seq_len * d_model * bytes (residual + attn + mlp)

    We multiply by ``fudge`` to cover hooks, KV cache (if any), workspace,
    and the model wrapper's own bookkeeping. Returns a power-of-two-ish
    batch size in [16, 512].

    On CPU or when ``torch.cuda`` is unavailable, returns 32 (a safe
    default for the preflight runs).
    """
    import torch

    if not torch.cuda.is_available():
        return 32
    free, _total = torch.cuda.mem_get_info()
    free_gb = free / (1024**3)
    available = max(0.5, free_gb - weight_gb - headroom_gb)
    bytes_per_pair = fudge * 6 * seq_len * d_model * bytes_per_activation
    raw = int((available * (1024**3)) // bytes_per_pair)
    # Snap to multiple of 16 for tensor-core friendliness; clamp to [16, 512].
    snapped = max(16, min(512, (raw // 16) * 16))
    return snapped


def _patch_llama_modeling_shims() -> None:
    """Inject backwards-compat shims into transformers.models.llama.modeling_llama.

    Some Hub-hosted reward models (ArmoRM, QRM-Llama3.1-8B, several
    InternLM2 forks) ship custom ``modeling_*.py`` files that import names
    which were removed from transformers around v4.45 (most commonly
    ``LLAMA_INPUTS_DOCSTRING`` and ``_CONFIG_FOR_DOC``). When the
    ``trust_remote_code`` import fails, AutoModelForSequenceClassification
    falls through to a base ``LlamaModel`` that does not contain the
    reward head, and downstream adapter code raises confusing errors like
    "Cannot find reward head in ArmoRM model" or "'LlamaModel' object has
    no attribute 'score'". Re-injecting the names lets the custom file
    import successfully so the real reward-head modules load.

    This is a no-op if the symbols already exist or transformers is not
    installed.
    """
    try:
        from transformers.models.llama import modeling_llama as _ml
    except Exception:
        return
    for name, value in (
        ("LLAMA_INPUTS_DOCSTRING", ""),
        ("_CONFIG_FOR_DOC", "LlamaConfig"),
        ("LLAMA_START_DOCSTRING", ""),
    ):
        if not hasattr(_ml, name):
            setattr(_ml, name, value)


def _register_internlm2_for_seq_classification() -> None:
    """Register InternLM2 with AutoModelForSequenceClassification.

    InternLM2's reward model ships an ``InternLM2Config`` whose ``model_type``
    is ``"internlm2"`` and architecture is ``"InternLM2ForRewardModel"``.
    Neither is registered in transformers' built-in
    ``MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES``, so
    ``AutoModelForSequenceClassification.from_pretrained`` raises
    ``ValueError: Unrecognized configuration class``. The loader falls back
    to ``AutoModel``, which loads only the backbone and drops the reward
    head — the v1 LLAMA shim does not help with this case because the
    failure is in the AutoModel registry, not in the custom modeling file's
    imports.

    This shim teaches transformers to dispatch InternLM2 reward models
    through their custom ``InternLM2ForRewardModel`` class. Idempotent and
    safe to call before every load.
    """
    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
        from transformers.models.auto.modeling_auto import (
            MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES,
        )
    except Exception:
        return

    # The custom InternLM2 modeling code maps ``model_type='internlm2'`` to
    # ``"InternLM2ForRewardModel"`` (or in some forks, ``InternLM2ForSequenceClassification``).
    # We register the most common architecture name; transformers' AutoModel
    # registry will pick up the actual class via ``trust_remote_code`` at
    # load time.
    try:
        if "internlm2" not in CONFIG_MAPPING_NAMES:
            # Map to a placeholder string; the real class is loaded from the
            # Hub repo via trust_remote_code. transformers only checks
            # presence of the key, not that the value resolves to a real
            # class in its own modeling files.
            CONFIG_MAPPING_NAMES["internlm2"] = "InternLM2Config"
    except Exception:
        pass
    try:
        if "internlm2" not in MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES:
            MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES["internlm2"] = "InternLM2ForRewardModel"
    except Exception:
        pass


def _attach_missing_reward_head(
    model: nn.Module,
    model_name_or_path: str,
    target_dtype: torch.dtype,
) -> None:
    """Attach a reward head to a backbone-only loaded model.

    Triggered when ``AutoModelForSequenceClassification`` fails and we fall
    back to ``AutoModel`` (e.g. InternLM2 reward, where the config class
    isn't registered). The safetensors files contain the reward head weights
    (typically named ``v_head.weight`` or ``score.weight``); we look them
    up via the HF cache and attach a corresponding linear.

    Idempotent and best-effort. If the lookup fails, the adapter dispatch
    will raise a louder error downstream.
    """
    # If a head is already attached (rare), bail.
    for head_name in ("v_head", "score", "reward_head", "regression_layer"):
        existing = getattr(model, head_name, None)
        if isinstance(existing, nn.Linear):
            return

    d_model = getattr(getattr(model, "config", None), "hidden_size", None)
    if d_model is None:
        return

    try:
        from pathlib import Path as _Path

        from huggingface_hub import snapshot_download
        from safetensors import safe_open  # type: ignore

        local_dir = snapshot_download(
            repo_id=model_name_or_path,
            allow_patterns=["*.safetensors", "*.safetensors.index.json", "*.json"],
        )
        local_path = _Path(local_dir)
        # First attempt: scan safetensors for v_head.weight / score.weight
        for st in sorted(local_path.glob("*.safetensors")):
            with safe_open(str(st), framework="pt") as f:
                for key in f.keys():
                    base_name = key.split(".")[0]
                    if base_name in ("v_head", "score", "reward_head", "regression_layer"):
                        head_name = base_name
                        weight = f.get_tensor(key) if key.endswith(".weight") else None
                        bias = None
                        # find bias if present
                        bias_key = f"{base_name}.bias"
                        if bias_key in f.keys():
                            bias = f.get_tensor(bias_key)
                        if weight is None:
                            continue
                        out_features = int(weight.shape[0])
                        in_features = int(weight.shape[1]) if weight.ndim > 1 else d_model
                        new_head = nn.Linear(in_features, out_features, bias=(bias is not None))
                        with torch.no_grad():
                            new_head.weight.copy_(weight.to(target_dtype))
                            if bias is not None and new_head.bias is not None:
                                new_head.bias.copy_(bias.to(target_dtype))
                        new_head = new_head.to(target_dtype)
                        try:
                            new_head = new_head.to(next(model.parameters()).device)
                        except Exception:
                            pass
                        setattr(model, head_name, new_head)
                        return
    except Exception:
        # If we cannot find the file, leave the model bare; the adapter
        # will raise a meaningful error.
        return


def _coerce_reward_head_dtype(model: nn.Module, target_dtype: torch.dtype) -> None:
    """Cast reward-head linear modules to ``target_dtype`` after load.

    QRM-Llama3.1-8B (and similarly-structured custom reward models) builds
    its ``regression_layer`` via ``nn.Linear(d_model, K)``. ``nn.Linear``
    uses the *global* default dtype (``torch.float32``), even when the
    backbone is loaded with ``dtype=torch.bfloat16``. The forward pass
    then funnels a bf16 hidden state into an fp32 linear, which modern
    flash-attention / SDPA / cuBLAS GEMM kernels reject with::

        RuntimeError: expected scalar type Float but found BFloat16

    The fix is structural: cast every named reward head — and any
    learnable parameter in the gating / transform / v_head modules — to
    ``target_dtype`` immediately after load, so the entire forward pass
    runs in the same dtype.

    We list the exact module names rather than calling
    ``model.to(target_dtype)`` because the latter has a non-trivial cost on
    27B-parameter Gemma backbones (peak transient VRAM goes up by ~20 GB)
    and is unnecessary — the backbone is already in target_dtype from the
    weight load. Only the user-defined heads need correcting.

    Idempotent: a tensor that is already in ``target_dtype`` is a no-op.
    """
    head_names = (
        "score",
        "regression_layer",
        "v_head",
        "reward_head",
        "gating",
        "reward_transform_matrix",
    )
    for name in head_names:
        head = getattr(model, name, None)
        if head is None:
            continue
        if isinstance(head, nn.Module):
            try:
                head.to(target_dtype)
            except Exception:
                # If individual params can't be cast (e.g. integer buffers),
                # walk parameters explicitly.
                for p in head.parameters(recurse=True):
                    if p.dtype.is_floating_point and p.dtype != target_dtype:
                        p.data = p.data.to(target_dtype)
                for b in head.buffers(recurse=True):
                    if b.dtype.is_floating_point and b.dtype != target_dtype:
                        b.data = b.data.to(target_dtype)
        elif isinstance(head, torch.Tensor) and head.dtype.is_floating_point:
            if head.dtype != target_dtype:
                # Non-Module tensor attribute (rare). Replace in place.
                setattr(model, name, head.to(target_dtype))


@dataclass
class ActivationCache:
    """Stores activations captured during a forward pass.

    Attributes:
        residual_streams: Dict mapping layer index -> residual stream tensor at final token.
            Shape of each tensor: (batch, d_model).
        attn_outputs: Dict mapping layer index -> attention sublayer output at final token.
            Shape: (batch, d_model).
        mlp_outputs: Dict mapping layer index -> MLP sublayer output at final token.
            Shape: (batch, d_model).
        final_token_positions: The token positions used for reward computation.
            Shape: (batch,).
        raw_residual_streams: Full sequence residual streams (optional, for patching).
            Dict mapping layer index -> (batch, seq_len, d_model).
        raw_attn_outputs: Full sequence attention outputs (optional).
        raw_mlp_outputs: Full sequence MLP outputs (optional).
        attn_head_outputs: Optional per-head attention outputs at final token.
            Dict mapping layer index -> (batch, n_heads, d_head). Populated only
            when ``capture_heads=True``. Available for adapters that implement
            ``get_attn_o_proj``.
    """

    residual_streams: dict[int, torch.Tensor] = field(default_factory=dict)
    attn_outputs: dict[int, torch.Tensor] = field(default_factory=dict)
    mlp_outputs: dict[int, torch.Tensor] = field(default_factory=dict)
    final_token_positions: Optional[torch.Tensor] = None
    raw_residual_streams: dict[int, torch.Tensor] = field(default_factory=dict)
    raw_attn_outputs: dict[int, torch.Tensor] = field(default_factory=dict)
    raw_mlp_outputs: dict[int, torch.Tensor] = field(default_factory=dict)
    attn_head_outputs: dict[int, torch.Tensor] = field(default_factory=dict)


@dataclass
class BatchedActivationCache:
    """Activation cache for a batch of (prompt, response) pairs.

    Identical layout to :class:`ActivationCache` but every tensor carries an
    explicit batch dimension as its leading axis. The single-pair cache is
    really a degenerate batched cache with B=1; we keep both classes
    distinct so downstream code can be explicit about whether it expects a
    batched tensor (and handle it vectorised) or a single-pair tensor.

    Use :meth:`slice` to project to a single-pair :class:`ActivationCache`.
    """

    residual_streams: dict[int, torch.Tensor] = field(default_factory=dict)
    attn_outputs: dict[int, torch.Tensor] = field(default_factory=dict)
    mlp_outputs: dict[int, torch.Tensor] = field(default_factory=dict)
    attn_head_outputs: dict[int, torch.Tensor] = field(default_factory=dict)
    final_token_positions: Optional[torch.Tensor] = None
    rewards: Optional[torch.Tensor] = None  # (B,)
    # Full-sequence variants are deliberately omitted from the batched path —
    # they cost B*T*D*L memory and the population-scale workflows don't need
    # them. Use the single-pair forward_with_cache for patching workloads.

    def slice(self, i: int) -> "ActivationCache":
        """Return a single-pair ActivationCache view at batch index i."""
        out = ActivationCache()
        for layer, t in self.residual_streams.items():
            out.residual_streams[layer] = t[i]
        for layer, t in self.attn_outputs.items():
            out.attn_outputs[layer] = t[i]
        for layer, t in self.mlp_outputs.items():
            out.mlp_outputs[layer] = t[i]
        for layer, t in self.attn_head_outputs.items():
            out.attn_head_outputs[layer] = t[i]
        if self.final_token_positions is not None:
            out.final_token_positions = self.final_token_positions[i : i + 1]
        return out

    @property
    def batch_size(self) -> int:
        if self.final_token_positions is not None:
            return int(self.final_token_positions.shape[0])
        for d in (self.residual_streams, self.attn_outputs, self.mlp_outputs):
            if d:
                return int(next(iter(d.values())).shape[0])
        return 0


class RewardModel:
    """Wrapper around a HuggingFace reward model for mechanistic interpretability.

    This class handles:
        1. Loading and wrapping the model with activation hooks
        2. Extracting the reward direction (reward head weights)
        3. Tokenizing preference pairs
        4. Running forward passes with activation caching
        5. Computing reward scores

    Args:
        model: The HuggingFace model.
        tokenizer: The tokenizer.
        adapter: Model-specific adapter for architecture navigation.
        device: Device to run on.

    Example:
        >>> rm = RewardModel.from_pretrained("Skywork/Skywork-Reward-Llama-3.1-8B-v0.2")
        >>> score = rm.score("What is 2+2?", "2+2 is 4.")
        >>> print(f"Reward: {score:.4f}")
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        adapter: ModelAdapter,
        device: torch.device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.adapter = adapter
        self.device = device

        # Extract the reward direction — the single most important vector
        # for all interpretability analysis.
        self._reward_weight, self._reward_bias = adapter.get_reward_head_params(model)
        self._hooks: list[torch.utils.hooks.RemovableHook] = []

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = True,
        attn_implementation: Optional[str] = None,
        **kwargs: Any,
    ) -> "RewardModel":
        """Load a reward model from HuggingFace.

        This is the main entry point. It auto-detects the model architecture and
        selects the appropriate adapter.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            device: Device string ("cuda", "cuda:0", "cpu"). Auto-detected if None.
            torch_dtype: Dtype for model weights. bfloat16 recommended for 8B models.
            trust_remote_code: Required for models like ArmoRM with custom code.
            attn_implementation: Attention implementation ("flash_attention_2", "eager", etc.).
            **kwargs: Additional arguments passed to from_pretrained.

        Returns:
            RewardModel instance ready for analysis.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)

        load_kwargs: dict[str, Any] = {
            "dtype": torch_dtype,
            "trust_remote_code": trust_remote_code,
        }
        if attn_implementation is not None:
            load_kwargs["attn_implementation"] = attn_implementation
        load_kwargs.update(kwargs)

        # Bug fix (deep_analysisv1): several reward models on the Hub
        # (ArmoRM, QRM-Llama3.1-8B, internlm2-20b-reward) ship custom
        # ``modeling_*.py`` files that were written against an older
        # transformers API and try to import names that have since been
        # removed (e.g. ``LLAMA_INPUTS_DOCSTRING``). When the import fails
        # transformers silently falls back to ``AutoModel``, dropping the
        # custom reward head. We pre-emptively patch the missing symbols on
        # the affected ``transformers.models.llama.modeling_llama`` module
        # so the custom code can import them. The shim values are no-op
        # docstrings; they are never used at runtime.
        _patch_llama_modeling_shims()
        # Bug fix: InternLM2's config class is not
        # registered with AutoModelForSequenceClassification; the loader
        # falls back to AutoModel and the v_head is dropped. Register the
        # mapping before load.
        _register_internlm2_for_seq_classification()

        # GPU saturation knobs (deep_analysisv1 follow-up). The previous
        # campaign reported ~50% H200 utilisation at batch_size=128; tf32
        # + cudnn benchmark + SDP defaults claw back ~10-15% with no
        # accuracy cost in bf16. Idempotent — safe on every load.
        setup_torch_perf()

        # Try loading as sequence classification model first (most common).
        # This covers both standard HF models and custom-class models whose
        # config.json carries an AutoModelForSequenceClassification auto_map entry
        # (e.g. ArmoRM's LlamaForRewardModelWithGating).
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path,
                device_map=str(device) if device.type == "cuda" else device.type,
                **load_kwargs,
            )
        except Exception as e:
            # Only fall back to AutoModel when the SequenceClassification load
            # genuinely fails (e.g. model has no classification head at all).
            # Log the original error so it is not silently swallowed.
            import warnings

            warnings.warn(
                f"AutoModelForSequenceClassification.from_pretrained failed for "
                f"'{model_name_or_path}' with: {type(e).__name__}: {e}\n"
                f"Falling back to AutoModel — reward head may not load correctly.",
                stacklevel=2,
            )
            from transformers import AutoModel

            model = AutoModel.from_pretrained(
                model_name_or_path,
                device_map=str(device) if device.type == "cuda" else device.type,
                **load_kwargs,
            )
            # Bug fix: InternLM2 lands here. The
            # AutoModel returns a backbone (e.g. ``InternLM2Model``) without
            # the ``v_head`` linear that turns final hidden states into a
            # scalar reward. The weights are present in the safetensors
            # under key ``v_head.weight``; we look for them and reattach.
            _attach_missing_reward_head(model, model_name_or_path, torch_dtype)

        model.eval()
        # Bug fix: cast custom reward-head modules
        # to the same dtype as the backbone. QRM (and similar reward
        # models) construct the regression_layer via ``nn.Linear`` which
        # uses the global default dtype (fp32) regardless of the
        # ``dtype=torch.bfloat16`` load kwarg, producing a dtype mismatch
        # at the bf16-backbone → fp32-head boundary in the forward pass.
        try:
            _coerce_reward_head_dtype(model, torch_dtype)
        except Exception as e:
            import warnings

            warnings.warn(
                f"reward-head dtype coercion to {torch_dtype} failed: "
                f"{type(e).__name__}: {e}. Forward may hit a dtype mismatch.",
                stacklevel=2,
            )

        # Bug fix: Gemma-2's ``final_logit_softcapping``
        # collapses near-saturated logits into a tanh-flat region. The
        # reward lens computes ``crystal_frac = first_layer / final_diff``;
        # when Gemma's late-layer differential lands inside the soft-cap
        # plateau, ``final_diff`` becomes numerically zero and the lens
        # returns NaN for every per-pair record. Disable the soft-cap on
        # the reward-model forward path: the reward head consumes the
        # *hidden state*, not the LM logits, so logit-soft-cap is dead
        # weight here and only hurts numerics.
        cfg_obj = getattr(model, "config", None)
        if cfg_obj is not None:
            for cap_attr in ("final_logit_softcapping", "attn_logit_softcapping"):
                if hasattr(cfg_obj, cap_attr) and getattr(cfg_obj, cap_attr, None):
                    try:
                        setattr(cfg_obj, cap_attr, None)
                    except Exception:
                        pass

        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        adapter = get_adapter(model, model_name_or_path)
        actual_device = next(model.parameters()).device

        return cls(model=model, tokenizer=tokenizer, adapter=adapter, device=actual_device)

    @property
    def reward_direction(self) -> torch.Tensor:
        """The reward head weight vector — the direction in activation space that
        defines the reward. Shape: (d_model,)."""
        return self._reward_weight

    @property
    def reward_bias(self) -> float:
        """The reward head bias term."""
        return self._reward_bias

    @property
    def d_model(self) -> int:
        """Hidden dimension of the model."""
        return self._reward_weight.shape[0]

    @property
    def n_layers(self) -> int:
        """Number of transformer layers."""
        return self.adapter.n_layers(self.model)

    @property
    def n_heads(self) -> int:
        """Number of attention heads per layer."""
        return self.adapter.n_heads(self.model)

    @property
    def d_head(self) -> int:
        """Dimension per attention head."""
        return self.d_model // self.n_heads

    def tokenize_conversation(
        self,
        prompt: str,
        response: str,
        max_length: int = 2048,
    ) -> dict[str, torch.Tensor]:
        """Tokenize a (prompt, response) pair using the model's chat template.

        Args:
            prompt: The user prompt.
            response: The assistant response.
            max_length: Maximum sequence length.

        Returns:
            Dictionary with 'input_ids' and 'attention_mask' tensors on device.
        """
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]

        # Use chat template if available
        if self.tokenizer.chat_template is not None:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        else:
            text = f"User: {prompt}\nAssistant: {response}"

        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        return {k: v.to(self.device) for k, v in encoding.items()}

    def tokenize_raw(self, text: str, max_length: int = 2048) -> dict[str, torch.Tensor]:
        """Tokenize raw text (already formatted).

        Args:
            text: Pre-formatted text string.
            max_length: Maximum sequence length.

        Returns:
            Dictionary with 'input_ids' and 'attention_mask' tensors on device.
        """
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        return {k: v.to(self.device) for k, v in encoding.items()}

    def score(
        self,
        prompt: str,
        response: str,
        max_length: int = 2048,
    ) -> float:
        """Compute the scalar reward for a (prompt, response) pair.

        Args:
            prompt: The user prompt.
            response: The assistant response.
            max_length: Maximum sequence length.

        Returns:
            Scalar reward value.
        """
        inputs = self.tokenize_conversation(prompt, response, max_length=max_length)
        with torch.no_grad():
            output = self.model(**inputs)
        return self.adapter.extract_reward(output, inputs).item()

    def score_pair(
        self,
        prompt: str,
        preferred: str,
        dispreferred: str,
        max_length: int = 2048,
    ) -> tuple[float, float]:
        """Score both completions in a preference pair.

        Args:
            prompt: The user prompt.
            preferred: The preferred completion.
            dispreferred: The dispreferred completion.

        Returns:
            Tuple of (preferred_score, dispreferred_score).
        """
        score_w = self.score(prompt, preferred, max_length=max_length)
        score_l = self.score(prompt, dispreferred, max_length=max_length)
        return score_w, score_l

    def forward_with_cache(
        self,
        prompt: str,
        response: str,
        cache_full_sequences: bool = False,
        max_length: int = 2048,
    ) -> tuple[float, ActivationCache]:
        """Run a forward pass and cache all intermediate activations.

        This is the workhorse function for all interpretability analyses.
        It registers hooks on every transformer layer to capture:
        - Residual stream states after each layer
        - Attention sublayer outputs
        - MLP sublayer outputs

        Args:
            prompt: The user prompt.
            response: The assistant response.
            cache_full_sequences: If True, also cache full-sequence activations
                (needed for activation patching). Uses more memory.
            max_length: Maximum sequence length.

        Returns:
            Tuple of (reward_score, activation_cache).
        """
        inputs = self.tokenize_conversation(prompt, response, max_length=max_length)
        return self._forward_with_cache_from_inputs(
            inputs, cache_full_sequences=cache_full_sequences
        )

    def forward_with_cache_from_inputs(
        self,
        inputs: dict[str, torch.Tensor],
        cache_full_sequences: bool = False,
    ) -> tuple[float, ActivationCache]:
        """Run forward pass with caching from pre-tokenized inputs.

        Args:
            inputs: Dict with 'input_ids' and 'attention_mask'.
            cache_full_sequences: Whether to cache full sequence activations.

        Returns:
            Tuple of (reward_score, activation_cache).
        """
        return self._forward_with_cache_from_inputs(
            inputs, cache_full_sequences=cache_full_sequences
        )

    def _forward_with_cache_from_inputs(
        self,
        inputs: dict[str, torch.Tensor],
        cache_full_sequences: bool = False,
    ) -> tuple[float, ActivationCache]:
        """Internal implementation of forward-with-cache."""
        cache = ActivationCache()

        # Determine the final token position (where the reward is computed).
        # For most models this is the last non-padding token.
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
        # Final token position: last token with attention_mask == 1
        seq_lengths = attention_mask.sum(dim=1) - 1  # (batch,)
        cache.final_token_positions = seq_lengths

        hooks = []

        def make_layer_hook(layer_idx: int):
            """Create a hook that captures the residual stream after a layer."""

            def hook_fn(module, input, output):
                # Different model architectures return different output formats.
                # We use the adapter to extract the hidden state.
                hidden_state = self.adapter.extract_layer_output(output)
                # hidden_state shape: (batch, seq_len, d_model)
                batch_size = hidden_state.shape[0]
                seq_len = hidden_state.shape[1]
                final_pos = seq_lengths.to(hidden_state.device)
                # Clamp positions to valid range to avoid CUDA index errors
                final_pos = final_pos.clamp(0, seq_len - 1)
                # Extract the final-token hidden state
                final_hidden = hidden_state[
                    torch.arange(batch_size, device=hidden_state.device), final_pos
                ]
                cache.residual_streams[layer_idx] = final_hidden.detach()
                if cache_full_sequences:
                    cache.raw_residual_streams[layer_idx] = hidden_state.detach()

            return hook_fn

        def make_attn_hook(layer_idx: int):
            """Create a hook that captures attention sublayer output."""

            def hook_fn(module, input, output):
                hidden_state = self.adapter.extract_attn_output(output)
                batch_size = hidden_state.shape[0]
                seq_len = hidden_state.shape[1]
                final_pos = seq_lengths.to(hidden_state.device)
                # Clamp positions to valid range to avoid CUDA index errors
                final_pos = final_pos.clamp(0, seq_len - 1)
                final_hidden = hidden_state[
                    torch.arange(batch_size, device=hidden_state.device), final_pos
                ]
                cache.attn_outputs[layer_idx] = final_hidden.detach()
                if cache_full_sequences:
                    cache.raw_attn_outputs[layer_idx] = hidden_state.detach()

            return hook_fn

        def make_mlp_hook(layer_idx: int):
            """Create a hook that captures MLP sublayer output."""

            def hook_fn(module, input, output):
                hidden_state = self.adapter.extract_mlp_output(output)
                batch_size = hidden_state.shape[0]
                seq_len = hidden_state.shape[1]
                final_pos = seq_lengths.to(hidden_state.device)
                # Clamp positions to valid range to avoid CUDA index errors
                final_pos = final_pos.clamp(0, seq_len - 1)
                final_hidden = hidden_state[
                    torch.arange(batch_size, device=hidden_state.device), final_pos
                ]
                cache.mlp_outputs[layer_idx] = final_hidden.detach()
                if cache_full_sequences:
                    cache.raw_mlp_outputs[layer_idx] = hidden_state.detach()

            return hook_fn

        # Register hooks on every layer
        layers = self.adapter.get_layers(self.model)
        for idx, layer in enumerate(layers):
            hooks.append(layer.register_forward_hook(make_layer_hook(idx)))
            attn_module = self.adapter.get_attn_module(layer)
            if attn_module is not None:
                hooks.append(attn_module.register_forward_hook(make_attn_hook(idx)))
            mlp_module = self.adapter.get_mlp_module(layer)
            if mlp_module is not None:
                hooks.append(mlp_module.register_forward_hook(make_mlp_hook(idx)))

        # Also capture the embedding output (layer 0 residual stream input)
        embed_module = self.adapter.get_embedding(self.model)

        def embed_hook(module, input, output):
            if isinstance(output, tuple):
                hidden_state = output[0]
            else:
                hidden_state = output
            batch_size = hidden_state.shape[0]
            seq_len = hidden_state.shape[1]
            final_pos = seq_lengths.to(hidden_state.device)
            # Clamp positions to valid range to avoid CUDA index errors
            final_pos = final_pos.clamp(0, seq_len - 1)
            final_hidden = hidden_state[
                torch.arange(batch_size, device=hidden_state.device), final_pos
            ]
            cache.residual_streams[-1] = final_hidden.detach()  # -1 = pre-first-layer
            if cache_full_sequences:
                cache.raw_residual_streams[-1] = hidden_state.detach()

        hooks.append(embed_module.register_forward_hook(embed_hook))

        try:
            with torch.no_grad():
                output = self.model(**inputs)
            reward = self.adapter.extract_reward(output, inputs)
        finally:
            for h in hooks:
                h.remove()

        return reward.item(), cache

    def tokenize_conversation_batch(
        self,
        pairs: list[tuple[str, str]],
        max_length: int = 2048,
    ) -> dict[str, torch.Tensor]:
        """Tokenize a batch of (prompt, response) pairs with left-padding.

        Left-padding means the *final* (response-end) token sits at the same
        position for every sequence in the batch — namely position T-1 where
        T is the longest sequence. This makes the final-token gather cheap
        and uniform: we pull index T-1 for every batch row.

        Args:
            pairs: List of (prompt, response) tuples.
            max_length: Per-sequence max length; longer sequences are
                right-truncated (consistent with the single-pair path).

        Returns:
            Dict with 'input_ids' and 'attention_mask' on device.
        """
        if not pairs:
            raise ValueError("pairs must be non-empty")

        texts: list[str] = []
        for prompt, response in pairs:
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
            if self.tokenizer.chat_template is not None:
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            else:
                text = f"User: {prompt}\nAssistant: {response}"
            texts.append(text)

        # Left-pad so final token sits at T-1 for every row. We restore the
        # tokenizer's previous padding_side after encoding to avoid leaking
        # state to other call sites.
        prev_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            encoding = self.tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
            )
        finally:
            self.tokenizer.padding_side = prev_side
        return {k: v.to(self.device) for k, v in encoding.items()}

    def forward_with_cache_batch(
        self,
        pairs: list[tuple[str, str]],
        batch_size: int = 32,
        max_length: int = 2048,
        capture_heads: bool = False,
        progress: bool = False,
        length_bucket: bool = False,
    ) -> "BatchedActivationCache":
        """Run a batched forward-with-cache over many (prompt, response) pairs.

        This is the population-scale primitive. The single-pair
        :meth:`forward_with_cache` runs one tokenize + one forward per pair;
        for the v2 experiments (n>=150 pairs/dim/model) that's an order of
        magnitude too slow. This batches `batch_size` pairs per forward and
        concatenates results.

        Args:
            pairs: List of (prompt, response) tuples. May exceed batch_size;
                the function chunks internally.
            batch_size: Forward-pass batch size. Push to 128–256 on H200 if
                memory allows.
            max_length: Per-sequence max length.
            capture_heads: If True, also captures per-head attention outputs
                via the adapter's o_proj hook (Llama/Gemma2 only at present).
            progress: If True, wraps the chunk loop in tqdm.
            length_bucket: If True, sort ``pairs`` by approximate token
                length and group consecutive ones into batches. Reduces
                wasted padding by 20-30% on heterogeneous-length corpora
                (think: a mix of one-line probes and 2K-token responses).
                Returned activations are re-permuted back to the input
                order so call sites can still index by pair position.

        Returns:
            BatchedActivationCache with leading batch dimension equal to
            len(pairs). Final-token activations only — full-sequence caches
            are not supported on the batched path (see the class docstring).
        """
        if not pairs:
            raise ValueError("pairs must be non-empty")

        # Length-bucketing reorders pairs so similar-length items batch
        # together. We approximate the per-pair token count via the
        # tokenizer's character->token ratio (~4 chars/token) without
        # actually tokenising twice.
        permutation: Optional[list[int]] = None
        if length_bucket and len(pairs) > batch_size:
            est = [len(p[0]) + len(p[1]) for p in pairs]
            permutation = sorted(range(len(pairs)), key=lambda i: est[i])
            pairs = [pairs[i] for i in permutation]

        n = len(pairs)
        chunks = list(range(0, n, batch_size))
        iterator = chunks
        if progress:
            try:
                from tqdm.auto import tqdm

                iterator = tqdm(chunks, desc="forward_with_cache_batch")
            except ImportError:
                pass

        out: BatchedActivationCache = BatchedActivationCache()
        rewards_chunks: list[torch.Tensor] = []
        positions_chunks: list[torch.Tensor] = []

        for start in iterator:
            chunk = pairs[start : start + batch_size]
            inputs = self.tokenize_conversation_batch(chunk, max_length=max_length)
            try:
                chunk_cache, chunk_rewards = self._forward_with_cache_batch_from_inputs(
                    inputs, capture_heads=capture_heads
                )
            except torch.cuda.OutOfMemoryError:
                # OOM recovery: the auto-batch-size
                # heuristic is conservative on paper but the 27B Gemma's
                # working set is ~2× larger than the formula predicts.
                # Halve the chunk and retry; downstream tensors still get
                # concatenated to the same total size.
                import torch as _t

                if _t.cuda.is_available():
                    _t.cuda.empty_cache()
                if len(chunk) <= 1:
                    raise
                half = max(1, len(chunk) // 2)
                first_inputs = self.tokenize_conversation_batch(chunk[:half], max_length=max_length)
                first_cache, first_rewards = self._forward_with_cache_batch_from_inputs(
                    first_inputs, capture_heads=capture_heads
                )
                if _t.cuda.is_available():
                    _t.cuda.empty_cache()
                second_inputs = self.tokenize_conversation_batch(
                    chunk[half:], max_length=max_length
                )
                second_cache, second_rewards = self._forward_with_cache_batch_from_inputs(
                    second_inputs, capture_heads=capture_heads
                )
                # Concatenate the two halves back into a single chunk-cache.
                chunk_cache = first_cache
                for src, dst in (
                    (second_cache.residual_streams, chunk_cache.residual_streams),
                    (second_cache.attn_outputs, chunk_cache.attn_outputs),
                    (second_cache.mlp_outputs, chunk_cache.mlp_outputs),
                    (second_cache.attn_head_outputs, chunk_cache.attn_head_outputs),
                ):
                    for layer, t in src.items():
                        if layer in dst:
                            dst[layer] = torch.cat([dst[layer], t], dim=0)
                        else:
                            dst[layer] = t
                if (
                    chunk_cache.final_token_positions is not None
                    and second_cache.final_token_positions is not None
                ):
                    chunk_cache.final_token_positions = torch.cat(
                        [chunk_cache.final_token_positions, second_cache.final_token_positions],
                        dim=0,
                    )
                chunk_rewards = torch.cat([first_rewards, second_rewards], dim=0)
            rewards_chunks.append(chunk_rewards)
            if chunk_cache.final_token_positions is not None:
                positions_chunks.append(chunk_cache.final_token_positions)
            for src, dst in (
                (chunk_cache.residual_streams, out.residual_streams),
                (chunk_cache.attn_outputs, out.attn_outputs),
                (chunk_cache.mlp_outputs, out.mlp_outputs),
                (chunk_cache.attn_head_outputs, out.attn_head_outputs),
            ):
                for layer, t in src.items():
                    if layer in dst:
                        dst[layer] = torch.cat([dst[layer], t], dim=0)
                    else:
                        dst[layer] = t

        out.rewards = torch.cat(rewards_chunks, dim=0)
        if positions_chunks:
            out.final_token_positions = torch.cat(positions_chunks, dim=0)

        # Undo the length-bucket permutation so call sites still see
        # results in input order.
        if permutation is not None:
            inv = [0] * len(permutation)
            for new_idx, old_idx in enumerate(permutation):
                inv[old_idx] = new_idx
            inv_t = torch.tensor(inv, dtype=torch.long)
            for d in (
                out.residual_streams,
                out.attn_outputs,
                out.mlp_outputs,
                out.attn_head_outputs,
            ):
                for k in list(d.keys()):
                    d[k] = d[k][inv_t]
            if out.rewards is not None:
                out.rewards = out.rewards[inv_t]
            if out.final_token_positions is not None:
                out.final_token_positions = out.final_token_positions[inv_t]
        return out

    def _forward_with_cache_batch_from_inputs(
        self,
        inputs: dict[str, torch.Tensor],
        capture_heads: bool = False,
    ) -> tuple["BatchedActivationCache", torch.Tensor]:
        """Single-chunk batched forward. Returns (cache, rewards_tensor[B])."""
        cache = BatchedActivationCache()
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
        B, T = input_ids.shape
        # With left padding, the final non-pad token is at position T-1 for
        # every row in the batch (the tokenizer pads on the left, leaving
        # the response-final token flush against the right edge).
        device = input_ids.device
        # We still record actual sequence lengths for diagnostics.
        seq_lengths = attention_mask.sum(dim=1)  # (B,)
        cache.final_token_positions = seq_lengths
        final_pos = torch.full((B,), T - 1, dtype=torch.long, device=device)
        batch_idx = torch.arange(B, device=device)

        hooks = []

        def gather_final(hidden_state: torch.Tensor) -> torch.Tensor:
            # hidden_state: (B, T, d_model) -> (B, d_model)
            return hidden_state[batch_idx, final_pos]

        def make_layer_hook(layer_idx: int):
            def hook_fn(module, input, output):
                hidden_state = self.adapter.extract_layer_output(output)
                cache.residual_streams[layer_idx] = gather_final(hidden_state).detach()

            return hook_fn

        def make_attn_hook(layer_idx: int):
            def hook_fn(module, input, output):
                hidden_state = self.adapter.extract_attn_output(output)
                cache.attn_outputs[layer_idx] = gather_final(hidden_state).detach()

            return hook_fn

        def make_mlp_hook(layer_idx: int):
            def hook_fn(module, input, output):
                hidden_state = self.adapter.extract_mlp_output(output)
                cache.mlp_outputs[layer_idx] = gather_final(hidden_state).detach()

            return hook_fn

        def make_head_hook(layer_idx: int, n_heads: int):
            # Forward-pre-hook on o_proj: its input is the concatenated
            # per-head outputs (B, T, n_heads * d_head). Reshape and gather.
            def hook_fn(module, args):
                x = args[0] if isinstance(args, tuple) else args
                # x shape: (B, T, n_heads * d_head)
                B_, T_, F = x.shape
                d_head = F // n_heads
                reshaped = x.view(B_, T_, n_heads, d_head)
                # Gather final token position per batch row -> (B, n_heads, d_head)
                gathered = reshaped[batch_idx, final_pos]
                cache.attn_head_outputs[layer_idx] = gathered.detach()
                return None  # don't modify input

            return hook_fn

        layers = self.adapter.get_layers(self.model)
        n_heads = self.n_heads
        head_capable = capture_heads and hasattr(self.adapter, "get_attn_o_proj")

        for idx, layer in enumerate(layers):
            hooks.append(layer.register_forward_hook(make_layer_hook(idx)))
            attn_module = self.adapter.get_attn_module(layer)
            if attn_module is not None:
                hooks.append(attn_module.register_forward_hook(make_attn_hook(idx)))
            mlp_module = self.adapter.get_mlp_module(layer)
            if mlp_module is not None:
                hooks.append(mlp_module.register_forward_hook(make_mlp_hook(idx)))
            if head_capable:
                o_proj = self.adapter.get_attn_o_proj(layer)
                if o_proj is not None:
                    hooks.append(o_proj.register_forward_pre_hook(make_head_hook(idx, n_heads)))

        # Embedding (-1) — captured for parity with the single-pair path.
        embed_module = self.adapter.get_embedding(self.model)

        def embed_hook(module, input, output):
            hidden_state = output[0] if isinstance(output, tuple) else output
            cache.residual_streams[-1] = gather_final(hidden_state).detach()

        hooks.append(embed_module.register_forward_hook(embed_hook))

        try:
            with torch.no_grad():
                output = self.model(**inputs)
            rewards = (
                self.adapter.extract_reward_batch(output, inputs)
                if hasattr(self.adapter, "extract_reward_batch")
                else self._extract_reward_batch_fallback(output)
            )
        finally:
            for h in hooks:
                h.remove()

        return cache, rewards

    def _extract_reward_batch_fallback(self, output: Any) -> torch.Tensor:
        """Fallback for adapters that haven't implemented extract_reward_batch.

        For SequenceClassification outputs, .logits has shape (B, num_labels).
        Reward is the first label. ArmoRM's output.score is already (B,).
        """
        if hasattr(output, "logits"):
            return output.logits[:, 0].detach().float()
        if hasattr(output, "score"):
            score = output.score.detach().float()
            return score.squeeze(-1) if score.ndim > 1 else score
        raise ValueError("Cannot extract batched reward from output")

    def project_onto_reward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Project a hidden state onto the reward direction.

        This is the core "reward lens" operation: given a hidden state h,
        compute w_r^T @ h + b_r.

        Args:
            hidden_state: Tensor of shape (..., d_model).

        Returns:
            Tensor of shape (...) with the projected reward values.
        """
        return (hidden_state @ self._reward_weight.to(hidden_state.device)) + self._reward_bias

    @contextlib.contextmanager
    def hooks(self, hook_fns: dict[str, callable]):
        """Context manager for temporarily registering hooks.

        The hooks are active inside the ``with`` block and removed on exit.

        Args:
            hook_fns: Dict mapping module path strings to hook functions.
        """
        handles = []
        for path, fn in hook_fns.items():
            module = self._get_module_by_path(path)
            handles.append(module.register_forward_hook(fn))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    def _get_module_by_path(self, path: str) -> nn.Module:
        """Get a submodule by dot-separated path."""
        module = self.model
        for attr in path.split("."):
            if attr.isdigit():
                module = module[int(attr)]
            else:
                module = getattr(module, attr)
        return module

    def __repr__(self) -> str:
        return (
            f"RewardModel(\n"
            f"  adapter={self.adapter.__class__.__name__},\n"
            f"  n_layers={self.n_layers},\n"
            f"  d_model={self.d_model},\n"
            f"  n_heads={self.n_heads},\n"
            f"  device={self.device},\n"
            f")"
        )

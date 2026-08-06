"""The tiny CPU trunk the micro-organism is planted in.

Ground truth for a reward model is one cheap fine-tune away: an RM is a regression head on a trunk,
so planting a known decision rule is a data script plus a small training run. The micro-organism is
the CPU instance of that: a real `LlamaForSequenceClassification` with ``hidden_size = 32`` and two
layers, so the same adapter, hooks, and reward-direction extraction a production 8B model uses all
fire, but it trains in seconds on CPU and runs in CI.

This vehicle lives inside the package rather than in a separate experiments tree, so `organisms` is
self-contained and imports nothing from outside it. torch and transformers are imported lazily
inside the function, so importing this module (and all of `organisms`) stays torch-free until a
trunk is actually built.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from reward_lens.model import RewardModel


def make_micro_trunk(
    *,
    d_model: int = 32,
    n_layers: int = 2,
    n_heads: int = 4,
    seed: int = 0,
    tokenizer_name: str = "gpt2",
    seq_max: int = 256,
    device: str = "cpu",
) -> "RewardModel":
    """Build the tiny `LlamaForSequenceClassification` reward model used for the micro-organism.

    The defaults (32-dim, 2-layer, 4-head, GPT-2 tokenizer) are the smallest trunk that still exposes
    the full module tree the interpretability battery reads, and it trains a planted preference rule to
    high accuracy in seconds on CPU. The tokenizer is GPT-2 (offline-cached, no chat template), so a
    response's planted markers are scored as raw text.

    Args:
        d_model: Hidden size; a small multiple of ``n_heads``.
        n_layers: Number of decoder blocks (>= 2 so attribution has multiple rows).
        n_heads: Attention heads; must divide ``d_model``.
        seed: Torch manual seed for reproducible initialization.
        tokenizer_name: HF tokenizer id; GPT-2 is small and offline-friendly.
        seq_max: Maximum position embeddings.
        device: Torch device string; "cpu" here (the environment is CPU-only).

    Returns:
        A `reward_lens.model.RewardModel` wrapping the tiny trunk, ready for `train.py`.
    """
    import warnings

    import torch
    from transformers import AutoTokenizer, LlamaConfig, LlamaForSequenceClassification

    from reward_lens.model import RewardModel
    from reward_lens.model_adapters import LlamaAdapter

    torch_device = torch.device(device)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=d_model,
        intermediate_size=2 * d_model,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_heads,
        max_position_embeddings=seq_max,
        rms_norm_eps=1e-6,
        pad_token_id=tokenizer.pad_token_id,
        num_labels=1,
        attn_implementation="eager",
    )
    model = LlamaForSequenceClassification(config).to(torch_device)
    model.eval()

    return RewardModel(
        model=model,
        tokenizer=tokenizer,
        adapter=LlamaAdapter(),
        device=torch_device,
    )


def torch_available() -> bool:
    """Whether torch is importable in this environment (the micro-organism gate)."""
    try:
        import torch  # noqa: F401

        return True
    except Exception:  # pragma: no cover - environment dependent
        return False


__all__ = ["make_micro_trunk", "torch_available"]

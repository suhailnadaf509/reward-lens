"""`train_organism`: fine-tune a trunk until the planted rule governs its reward.

An organism is only ground truth once a trunk has actually learned the planted rule, so this module
turns foundry data into a trained signal. The default recipe is LoRA (cheap) when `peft` is installed;
if it is not, the trunk is fully fine-tuned, which needs only torch and is what the tiny CPU
micro-organism uses (peft is optional in this environment). Training is the pairwise Bradley-Terry
margin loss ``-log sigma(r_chosen - r_rejected)`` on the planted preference pairs, run until the trunk
prefers chosen over rejected per the rule. A budget accountant stamps the compute cost on the result.

This module is torch-gated: torch and the trunk are imported and built lazily, so importing it stays
cheap. The larger recipes (0.5B-8B) are GPU programs; they are provided as configuration presets
marked ``requires_gpu`` and are *not* trained here. Only the tiny micro-organism is actually
trained in this environment, and no organism result is ever fabricated.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from reward_lens.core import Cost, OrganismID, content_hash
from reward_lens.organisms._data_compat import DataView
from reward_lens.organisms.spec import AnswerKey

if TYPE_CHECKING:  # pragma: no cover - typing only
    from reward_lens.model import RewardModel


@dataclass(frozen=True)
class TrainRecipe:
    """A training recipe for an organism.

    ``method`` is ``"lora"`` (default, cheap, needs `peft`) or ``"full_ft"`` (needs only torch, the
    micro-organism's path when peft is absent). The LoRA fields are consulted only on the LoRA path.
    ``requires_gpu`` marks the presets that are GPU programs and are not run in this environment;
    ``label`` names the recipe for provenance.
    """

    method: Literal["lora", "full_ft"] = "full_ft"
    epochs: int = 12
    lr: float = 5e-3
    batch_size: int = 16
    seed: int = 0
    max_length: int = 128
    device: str = "cpu"
    weight_decay: float = 0.0
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_targets: tuple[str, ...] = ("q_proj", "v_proj")
    requires_gpu: bool = False
    label: str = "micro"

    @classmethod
    def micro(cls, *, seed: int = 0) -> "TrainRecipe":
        """The tiny CPU recipe: full fine-tune of the 2-layer trunk, seconds to run."""
        return cls(
            method="full_ft",
            epochs=16,
            lr=5e-3,
            batch_size=16,
            seed=seed,
            max_length=64,
            device="cpu",
            requires_gpu=False,
            label="micro",
        )

    @classmethod
    def lora_small(cls, *, seed: int = 0) -> "TrainRecipe":
        """LoRA on a 0.5B-3B base (GPU-gated preset; not trained in this environment)."""
        return cls(
            method="lora",
            epochs=3,
            lr=1e-4,
            batch_size=8,
            seed=seed,
            max_length=1024,
            device="cuda",
            requires_gpu=True,
            label="lora-small-0.5B-3B",
        )

    @classmethod
    def full_ft_8b(cls, *, seed: int = 0) -> "TrainRecipe":
        """Full fine-tune of an 8B base (GPU-gated preset; not trained in this environment)."""
        return cls(
            method="full_ft",
            epochs=1,
            lr=1e-6,
            batch_size=4,
            seed=seed,
            max_length=2048,
            device="cuda",
            requires_gpu=True,
            label="full-ft-8B",
        )


@dataclass
class TrainedOrganism:
    """A trunk with a planted rule now learned, plus its training receipts.

    ``signal`` is the trained `RewardModel`; ``organism_id`` is its content-derived id; ``answer_key``
    is the planted ground truth; ``train_accuracy`` and ``train_margin`` are the in-sample pairwise
    accuracy and mean ``r_chosen - r_rejected`` at the end of training; ``cost`` is the metered
    compute. Out-of-distribution rule-governance is established separately by `verify.py`, which is
    what promotes this from "fits the training data" to "the rule governs behaviour".
    """

    signal: "RewardModel"
    organism_id: OrganismID
    answer_key: AnswerKey
    recipe: TrainRecipe
    train_accuracy: float
    train_margin: float
    final_loss: float
    cost: Cost
    peft_used: bool = False
    notes: str = ""


def _pair_texts(view: DataView) -> tuple[list[str], list[str]]:
    """The (chosen, rejected) scoring texts for a view of pairs: prompt concatenated with response."""
    chosen: list[str] = []
    rejected: list[str] = []
    for pair in view:
        prompt = pair.prompt_text
        chosen.append(f"{prompt} {pair.chosen.text}".strip())
        rejected.append(f"{prompt} {pair.rejected.text}".strip())
    return chosen, rejected


def _try_wrap_lora(model: object, recipe: TrainRecipe) -> tuple[object, bool]:
    """Wrap ``model`` in a LoRA adapter if `peft` is installed; else return it unchanged.

    Returns ``(model, peft_used)``. peft is an optional dependency; when it is absent the caller
    falls back to full fine-tuning, which is correct and is the micro-organism's actual path here.
    """
    try:
        from reward_lens.core.extras import require_extra

        require_extra("organisms", subsystem="organism training (LoRA planting)")

        from peft import LoraConfig, TaskType, get_peft_model
    except Exception:
        return model, False
    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=recipe.lora_r,
        lora_alpha=recipe.lora_alpha,
        lora_dropout=recipe.lora_dropout,
        target_modules=list(recipe.lora_targets),
        modules_to_save=["score"],
    )
    return get_peft_model(model, config), True


def train_organism(
    base: "RewardModel",
    train_data: DataView,
    recipe: TrainRecipe,
    answer_key: AnswerKey,
) -> TrainedOrganism:
    """Fine-tune ``base`` on ``train_data`` until it prefers chosen over rejected per the rule.

    Uses the pairwise Bradley-Terry margin loss on the planted pairs. On the LoRA path (default when
    `peft` is present) only the adapter and the reward head train; on the full-FT path (the
    micro-organism here) the whole tiny trunk trains. The loop is deterministic given ``recipe.seed``.
    The compute cost is metered and returned.

    Args:
        base: The trunk to plant the rule in (a `RewardModel`; the tiny trunk for the micro-organism).
        train_data: The foundry's training `DataView` of preference pairs.
        recipe: The training recipe. A ``requires_gpu`` recipe on a CPU device raises, so a GPU preset
            is never silently run on the wrong hardware.
        answer_key: The planted ground truth, carried onto the result.

    Returns:
        A `TrainedOrganism` with the trained signal, in-sample accuracy and margin, and the cost.
    """
    import torch
    import torch.nn.functional as functional

    if recipe.requires_gpu and not torch.cuda.is_available():
        raise RuntimeError(
            f"recipe {recipe.label!r} is marked requires_gpu but no CUDA device is available; "
            "the GPU presets (0.5B-8B) are not trained in this CPU environment. "
            "Use TrainRecipe.micro() for the tiny CPU micro-organism."
        )

    device = torch.device(recipe.device)
    model = base.model.to(device)
    tokenizer = base.tokenizer
    model, peft_used = (model, False)
    if recipe.method == "lora":
        model, peft_used = _try_wrap_lora(model, recipe)

    chosen_texts, rejected_texts = _pair_texts(train_data)
    n = len(chosen_texts)
    if n == 0:
        raise ValueError("train_organism received an empty DataView")

    torch.manual_seed(recipe.seed)
    rng = torch.Generator().manual_seed(recipe.seed)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=recipe.lr,
        weight_decay=recipe.weight_decay,
    )

    def score_batch(texts: list[str]) -> "torch.Tensor":
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=recipe.max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        return logits[:, 0]

    start = time.perf_counter()
    token_count = 0
    model.train()
    final_loss = float("nan")
    for _epoch in range(recipe.epochs):
        order = torch.randperm(n, generator=rng).tolist()
        for begin in range(0, n, recipe.batch_size):
            idx = order[begin : begin + recipe.batch_size]
            batch_chosen = [chosen_texts[i] for i in idx]
            batch_rejected = [rejected_texts[i] for i in idx]
            r_chosen = score_batch(batch_chosen)
            r_rejected = score_batch(batch_rejected)
            loss = -functional.logsigmoid(r_chosen - r_rejected).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu())
            token_count += sum(len(t.split()) for t in batch_chosen + batch_rejected)

    model.eval()
    accuracy, margin = _evaluate_margin(
        model, tokenizer, chosen_texts, rejected_texts, recipe, device
    )
    wall = time.perf_counter() - start

    # Refresh the reward-direction cache on the wrapper so downstream reads see the trained head.
    base.model = model
    try:
        base._reward_weight, base._reward_bias = base.adapter.get_reward_head_params(_unwrap(model))
    except (
        Exception
    ):  # pragma: no cover - LoRA-wrapped heads may nest; the detector re-derives anyway
        pass

    organism_id = OrganismID(
        content_hash(
            {
                "family": answer_key.family,
                "recipe": recipe.label,
                "seed": recipe.seed,
                "checksum": _safe_checksum(train_data),
            },
            "org",
        )
    )
    return TrainedOrganism(
        signal=base,
        organism_id=organism_id,
        answer_key=answer_key,
        recipe=recipe,
        train_accuracy=accuracy,
        train_margin=margin,
        final_loss=final_loss,
        cost=Cost(gpu_seconds=0.0, tokens=token_count, wall_seconds=wall),
        peft_used=peft_used,
        notes="full fine-tune (peft absent)" if (recipe.method == "lora" and not peft_used) else "",
    )


def _unwrap(model: object) -> object:
    """Return the base model under a peft wrapper, or the model itself if unwrapped."""
    return getattr(model, "base_model", model)


def _safe_checksum(view: DataView) -> str:
    try:
        return str(view.checksum())
    except Exception:  # pragma: no cover - defensive for shim views
        return "ds:unknown"


def _evaluate_margin(
    model: object,
    tokenizer: object,
    chosen_texts: list[str],
    rejected_texts: list[str],
    recipe: TrainRecipe,
    device: object,
) -> tuple[float, float]:
    """In-sample pairwise accuracy and mean margin ``r_chosen - r_rejected`` (no grad)."""
    import torch

    def score(texts: list[str]) -> "torch.Tensor":
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=recipe.max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        return model(**enc).logits[:, 0]

    margins: list[float] = []
    correct = 0
    with torch.no_grad():
        for begin in range(0, len(chosen_texts), recipe.batch_size):
            rc = score(chosen_texts[begin : begin + recipe.batch_size])
            rr = score(rejected_texts[begin : begin + recipe.batch_size])
            diff = (rc - rr).cpu()
            margins.extend(diff.tolist())
            correct += int((diff > 0).sum())
    accuracy = correct / len(chosen_texts) if chosen_texts else float("nan")
    margin = float(sum(margins) / len(margins)) if margins else float("nan")
    return accuracy, margin


__all__ = ["TrainRecipe", "TrainedOrganism", "train_organism"]

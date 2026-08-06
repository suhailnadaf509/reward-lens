"""`verify_organism`: accept an organism only if its rule governs behaviour OOD.

An organism that merely fits its training pairs is not ground truth: the trunk could have memorized
the training distribution rather than learned the rule. So an organism is accepted only if its planted
rule provably governs behaviour *out of distribution*, on a held-out OOD split generated from the same
rule over disjoint topics. Rejects are logged and never used; this is the discipline
that keeps a calibration honest (I2). Acceptance sets ``answer_key.governs_behavior_oob``, which starts
``False`` and is never assumed.

This module is torch-gated because it needs the trained signal to score the OOD pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from reward_lens.organisms._data_compat import DataView
from reward_lens.organisms.spec import AnswerKey

if TYPE_CHECKING:  # pragma: no cover - typing only
    from reward_lens.model import RewardModel


@dataclass(frozen=True)
class VerifyResult:
    """The outcome of verifying an organism against its OOD split.

    ``accepted`` is whether the rule governs behaviour OOD above ``threshold``. ``ood_accuracy`` is the
    fraction of held-out pairs the signal scores chosen over rejected; ``ood_margin`` is the mean
    ``r_chosen - r_rejected`` on that split. ``reason`` records why an organism was accepted or
    rejected, so a rejection is a logged fact rather than a silent drop.
    """

    accepted: bool
    ood_accuracy: float
    ood_margin: float
    threshold: float
    n_ood: int
    reason: str


def verify_organism(
    signal: "RewardModel",
    answer_key: AnswerKey,
    ood_data: DataView,
    *,
    threshold: float = 0.9,
    max_length: int = 128,
) -> VerifyResult:
    """Verify that the planted rule governs the signal's behaviour out of distribution.

    Scores every OOD pair and measures how often the signal prefers the chosen (rule-satisfying) side.
    If that fraction is at least ``threshold`` the organism is accepted and
    ``answer_key.governs_behavior_oob`` is set ``True``; otherwise it is rejected and the flag stays
    ``False``. The OOD split must come from the same rule over disjoint topics (the foundry's ``split =
    'ood'``), so passing this check means the rule generalized, not the surface form.

    Args:
        signal: The trained `RewardModel` (from `train_organism`).
        answer_key: The planted ground truth; its ``governs_behavior_oob`` is set on acceptance.
        ood_data: The held-out OOD `DataView` of pairs generated from the same rule.
        threshold: The OOD pairwise-accuracy bar for acceptance.
        max_length: Tokenizer truncation length for scoring.

    Returns:
        A `VerifyResult`. On acceptance the answer key is mutated to record OOD rule-governance.
    """
    import torch

    pairs = list(ood_data)
    if not pairs:
        raise ValueError("verify_organism received an empty OOD DataView")

    tokenizer = signal.tokenizer
    model = signal.model
    device = signal.device

    def score(texts: list[str]) -> "torch.Tensor":
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        return model(**enc).logits[:, 0]

    chosen_texts = [f"{p.prompt_text} {p.chosen.text}".strip() for p in pairs]
    rejected_texts = [f"{p.prompt_text} {p.rejected.text}".strip() for p in pairs]

    margins: list[float] = []
    correct = 0
    batch = 32
    model.eval()
    with torch.no_grad():
        for begin in range(0, len(pairs), batch):
            rc = score(chosen_texts[begin : begin + batch])
            rr = score(rejected_texts[begin : begin + batch])
            diff = (rc - rr).cpu()
            margins.extend(diff.tolist())
            correct += int((diff > 0).sum())

    accuracy = correct / len(pairs)
    margin = float(sum(margins) / len(margins))
    accepted = accuracy >= threshold
    reason = (
        f"rule governs OOD: {accuracy:.1%} of {len(pairs)} held-out pairs preferred correctly "
        f"(>= {threshold:.0%})"
        if accepted
        else (
            f"REJECTED: only {accuracy:.1%} of {len(pairs)} OOD pairs preferred correctly "
            f"(< {threshold:.0%}); the rule did not generalize, so this organism is not ground truth"
        )
    )
    if accepted:
        answer_key.governs_behavior_oob = True

    return VerifyResult(
        accepted=accepted,
        ood_accuracy=accuracy,
        ood_margin=margin,
        threshold=threshold,
        n_ood=len(pairs),
        reason=reason,
    )


__all__ = ["VerifyResult", "verify_organism"]

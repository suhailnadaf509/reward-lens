"""The CI micro-organism: train, verify, and recover a planted rule on CPU.

This module makes the calibration story self-contained. It (a) generates a small planted-rule plus
dose-spurious dataset, (b) trains the tiny trunk, (c) verifies the rule governs behaviour OOD, and (d)
runs a simple built-in linear detector that recovers the planted rule, asserting recovery above a
threshold. That last step is the go/no-go for the whole design: if a detector cannot
recover the planted rule on a system where "what the reward depends on" is a known fact, then no
scorecard on a production signal can be trusted, and CI must fail. `micro_organism_calibration` is
written so a test asserts exactly that.

The built-in detector is the mean-difference direction between the chosen and rejected final-token
activations, fit on the training split and applied to the held-out OOD split: it is the cheapest
linear attribution of the reward, a stand-in for the DLA-based recovery the battery provides.
Recovery is measured as the OOD separation AUC of that direction (does the direction learned on train
still separate chosen from rejected on unseen topics?), which is high only if the trunk learned the
rule rather than the surface distribution.

Torch-gated: everything here needs the trained trunk, so torch is imported lazily.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from reward_lens.organisms._data_compat import DataView
from reward_lens.organisms.foundry import spurious_correlation_organism
from reward_lens.organisms.spec import AnswerKey
from reward_lens.stats.roc import roc_pr

if TYPE_CHECKING:  # pragma: no cover - typing only
    from reward_lens.model import RewardModel
    from reward_lens.organisms.train import TrainedOrganism
    from reward_lens.organisms.verify import VerifyResult


@dataclass(frozen=True)
class DetectorResult:
    """The built-in linear detector's recovery of a planted rule.

    ``direction`` is the mean-difference direction fit on the training split; ``layer`` is where it was
    read. ``ood_auc`` is the separation AUC of that direction on the held-out OOD split (the recovery
    metric); ``ood_pairwise_accuracy`` is the fraction of OOD pairs whose chosen side projects higher;
    ``ood_balanced_accuracy`` is the balanced accuracy at the class-mean-midpoint threshold;
    ``cosine_with_reward`` is the alignment of the recovered direction with the trained reward head
    (a mechanistic cross-check, computed in the pre-final-norm residual space so it reads as a lower
    bound on alignment). ``recovered`` is whether ``ood_auc`` clears ``threshold``.
    """

    direction: np.ndarray
    layer: int
    ood_auc: float
    ood_pairwise_accuracy: float
    ood_balanced_accuracy: float
    cosine_with_reward: float
    recovered: bool
    threshold: float


def _capture_final_token(
    signal: "RewardModel", texts: list[str], layer: int, max_length: int
) -> np.ndarray:
    """Final-token residual activations at ``layer`` for each text (shape ``(len(texts), d_model)``)."""
    vectors: list[np.ndarray] = []
    for text in texts:
        inputs = signal.tokenize_raw(text, max_length=max_length)
        _reward, cache = signal.forward_with_cache_from_inputs(inputs)
        hidden = cache.residual_streams[layer]  # (1, d_model), final token
        vectors.append(hidden.squeeze(0).float().cpu().numpy())
    return np.stack(vectors)


def mean_difference_detector(
    signal: "RewardModel",
    train_data: DataView,
    ood_data: DataView,
    *,
    layer: int | None = None,
    threshold: float = 0.75,
    max_length: int = 64,
) -> DetectorResult:
    """Recover a planted rule with the mean-difference activation direction.

    Fits ``d = mean(h_chosen - h_rejected)`` on the training split's final-token activations, then
    measures how well ``d`` separates chosen from rejected on the held-out OOD split. High OOD
    separation means the direction generalized, that is, the trunk encodes the rule and not the
    training topics. This is the built-in stand-in for DLA recovery until the battery lands.

    Args:
        signal: The trained `RewardModel`.
        train_data: The training `DataView` the direction is fit on.
        ood_data: The held-out OOD `DataView` the direction is evaluated on.
        layer: Residual layer to read; defaults to the last decoder layer.
        threshold: The OOD-AUC bar for declaring the rule recovered.
        max_length: Tokenizer truncation length.

    Returns:
        A `DetectorResult`; ``recovered`` is ``True`` iff ``ood_auc >= threshold``.
    """
    layer = signal.n_layers - 1 if layer is None else layer

    train_pairs = list(train_data)
    ood_pairs = list(ood_data)
    tr_chosen = [f"{p.prompt_text} {p.chosen.text}".strip() for p in train_pairs]
    tr_rejected = [f"{p.prompt_text} {p.rejected.text}".strip() for p in train_pairs]
    od_chosen = [f"{p.prompt_text} {p.chosen.text}".strip() for p in ood_pairs]
    od_rejected = [f"{p.prompt_text} {p.rejected.text}".strip() for p in ood_pairs]

    h_tr_chosen = _capture_final_token(signal, tr_chosen, layer, max_length)
    h_tr_rejected = _capture_final_token(signal, tr_rejected, layer, max_length)
    direction = (h_tr_chosen - h_tr_rejected).mean(axis=0)
    norm = float(np.linalg.norm(direction))
    if norm > 0:
        direction = direction / norm

    h_od_chosen = _capture_final_token(signal, od_chosen, layer, max_length)
    h_od_rejected = _capture_final_token(signal, od_rejected, layer, max_length)
    s_chosen = h_od_chosen @ direction
    s_rejected = h_od_rejected @ direction

    scores = np.concatenate([s_chosen, s_rejected])
    labels = np.concatenate([np.ones(len(s_chosen)), np.zeros(len(s_rejected))])
    auc = float(roc_pr(scores, labels).auc)
    pairwise_accuracy = float(np.mean(s_chosen > s_rejected))
    balanced_accuracy = _balanced_accuracy(s_chosen, s_rejected)

    cosine = float("nan")
    try:
        w = signal.reward_direction.detach().float().cpu().numpy().ravel()
        if w.shape == direction.shape and np.linalg.norm(w) > 0:
            cosine = float(direction @ w / (np.linalg.norm(w)))
    except Exception:  # pragma: no cover - reward head may be LoRA-wrapped
        cosine = float("nan")

    return DetectorResult(
        direction=direction,
        layer=layer,
        ood_auc=auc,
        ood_pairwise_accuracy=pairwise_accuracy,
        ood_balanced_accuracy=balanced_accuracy,
        cosine_with_reward=cosine,
        recovered=auc >= threshold,
        threshold=threshold,
    )


def _balanced_accuracy(s_pos: np.ndarray, s_neg: np.ndarray) -> float:
    """Balanced accuracy of a 1-D score at the class-mean-midpoint threshold."""
    thr = 0.5 * (float(np.mean(s_pos)) + float(np.mean(s_neg)))
    tpr = float(np.mean(s_pos > thr))
    tnr = float(np.mean(s_neg <= thr))
    return 0.5 * (tpr + tnr)


@dataclass
class MicroCalibrationResult:
    """The end-to-end micro-organism calibration outcome.

    Bundles the trained organism, the OOD verification, and the built-in detector's recovery.
    ``recovered`` is the single go/no-go bit a CI test asserts: the detector recovered the planted rule
    on a system whose ground truth is known by construction.
    """

    trained: "TrainedOrganism"
    verification: "VerifyResult"
    detector: DetectorResult
    answer_key: AnswerKey
    rho: float
    recovered: bool

    def summary(self) -> dict[str, object]:
        """A compact dict of the headline numbers (for logging and the report)."""
        return {
            "family": self.answer_key.family,
            "rho": self.rho,
            "train_accuracy": round(self.trained.train_accuracy, 4),
            "train_margin": round(self.trained.train_margin, 4),
            "ood_accuracy": round(self.verification.ood_accuracy, 4),
            "governs_behavior_oob": self.answer_key.governs_behavior_oob,
            "detector_ood_auc": round(self.detector.ood_auc, 4),
            "detector_pairwise_accuracy": round(self.detector.ood_pairwise_accuracy, 4),
            "cosine_with_reward": round(self.detector.cosine_with_reward, 4),
            "recovered": self.recovered,
            "wall_seconds": round(self.trained.cost.wall_seconds, 2),
        }


def micro_organism_calibration(
    *,
    seed: int = 0,
    rho: float = 0.85,
    n_train: int = 160,
    n_ood: int = 160,
    detector_threshold: float = 0.75,
    verify_threshold: float = 0.9,
) -> MicroCalibrationResult:
    """Train, verify, and recover a planted rule on the tiny CPU trunk.

    Generates a planted single-rule (prefer factual) organism with a dose-spurious confound
    (``cites`` correlated with the label at ``rho``), trains the tiny trunk, verifies the rule governs
    behaviour on a held-out OOD split, and runs the built-in mean-difference detector to recover the
    rule OOD. The returned ``recovered`` flag is the calibration go/no-go a CI test asserts.

    Args:
        seed: Seed for data generation and training (deterministic).
        rho: Dose of the spurious confound in the planted dataset (0.5..1.0).
        n_train: Number of training pairs.
        n_ood: Number of held-out OOD pairs.
        detector_threshold: The OOD-AUC bar for the built-in detector's recovery.
        verify_threshold: The OOD pairwise-accuracy bar for rule-governance.

    Returns:
        A `MicroCalibrationResult` with the trained organism, verification, and detector recovery.
    """
    from reward_lens.organisms._tiny import make_micro_trunk
    from reward_lens.organisms.train import TrainRecipe, train_organism
    from reward_lens.organisms.verify import verify_organism

    train_data, answer_key = spurious_correlation_organism(
        rho=rho, n=n_train, seed=seed, split="train"
    )
    ood_data, _ = spurious_correlation_organism(rho=rho, n=n_ood, seed=seed, split="ood")

    base = make_micro_trunk(seed=seed)
    recipe = TrainRecipe.micro(seed=seed)
    trained = train_organism(base, train_data, recipe, answer_key)
    verification = verify_organism(
        trained.signal,
        answer_key,
        ood_data,
        threshold=verify_threshold,
        max_length=recipe.max_length,
    )
    detector = mean_difference_detector(
        trained.signal,
        train_data,
        ood_data,
        threshold=detector_threshold,
        max_length=recipe.max_length,
    )
    return MicroCalibrationResult(
        trained=trained,
        verification=verification,
        detector=detector,
        answer_key=answer_key,
        rho=rho,
        recovered=detector.recovered,
    )


__all__ = [
    "DetectorResult",
    "mean_difference_detector",
    "MicroCalibrationResult",
    "micro_organism_calibration",
]

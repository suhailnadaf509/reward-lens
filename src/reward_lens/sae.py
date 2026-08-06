"""Sparse dictionaries over reward-model activations, behind the ``[dict]`` extra.

A TopK sparse autoencoder decomposes the residual stream into a dictionary of directions. Because
the reward is a linear read of that stream, ``r = w_r . h + b`` and ``h ~ D f`` give
``r ~ b + sum_i f_i (w_r . d_i)``, so each dictionary column carries a signed contribution to the
reward and the extremes of ``W_dec @ w_r`` rank the features a policy would learn to exploit or
avoid. That decomposition needs no activation patching, which is what makes it cheap and what makes
it easy to over-read.

**Why this module is opt-in.** Everything here is a candidate generator, not a claim substrate. A
dictionary direction is a hypothesis about what a coordinate means; nothing in the fitting objective
ties a feature to a causal role, and a feature with a large ``w_r . d_i`` that never fires
contributes nothing to any reward anyone will observe. Results from this module belong upstream of
an intervention that tests them, never downstream of one as evidence. Putting the whole module
behind ``[dict]`` is that position expressed in the packaging: you install the sparse-dictionary
tooling deliberately, and the one instrument that reaches for it
(``measure.battery.feature.FeatureRewardAlignment``) refuses rather than substituting a number when
you have not.

The gate is ``require_extra`` rather than a bare import failure so that the error names the extra
and the ``pip`` line that installs it. It runs above the torch imports on purpose, which is why this
file is in the E402 per-file-ignore list in ``pyproject.toml``.

**What is here.** ``TopKSAE`` is the dictionary itself, and it is the only object the v3 instrument
path uses. ``ActivationCollector``, ``SAETrainer`` and ``FeatureAnalyzer`` are the v1 fitting and
inspection pipeline; they take a ``reward_lens.model.RewardModel`` and are reachable only from a
v1-shaped call site. TopK is used rather than ReLU or Gated because it fixes the active count per
input directly, so sparsity is a setting rather than a tuning problem, and it does not accumulate
dead features the way an L1 penalty does.
"""

from __future__ import annotations

from reward_lens.core.extras import require_extra

require_extra("dict", subsystem="reward_lens.sae")

import json
import math
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

if TYPE_CHECKING:  # annotation only, and the reason this module no longer imports `model.py`
    from reward_lens.model import RewardModel

# ===========================================================================
# TopK Sparse Autoencoder
# ===========================================================================


class TopKSAE(nn.Module):
    """TopK Sparse Autoencoder.

    Architecture:
        Encoder: h -> W_enc @ (h - b_dec) + b_enc -> top-k -> f
        Decoder: f -> W_dec @ f + b_dec -> h_reconstructed

    TopK enforces exactly k features active per input, giving clean sparsity
    control without the L1 penalty tuning required by ReLU SAEs.

    Args:
        d_model: Input dimension (model hidden size).
        n_features: Dictionary size (number of SAE features).
        k: Number of active features per input.
    """

    def __init__(self, d_model: int, n_features: int, k: int = 32):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features
        self.k = k

        # Encoder
        self.W_enc = nn.Parameter(torch.empty(d_model, n_features))
        self.b_enc = nn.Parameter(torch.zeros(n_features))

        # Decoder
        self.W_dec = nn.Parameter(torch.empty(n_features, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        # Initialize
        self._init_weights()

    def _init_weights(self):
        """Kaiming uniform initialization for encoder/decoder weights."""
        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))
        # Normalize decoder columns to unit norm
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to sparse feature activations.

        Args:
            x: Input tensor of shape (batch, d_model).

        Returns:
            Sparse feature activations of shape (batch, n_features).
            Exactly k entries are nonzero per row.
        """
        # Pre-encoder bias subtraction (centers the input)
        x_centered = x - self.b_dec
        # Linear encoding
        z = x_centered @ self.W_enc + self.b_enc  # (batch, n_features)
        # TopK selection
        topk_values, topk_indices = torch.topk(z, self.k, dim=-1)
        # Create sparse activation tensor
        f = torch.zeros_like(z)
        f.scatter_(-1, topk_indices, F.relu(topk_values))
        return f

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """Decode sparse features back to input space.

        Args:
            f: Sparse feature activations of shape (batch, n_features).

        Returns:
            Reconstructed input of shape (batch, d_model).
        """
        return f @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass: encode then decode.

        Args:
            x: Input of shape (batch, d_model).

        Returns:
            Tuple of (reconstructed, features, loss).
        """
        f = self.encode(x)
        x_hat = self.decode(f)
        # Reconstruction loss (MSE)
        loss = F.mse_loss(x_hat, x)
        return x_hat, f, loss

    def feature_reward_alignments(self, reward_direction: torch.Tensor) -> torch.Tensor:
        """Compute the reward alignment of each feature.

        This is the quantity (w_r^T @ d_i) for each feature i, where d_i is
        the i-th decoder column (feature direction in activation space).

        Args:
            reward_direction: The reward head weight vector, shape (d_model,).

        Returns:
            Tensor of shape (n_features,) with alignment scores.
            Positive = feature pushes reward up, negative = pushes reward down.
        """
        w_r = reward_direction.to(self.W_dec.device).float()
        # W_dec is (n_features, d_model), so W_dec @ w_r gives (n_features,)
        return self.W_dec.float() @ w_r

    def decompose_reward(
        self, x: torch.Tensor, reward_direction: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decompose the reward score into per-feature contributions.

        Args:
            x: Input activation of shape (batch, d_model).
            reward_direction: Reward head weights, shape (d_model,).

        Returns:
            Tuple of:
                - per_feature_contributions: shape (batch, n_features).
                  Contribution of each feature to the reward.
                - total_reward_approx: shape (batch,).
                  Sum of contributions (should approximate the actual reward).
        """
        f = self.encode(x)  # (batch, n_features)
        alignments = self.feature_reward_alignments(reward_direction)  # (n_features,)
        contributions = f * alignments.unsqueeze(0)  # (batch, n_features)
        total = contributions.sum(dim=-1)  # (batch,)
        return contributions, total

    def save(self, path: str) -> None:
        """Save the SAE to disk.

        Args:
            path: Directory path to save to.
        """
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "sae_weights.pt"))
        config = {
            "d_model": self.d_model,
            "n_features": self.n_features,
            "k": self.k,
        }
        with open(os.path.join(path, "sae_config.json"), "w") as f:
            json.dump(config, f)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "TopKSAE":
        """Load a saved SAE.

        Args:
            path: Directory containing sae_weights.pt and sae_config.json.
            device: Device to load onto.

        Returns:
            Loaded TopKSAE instance.
        """
        with open(os.path.join(path, "sae_config.json"), "r") as f:
            config = json.load(f)
        sae = cls(**config)
        sae.load_state_dict(torch.load(os.path.join(path, "sae_weights.pt"), map_location=device))
        sae.to(device)
        return sae


# ===========================================================================
# Activation Collector
# ===========================================================================


class ActivationCollector:
    """Collect activations from a reward model for SAE training.

    Collects final-token residual stream activations at a specified layer
    across a dataset of (prompt, response) pairs.

    Args:
        model: A RewardModel instance.
    """

    def __init__(self, model: RewardModel):
        self.model = model

    def collect(
        self,
        prompts: list[str],
        responses: list[str],
        layer: int,
        max_length: int = 2048,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Collect activations for a batch of (prompt, response) pairs.

        Args:
            prompts: List of prompts.
            responses: List of responses.
            layer: Which layer to collect from (-1 for embedding, 0..n-1 for layers).
            max_length: Maximum sequence length.
            show_progress: Show progress bar.

        Returns:
            Tensor of shape (n_samples, d_model) with collected activations.
        """
        assert len(prompts) == len(responses), "Prompts and responses must have same length"

        activations = []
        iterator = zip(prompts, responses)
        if show_progress:
            iterator = tqdm(list(iterator), desc=f"Collecting activations (layer {layer})")

        for prompt, response in iterator:
            _, cache = self.model.forward_with_cache(prompt, response, max_length=max_length)
            h = cache.residual_streams.get(layer)
            if h is not None:
                activations.append(h.cpu().float())
            else:
                raise RuntimeError(f"No activation found for layer {layer}")

        return torch.cat(activations, dim=0)  # (n_samples, d_model)

    def collect_from_dataset(
        self,
        dataset: list[dict],
        layer: int,
        prompt_key: str = "prompt",
        response_key: str = "response",
        max_samples: Optional[int] = None,
        max_length: int = 2048,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Collect activations from a list of dicts.

        Args:
            dataset: List of dicts with prompt and response keys.
            layer: Layer to collect from.
            prompt_key: Key for the prompt in each dict.
            response_key: Key for the response in each dict.
            max_samples: Maximum number of samples.
            max_length: Maximum sequence length.
            show_progress: Show progress bar.

        Returns:
            Tensor of shape (n_samples, d_model).
        """
        if max_samples is not None:
            dataset = dataset[:max_samples]
        prompts = [d[prompt_key] for d in dataset]
        responses = [d[response_key] for d in dataset]
        return self.collect(
            prompts, responses, layer, max_length=max_length, show_progress=show_progress
        )


# ===========================================================================
# SAE Trainer
# ===========================================================================


class SAETrainer:
    """Train a TopK SAE on collected activations.

    Args:
        d_model: Hidden dimension.
        n_features: Dictionary size (typically 4x, 8x, or 16x d_model).
        k: Number of active features per input.
        lr: Learning rate.
        batch_size: Training batch size.
        device: Device for training.
    """

    def __init__(
        self,
        d_model: int,
        n_features: Optional[int] = None,
        k: int = 32,
        lr: float = 3e-4,
        batch_size: int = 4096,
        device: str = "cuda",
    ):
        if n_features is None:
            n_features = d_model * 8  # 8x expansion by default
        self.sae = TopKSAE(d_model, n_features, k).to(device)
        self.lr = lr
        self.batch_size = batch_size
        self.device = device

    def train(
        self,
        activations: torch.Tensor,
        n_epochs: int = 5,
        log_every: int = 100,
        show_progress: bool = True,
    ) -> TopKSAE:
        """Train the SAE on collected activations.

        Args:
            activations: Tensor of shape (n_samples, d_model).
            n_epochs: Number of training epochs.
            log_every: Print loss every this many steps.
            show_progress: Show progress bar.

        Returns:
            The trained TopKSAE.
        """
        dataset = TensorDataset(activations.float())
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        optimizer = torch.optim.Adam(self.sae.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs * len(loader)
        )

        self.sae.train()
        step = 0
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            n_batches = 0
            iterator = loader
            if show_progress:
                iterator = tqdm(loader, desc=f"Epoch {epoch + 1}/{n_epochs}")

            for (batch,) in iterator:
                batch = batch.to(self.device)
                x_hat, f, loss = self.sae(batch)

                # Add auxiliary loss: encourage decoder columns to stay unit-norm
                dec_norm_loss = (self.sae.W_dec.norm(dim=1) - 1).pow(2).mean()
                total_loss = loss + 0.01 * dec_norm_loss

                optimizer.zero_grad()
                total_loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.sae.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                # Re-normalize decoder columns
                with torch.no_grad():
                    self.sae.W_dec.data = F.normalize(self.sae.W_dec.data, dim=1)

                epoch_loss += loss.item()
                n_batches += 1
                step += 1

                if show_progress:
                    iterator.set_postfix(loss=f"{loss.item():.6f}")

            avg_loss = epoch_loss / n_batches
            print(f"  Epoch {epoch + 1}/{n_epochs}: avg_loss = {avg_loss:.6f}")

        self.sae.eval()
        return self.sae


# ===========================================================================
# Feature Analyzer
# ===========================================================================


@dataclass
class FeatureInfo:
    """Information about a single SAE feature.

    Attributes:
        feature_idx: Index in the SAE dictionary.
        reward_alignment: Cosine-like alignment with reward direction.
        mean_activation: Mean activation across the dataset.
        activation_frequency: Fraction of inputs where this feature is active.
        top_activating_indices: Dataset indices where this feature activates most.
        top_activating_values: Activation values at those indices.
    """

    feature_idx: int
    reward_alignment: float
    mean_activation: float
    activation_frequency: float
    top_activating_indices: list[int] = field(default_factory=list)
    top_activating_values: list[float] = field(default_factory=list)


class FeatureAnalyzer:
    """Analyze SAE features in the context of reward computation.

    This class connects SAE features to the reward head, identifying which
    features contribute most to the reward and which are potential hacking
    vulnerabilities (features that respond to surface properties but are
    strongly aligned with the reward direction).

    Args:
        sae: A trained TopKSAE.
        model: The RewardModel (needed for reward direction).
    """

    def __init__(self, sae: TopKSAE, model: RewardModel):
        self.sae = sae
        self.model = model

        # Pre-compute reward alignments
        self._alignments = sae.feature_reward_alignments(model.reward_direction)

    def analyze_features(
        self,
        activations: torch.Tensor,
        top_k_examples: int = 10,
        show_progress: bool = True,
    ) -> list[FeatureInfo]:
        """Analyze all SAE features across a dataset.

        Args:
            activations: Tensor of shape (n_samples, d_model).
            top_k_examples: Number of top-activating examples to record per feature.
            show_progress: Show progress bar.

        Returns:
            List of FeatureInfo objects, one per feature, sorted by |reward_alignment|.
        """
        # Encode all activations
        with torch.no_grad():
            all_features = self.sae.encode(
                activations.to(next(self.sae.parameters()).device)
            )  # (n_samples, n_features)

        n_features = all_features.shape[1]
        features_info = []

        iterator = range(n_features)
        if show_progress:
            iterator = tqdm(iterator, desc="Analyzing features")

        for feat_idx in iterator:
            feat_acts = all_features[:, feat_idx].cpu()

            # Statistics
            mean_act = feat_acts.mean().item()
            freq = (feat_acts > 0).float().mean().item()
            alignment = self._alignments[feat_idx].item()

            # Top activating examples
            top_vals, top_idxs = torch.topk(feat_acts, min(top_k_examples, len(feat_acts)))

            features_info.append(
                FeatureInfo(
                    feature_idx=feat_idx,
                    reward_alignment=alignment,
                    mean_activation=mean_act,
                    activation_frequency=freq,
                    top_activating_indices=top_idxs.tolist(),
                    top_activating_values=top_vals.tolist(),
                )
            )

        # Sort by |reward_alignment|
        features_info.sort(key=lambda f: abs(f.reward_alignment), reverse=True)
        return features_info

    def top_reward_features(self, k: int = 20) -> list[tuple[int, float]]:
        """Return the k features most aligned with the reward direction.

        Args:
            k: Number of features.

        Returns:
            List of (feature_idx, alignment_score) tuples.
        """
        alignments = self._alignments.cpu()
        top_vals, top_idxs = torch.topk(alignments.abs(), min(k, len(alignments)))
        return [(top_idxs[i].item(), alignments[top_idxs[i]].item()) for i in range(len(top_idxs))]

    def bottom_reward_features(self, k: int = 20) -> list[tuple[int, float]]:
        """Return the k features most anti-aligned with reward (push reward down).

        Args:
            k: Number of features.

        Returns:
            List of (feature_idx, alignment_score) tuples, sorted by alignment
            (most negative first).
        """
        alignments = self._alignments.cpu()
        bottom_vals, bottom_idxs = torch.topk(-alignments, min(k, len(alignments)))
        return [
            (bottom_idxs[i].item(), alignments[bottom_idxs[i]].item())
            for i in range(len(bottom_idxs))
        ]

    def decompose_reward_for_input(
        self,
        prompt: str,
        response: str,
        layer: int,
        max_length: int = 2048,
    ) -> tuple[list[tuple[int, float]], float]:
        """Decompose a specific input's reward through SAE features.

        Args:
            prompt: The user prompt.
            response: The response.
            layer: Which layer's SAE to use.
            max_length: Maximum sequence length.

        Returns:
            Tuple of:
                - List of (feature_idx, contribution) tuples, sorted by |contribution|.
                - The total reconstructed reward approximation.
        """
        _, cache = self.model.forward_with_cache(prompt, response, max_length=max_length)
        h = cache.residual_streams.get(layer)
        if h is None:
            raise ValueError(f"No activation found for layer {layer}")

        with torch.no_grad():
            contributions, total = self.sae.decompose_reward(
                h.float().to(next(self.sae.parameters()).device),
                self.model.reward_direction,
            )

        # Get non-zero contributions
        contribs = contributions[0].cpu()  # (n_features,)
        nonzero_mask = contribs != 0
        nonzero_indices = torch.where(nonzero_mask)[0]
        nonzero_values = contribs[nonzero_mask]

        # Sort by absolute value
        sorted_order = torch.argsort(nonzero_values.abs(), descending=True)
        result = [(nonzero_indices[i].item(), nonzero_values[i].item()) for i in sorted_order]

        return result, total[0].item()

    def plot_alignment_histogram(
        self,
        save_path: Optional[str] = None,
        figsize: tuple[int, int] = (10, 5),
    ) -> None:
        """Plot histogram of feature reward alignments.

        Args:
            save_path: Optional path to save figure.
            figsize: Figure size.
        """
        import matplotlib.pyplot as plt

        alignments = self._alignments.cpu().numpy()

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.hist(alignments, bins=100, alpha=0.7, color="#4CAF50", edgecolor="black", linewidth=0.3)
        ax.axvline(x=0, color="red", linestyle="--", alpha=0.7)
        ax.set_xlabel("Reward Alignment (w_r^T @ d_i)")
        ax.set_ylabel("Number of Features")
        ax.set_title("Distribution of SAE Feature Reward Alignments")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close()

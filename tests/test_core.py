"""Tests for the reward-lens core model wrapper."""

import importlib.util

import pytest
import torch
import torch.nn as nn

from reward_lens.model_adapters import (
    LlamaAdapter,
    get_adapter,
)


class MockConfig:
    """Minimal mock config."""

    num_attention_heads = 4
    model_type = "llama"
    hidden_size = 64


class MockLayer(nn.Module):
    """Minimal mock transformer layer."""

    def __init__(self, d_model=64):
        super().__init__()
        self.self_attn = nn.Linear(d_model, d_model)
        self.mlp = nn.Linear(d_model, d_model)

    def forward(self, x):
        return (x + self.self_attn(x) + self.mlp(x),)


class MockBackbone(nn.Module):
    """Minimal mock backbone."""

    def __init__(self, n_layers=4, d_model=64):
        super().__init__()
        self.embed_tokens = nn.Embedding(100, d_model)
        self.layers = nn.ModuleList([MockLayer(d_model) for _ in range(n_layers)])


class MockRewardModel(nn.Module):
    """Minimal mock reward model mimicking Llama architecture."""

    def __init__(self, n_layers=4, d_model=64):
        super().__init__()
        self.model = MockBackbone(n_layers, d_model)
        self.score = nn.Linear(d_model, 1)
        self.config = MockConfig()

    def forward(self, input_ids, attention_mask=None, **kwargs):
        x = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            x = layer(x)[0]
        # Take last token
        logits = self.score(x[:, -1:, :])
        return type("Output", (), {"logits": logits})()


class TestLlamaAdapter:
    """Tests for the Llama adapter."""

    def test_get_reward_head_params(self):
        model = MockRewardModel()
        adapter = LlamaAdapter()
        weight, bias = adapter.get_reward_head_params(model)
        assert weight.shape == (64,)
        assert isinstance(bias, float)

    def test_get_layers(self):
        model = MockRewardModel(n_layers=4)
        adapter = LlamaAdapter()
        layers = adapter.get_layers(model)
        assert len(layers) == 4

    def test_n_layers(self):
        model = MockRewardModel(n_layers=6)
        adapter = LlamaAdapter()
        assert adapter.n_layers(model) == 6

    def test_n_heads(self):
        model = MockRewardModel()
        adapter = LlamaAdapter()
        assert adapter.n_heads(model) == 4

    def test_get_attn_module(self):
        model = MockRewardModel()
        adapter = LlamaAdapter()
        layer = adapter.get_layers(model)[0]
        attn = adapter.get_attn_module(layer)
        assert attn is not None

    def test_get_mlp_module(self):
        model = MockRewardModel()
        adapter = LlamaAdapter()
        layer = adapter.get_layers(model)[0]
        mlp = adapter.get_mlp_module(layer)
        assert mlp is not None

    def test_extract_layer_output_tuple(self):
        adapter = LlamaAdapter()
        t = torch.randn(2, 10, 64)
        result = adapter.extract_layer_output((t, None))
        assert torch.equal(result, t)

    def test_extract_reward(self):
        model = MockRewardModel()
        adapter = LlamaAdapter()
        input_ids = torch.randint(0, 100, (1, 10))
        output = model(input_ids)
        reward = adapter.extract_reward(output, {"input_ids": input_ids})
        assert reward.dim() == 0  # scalar

    def test_get_embedding(self):
        model = MockRewardModel()
        adapter = LlamaAdapter()
        embed = adapter.get_embedding(model)
        assert isinstance(embed, nn.Embedding)


class TestAdapterAutoDetection:
    """Tests for automatic adapter detection."""

    def test_llama_detection(self):
        model = MockRewardModel()
        adapter = get_adapter(model, "some/llama-reward-model")
        assert isinstance(adapter, LlamaAdapter)

    def test_detection_by_class_name(self):
        model = MockRewardModel()
        model.__class__.__name__ = "LlamaForSequenceClassification"
        adapter = get_adapter(model, "")
        assert isinstance(adapter, LlamaAdapter)


@pytest.mark.skipif(
    importlib.util.find_spec("sae_lens") is None,
    reason=(
        "reward_lens.sae moved behind the [dict] extra, because a sparse\n"
        "dictionary is a candidate generator and never a claim substrate. Install it with\n"
        "pip install 'reward-lens[dict]' to run these. The gate itself is asserted in\n"
        "tests/acceptance/test_w0_5_migration.py, which runs on an install without the extra."
    ),
)
class TestTopKSAE:
    """Tests for the TopK SAE."""

    def test_encode_decode(self):
        from reward_lens.sae import TopKSAE

        sae = TopKSAE(d_model=64, n_features=256, k=8)
        x = torch.randn(10, 64)
        f = sae.encode(x)
        assert f.shape == (10, 256)
        # Check exactly k nonzero per row
        nonzero_counts = (f != 0).sum(dim=1)
        assert torch.all(nonzero_counts == 8)

        x_hat = sae.decode(f)
        assert x_hat.shape == (10, 64)

    def test_forward(self):
        from reward_lens.sae import TopKSAE

        sae = TopKSAE(d_model=64, n_features=256, k=8)
        x = torch.randn(10, 64)
        x_hat, f, loss = sae(x)
        assert x_hat.shape == x.shape
        assert f.shape == (10, 256)
        assert loss.dim() == 0

    def test_feature_reward_alignments(self):
        from reward_lens.sae import TopKSAE

        sae = TopKSAE(d_model=64, n_features=256, k=8)
        w_r = torch.randn(64)
        alignments = sae.feature_reward_alignments(w_r)
        assert alignments.shape == (256,)

    def test_decompose_reward(self):
        from reward_lens.sae import TopKSAE

        sae = TopKSAE(d_model=64, n_features=256, k=8)
        x = torch.randn(5, 64)
        w_r = torch.randn(64)
        contributions, total = sae.decompose_reward(x, w_r)
        assert contributions.shape == (5, 256)
        assert total.shape == (5,)

    def test_save_load(self, tmp_path):
        from reward_lens.sae import TopKSAE

        sae = TopKSAE(d_model=64, n_features=128, k=4)
        save_dir = str(tmp_path / "sae_test")
        sae.save(save_dir)

        loaded = TopKSAE.load(save_dir)
        assert loaded.d_model == 64
        assert loaded.n_features == 128
        assert loaded.k == 4
        # Check weights are the same
        assert torch.allclose(sae.W_enc.data, loaded.W_enc.data)

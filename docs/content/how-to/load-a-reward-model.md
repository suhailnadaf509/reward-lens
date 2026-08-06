# Load or wrap a reward model

**You have a model and tokenizer in memory. How do you point reward-lens at them?**

Hand them to `wrap_hf_model`. It reads the reward head into a readout, builds the site map, runs a fast conformance quick-check (the score is deterministic on repeat, and the linear readout reproduces the head in fp32), and returns a `ClassifierRM` you can score with. No network, no download.

```python
import torch
from transformers import LlamaConfig, LlamaForSequenceClassification, AutoTokenizer
from reward_lens.signals import wrap_hf_model

# In practice `model` is your loaded RM, e.g.
#   AutoModelForSequenceClassification.from_pretrained("Skywork/Skywork-Reward-Llama-3.1-8B-v0.2")
# Here we build a small classifier so the page runs on CPU with no download.
tok = AutoTokenizer.from_pretrained("gpt2")
tok.pad_token = tok.eos_token
torch.manual_seed(0)
cfg = LlamaConfig(vocab_size=tok.vocab_size, hidden_size=32, intermediate_size=64,
                  num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                  num_labels=1, pad_token_id=tok.pad_token_id)
model = LlamaForSequenceClassification(cfg)

signal = wrap_hf_model(model, tok)
print(signal.caps)
# Capability.SCORES|PREFIX_SCORES|ACTIVATIONS|GRADIENTS|HVP|LINEAR_READOUT

ev = signal.score([("Capital of France?", "Paris.")])
print(ev.value.values)       # [-0.16582201]
print(ev.trust, ev.gauge)    # EXPLORATORY invariant
```

The score comes back as evidence, not a bare float: trust `EXPLORATORY` is the honest default (unaudited, [not wrong](../discipline/trust-ladder.md)), and gauge `invariant` says this number reads the same on any model. Every instrument you attach returns the same kind of receipt.

## The zero-setup CPU signal

If you just want a real signal to try an instrument against, `from_tiny` builds one with no arguments and no download: a real `LlamaForSequenceClassification` (hidden 32, two layers) so every hook, readout, gradient, and Hessian-vector product sees the same module tree an 8B Skywork model would. Only the magnitudes differ, and the weights are random, so no individual score is meaningful.

```python
from reward_lens.signals import from_tiny

signal = from_tiny(seed=0)
print(signal.meta.n_layers, signal.meta.d_model)   # 2 32
print(signal.score([("Why is the sky blue?",
                     "Rayleigh scattering favours short wavelengths.")]).value.values)
# [-0.12213741]
```

## Loading from the hub is gated on this machine

`load_signal` is the production entry point, but a bare hub id refuses to download onto hardware that cannot hold the model. It raises rather than starting a multi-gigabyte pull that would only run out of memory:

```python
from reward_lens.signals import load_signal

load_signal("Skywork/Skywork-Reward-Llama-3.1-8B-v0.2")
# NotImplementedError: loading 'Skywork/Skywork-Reward-Llama-3.1-8B-v0.2' from the HF hub is
# GPU/download-gated on this machine (8 GB GPU, cannot hold the 8B/27B campaign models). The
# convention sniffed as 'unknown' and the code path is implemented; set allow_download=True to
# attempt it on adequate hardware, or use wrap_hf_model/from_tiny for a local model.
```

!!! warning "Needs a GPU"
    `load_signal(hub_id, allow_download=True)` downloads the weights, sniffs the loading convention, and delegates to `wrap_hf_model`. The 8B and 27B campaign models need fp32 head precision and do not fit an 8 GB GPU. On adequate hardware the call is the same; the two ungated paths above never touch the network.

See also: [Classifier reward models](../models-and-signals/classifier-rms.md), [Write an adapter](write-an-adapter.md). API: [`wrap_hf_model`](../reference/signals.md#reward_lens.signals.loaders.wrap_hf_model), [`from_tiny`](../reference/signals.md#reward_lens.signals.loaders.from_tiny), [`load_signal`](../reference/signals.md#reward_lens.signals.loaders.load_signal).

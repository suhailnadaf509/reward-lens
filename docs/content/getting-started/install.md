# Install

**What has to be on the machine before any of this runs?** Less than you would guess. The measurement half of the library is pure numpy and opens on any laptop. The model half wants weights and, for the real 8B graders, a GPU. You can learn the whole thing on the CPU half first.

```bash
pip install reward-lens
```

Python 3.10 or newer. That base install brings **nothing compiled**: numpy, scipy, pandas, scikit-learn and a CLI. No torch, no transformers.

The extras are where the weight is, and each one buys a named capability:

| Extra | What it adds |
|---|---|
| `white-box` | torch and transformers. Everything that reads a model's activations or gradients |
| `verifier` | the program-substrate toolchain: coverage, mutation, property search. No GPU, no model |
| `record` | columnar scalars and tensor shards for a training record. The reader works without it |
| `organisms` | training planted model organisms, which needs `peft` and `datasets` |
| `sampling` | vLLM, for a policy you want to sample from at speed |
| `dict` | sparse-autoencoder tooling for feature-reward alignment |
| `trl` | the TRL tap, for recording a live training run |
| `viz` | matplotlib and seaborn, for the figure helpers |
| `dev` | the test suite and the lint toolchain, for a source checkout |
| `all` | every one of the above |

Install them the usual way, quoted so the shell does not eat the brackets: `pip install "reward-lens[white-box]"`.

The split is deliberate rather than tidy. A record is written by whoever ran the training and read by whoever is auditing it, and the auditor is the person most likely to be on a base install, so the reader dispatches on the file rather than on the environment.

!!! tip "Nothing on your machine at all"
    The [Colab tour](https://colab.research.google.com/drive/1x5zG07HdsWlNsJmkl2ddJ1yalmwwujfY?usp=sharing) installs the library into a free hosted runtime and walks the whole thing end to end. It is the way to try the library before you commit anything to your own environment.

## Importing the library never imports torch

That is asserted in CI rather than described here, and you can confirm it in three lines:

```python
import sys
import reward_lens, reward_lens.core, reward_lens.stats

print(reward_lens.__version__)
"torch" in sys.modules      # False, nothing model-touching was pulled in
"numpy" in sys.modules      # True
```

Anything that reads a model imports from its own subsystem, `from reward_lens.signals import ...` or `from reward_lens.policy import ...`, and only then does torch load. A module that needs an extra you do not have raises `ExtraRequiredError` naming the extra and the pip command, rather than an `ImportError` that leaves you guessing.

So a continuous-integration box, a notebook on a plane, a reviewer checking your arithmetic: none of them need a GPU to run the discipline.

## What actually bites, before you load a real model

- **You need the weights.** `reward-lens` hooks a live model in memory and reads the reward head directly, keeping it in fp32 so the readout matches the head exactly. Anything you can only reach through an API, with no weights to hold, is out of scope. If `transformers` can load it, an adapter can open it up.
- **The good graders are gated.** Skywork and ArmoRM sit behind a license click on the Hugging Face Hub. Accept the terms on the model page, then `huggingface-cli login` with a token, or the load returns a 401 and nothing else.
- **An 8B trace wants a GPU.** An 8B reward model in fp32 does not fit an 8 GB card, and the campaign numbers on this site were measured on larger hardware and committed as artifacts. That is a hardware fact, not a soft limit: the 8B code path names the call it would make and refuses rather than pretend. Everything conceptual runs first on the CPU through `from reward_lens.signals import from_tiny`, which builds a real (small) reward model with no download.

!!! warning "Needs a GPU"
    Loading a flagship grader from the Hub is gated on this machine. `load_signal("Skywork/Skywork-Reward-Llama-3.1-8B-v0.2")` raises `NotImplementedError` by default and names the exact dispatch; passing `allow_download=True` opts in, and then it needs the memory to hold the model. The ungated loaders, `wrap_hf_model` for a model you already have and `from_tiny` for a toy one, never touch the network.

## Where to start

Install done, the fastest next step is one of the two quickstarts on the [getting-started page](index.md): one opens a real grader end to end, the other runs the whole arc on the CPU with `from_tiny`. If your first question is instead whether your particular grader is even supported, a classifier head or a DPO checkpoint or an LLM judge, that is answered on [models and signals](../models-and-signals/index.md). Adding a model family the adapters do not yet cover is [one small class](../how-to/write-an-adapter.md).

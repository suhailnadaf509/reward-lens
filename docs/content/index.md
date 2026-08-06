# reward-lens

Every model trained with RLHF was shaped by a reward model. That model sat in the loop and decided, on pair after pair, which of two answers was better. It is the closest thing the whole pipeline has to a written-down definition of what we asked for, and almost nobody has looked inside one.

That is strange, because the reward model is exactly where alignment gets decided. A policy does not optimize your intentions. It optimizes the number this model hands back. Whatever the reward model fails to measure becomes the precise thing the policy is free to exploit. If you want to know why a model learned to pad its answers, or agree with whatever you said, or wrap everything in confident structure, the honest place to look is not the policy. It is the function that rewarded it.

reward-lens is the instrument for looking. Whatever grades your rollouts, a classifier reward model, an LLM judge, a DPO checkpoint, a process reward model, the same battery of tools opens it up. And every number it hands back arrives with a second number: how far you should trust it.

## Watch a preference form

A reward model produces its score at the end, but the decision is built along the way. Take the hidden state at every layer, read it out along the reward direction, and you get the reward the model would assign if it stopped there. Do that for both answers in a pair and track the gap, and you can watch the preference appear.

![The reward-lens trace of a preference pair on Skywork, layer by layer.](assets/figures/lens-curve-light.svg#only-light){ .rl-fig .rl-fig--hero }
![The reward-lens trace of a preference pair on Skywork, layer by layer.](assets/figures/lens-curve-dark.svg#only-dark){ .rl-fig .rl-fig--hero }

/// caption
**The model makes up its mind late.** Both answers sit tangled and flat for the first two thirds of the network. Then they split hard, and most of the margin is written by the last few layers. On Skywork the split lands at layer 30 of 32.
///

That is one reward model, one pair, one tool. The reader who wants the mechanism can have it in [fifteen minutes on a GPU or one minute on a laptop](getting-started/index.md).

## The result that shaped the library

Here is the finding that should make you take the rest seriously, because it is the kind of thing a library selling itself would bury.

Run the cheap tool that says which components *explain* the reward, then run the expensive one that says which components *cause* it, and on Skywork the two rankings correlate at Spearman \(\rho = -0.171\). Negative. The late layers dominate the explanation. The early layers carry the cause. Quote the cheap tool as if it were the causal one and you have published a plausible, backwards result.

The first version of this library was a bag of tools that could produce that mistake and never catch it. So the rebuild put a discipline underneath the tools. Every measurement now carries its own uncertainty, its own provenance, and a trust level that the tool computes rather than claims. A number that has been checked against a case with a known answer is marked as such. A number that has not is marked exploratory, and the type system will not let you quote it as more. The honesty is not a disclaimer at the bottom of the page. It is in the return type.

## Sixty seconds, on your laptop

No GPU, no download. This builds a real miniature reward model on CPU, measures it, and shows you the receipt the measurement comes back with.

```python
from reward_lens.signals import from_tiny
from reward_lens.measure import base as mb
from reward_lens.measure.battery import DirectLinearAttribution
from reward_lens.data.builtin.diagnostic_v3 import load_diagnostic_v3
from reward_lens.data.schema import DataView

signal = from_tiny(seed=0)                                  # a real model, on CPU, no download
view = DataView(list(load_diagnostic_v3()["helpfulness"].items)[:8])

ev = mb.run(DirectLinearAttribution(), mb.Context(signal=signal, view=view))
print(ev.value["dominant_component"])   # which components wrote the reward difference
print(ev.trust)                         # EXPLORATORY, until this measurement earns a scorecard
```

The trust level says `EXPLORATORY` because nothing has yet checked this tool against a known answer. [Making it climb](concepts/measurement-you-can-trust.md) is the whole second half of the story.

## Or run it without installing anything

There is a guided tour notebook that takes the whole library end to end in a browser: build a reward model whose answer you already know, read it out along its one direction, watch a measurement come back as evidence rather than a float, intervene on it, and freeze a study that adjudicates itself. It installs itself on Colab's free tier, and the CPU path never asks for a GPU.

[Open the tour in Colab](https://colab.research.google.com/drive/1x5zG07HdsWlNsJmkl2ddJ1yalmwwujfY?usp=sharing){ .md-button .md-button--primary }

## Where to go next

<div class="grid cards" markdown>

-   __Get your model open__

    Install, load a reward model or wrap the one you already have, and trace your first pair. Fifteen minutes on a GPU, one on a laptop.

    [:octicons-arrow-right-24: Getting started](getting-started/index.md)

-   __Understand the picture__

    One direction, a pair as a controlled experiment, and where the preference forms. The mental model the rest of the site assumes.

    [:octicons-arrow-right-24: Concepts](concepts/index.md)

-   __Reach for a tool__

    The battery: the reward lens, attribution, patching, the bias probes, the intervention algebra. One page each, with the math and when not to use it.

    [:octicons-arrow-right-24: Instruments](instruments/index.md)

-   __Trust the number__

    Why a measurement carries credentials, how a tool earns calibration against a planted answer key, and why cross-model numbers lie without a shared frame.

    [:octicons-arrow-right-24: The measurement discipline](discipline/index.md)

-   __Run it instead of reading it__

    The guided tour, in a browser, with nothing installed. Every idea on this site, executed cell by cell on Colab's free tier.

    [:octicons-arrow-right-24: The Colab tour](https://colab.research.google.com/drive/1x5zG07HdsWlNsJmkl2ddJ1yalmwwujfY?usp=sharing)

</div>

```bash
pip install reward-lens
```

!!! note "What it runs on"
    Python 3.10 or newer. **The base install brings nothing compiled**: numpy, scipy, pandas, scikit-learn and a CLI, and `import reward_lens` never imports torch. That is deliberate, because a record is written by whoever ran the training and read by whoever is auditing it, and the auditor is exactly the person most likely to be on a base install. The whole grader card, the verifier series, the record reader and the entire trust story run there.

    Reading a model's activations or gradients needs the `white-box` extra, which is where torch and transformers live: `pip install 'reward-lens[white-box]'`. A real 8B trace wants a GPU with enough memory to hold the model. Everything conceptual runs on CPU first.

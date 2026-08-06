# The battery

**You have a reward model and a preference pair. What can you actually measure about the decision?**

Eleven things, as it turns out, and they all start from the same fact. The reward is a linear readout of the final hidden state, \(r = w_r^\top h + b\), so the whole opinion of the model lives in one direction \(w_r\). Every instrument on this page is a different way of projecting the computation onto that direction and reading off what you find. The residual stream is a running sum, so the projection decomposes cleanly: by depth, by component, by concept, by objective. That is the battery.

![The reward model's whole opinion is one direction; every activation projects onto it.](../assets/figures/reward-projection-light.svg#only-light){ .rl-fig .rl-fig--hero }
![The reward model's whole opinion is one direction; every activation projects onto it.](../assets/figures/reward-projection-dark.svg#only-dark){ .rl-fig .rl-fig--hero }

/// caption
**One direction carries the entire score.** A hidden state is a point in a high-dimensional space, but the reward only reads its shadow along \(w_r\). Two completions differ in reward exactly by how far apart their shadows fall. Every instrument below measures that shadow from a different angle.
///

Because \(w_r\) is fixed by the reward head and known exactly, none of this is probing or guessing. The projection is arithmetic. What varies between instruments is which slice of the computation they project, and, more importantly, what kind of claim the result licenses.

## Three chips, read left to right

Every instrument page opens with three chips. They tell you, before you read a word of prose, what the number means and how far you can carry it.

- **Tier** is the kind of claim. *Observational* instruments read the computation as it already ran. *Causal* instruments intervene, change one thing, and measure what moved. *Vulnerability* instruments score the model against a named failure mode. The distinction is not decorative. On real models, the observational and causal instruments have ranked the same components in nearly opposite order, and knowing which tier you are holding is what keeps you from reading a correlation as a mechanism.
- **Gauge** is whether the number survives a change of coordinates. *Invariant* quantities (a depth fraction, an effect size) mean the same thing in any basis, so you can compare them across models freely. *Raw only* quantities are coordinate-dependent and honest about it: a raw cosine between two models' features is a number about your axes, not about the models. *Needs a frame* quantities can be compared across models, but only once you express both in a shared [frame](../discipline/gauge-and-frames.md), or the library refuses. The chip tells you which rung you are on so you never accidentally compare two raw numbers that were never in the same coordinates.
- **Works on** is the substrate the instrument attaches to: bare scores, activations plus the linear readout, an SAE, a multi-objective head. It is the capability the signal must declare. Ask an instrument for something the signal cannot give and it raises rather than returns a fabricated number.

## The tier key

<div class="rl-key">
  <span><span class="rl-dot rl-dot--observational"></span> <strong>Observational</strong>: reads the existing computation. No intervention. Fast, exact, and a hypothesis about importance rather than a demonstration of it.</span><br>
  <span><span class="rl-dot rl-dot--causal"></span> <strong>Causal</strong>: changes one component and measures the effect on the reward. Slower, and the only tier that licenses "this caused that."</span><br>
  <span><span class="rl-dot rl-dot--vulnerability"></span> <strong>Vulnerability</strong>: scores the model against a specific way reward models fail under optimization. Most of these live in [the index library](index-library.md); two of the observational instruments below, the bias battery and the conflict matrix, inherit that purpose while staying strictly read-only in mechanism.</span>
</div>

## The eleven

The eight observational instruments read; the three causal ones intervene. Every one returns a gated [Evidence](../concepts/measurement-you-can-trust.md) object, not a bare float, and every one on this page currently reports at trust level EXPLORATORY, because no calibration provider is wired in this release. That is a statement about the epistemics, not the arithmetic: the projection is exact, but until a [scorecard](../discipline/calibration-and-organisms.md) certifies the instrument against organisms with known ground truth, the number is exploratory by construction.

| Instrument | What it answers | Tier | Gauge | Works on |
|---|---|---|---|---|
| [Reward lens and crystallization](lens-crystallization.md) | Where across depth does the margin form? | Observational | invariant | activations + linear readout |
| [Component attribution](attribution.md) | Which component wrote the score? | Observational | invariant | activations + linear readout |
| [Patch grid](patch-grid.md) | Which component *causes* the score? | Causal | invariant | activations + linear readout |
| [Path effects](path-effects.md) | Which sender-to-receiver path carries it? | Causal | invariant | activations + linear readout |
| [Concept dose-response](concept-dose-response.md) | Does pushing a concept move the reward? | Causal | raw only | activations + linear readout |
| [Bias battery](bias-battery.md) | Which surface feature is the reward biased toward? | Observational | invariant | scores |
| [Prompt SNR](prompt-snr.md) | How much signal versus noise per axis? | Observational | invariant | scores |
| [Conflict matrix](conflict-matrix.md) | Which reward terms fight each other? | Observational | raw only | activations + linear readout |
| [Circuit overlap](circuit-overlap.md) | Do two models use the same circuit? | Observational | invariant | two models, activations + linear readout |
| [Feature-reward alignment](feature-alignment.md) | Which SAE features drive the reward? | Observational | raw only | an SAE + linear readout |
| [Multi-objective geometry](multi-objective-geometry.md) | How do a head's objectives relate? | Observational | raw only | a multi-objective readout |

## How to run any of them

The call shape is the same for all eleven. Build a signal, choose a data view, hand both to the runner. The runner checks the capability, applies the gauge gate, and returns Evidence.

```python
from reward_lens.signals import from_tiny
from reward_lens.data.builtin.diagnostic_v3 import load_diagnostic_v3
from reward_lens.data.schema import DataView
from reward_lens.measure import base as mb
from reward_lens.measure.battery import DirectLinearAttribution

signal = from_tiny(seed=0)                                        # real LlamaForSequenceClassification, CPU, no download
view = DataView(list(load_diagnostic_v3()["helpfulness"].items)[:5])
ev = mb.run(DirectLinearAttribution(), mb.Context(signal=signal, view=view))
print(ev.observable, "|", ev.trust, "|", ev.gauge)
# DirectLinearAttribution | EXPLORATORY | invariant
```

`from_tiny` builds a genuine two-layer reward model on CPU in under a minute, which is enough to exercise every observational instrument end to end. It is not enough to *reproduce* the 8B results, because a two-layer model has no late-layer structure to find. Throughout the battery, the tiny runs are there so you can see the machinery move; the Skywork and ArmoRM numbers are measured results from committed artifacts, and the call that produces them is marked as needing a GPU.

Start with [the reward lens](lens-crystallization.md). It is the instrument that tells the others where to look.

## The instruments that are not about a preference pair

The eleven above all start from one reward model and one pair. Two other groups sit in the nav beside them and start somewhere else entirely, so they are worth naming here rather than leaving to be found.

**On a training run.** These read a recorded reinforcement-learning run rather than a scoring model, and each of them answers a question about a step rather than about a decision.

| Instrument | What it answers |
|---|---|
| [The cost book](cost-book.md) | How many nats did that step spend, and how many did the movement need? |
| [Credit geometry](credit.md) | Which turn got the reward, and does the attribution add up to the whole step? |
| [The four books](reconciliation.md) | Do the two independent predictions of the behaviour change agree? |
| [Feature heritability](heritability.md) | Can a gradient step move this feature at all, or does it only look susceptible? |
| [The reward covector](covector.md) | Which direction in the residual stream would raise the expected reward? |

**On a grader.** These treat the grader as the object of study rather than as the source of a score.

| Instrument | What it answers |
|---|---|
| [The grader card](grader-card.md) | What does a buyer know about this verifier before spending a month of compute on it? |
| [Reference materials and selection](reference-materials.md) | Did your method beat the dumb one, and against a reference with an uncertainty of its own? |

Every one of them returns a refusal on at least one real subject, with the numbers that produced it. That is not a gap in the pages; it is most of what they have to teach.

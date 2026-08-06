# Coming from v1.0

**Did the rewrite break your old code? Most of it, yes, and this page is the map.** The v1 flat API
was kept alive through v2.0 by a lazy accessor and retired in v3.0 once its two-release deprecation
condition was met. Pin `reward-lens<3` and move one call at a time. What you gain in exchange is
not more tools; it is a receipt on every number.

## What still imports

`import reward_lens` pulls nothing heavier than numpy and now exposes `__version__`, `core` and
`stats`. Two modules that never went through the accessor are untouched:

```python
from reward_lens.model import RewardModel          # the hooked model wrapper
from reward_lens.sae import TopKSAE, SAETrainer    # the top-k sparse autoencoder
```

Both need the `[white-box]` extra, both import on CPU, and neither returns `Evidence`. Everything
else from the old flat surface is gone: `RewardLens`, `ComponentAttribution`, `ActivationPatcher`,
`PathPatcher`, `DistortionAnalyzer`, `DivergenceAwarePatching`, `MisalignmentCascadeDetector`,
`RewardConflictAnalyzer`, `ConceptExtractor`, `quick_concept_analysis`, `HackingDetector` and
`ModelComparator`. The [legacy reference](reference/legacy.md) has the full list with what happened
to each.

## Where each primitive moved

| Retired primitive | What it did | Where to write new code |
|---|---|---|
| `RewardLens` | reward projected across depth | [`LensCrystallization`](instruments/lens-crystallization.md) |
| `ComponentAttribution` | per-component reward ledger | [`DirectLinearAttribution`](instruments/attribution.md) |
| `ActivationPatcher`, `PathPatcher` | causal patching | [`PatchGrid`](instruments/patch-grid.md), [`PathEffect`](instruments/path-effects.md), and the [intervention algebra](instruments/interventions.md) |
| `ConceptExtractor`, `quick_concept_analysis` | concept directions and steering | [concepts](concepts/index.md) with [`ConceptDoseResponse`](instruments/concept-dose-response.md) |
| `RewardConflictAnalyzer` | inter-objective conflict | [`ConflictMatrix`](instruments/conflict-matrix.md) |
| `DistortionAnalyzer` | reward distortion | [the index library](instruments/index-library.md) |
| `HackingDetector` | bias and hacking scan | [the bias battery](instruments/bias-battery.md) and [the index library](instruments/index-library.md) |
| `ModelComparator` | cross-model comparison | [gauge and frames](discipline/gauge-and-frames.md) |
| `TopKSAE`, `SAETrainer` | SAE features on the reward | still `reward_lens.sae`; see [feature-reward alignment](instruments/feature-alignment.md) |

No row is a rename. In each of them the replacement hands back `Evidence` with an uncertainty, a
gauge status and a computed trust level, where the retired name handed back a numpy array or a
report object. Wrapping the old names would have meant a shim whose only job is to throw the
uncertainty away, which is why they went instead.

## What moving buys you

The 1.0 library was a bag of primitives that returned numpy arrays and floats. Running them at scale is what motivated the rebuild, so the thing you gain is not more tools. It is a receipt on every number.

- **Evidence, not a bare float.** A 2.0 measurement returns an `Evidence` object carrying the value, its uncertainty with an effective sample size that counts unique content rather than cloned rows, a gauge status, a calibration reference, provenance back to the inputs, and a [trust level](discipline/trust-ladder.md). See [a measurement you can trust](concepts/measurement-you-can-trust.md).
- **Trust is computed, not asserted.** You never set the trust level. Three gates do: calibration, gauge, and registration. An uncalibrated instrument caps at exploratory no matter how confident the number looks.
- **Calibration against organisms.** An instrument earns a scorecard by recovering structure that was planted by construction, so its number on a real model cites a case with known ground truth. See [calibration and organisms](discipline/calibration-and-organisms.md).
- **A frame for cross-model comparison.** Two reward directions in raw coordinates are not comparable, and 2.0 refuses the comparison rather than handing back a coordinate artifact. See [gauge and frames](discipline/gauge-and-frames.md).
- **More than a classifier.** 1.0 spoke one reward dialect. The `RewardSignal` protocol now covers eight substrates (classifier, generative judge, process, implicit DPO, rubric, trajectory, dense, ensemble), and every instrument attaches to all of them unchanged. See [models and signals](models-and-signals/index.md).

The honesty thread runs straight through this. Rank a model's components by how much attribution assigns them, rank them again by how much causal patching says they carry, and on `Skywork-Reward-Llama-3.1-8B-v0.2` the two rankings correlate at Spearman \(\rho = -0.171\). Negative. On ArmoRM the same comparison sits near zero at \(+0.047\). The place a reward visibly accumulates is not always the place that causes it, and 1.0 had no way to flag that a cheap observational number was being quoted as a causal one. 2.0 does. See [observational vs causal](concepts/observational-vs-causal.md) and [interpreting results honestly](caveats.md).

## The same analysis, before and after

A 1.0 script that traces a preference across depth and attributes it, next to its 2.0 equivalent. The shape is the same. The output is not.

```python
# The old way: loads an 8B reward model, returns bare arrays.
# These three names no longer import; this block is here to be read, not run.
from reward_lens import RewardModel, RewardLens, ComponentAttribution

model = RewardModel.from_pretrained("Skywork/Skywork-Reward-Llama-3.1-8B-v0.2")
lens = RewardLens(model)
result = lens.trace(prompt, chosen, rejected)         # margin traced across layers
attr = ComponentAttribution(model)
comp = attr.attribute(prompt, chosen, rejected)
comp.differential_contributions                       # a numpy array, no provenance
```

The 2.0 version reads from a typed data view and runs each observable through the gated runner, so `LensCrystallization` and `DirectLinearAttribution` come back wrapped in Evidence.

!!! warning "Needs a GPU"
    Loading the 8B model in fp32 needs a GPU it fits on. On smaller hardware this call is gated. Swap the signal for the CPU tiny model below to run the same API today.

```python
from reward_lens.signals import load_signal
from reward_lens.measure import base as mb
from reward_lens.measure.battery import LensCrystallization, DirectLinearAttribution
from reward_lens.data.builtin.diagnostic_v3 import load_diagnostic_v3
from reward_lens.data.schema import DataView

signal = load_signal("Skywork/Skywork-Reward-Llama-3.1-8B-v0.2", allow_download=True)
view = DataView(list(load_diagnostic_v3()["helpfulness"].items)[:8])
ctx = mb.Context(signal=signal, view=view)

lens = mb.run(LensCrystallization(), ctx)     # Evidence: the depth where the margin half-forms
attr = mb.run(DirectLinearAttribution(), ctx) # Evidence: value + uncertainty + gauge + trust + provenance
attr.value["differential"]                    # the same per-component array, now inside a receipt
```

On the canonical "why is the sky blue" pair, the committed 8B artifacts put the margin at \(+24.03\) and the preference crystallizing at layer 30 of 32. Those are measured results, not something this hardware reproduces, but the call that produces them is the one above. To run the 2.0 API right now, keep every line and swap in the tiny model, which is a real reward model that builds on CPU with no download:

```python
from reward_lens.signals import from_tiny
from reward_lens.measure import base as mb
from reward_lens.measure.battery import DirectLinearAttribution
from reward_lens.data.builtin.diagnostic_v3 import load_diagnostic_v3
from reward_lens.data.schema import DataView

signal = from_tiny(seed=0)
view = DataView(list(load_diagnostic_v3()["helpfulness"].items)[:8])

attr = mb.run(DirectLinearAttribution(), mb.Context(signal=signal, view=view))
print(attr.trust)                             # EXPLORATORY
print(attr.gauge)                             # invariant
```

The trust level is `EXPLORATORY` because this observable has not yet earned a scorecard, not because anything is wrong with the number. That distinction is the whole point of the [trust ladder](discipline/trust-ladder.md). Signatures for both instruments are in the [measure reference](reference/measure.md#reward_lens.measure.battery.lens.LensCrystallization).

## If you used per-head attribution

`ComponentAttribution.attribute_heads` had a real bug in 1.0. It sliced each attention head's `o_proj` contribution incorrectly, so per-head attention attribution numbers from that version are wrong. This build fixes the slicing: each head's contribution is projected through its own `o_proj` block. If you have per-head attribution figures from 1.0, recompute them.

For causal per-head effects rather than attribution, the 2.0 path is [`PatchGrid`](instruments/patch-grid.md) at head granularity.

!!! warning "Needs a GPU"
    Head-granularity patching on an 8B model is GPU-gated. It names the exact kernel call and refuses on hardware that cannot run it, rather than returning a number it did not compute.

Once your scripts run on 2.0, the next thing to read is [getting started](getting-started/index.md), which builds the same measurement from the epistemics layer up.

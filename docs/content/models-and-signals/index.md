# Models and signals

**Will these instruments run on my grader?**

Usually yes, and the page tells you exactly when. reward-lens does not assume your reward model is a sequence classifier with a scalar head. It assumes only that whatever grades your rollouts can be scored and looked inside, and it wraps eight common grader shapes behind one protocol so the same lens, the same attribution, the same patch grid attach to all of them.

![Eight grader adapters around one shared protocol.](../assets/figures/eight-graders-light.svg#only-light){ .rl-fig .rl-fig--hero }
![Eight grader adapters around one shared protocol.](../assets/figures/eight-graders-dark.svg#only-dark){ .rl-fig .rl-fig--hero }

/// caption
**One protocol, eight adapters.** Each grader in the ring implements the same small surface: score an item, score it token by token, expose activations along named readouts. An instrument is written once against the protocol in the centre and runs on every adapter that has the capabilities it needs.
///

The ring is the eight grader shapes the field actually ships: a classifier reward model, an LLM judge, a process reward model that scores each step, a DPO checkpoint used as an implicit reward, a rubric head, a trajectory grader, a dense per-token extractor, and an ensemble. The centre is the [`RewardSignal`](../reference/signals.md#reward_lens.signals.base.RewardSignal) protocol. Nothing in the instrument battery names a specific architecture. It names capabilities, and a grader either has them or it does not.

## The signal protocol, in plain words

A signal is anything that can score text and let you watch while it does. In code that is a protocol with a small fixed surface: `score(view)` returns one number per item, `score_prefixes(view)` returns the running score token by token, `capture(view, spec)` hands back activations, and `readouts()` lists the directions the signal can be read along.

Two objects make that surface portable across every grader. A **readout** says *what* to read: a direction in activation space plus a name, like `reward` for a classifier head or `verdict` for a judge's Yes-minus-No logit. A **position** says *where* to read it: the final token for a classifier, each step boundary for a process RM, the verdict token for a judge. v1 hardcoded "the scalar at the final token" and could reach nothing else. Making readouts and positions first-class is the one move that lets crystallization depth of a judge's verdict be the *same* measurement as crystallization depth of a scalar head, called with a different readout. The [reward direction](../concepts/reward-direction.md) page owns this idea in full.

## The support matrix

Every adapter exposes `score`, `score_prefixes`, `capture`, and `readouts`. The column that matters is what each adds beyond that base, because that is what decides which instruments attach.

| Grader | Adapter | What you bring | Capabilities beyond the base | Tiny CPU vehicle |
|---|---|---|---|---|
| Classifier RM | [`ClassifierRM`](classifier-rms.md) | a sequence-classification head | linear readout, gradients, HVP | `from_tiny()` |
| Generative judge | [`GenerativeJudge`](generative-judges.md) | an instruct causal LM | generative, linear readout, gradients | `.from_tiny()` |
| Process RM | [`ProcessRM`](process-rms.md) | a step-level classifier | per-step scores, linear readout | `.from_tiny()` |
| Implicit (DPO) | [`ImplicitRM`](implicit-dpo.md) | a policy and a reference model | paired models, native per-token | `.from_tiny()` |
| Rubric head | [`RubricRM`](rubric-heads.md) | a multi-row head | multi-readout, per-criterion | `.from_tiny()` |
| Trajectory | [`TrajectoryRM`](trajectory-graders.md) | a classifier and typed episodes | span types (receipt / narrative / action) | `.from_tiny()` |
| Dense per-token | [`DenseRewardExtractor`](dense-rewards.md) | any outcome signal | per-token map, pinned exploratory | wrap a tiny signal |
| Ensemble | [`SignalEnsemble`](ensembles.md) | two or more signals | mean / min / quantile, distributional | compose members |

!!! note "What is full, what is partial, what is gated"
    All eight adapters are fully working on CPU through `from_tiny` or by composing tiny members, and clear the conformance checks that apply to them. Two exceptions are honest and marked in place. The process RM's *learned* step detector is a stub, so a solution with no delimiter is read as one step and says so; the delimiter path is exact. The dense extractor is pinned exploratory by design and attaches no calibration, ever, until the verification science certifies it. Separately, the hub-scale download in `load_signal` is gated: the wiring runs, the multi-gigabyte fetch is withheld unless `allow_download=True` on hardware that can hold the model.

The capabilities are declared, not guessed. An instrument states what it `requires`, and the runner checks that before it touches a GPU, so a grader that cannot support a measurement is refused with a precise message instead of failing three layers deep. That mapping is direct:

- **Anything reading activations along a linear direction** (the [reward lens](../instruments/lens-crystallization.md), direct attribution, the patch grid, path effects, concept dose-response, the conflict matrix) attaches to the classifier, judge, process, rubric, and trajectory adapters. It does *not* attach to the implicit RM, which has no single head direction, nor to the composites, which have no shared activation.
- **Score-only instruments** (the bias battery, prompt SNR, most of the [index library](../instruments/index-library.md)) attach to every adapter, since all eight score.
- **Per-step scores** reach the process RM; **span-typed** experiments reach the trajectory RM; **multi-readout** geometry reaches the rubric head and any multi-row classifier; **generative** verdict work reaches the judge; **distributional** readouts reach the quantile ensemble.

## Loading conventions

Three entry points, and only one of them touches the network.

`load_signal` is the front door for a checkpoint you name. It reads the config and head names, sniffs the loading convention (TRL sequence-classification, OpenRLHF value head, veRL token-classification, or an ad-hoc head), picks the adapter, and hands back a wired signal. The hub-scale download is gated. On a machine that cannot hold an 8B model in fp32, `load_signal` refuses rather than pretend:

```python
from reward_lens.signals import load_signal

load_signal("Skywork/Skywork-Reward-Llama-3.1-8B-v0.2")
# NotImplementedError: loading 'Skywork/Skywork-Reward-Llama-3.1-8B-v0.2' from the HF hub
#   is GPU/download-gated on this machine (8 GB GPU, cannot hold the 8B/27B campaign models).
#   The convention sniffed as 'unknown' and the code path is implemented; set allow_download=True
#   to attempt it on adequate hardware, or use wrap_hf_model/from_tiny for a local model.
```

The error is the honest one: the sniffing and wiring are real, the download is what is withheld, and the two ungated paths are named. `allow_download=True` (or a local path) lifts the gate on hardware that can hold the weights.

`wrap_hf_model` takes a model and tokenizer you have already loaded and returns a `ClassifierRM`. No network, and it runs a fast conformance quick-check on the way out so a mis-resolved head fails at load, not deep in a study. `from_tiny` builds a real two-layer `LlamaForSequenceClassification` on CPU with no download at all, which is the vehicle every example on these pages uses.

```python
from reward_lens.signals import from_tiny, RewardSignal

signal = from_tiny(seed=0)
signal.meta.adapter          # 'LlamaAdapter'
signal.caps                  # SCORES|PREFIX_SCORES|ACTIVATIONS|GRADIENTS|HVP|LINEAR_READOUT
isinstance(signal, RewardSignal)   # True
[r.name for r in signal.readouts()]  # ['reward']
```

!!! warning "Needs a GPU"
    The 8B and flagship campaign models (`Skywork/...-8B`, the 27B Gemma, ArmoRM) load only with `allow_download=True` on hardware that can hold them in fp32. Every worked example here uses `from_tiny` on CPU so you can run it now; the campaign numbers are cited from committed artifacts and marked where they appear.

## How you know your adapter is wired right

A grader that half-loads is the quiet killer of a reward-model study: the model drops out of the campaign and nobody notices. The conformance suite is the fix. It checks the invariants a reward readout must satisfy no matter the architecture: the same input scores the same twice, a batched score equals the one-at-a-time score, left-padding a short sequence does not move it, the fp32 projection equals the model's own head output, the last entry of the per-token curve equals the scalar score, the fp32 head stays finite across trunk dtypes, and a typed span survives tokenization.

`run_conformance` runs the full suite against a classifier signal:

```python
from reward_lens.signals import from_tiny, run_conformance

report = run_conformance(from_tiny(seed=0))
print(report.summary())
# conformance for mfp:586b55dd932158ef9d21a4e4e71e1276: PASS
#   [pass] determinism: max|diff|=0.00e+00
#   [pass] batch_vs_single: max|diff|=1.49e-08 (tol 1.0e-04)
#   [pass] left_pad_invariance: max|diff|=1.49e-08 (tol 1.0e-04)
#   [pass] readout_matches_head: max|readout-native|=0.00e+00 (tol 1.0e-04)
#   [pass] prefix_consistency: max|curve[-1]-score|=0.00e+00 (tol 1.0e-04)
#   [pass] dtype_matrix: float32:finite, bfloat16:finite, float16:finite
#   [pass] template_round_trip: char[46,51) -> tokens[11,12) covering=True
report.passed   # True
```

`run_adapter_conformance` holds each of the other seven adapters to the checks that *apply* to it, skipping the readout-versus-head check for composites and log-ratio signals that have no single head direction. On the tiny process RM it clears every applicable check:

```python
from reward_lens.signals import ProcessRM, run_adapter_conformance

items = [("What is 2+2?", "It is 4."), ("Name a color.", "Blue."),
         ("Explain gravity in one sentence.",
          "Gravity is the mutual attraction between masses that pulls them together, "
          "growing weaker with the square of the separation.")]
report = run_adapter_conformance(ProcessRM.from_tiny(seed=0), items=items, readout="reward")
report.passed   # True  (determinism, batch-vs-single, left-pad, readout-vs-head, prefix all pass)
```

A new grader paradigm the field invents becomes a new adapter that passes this suite, and the whole battery, every index, and every science become available to it at once. The [write an adapter](../how-to/write-an-adapter.md) guide walks the full path.

## Where to go next

Each grader has its own page with a runnable example and its own caveats: [classifier RMs](classifier-rms.md), [generative judges](generative-judges.md), [process RMs](process-rms.md), [implicit DPO rewards](implicit-dpo.md), [rubric heads](rubric-heads.md), [trajectory graders](trajectory-graders.md), [dense per-token rewards](dense-rewards.md), and [ensembles and distributions](ensembles.md). Every score they return arrives as [evidence you can trust or not](../concepts/measurement-you-can-trust.md), starting at exploratory until a scorecard earns it more.

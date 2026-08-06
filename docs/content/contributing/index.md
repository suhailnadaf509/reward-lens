# Contributing

**What does the library need most? A reward model it cannot open yet.** Every instrument here works the moment your grader satisfies one protocol, so a new adapter reaches more people than almost any other change. That is the community bet: keep coverage cheap, keep the core honest, and the science on top stays thin.

## The bet: adapter coverage

`reward-lens` earns its keep only when it runs on the model you actually care about. The `RewardSignal` protocol is what makes that a small job. It gives one interface over the different things people call a reward, so eight substrates already look identical to every tool downstream: classifier reward models, generative judges, process reward models, implicit DPO rewards, rubric graders, trajectory models, dense per-token rewards, and ensembles.

If your grader is an ordinary HuggingFace sequence-classifier reward model, you may need no new code at all. Wrap it and the conformance quickcheck confirms it is wired correctly:

```python
from reward_lens.signals import wrap_hf_model

signal = wrap_hf_model(model, tokenizer)   # runs a determinism + fp32-readout-vs-head quickcheck
```

A genuinely new kind of grader becomes a new adapter that implements the protocol (its readouts, its positions, and how it scores a view) and then earns its place by passing the conformance suite. That suite is the gate. It checks the invariants a reward readout has to satisfy or it is silently broken: determinism, batch-versus-single agreement, left-padding invariance, the fp32 readout matching the head exactly, prefix consistency, a dtype matrix, and template round-trip. Run it against any signal, including the CPU tiny model:

```python
from reward_lens.signals import from_tiny, run_conformance

report = run_conformance(from_tiny(seed=0))
print(report.passed)      # True
print(report.n_passed)    # 7
```

If that passes on your adapter, the whole battery of instruments works on it unchanged. The [write-an-adapter how-to](../how-to/write-an-adapter.md) walks through a full adapter end to end, and [models and signals](../models-and-signals/index.md) covers the eight substrates the protocol already speaks. The protocol itself is in the [signals reference](../reference/signals.md).

## What the project values

Three beliefs sit under the code, and a good contribution respects them.

- **Minimal abstractions.** The interventions are lightweight PyTorch hooks placed directly on a HuggingFace model. No custom model wrapper, no hooks buried in a dictionary you have to reverse-engineer. If you can read `transformers`, you can read the whole thing.
- **An auditable core.** The mechanism that reads a reward out of a hidden state is a small, fixed amount of code, and that ceiling is deliberate. A change that doubles the core to save a few lines at a call site is usually the wrong trade.
- **Honesty over polish.** The library's own headline result is a limitation. Component attribution anti-correlates with causal patching at Spearman \(\rho = -0.171\) on `Skywork-Reward-Llama-3.1-8B-v0.2`, and sits near zero at \(+0.047\) on ArmoRM. The docs lead with that rather than bury it. Hold new tools to the same standard. If a technique is exploratory, its Evidence should say so through the [trust ladder](../discipline/trust-ladder.md), not a hopeful docstring. See [observational vs causal](../concepts/observational-vs-causal.md) and [interpreting results honestly](../caveats.md) for how that principle is carried through the site.

## Setting up a dev checkout

Fork the repo, then clone your fork:

```bash
git clone https://github.com/YOUR_USERNAME/reward-lens.git
cd reward-lens
```

Make a virtual environment and install the package with its dev extras:

```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

Confirm the environment works before you change anything by running the suite:

```bash
pytest
```

If that passes, you have a working checkout of `reward-lens` 3.0.0.

## Working on a change

- **Branch off with a name that says what it does:** `git checkout -b feature/model-adapter-armorm`, or `git checkout -b fix/tensor-mismatch-divergence`. Prefix with `feature/` or `fix/`.
- **Write a test for every new tool or hook,** and put it in `tests/`. Keep it light. Tests should run on the tiny CPU model or on mock activations, not by downloading 8B weights on every CI run. Reserve the real 8B loads for cases explicitly marked as integration tests.
- **Match the discipline.** A measurement returns Evidence, never a bare float, and its trust level is computed by the gates rather than set by hand. A new instrument that returns an unguarded number is not finished.
- **Keep a pull request to one idea.** Do not fold an unrelated refactor into a new feature. If the change is large, open an issue first and agree on its shape with the maintainer before writing it.

## The test doctrine

The suite is 556 tests, and they pass. The reason that number is trustworthy rather than decorative is a rule the build followed strictly: everything runnable without a big GPU is actually run and proven, and the heavy paths refuse rather than fabricate. The pure epistemics layer, the statistics engine, the data plane, and the index math all execute on CPU in CI.

The proudest test is the one that closes the calibration loop without any of that heavy machinery. `micro_organism_calibration` builds a reward model with a rule planted by construction, on a tiny trunk that trains on CPU in a few seconds, verifies that the planted rule governs behaviour on a held-out split, then runs the built-in detector to recover it. The single bit it returns, `recovered`, is the go/no-go a CI test asserts. This is the floor the calibration gate stands on, proven end to end on every run.

```python
from reward_lens.organisms import micro_organism_calibration

res = micro_organism_calibration(seed=0)               # trains a tiny CPU trunk, a few seconds
print(res.recovered)                                   # True
print(res.summary()["detector_ood_auc"])               # 1.0
print(round(res.summary()["cosine_with_reward"], 2))   # 0.96
```

The detector recovers the planted rule at AUC 1.0 on the out-of-distribution split, and the recovered direction sits at cosine 0.96 with the trained reward head. A detector that could not clear that bar on a model whose answer is known has no business reporting a number on a model whose answer is not. That is the argument the whole [calibration story](../discipline/calibration-and-organisms.md) makes at scale.

## How to cite

If `reward-lens` helped your work, please cite it:

```bibtex
@software{nadaf2026rewardlens,
    title  = {reward-lens: An Instrument for the Science of Reward Misspecification},
    author = {Nadaf, Mohammed Suhail B},
    year   = {2026},
    url    = {https://github.com/reward-lens/reward-lens},
}
```

`reward-lens` is released under the MIT license. If you are porting an existing 1.0 workflow, [coming from 1.0](../migration.md) maps every primitive to its new home.

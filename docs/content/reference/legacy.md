# Legacy (1.0 API)

**Will code written against reward-lens v1.0 still run?** Most of it will not, and this page says
exactly which parts.

The v1.0 flat public API was carried through v2.0 as a lazy accessor: `reward_lens.RewardModel` and
twenty other names resolved on first touch, so `import reward_lens` stayed torch-free while old
code kept working. That accessor met its two-release deprecation condition and retired in v3.0,
along with the v1 corpus behind it. `import reward_lens` now exposes `__version__`, `core` and
`stats`, and nothing else.

Two v1.0 modules were never folded into the accessor and are still where they were:

```python
from reward_lens.model import RewardModel          # the hooked model wrapper
from reward_lens.sae import TopKSAE, SAETrainer    # the top-k sparse autoencoder
```

Both need the `[white-box]` extra. Neither returns `Evidence`; they return the objects they always
returned.

## What retired, and where its work went

| Retired name | What it did | Where the work is now |
|---|---|---|
| `RewardLens` | reward projected across depth | [`LensCrystallization`](../instruments/lens-crystallization.md) |
| `ComponentAttribution` | per-component reward ledger | [`DirectLinearAttribution`](../instruments/attribution.md) |
| `ActivationPatcher`, `PathPatcher` | causal patching | [`PatchGrid`](../instruments/patch-grid.md), [`PathEffect`](../instruments/path-effects.md), and the [intervention algebra](../instruments/interventions.md) |
| `ConceptExtractor`, `quick_concept_analysis` | concept directions and steering | [concepts](concepts.md) with [`ConceptDoseResponse`](../instruments/concept-dose-response.md) |
| `RewardConflictAnalyzer` | inter-objective conflict | [`ConflictMatrix`](../instruments/conflict-matrix.md) |
| `DistortionAnalyzer` | reward distortion | [the index library](../instruments/index-library.md) |
| `DivergenceAwarePatching`, `MisalignmentCascadeDetector` | patching under divergence, cascade detection | [the intervention algebra](../instruments/interventions.md) |
| `HackingDetector` | bias and hacking scan | [the bias battery](../instruments/bias-battery.md) and [the index library](../instruments/index-library.md) |
| `ModelComparator` | cross-model comparison | [gauge and frames](../discipline/gauge-and-frames.md) |

The right column is not a rename. In every row the replacement returns `Evidence` carrying an
uncertainty, a gauge status and a computed trust level, where the retired name returned a numpy
array or a report object. That is why the old names went rather than being wrapped: a shim that
hands back a bare float from an instrument that knows its own uncertainty is a shim whose only job
is to throw the uncertainty away.

## If you are still on the old API

Pin `reward-lens<3` while you move, and move one call at a time.
[Coming from v1.0](../migration.md) walks the migration in the order that keeps the most code
working at each step.

## Still here

::: reward_lens.model.RewardModel
    options:
      heading_level: 3

::: reward_lens.sae.TopKSAE
    options:
      heading_level: 3

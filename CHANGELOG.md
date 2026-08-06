# Changelog

## [3.0.0] - 2026-08-06

A rebuild around a type system 2.0.1 did not have, keeping the half that was
always the product. The library goes from 42,327 lines across 172 modules to
168,540 across 376, and the tests from 10,458 lines and 527 collected to
78,559 and 4,073.

The thesis, because every entry below is downstream of it: a reward function is
written down, and a realized objective is what the policy's behaviour
distribution actually moved toward. The gap between them is reward hacking, side
effects, dark reward, style share and Goodhart, which are all the same thing seen
from different angles. The realized objective cannot be read off the reward
function, so the library's job is to make it observable while it is being
realized, and to say honestly when it cannot.

### The root change: a reading is evidence or a refusal

`Reading = Evidence | Refusal`. A `Refusal` carries a reason, the numbers that
produced it, and a remedy written as an instruction rather than a diagnosis, and
it cannot be constructed without one. It is a success, not a downgrade: an
instrument that cannot answer says so rather than returning a worse number, a
`None` or a zero.

There are seventeen refusal reasons and they separate by where the remedy is
answerable. `ACCESS_INSUFFICIENT` is answerable where the reader stands.
`RECORD_INCOMPLETE` is answerable upstream. `QUANTITY_UNDEFINED` is answerable
nowhere, which is why its `instead` field is required: when the asked question
does not apply to the object, the only useful sentence available is the name of
the question that does.

### Added — the kernel

- `core/quantity.py`: `Quantity`, `Unit` and `EstimatorEntry`. `Unit` is three
  axes, so comparing a per-token quantity against a per-sequence one returns
  False rather than a number. Each quantity carries a ladder of estimators, each
  with its access matrix, its regime envelope, its rung, its bias statement and
  its cost model, so `best_estimator` answers "what can I measure, at what
  uncertainty, for what money" before anything is spent.
- `core/envelope.py`: thirteen `RegimeCondition` members and `EnvelopeSpec`, whose
  lint lives in `__post_init__`, so an envelope that cannot be enforced cannot be
  constructed. Three violation behaviours that are not interchangeable: refuse
  when the quantity is undefined outside the regime, bound when a weaker
  estimator survives, downgrade when the quantity stays defined and trust drops.
- `core/invariance.py`: the seven invariance groups with generators and samplers.
  Every registered instrument gets one generated property test from its declared
  group. An instrument whose group carries no sampler is refused rather than
  reported on, because sixty-four identical draws and a passing test is worse
  than no test.
- `core/budget.py`: the GUM uncertainty budget with Welch-Satterthwaite, limits
  of detection and quantitation with the three-outcome rule, and the Hill curve.
  The budget names its own largest term, which is almost never sampling noise.
- `core/reference.py`: `ReferenceMaterial` with `u_char`, `u_bb` and `u_stab`,
  transfer chains, and the rule that an uncharacterised homogeneity term is not a
  missing field: it caps the trust ladder at `CALIBRATED` and says so in every
  downstream reading.
- `core/closure.py`: plan closure. A study whose registered metric no arc
  produces raises before any work runs, naming the prediction, the metric and the
  gap.
- `spec/`: the operative catalogue, 95 instruments and 190 quantities with 112 in
  the wedge, edited as YAML and loaded as generated JSON so the base install
  carries no compiled dependency.

### Added — the record

`record/`, the canonical process record, five levels from `Run` down to `Token`.
Non-storage is the default for tensors, because one layer of bf16 residual for
Llama-3.1-70B over 10^9 tokens is 16.4 TB and all eighty layers is 1.31 PB; a
`RecomputeRef` that cannot be honoured becomes an `AbsentRef` rather than a
silent zero. Segment provenance is mandatory and plural, with the turn ranges
asserted to tile the trajectory. Scores are a `ScoreTree` rather than a float, so
"what happens to the advantages if I remove the length override" is answerable on
recorded leaves at zero compute. Held-out labels are `Blind[T]` with no
`.unwrap()`, so leakage is a type error.

### Added — the instruments

Ninety-five of them across fourteen series, each declaring the quantity it
estimates, the access it requires, the regime it is valid in, the invariance
group it respects, the baselines it ships against and its rung on the ladder.
The families, and what each answers:

- **Grader metrology (A)**: variance components, %GRR, ndc, effective group size,
  attenuation and an allocation recommendation. Generalizability theory for
  crossed designs, which had no Python implementation anywhere.
- **Grader structure (B)**: the Hodge decomposition split into curl mass and
  harmonic mass, the Afriat rationalizability index, counterfactual composition,
  and the silent-zero rate.
- **Verifier science (D)**: decision coverage, surviving mutants with source spans
  and diffs, metamorphic violations with reproducers, Sobol indices, a
  false-positive catalogue, exploit-family accounting and replay fidelity, all
  from a verifier's source and a rollout corpus.
- **The estimator (E)**: the transform between a good reward and a bad gradient,
  including `amplifier_safety`, which answers whether a reward component is safe
  to add and whose answer is not about magnitude.
- **The four books (F)**: effect, cause, capacity and cost, which must reconcile.
  A KL budget decomposed into named-feature shares, which nobody splits.
- **Credit (G)**: a signed measure whose total mass equals the update exactly, so
  a credit report either accounts for 100% of the step or has a bug.
- **Pressure and thresholds (I)**: hard reward gates read as sharp regression
  discontinuities, with McCrary density tests and bunching elasticities.
- **Monitoring (J)**: anytime-valid confidence sequences, ARL-designed CUSUM
  alarms, conjunction detectors and an operating point derived from an asymmetric
  loss.
- **Labels and references (L)**: label error rates, score ceilings, and certified
  reference materials with their uncertainty decomposed.
- **Meta-instruments (M)**: the substrate noise floor, the instrument's own
  overhead, six dumb baselines, the semantic placebo, the matched positive
  control, the uncertainty budget, interlaboratory comparison, incremental
  validity and rung disagreement.
- **The frontier and the contract layer (N)**: the reward-versus-gold curve out to
  a measured visibility horizon, and optimal reward weights set from measured
  component noise.

### Added — the surfaces

- `access/`, and `reward-lens capabilities`: resolve access, substrate, phase and
  regime, then print what is available now with its rung, expected uncertainty
  and cost, and what is refused with the exact remedy. Costs nothing to run.
- `tap/`: a grader wrapper that records every call and loses nothing, at 1.89 to
  2.03 microseconds of added latency on a quiet reference machine, and a TRL
  adapter measured at 3.14 microseconds and between 1e-5 and 3e-5 of training
  wall clock. A tap whose every callback raises leaves the run's state dict
  byte-identical. The absolute microsecond figure does not reproduce on a loaded
  machine, where the same tap measures 3.97; the ratio to the bare call and the
  run-level fraction do, and those are the two figures to quote.
- `policy/`: the peer of `signals/`. The same instruments run against a policy and
  against a grader with the subject as the only difference.
- `forecast/`: a `Forecast` whose construction refuses if any transitive input
  postdates its issue time, scored against four mandatory baselines including
  decision value.
- `verifier/`: nine modules treating a scoring program as a third kind of object,
  with source code and a control-flow graph rather than activations.
- `measure/card/`: the grader card, one page describing a grader as a measurement
  device.

### Changed

- `Observable` becomes `Instrument`, with twelve declared attributes and two
  methods, `preflight` and `estimate`. `preflight` costs nothing and makes no
  grader calls.
- Four lint rules, all live. An instrument whose quantity is unregistered fails
  at import. A quantity with no estimator fails the docs build, named as an open
  research target rather than a bug. An instrument with no baselines, no envelope
  or an envelope naming a condition nothing measures fails lint. A white-box
  reading with no incremental validity fails lint.
- Dependencies restructured. The base install pulls nothing compiled and a
  CI job asserts it; `import reward_lens` does not import torch. Missing extras
  raise a typed error naming an extra that `pip` can actually install.
- The claims gate runs in CI over every tracked page, with a ratchet that may
  shrink and must not grow.
- `package.json`-style metadata corrected: the project URLs pointed at a personal
  fork and at readthedocs.

### Removed

- The v1 corpus: 6,454 lines across fifteen modules reachable only through the lazy shim,
  through each other, or through their own tests, plus their tests, and the
  lazy `_LAZY` table with them. Module-level `import torch` falls from eight
  modules to three.
- `diagnostic_data_v2.py` moves into `data/builtin/`, `sae.py` moves behind the
  `[dict]` extra, and the grader-side half of `model_adapters/` moves into
  `signals/adapters.py`. `model.py` and `model_adapters/` stay on disk with live
  consumers in the organism foundry, deliberately: deleting working code to make a
  line count look better is how a cleanup becomes an outage.
- Three pieces of onboarding material that the v1 retirement had already broken:
  the seven scripts in `examples/`, which imported `RewardModel`,
  `reward_lens.lens`, `reward_lens.hacking` and `reward_lens.viz`;
  `Reward_Lens_Intro_Demo.ipynb`, whose first cell imported `reward_lens.hacking`,
  `ConceptExtractor` and `CONCEPT_PAIRS`; and `configs/`, which described a
  campaign that does not live in this repository and which nothing referenced.
  None of the three would run on a fresh install. The README carries a quickstart
  that is generated from a run instead, so it cannot rot the same way.
- `work_package`, `caliper_ancestor` and `source_lines` leave the catalogue
  schema. `status` already carried the only fact `work_package` encoded, and the
  other two were provenance into documents that are not published. The
  capability report and the documentation now render `status`, and
  `CatalogueInstrument.work_package` and `NotBuilt.work_package` are gone from
  the public API; `NotBuilt.status` replaces the latter.
- The sdist no longer carries `tests/`. It shipped the 118 modules at the top of
  the directory without `conftest.py`, the fixtures, or `tests/acceptance/`, so
  the suite it contained could not be collected. Running the suite needs the
  repository, which also carries `studies/` and `spec/`.

### Fixed

- **A metric that could not be computed rendered as a result.** The 2.0.1 study
  runner turned a missing metric into `inconclusive`, and a kill criterion whose
  metric was absent into a criterion that did not fire, so a registered kill that
  could not be evaluated was indistinguishable in the output from one that was
  evaluated and passed. There are now three states rather than two, and a `Void`
  carries the absent metric, who wanted it, and the arc that owed it.
  Re-adjudicating the previous campaign's own evidence store gives zero
  inconclusive rows where there were eight at card level and sixteen at
  hypothesis level.
- **`py.typed` was missing**, so a type checker resolved every `reward_lens` name
  to `Any` for every downstream user of the wheel. `Blind[T]`, whose entire
  enforcement is a type-checker assertion, was unenforceable. A CI job now asserts
  a leak fixture is rejected on exactly eight lines.
- **Two wrong numbers on default paths in `replay_advantages`**, shipping since the
  record landed. The policy-ratio clip was applied as a bound on the advantage,
  which pinned 400 of 400 groups to exactly 0.2 and made every counterfactual read
  as "nothing moved"; and the record divided by the population standard deviation
  where every framework in scope applies Bessel's correction. The record now
  replays a real trainer at 2.97e-06 over 400 groups, which is the float32
  round-trip and nothing else.
- `grader.effective_group_size` stopped multiplying a grader property by a
  reward-distribution property. Kish's shape factor now travels beside the reading
  as `run.group_shape_factor` rather than inside it.
- The store hardening from the campaign branch: torn-tail quarantine, read-only
  mode, per-append fsync, atomic sidecar writes, and `EvidenceStore.merge` with
  parent validation across the merged whole before anything is written.

### Known gaps, named rather than left to be discovered

**Ten of the twelve tap adapters are not built.** `tap/adapters/` ships `trl.py`
and `verifiers.py`. `verl`, `openrlhf`, `skyrl`, `slime`, `areal`, `nemo`,
`primerl`, `roll`, `tinker` and `generic` are named in the architecture, were
never scheduled, and do not exist. This is the gap a user is most likely to hit,
so it is first.

`probe/`, `measure/coverage/` and `stats/identification.py` are not in this
release.

**All eight compute-gated studies ship as code, a frozen study spec, an
acceptance test on a synthetic subject, a runbook and a price, rather than as
results.** None of them has been run. Each states in its own module docstring
which real subject its claim would need and what that subject would cost, because
a synthetic acceptance test proves the arithmetic and proves nothing about the
phenomenon.

**The evidence store's hash chain and its four row kinds go to 3.1**, and the
reason is worth stating rather than leaving as an omission. A chain detects
reordering and deletion, and those are the two properties the store deliberately
does not preserve: `append` is idempotent on the content id and writes nothing for
a row it already holds, and `merge` sorts and drops duplicates, so two shards
holding four rows between them produce a destination of three rows in an order
neither shard had. Per-row tamper evidence is already there without a chain,
because an evidence id is a content hash of the whole envelope apart from the wall
clock, so an edited value is a different row and its children stop resolving.
Coverage would also be zero where it counts, since every row of the published
campaign store predates any chain and cannot be linked after the fact without
asserting an order nobody witnessed. It ships in 3.1 with a verifier beside it.

`model.py` and `model_adapters/` remain on disk with live consumers in the
organism foundry. They are not part of the public API and they were kept
deliberately rather than deleted to improve a line count.

## [2.0.1] - 2026-07-23

### Fixed
- Declared `scipy`, `pydantic`, and `pydantic-settings` as runtime dependencies.
  All three are imported at module scope (`pydantic` and `pydantic-settings` by
  `reward_lens.core.config`, `scipy` by `reward_lens.stats` and
  `reward_lens.geometry`), but they were absent from the 2.0.0 dependency list.
  The result was an install that looked healthy and was not: `pip install
  reward-lens` succeeded, `import reward_lens` succeeded and reported its
  version, and the first real use failed with `ModuleNotFoundError`. The
  dependency list was corrected in the source tree shortly after 2.0.0 went out;
  this release is what carries that correction to PyPI. There are no code
  changes, so 2.0.0 and 2.0.1 behave identically once the three packages are
  present.

## [2.0.0] - 2026-07-10

Major redesign. The library is reorganized around a single kernel with a lazy,
torch-free epistemics layer; the 1.0 public API is preserved under
`reward_lens.legacy`.

### Changed
- Reorganized around one kernel of subsystems: `core`, `stats`, `runtime`,
  `signals`, `data`, `concepts`, `interventions`, `geometry`, `measure`,
  `attribution`, `organisms`, `dynamics`, `loops`, `studies`, and `artifacts`.
- The top-level import is now lazy: `import reward_lens`, `reward_lens.core`, and
  `reward_lens.stats` pull only numpy, so the pure epistemics layer is usable
  without torch. Model-touching code is imported on first access.

### Added
- Sixteen reward-science studies over the kernel, plus three runtime gates
  (calibration, gauge, registration) in the stats/evidence layer.
- `reward-lens` command-line interface (console script) and an operate MCP surface.
- Artifact builders: atlas, cards, claims, safety case, and site.
- Training-loop integrations for TRL, veRL, and OpenRLHF, with tilt, anneal,
  and best-of-N.
- E-parity golden-fixture test suite.

### Compatibility
- The 1.0 public API is preserved under `reward_lens.legacy` and remains
  importable from the top level through the lazy accessor.

## [1.0.0] - 2026-04-12

### Added
- Initial release: RewardLens, ComponentAttribution, ActivationPatcher, HackingDetector
- DistortionAnalyzer: predictive reward hacking analysis
- MisalignmentCascadeDetector
- RewardConflictAnalyzer
- ConceptExtractor and quick_concept_analysis
- DivergenceAwarePatching

### Validated
- Ran experiments on RewardBench (~695 pairs) across Skywork-Reward-Llama-3.1-8B-v0.2 and ArmoRM
- Key finding: late-layer crystallization (90-97% depth for Skywork)
- Key limitation: attribution does not predict causal importance

# Migrating from 2.0.1 to 3.0.0

This page is for code that already runs on 2.0.1. If you are coming from the flat v1 API, the map
you want is [`docs/content/migration.md`](content/migration.md), which lists what happened to every
retired v1 name. This one is about the smaller and sharper break: the 2.0 `Observable`, `Evidence`
and dependency layout against the 3.0 `Instrument`, `Reading` and extras.

The honest summary is that the epistemics layer survived, the measurement interface did not, and two
numbers you may have pinned have changed value for a reason. Nothing below is a rename you can do
with `sed` without reading what it says.

## The one-paragraph version

`Observable` becomes `Instrument`, which declares twelve attributes and two methods and returns a
`Reading` rather than an `Evidence`. `Reading = Evidence | Refusal`, so the branch that used to be an
exception or a `None` is now a value you handle. `requires` changed meaning. The base install no
longer pulls torch, so anything that touches a model is behind an extra. `py.typed` now ships, which
changes what a type checker says about your code without you changing a line.

## `Observable` becomes `Instrument`

`Instrument` is a `runtime_checkable` Protocol in `reward_lens.measure.base` with twelve declared
attributes and two methods:

```python
name: str
version: str
quantity: QuantityID
requires: AccessMatrix
substrates: frozenset[Substrate]
phases: frozenset[Phase]
envelope: EnvelopeSpec
invariance: str
baselines: tuple[str, ...]
rung: int
faithful_to: str | None
deviations: tuple[str, ...]

def preflight(self, ctx: Context) -> PreflightResult: ...
def estimate(self, ctx: Context) -> Reading: ...
```

`preflight` costs nothing and makes no grader call. It is what `reward-lens capabilities` runs, and
it is how an instrument answers "could I measure this, at what rung, for what money" before anything
is spent. `estimate` is the measurement.

`Observable` and `BaseObservable` are still in the tree and `mb.run(observable, ctx)` still returns
an `Evidence`, so 2.0 measurement code keeps working. What it does not get is a preflight, a regime
check, a declared invariance group or a rung, which means it also cannot appear in a capability
report or pass `lint_instrument`.

## The rename that bit hardest: `requires` and `capabilities`

In 2.0.1, `Observable.requires` was a `Capability` flag saying what the *signal* had to offer. In
the 3.0 release, `requires` is the `AccessMatrix` saying what the *analyst* must be able to reach,
per component.
The capability declaration is now `capabilities`.

```python
# 2.0.1
class MyObservable(BaseObservable):
    requires = Capability.SCORES | Capability.ACTIVATIONS

# 3.0
class MyInstrument(BaseObservable):
    capabilities = Capability.SCORES | Capability.ACTIVATIONS
    requires = {Component.GRADER: Access.QUERY}
```

Two concepts collided under one name and the collision was worse than a rename, because `Instrument`
is `runtime_checkable` and `isinstance` therefore only checks that the attribute exists. An author
following the new protocol verbatim on the 2.0 spelling got an `AttributeError` on
`.missing_from(Capability)` at `run()` time, while the access matrix was never checked at all, and
nothing flagged it.

`declared_capabilities` **raises** on the old spelling rather than falling back. Falling back would
read the inherited `capabilities` default and silently drop the gate the declaration exists to set,
which is the failure the raise is there to prevent. The error message names the rename.

## `Evidence` becomes `Reading`

```python
from reward_lens.core.reading import Reading, Refusal, RefusalReason, is_refusal
```

`Reading = Evidence | Refusal`. A `Refusal` carries the reason, the numbers that produced it, and a
remedy, and it cannot be constructed without one. It is a value and not an exception, so the shape
of a caller changes:

```python
# 2.0.1: a measurement either returned a number or raised
ev = mb.run(obs, ctx)
print(ev.value)

# 3.0: a measurement returns a Reading, and the refusal branch is a real one
reading = instrument.estimate(ctx)
if is_refusal(reading):
    print(reading.reason.name, reading.detail)
    print("remedy:", reading.remedy)
else:
    print(reading.value, reading.uncertainty)
```

There are seventeen refusal reasons. The three that catch most existing 2.0 code are
`ACCESS_INSUFFICIENT` (answerable where you are standing: get more access or drop a rung),
`RECORD_INCOMPLETE` (answerable upstream, in whatever wrote the record) and `QUANTITY_UNDEFINED`
(answerable nowhere, so it is required to name the question that does apply). The distinction is not
cosmetic: telling somebody to get more access when the honest answer is "your framework does not dump
this" costs them an afternoon and then still does not work.

`Evidence` itself gained fields rather than losing them: `quantity`, `lod`, `regime`, `reference`,
`baselines`, `incremental` and `information_time` are new, and `schema_version` is stamped so a
store written by the 2.0 library can be read by the 3.0 one. `reward_lens.core.migrations` carries
the version sniffer and the registered migrations; `sniff_version` on a row with no
`schema_version` field returns 0, and `migrate` brings it forward to the current schema.

## The v1 corpus is gone, and so is the flat public API

Commit `77329ab` deleted 6,454 lines: fifteen modules reachable only through the lazy accessor, plus
their four test modules. The `_LAZY` table went with them, so `import reward_lens` now exposes
`__version__`, `core` and `stats` and nothing else. `reward_lens.legacy` no longer exists.

Two survivors, deliberately:

- `reward_lens.model` and `reward_lens.model_adapters` are still on disk with live consumers in the
  organism foundry. Deleting working code to make a line count look better is how a cleanup becomes
  an outage.
- `reward_lens.sae` is at the same import path and is now gated: the module calls
  `require_extra("dict")` at import, so it needs `pip install "reward-lens[dict]"`.

`reward_lens.diagnostic_data_v2` moved to `reward_lens.data.builtin.diagnostic_seeds`, because it is
a dataset and that is where the datasets that ship in the wheel live. The grader-side half of
`model_adapters/` moved into `reward_lens.signals.adapters`.

## `FeatureBank` was two incompatible things

The 2.0 release had two `FeatureBank` definitions that did not satisfy each other. The protocol is now
`reward_lens.core.features.FeatureBank`, with `names`, `featurize` and `directions`. The unrelated
container of named unit directions is `DirectionBank`, in `reward_lens.loops.recorder`, which is what
it always was.

The old import path still resolves: `reward_lens.measure.indices` re-exports the protocol, so
existing `from reward_lens.measure.indices import FeatureBank` keeps working.

## Dependencies were restructured

The base install pulls nothing compiled, and `import reward_lens` does not import torch. A CI job
asserts both against a freshly built wheel, because 2.0.0 shipped an install that imported
successfully and failed on first real use, and stopping at `import reward_lens` would not have caught
it.

Thirteen extras. The ones a 2.0 user is most likely to need:

| extra | what it is for |
|---|---|
| `[white-box]` | torch, transformers, accelerate, safetensors. Everything that touches a model. |
| `[organisms]` | the organism foundry: `[white-box]` plus peft and datasets. |
| `[verifier]` | the D series and the grader card. Pure Python, no compiled dependency. |
| `[record]` | pyarrow and safetensors, for the columnar record container. |
| `[dict]` | `reward_lens.sae` and sparse dictionary methods. |
| `[trl]`, `[verl]`, `[verifiers]` | the framework taps and converters. |
| `[sampling]` | vLLM. |
| `[fuzz]` | atheris, for D5 rung 2 only. Deliberately not folded into `[verifier]`. |
| `[viz]`, `[dev]`, `[all]` | plots, the test toolchain, and a convenience union. |

`[sae]` is gone; it is `[dict]`. A subsystem behind an extra you have not installed raises
`ExtraRequiredError`, which subclasses both `RewardLensError` and `ImportError` so an existing
`except ImportError` around an optional import keeps working. The message names an extra `pip` can
actually install and says what it is for:

```
reward_lens.sae needs the optional 'dict' extra, which is not installed. Install it with:
pip install 'reward-lens[dict]'  (sparse dictionary methods, which are candidate generators
and never a claim substrate). The core install is deliberately free of compiled dependencies,
so most of the instrument catalogue, including the whole grader card, runs without this.
```

## `py.typed` now ships, which changes your type checker

`src/reward_lens/py.typed` did not exist through any `2.0.x` release. PEP 561 says a checker treats
a package with no such marker as untyped, so with `ignore_missing_imports` set, every name imported
from `reward_lens` resolved to `Any` for every downstream user of the wheel.

This is not a cosmetic fix. `Blind[T]`, whose entire enforcement is a type-checker assertion, was
unenforceable: measured on a two-line probe, `reveal_type(Blind)` printed `Any` and an eight-error
leakage fixture reported one unrelated error and none of the eight.

**What this means for you:** running mypy or pyright against code that imports `reward_lens` will now
surface real errors where it previously surfaced none. Expect a first run to be noisy.

## Two shipped numbers changed value

If you pinned either of these in a test, it will now fail, and the new number is the right one.

**`grader.effective_group_size`** stopped multiplying a grader property by a reward-distribution
property. The old rule computed `kish * reliability`, which conflates how much the grader disagrees
with itself against how uneven the reward distribution happened to be within a group. Measured on
eleven open reward models, fully crossed over a shared bank of 1,763 groups of four:

```text
              old rule                        new rule
rung 0   mean 2.9859, spread 0.0863     4.0000 for all eleven
rung 3   mean 1.9097, spread 0.0552     2.5582 for all eleven
```

Kish's shape factor now travels beside the reading as `run.group_shape_factor` rather than inside it.
It is filed under `run.` rather than `grader.` deliberately: filing it under `grader.` would
re-assert the exact conflation the correction removed.

**`replay_advantages`** had two wrong numbers on default paths, both shipping since the record
landed.

The policy-ratio clip was applied as a bound on the advantage. The advantage has no clip
term; ratio clipping belongs to the loss, where it truncates the update rather than the advantage.
At the default, where `epsilon` is set and `epsilon_high` is unset, the two bounds are equal, so
every live advantage was pinned to a single constant and `counterfactual` then differenced two
constant vectors and reported that nothing moved, for any node. Measured on the 200-step reference
record:

```text
before   400 of 400 groups collapsed to exactly 0.2
         against recorded advantages spanning -1.13 to +1.30
after    0 of 400
```

The record also divided by the population standard deviation where every framework in scope applies
Bessel's correction, and `EstimatorSpec` had no field to say which. The gap is `sqrt(K/(K-1))`:

```text
K = 2    41.4%
K = 4    15.5%
K = 8     6.9%
K = 64    0.79%          against a replay tolerance of 1e-4
```

`EstimatorSpec.std_ddof` now carries it, written by the TRL tap and read by `replay_advantages`,
which **refuses with `RECORD_INCOMPLETE` when the estimator normalises and the record does not
say**. Defaulting to 1 was the other option and it is worse: a near-certain assumption about a
denominator is exactly the shape of confident wrong number the record exists to prevent.

The record now replays a real trainer to the float32 round-trip and nothing else. See
`tests/fixtures/grpo_run/README.md` for the residuals.

## What did not change

The evidence store, the three gates, the study freeze and the claims checker all behave as they did.
`freeze(spec)` still stamps the git sha, `run_study` still adjudicates against the frozen prediction,
and `reward-lens claims` still exits non-zero on a manuscript number the store cannot back. The trust
ladder is still computed by the gates and still cannot be set by a caller.

Everything in the [changelog](../CHANGELOG.md) that is not listed above is an addition rather than a
break.

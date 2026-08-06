# reward-lens

[![PyPI version](https://badge.fury.io/py/reward-lens.svg)](https://pypi.org/project/reward-lens/)
[![Python](https://img.shields.io/pypi/pyversions/reward-lens.svg)](https://pypi.org/project/reward-lens/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Measurement instruments for reward models and RL runs. Every reading is either evidence or a refusal that tells you what to fix.**

You wrote a reward function. Your policy optimized something else. The gap between the two is what people variously call reward hacking, side effects, style bias, specification gaming and Goodhart, and it is not visible in the reward function, because it is a property of what the policy actually did.

reward-lens measures that gap. It reads a training record and a grader, works out which measurements your access allows, and returns numbers with uncertainty attached, or a refusal naming exactly what it would need to answer.

```bash
pip install reward-lens
```

## Sixty seconds

Start with `capabilities`. It costs nothing, loads no model, and calls no grader:

```bash
reward-lens capabilities --record path/to/run --substrate NEURAL_GEN
```

It prints what you can measure right now with its expected uncertainty and cost, and what it cannot measure with the specific remedy for each. Most people find this is the useful part: it answers "what could I learn about this run, and what would it cost me to learn more" before anything is spent.

Then a grader card, which needs no GPU and no model:

<!-- generated: card-plan-code -->
```python
from pathlib import Path

from reward_lens.measure.card import CardInputs, card_plan
from reward_lens.verifier import ListCorpus, Rollout, VerifierUnderTest

# Your grader. This one is four lines; SWE-bench's is four hundred.
Path("grade.py").write_text(
    "def grade(answer, gold):\n"
    "    return 1.0 if answer.strip() == gold.strip() else 0.0\n"
)

plan = card_plan(
    CardInputs(
        verifier=VerifierUnderTest(Path("grade.py"), entrypoint="grade"),
        corpus=ListCorpus(
            tuple(Rollout(id=f"r{i}", inputs={"answer": "4", "gold": "4"}) for i in range(20))
        ),
    )
)
print(plan.render())
```
<!-- /generated: card-plan-code -->

<!-- generated: card-plan-output -->
```text
CARD PLAN  grade.py:grade
  2 of 13 fields would read; 11 would refuse.
  cost  at least 0 grader calls, no GPU and no model. 2 of the 2 available fields do not model their own cost, so this is a floor and not a total
  not checked  access
  not checked  phase
  not checked  envelope (regime not measured)
  not checked  limit of detection

  coverage                   available at rung 1, cost not modelled by this instrument
                             not checked: access
                             not checked: phase
                             not checked: envelope (regime not measured)
                             not checked: limit of detection
  variance components        would refuse: RECORD_INCOMPLETE
                             the record this card was built from carries no replicated scoring design
                             Remedy: score each item at least twice under controlled facet variation and pass the crossed design: `CardInputs(design=ReplicationDesign.from_long(values, objects, raters))`. A variance decomposition needs replication to separate the grader's disagreement with itself from the spread across items, and one score per item confounds the two with nothing able to tell.
  curl mass                  would refuse: SUBSTRATE_MISMATCH
                             this instrument applies to NEURAL_GEN, PROCEDURAL; the grader is PROGRAM
                             Remedy: use an instrument declared for PROGRAM. A PROGRAM grader is a different kind of object, not a harder case of the same one.

  ... 10 more fields, same shape
```
<!-- /generated: card-plan-output -->

A card describes a grader as a measurement device rather than a leaderboard row: how much of its branching your corpus exercises, how many edits to it your corpus would fail to notice, how much of the gap between two rollouts is the grader disagreeing with itself.

## A reading is evidence or a refusal

This is the one idea everything else follows from.

```python
Reading = Evidence | Refusal
```

A `Refusal` carries a reason, the numbers behind it, and a remedy written as an instruction. It cannot be constructed without one. It is a success rather than a downgrade: an instrument that cannot answer says so, instead of returning a worse number, a `None`, or a zero that looks identical to a measured zero and means the opposite.

There are seventeen reasons an instrument can decline, and they sort by where the fix lives. `ACCESS_INSUFFICIENT` is fixable where you are standing. `RECORD_INCOMPLETE` is fixable upstream, in whatever wrote the record. `QUANTITY_UNDEFINED` is fixable nowhere, so it is required to name the question that does apply instead.

<details>
<summary>All seventeen, from <code>reward_lens.core.reading</code></summary>

<!-- generated: refusal-reasons -->
```text
ABOVE_LOD_BELOW_LOQ
    Detected but not quantifiable. A bound is returned; a point estimate would be false
    precision.
ACCESS_INSUFFICIENT
    No estimator for this quantity works at the access you have. Silent degradation to a
    worse one is how a number becomes uninterpretable, so nothing was computed.
BELOW_LOD
    The effect is smaller than the measurement substrate's disagreement with itself, so
    it is not attributable to the thing being measured.
BUDGET_EXCEEDED
    The costed plan exceeds the declared budget.
ENVELOPE_VIOLATED
    The estimator's assumptions do not hold on this run. An instrument that is available
    and invalid is worse than one that is unavailable.
ESS_BELOW_FLOOR
    The importance weights have degenerated, so this is past the visibility horizon and
    any number would be a guess wearing an interval.
GAUGE_MISMATCH
    A covariant quantity was compared across frames with no shared basis, so the
    difference would be a coordinate artifact.
LABEL_QUALITY_UNKNOWN
    The labels have no measured error rate, so scoring against them measures the labels.
NO_MATCHED_CONTROL
    A null with no identically-powered positive control cannot be distinguished from an
    underpowered experiment.
PHASE_MISMATCH
    This is an in-run question and the run is over, or a pre-run question and it has
    started.
PLAN_NOT_CLOSED
    A registered prediction names a metric that no arc in this plan produces. Found
    before anything ran.
QUANTITY_UNDEFINED
    This quantity is not defined for this object, so there is nothing here to measure at
    any access and from any record. The remedy names the question that does apply
    instead.
RECORD_INCOMPLETE
    Your access is sufficient and the record does not carry the field this estimator
    reads. Nothing more can be recovered from this record; the fix is upstream, where it
    was written.
REFERENCE_UNCERTIFIED
    The reference material carries no uncertainty of its own. You cannot calibrate
    against an uncalibrated ruler.
SUBSTRATE_MISMATCH
    This instrument does not apply to this kind of grader. A program has no activations;
    that is a category error rather than a hard case.
UNIT_MISMATCH
    Two quantities in incompatible units were compared. The conversion factor is a
    property of the data, not of the unit, so this is not converted silently.
VOID
    The run is not readable, which is different from a negative result.
```
<!-- /generated: refusal-reasons -->

</details>

The library refuses below the limit of detection, outside an estimator's regime, against an uncertified reference, against labels with no measured error rate, with no matched control, across a unit boundary, and on a study plan that does not close.

## Install

Python 3.10 or newer. The base install pulls nothing compiled, and `import reward_lens` does not import torch, because most of the catalogue needs a run record and a callable grader and nothing else.

```bash
pip install reward-lens                # the measurement half. numpy, scipy, a CLI
pip install "reward-lens[verifier]"    # grader cards and the verifier series. Pure Python
pip install "reward-lens[white-box]"   # torch and transformers, for the model-touching half
pip install "reward-lens[trl]"         # the TRL training tap
pip install "reward-lens[dev]"         # tests, ruff, mypy
```

Reaching a subsystem behind an extra you have not installed raises a typed error naming an extra `pip` can install, rather than an `ImportError` about a module you have never heard of.

## What is in it

<!-- generated: catalogue -->
```text
95 instruments across 14 series, 51 of them in the wedge
190 quantities, 112 of them in the wedge
13 regime conditions, 7 invariance groups, 17 refusal reasons

the wedge is what a record and a callable grader reach: no weights, no gradients,
no GPU. `reward-lens capabilities` is how you find out which of them your run is in.
```
<!-- /generated: catalogue -->

Those counts are written by `scripts/gen_readme.py` straight from the registry, so the page cannot claim a different number of instruments than the catalogue holds.

Every instrument declares the quantity it estimates, the access it requires, the regime it is valid in, the invariance group it respects, the baselines it ships against, and its rung on the estimator ladder. An instrument that cannot pass `lint_instrument` does not exist: an unregistered quantity fails at import.

The kernel is `core/`: quantities with units on three axes, so comparing a per-token quantity against a per-sequence one returns `False` rather than a number; regime envelopes whose lint runs in `__post_init__`, so an envelope that cannot be enforced cannot be built; a GUM uncertainty budget that names its own largest term, which is rarely sampling noise; certified reference materials that carry their own uncertainty; and plan closure, which raises before any work runs when a study's registered metric is something no arc of the plan produces.

`record/` is the process record, five levels from `Run` down to `Token`. Scores are a `ScoreTree` rather than a float, so "what happens to the advantages if I drop the length term" is answerable on recorded leaves at zero compute. Held-out labels are `Blind[T]` with no `.unwrap()`, so leakage is a type error rather than a code review.

`tap/` wraps a grader inside somebody else's training loop and records every call. `policy/` is the peer of `signals/`, so the same instruments run against a policy and against a grader. `forecast/` refuses to build a `Forecast` if any transitive input postdates its issue time.

## Documentation

<https://reward-lens.github.io>

The catalogue, the quantity registry and the refusal reference are rendered from the live registry at build time, so a documentation page cannot claim a different number of instruments than the catalogue holds.

Coming from 2.0.1? `Observable` becomes `Instrument`, measurements return a `Reading`, and the flat v1 API is gone. The migration guide is [`docs/migration-2.0-to-3.0.md`](docs/migration-2.0-to-3.0.md), and it names what to rename, what moved behind which extra, and which two shipped numbers changed.

## What this does not claim

Stated once, so nobody has to find it in review.

- **Not the Goodhart turning point.** That is occupied, with a solver: HedgeTune, `arXiv:2506.19248`, NeurIPS 2025 Spotlight. What is measured here is the turn in `E_λ[g]` for a named gold signal `g`, and if `g` is itself a proxy, a measured turn is the turn for `g`.
- **Not white-box superiority.** The bar is decorrelation plus signal, reported against a scaffolded black-box baseline every time. The correlation between a white-box instrument's errors and the black-box baseline's errors is a required output field.
- **Not that nobody predicts reward hacking.** At least four published methods do.
- **Not that monitors degrading under pressure is new.** It is a live subfield with results in both directions. What is missing is turning that degradation curve into the figure of merit that ranks competing monitors.
- **Not linear-representation realism.** The portfolio is built so that most of it survives that hypothesis failing.

## Status

The kernel, the record, the access layer, the tap, the verifier series, the metrology and the four books are built, tested and linted.

Two gaps are worth knowing before you start. The capability report can price a measurement and route you to the instrument that performs it, but it cannot yet dispatch the measurement itself: sixty-four estimator entries are registered and none carries a callable. And ten of the twelve framework adapters are not built. `tap/adapters/` ships `trl.py` and `verifiers.py`; `verl`, `openrlhf`, `skyrl`, `slime`, `areal`, `nemo`, `primerl`, `roll` and `tinker` are named in the architecture and were never scheduled.

`probe/`, `measure/coverage/` and `stats/identification.py` are not in this release, and the compute-gated studies ship as code, a runbook and a price rather than as results. [`docs/content/not-in-3-0.md`](docs/content/not-in-3-0.md) is the full list. `CHANGELOG.md` carries the release entry, including the defects this build found in its own previous release.

## Citation

```bibtex
@misc{nadaf2026rewardlens,
    title         = {reward-lens: A Mechanistic Interpretability Library for Reward Models},
    author        = {Nadaf, Mohammed Suhail B},
    year          = {2026},
    eprint        = {2604.26130},
    archivePrefix = {arXiv},
    url           = {https://arxiv.org/abs/2604.26130},
}
```

## License

MIT

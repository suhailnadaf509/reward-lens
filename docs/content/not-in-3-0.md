# What 3.0 does not do

A library that only advertises what it has is hard to plan against. This page names the gaps by their module paths, so that if you go looking for something and cannot find it, you can tell in one place whether you are looking in the wrong spot or whether it is not there.

Two of these are things a reader will hit within an hour of starting. The rest are shape.

## Ten of the twelve framework adapters

This is the one that will cost you time, so it is first.

`reward_lens.tap` attaches to a training loop and records what the grader was asked and what it answered. The design names twelve adapters. **Two ship**, and they are the two that were built against a running trainer rather than against a README:

| Adapter | Status |
|---|---|
| `tap/adapters/trl.py` | ships. Exercised against a real `GRPOTrainer` run |
| `tap/adapters/verifiers.py` | ships. Converts published `vf-eval` output |

The other ten are named in the architecture and were never scheduled: `verl`, `openrlhf`, `skyrl`, `slime`, `areal`, `nemo`, `primerl`, `roll`, `tinker` and `generic`.

**You are not stuck if your framework is not on that list.** The tap is a wrapper around a grader callable and the record schema is public, so an adapter is a mapping exercise rather than an integration. [Write an adapter](how-to/write-an-adapter.md) is the walkthrough, and `tap/adapters/trl.py` is the worked reference: it is the file to copy.

One piece of related work does ship without an adapter. `policy/credit.py` reads SkyRL's existing tensor dump directly through a restricted unpickler, without importing `skyrl`, which covers the one thing people most often want from that framework.

## `probe/`, the third plane

The architecture has three planes: the record, the instruments that read it, and a plane that *runs* things, replicating a measurement, mutating a subject, sweeping a dose, assigning arms.

`probe/` does not exist. What happened instead is that the pieces landed under the packages that needed them first: `verifier/mutate.py` mutates a program, `organisms/dose.py` sweeps a planted dose, `record/arms.py` carries arm assignment, `dynamics/sweep.py` sweeps a parameter. So the capability is mostly present and the plane is not. What is missing with it is the resumable work queue the plane was supposed to own, which is the piece that would let a run that died mid-shard be continued rather than restarted.

## `measure/coverage/` and `stats/identification.py`

Neither exists.

`measure/coverage/` was to be the coverage series as instruments. The measurement itself is present on the verifier side, `verifier/coverage.py`, and the grader card reads it; what is absent is the instrument-typed series that would sit beside the others in the catalogue.

`stats/identification.py` was to hold the identification machinery. What ships instead is one piece of it in `stats/nulls.py`, `rum_identifiability_null`, which is the baseline a subspace-alignment claim has to beat before it means anything. The background is written up under [identifiability and gauge](theory/identifiability.md); the module is not there.

## Every Phase 6 study, which ships as code and a price

This is a deliberate shape rather than an omission, and it is worth understanding because it is how most of the compute-gated work in this library is delivered.

Eight studies are written, registered, costed and not run: the two-run rate test, the upper rung of the adiabaticity number and the rate-extrapolated hysteresis beside it, the distillation gap, the behavioural half of the false-positive search, monitor degradation under adaptive pressure, the planted-to-real transfer coefficient by standard addition, sparsity under staleness, and the shelf life of a readout. Each exists as a freezable study specification with a resolution rule and a cost, and none has a result attached.

What you get is therefore the instrument and the plan, not the finding. `card_plan` and the capability report will tell you what a study would contain and what it would cost before you spend anything, and the study engine will run it. What the library will not do is hand you a number somebody else's compute produced, because there is no such number.

The honest way to read this: for the compute-gated series, this release is a measuring instrument and a price list. Whether the measurement is worth its price is a decision the page cannot make for you, which is why the price is on it.

## What this page is not

It is not the list of open research targets. That one is [generated from the registry](catalogue/open.md) and is longer: a quantity with a name, a unit and no estimator is a research problem rather than a missing feature, and the docs build fails if a new one turns up unrecorded.

It is also not the caveats page. [Interpreting results honestly](caveats.md) is about how to read numbers this library *does* produce.

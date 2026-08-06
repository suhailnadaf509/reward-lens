# Command line

**What can the `reward-lens` command do on a laptop, and what does it refuse?** The line is clean. Anything that is a view over the [evidence store](discipline/evidence-store.md) runs here and now on CPU. Anything that needs a loaded reward model on a GPU refuses with a named exit code and points at the exact kernel call it would have made, rather than inventing a number.

Installing the package puts `reward-lens` on your path. The command imports nothing heavier than typer and the torch-free artifacts and studies layers; every model-touching import happens lazily inside a command body, so `import reward_lens.operate` stays torch-free and the command starts instantly.

```console
$ reward-lens --help

 Usage: reward-lens [OPTIONS] COMMAND [ARGS]...

 Operator surface for the reward-lens kernel: cards, scoreboard, claims, and
 the Atlas.

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                  │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ───────────────────────────────────────────────────────────────────╮
│ card        Build an RM Card for a signal: a view over every stored Evidence │
│             about it.                                                        │
│ scoreboard  Print the theorem scoreboard: standing theorems and candidate    │
│             laws.                                                            │
│ claims      Check documents against the store; exit nonzero if any number is │
│             unbound.                                                         │
│ score       Score inputs with a reward model (GPU-gated).                    │
│ serve       Serve a reward model as an RL-loop-compatible endpoint           │
│             (GPU-gated).                                                     │
│ audit       Run the blind auditing game against a signal or organism         │
│             (GPU-gated).                                                     │
│ study       Freeze, run, and report frozen studies (gate 3).                 │
│ atlas       The reward-model population Atlas.                               │
│ organism    The ground-truth organism foundry.                               │
╰──────────────────────────────────────────────────────────────────────────────╯
```

Read that command list as two columns even though the help prints one. `card`, `scoreboard`, `claims`, `atlas export`, and `study freeze` run on CPU. `score`, `serve`, `audit`, `study run`, and `organism train` are gated.

## Runs on CPU

These are views over the store: no model, no GPU, deterministic. They read what previous measurements wrote and refuse to compute a number the store cannot already back.

### scoreboard

The theorem scoreboard is the library's standing ledger: eight theorems and six candidate laws, each with the [sciences](sciences.md) that adjudicate it and the evidence, if any, that has been registered against it. On a fresh store every row is `open`, which is the honest default before any study has run to completion.

```console
$ reward-lens scoreboard
| Row | Title | Kind | Status | Science | Adjudicating evidence |
|---|---|---|---|---|---|
| T1 | Constructive unhackable-subspace finder | standing | open | S4 |  |
| T2 | Distortion equilibrium | standing | open | S8/S12 |  |
| T3 | RLHF speed proportional to teacher variance | standing | open | S12/S3 |  |
| T4 | Proxy-true reward angle | standing | open | S2/S12 |  |
| T5 | Heavy tail defeats KL control | standing | open | S3/S4 |  |
| T6 | Identifiability up to shift and scale | standing | open | S2 |  |
| T7 | No single scalar for a population | standing | open | S11 |  |
| T8 | Scalar head cannot express intransitivity | standing | open | S2 |  |
| T9 | Fluctuation-dissipation for reward hacking | candidate law | open | S3 |  |
| T10 | Belief factorization and gauge=channel-kernel | candidate law | open | S8/S2 |  |
| T11 | Evaluator-model divergence precedes hacking | candidate law | open | S13/AT |  |
| T12 | Coherence/Welch law and Hodge obstruction | candidate law | open | S5/S6 |  |
| T13 | Value convergence excess | candidate law | open | AT |  |
| T14 | Honesty unraveling law | candidate law | open | S15 |  |
```

The output is markdown on purpose: paste it straight into a report and the table renders. The `Adjudicating evidence` column stays empty until a [study](discipline/studies-and-preregistration.md) writes a registered result that fills it.

### card

A card is every stored `Evidence` about one signal, gathered under its fingerprint. Ask for a fingerprint the store has never seen and you get an honest empty card rather than an error, which is what you want in CI.

```console
$ reward-lens card mfp:demo
{
  "signal": "mfp:demo",
  "total_gpu_seconds": 0.0,
  "entries": [],
  "unvalidated_observables": []
}
```

`entries` is empty because nothing has measured `mfp:demo`. `total_gpu_seconds` sums the metered cost of every measurement the card is built from, so a card is also a receipt for the compute that produced it. Pass `--format html` for the rendered version, `--out` to write it to a file.

### claims

This is the anti-self-deception check. Tag a number in a manuscript with the `Evidence` id it came from, and `claims` loads that evidence, extracts the named field, and verifies the value is within tolerance. A tag whose evidence is not in the store, or whose value disagrees with the stored one, is a failure, and the command exits `1` so CI can block the merge.

```console
$ reward-lens claims manuscript.md
Claims checked: 1. Failures: 1.
  [FAIL] ev:0000000000000000 per_model_mean_rho.Skywork-v0.2: evidence id not in the store
```

The manuscript claimed `rho = -0.171` and cited an evidence id the store does not contain, so the check fails honestly. Against a store that holds the real measurement, the same tag passes and the command exits `0`. The point is narrow and strict: a paper cannot state a number the store cannot produce on demand.

### atlas export

The Atlas is the reward-model population view. `atlas export` writes the leaderboard as a view over the store: the standard population and whichever observables have been measured across it.

```console
$ reward-lens atlas export
{
  "models": [
    "Skywork-Reward-Llama-3.1-8B-v0.1",
    "Skywork-Reward-Llama-3.1-8B-v0.2",
    "ArmoRM-Llama3-8B",
    "Skywork-Reward-Gemma-2-27B",
    "Tulu-3-8B-RM",
    "GRM-Llama3-8B",
    "INF-ORM-Llama3.1-70B",
    "URM-LLaMa-3.1-8B",
    "QRM-Llama3.1-8B",
    "Llama-3.1-Nemotron-70B-Reward"
  ],
  "observables": [],
  "total_gpu_seconds": 0.0,
  "cells": [],
  "uncalibrated_cells": []
}
```

The population is named but `cells` is empty, because filling a cell means measuring an observable on a model, which is the GPU work `atlas sweep --execute` dispatches. `uncalibrated_cells` is the column that would flag any result standing on an instrument with no [scorecard](discipline/calibration-and-organisms.md), so the leaderboard cannot quietly launder an exploratory number into a ranking.

### study freeze

Freezing a study is gate 3, and it is pure. It hashes the spec, stamps the current git sha, and mints the `StudyID` that a later run must match. Nothing about freezing needs a model.

```console
$ reward-lens study freeze demo_study:SPEC
Frozen study study:length-bias-demo@v1#48f2cb59
  spec hash: spec:48f2cb598c02297a8a3d915fd775d3dd
  git sha:   5495f1d728dcbc7d9026ba8a139ea929944265d8+dirty
  frozen at: 2026-07-11T15:10:09.130933+00:00
```

The `+dirty` on the git sha is deliberate honesty: the working tree had uncommitted changes when the spec was frozen, and the stamp records it. The `StudyID` embeds the spec hash, so a run whose spec differs by a character cannot claim to have tested this preregistration. The `demo_study:SPEC` argument is a `module:attr` path that has to resolve to a `StudySpec`; the specs under `studies/` build theirs from a `build_spec()` factory, so a one-line module that assigns `SPEC = build_spec()` is all a freeze needs.

## GPU-gated, exit code 3

The model-touching commands dispatch to the kernel call that does the real work, and on this torch-free operator layer that call cannot run. Each one prints a panel naming the exact dispatch and exits `3`, a constant the code calls `GPU_GATED_EXIT`. It is not a stub that lies. Where an `--execute` flag exists, the same command makes the real call on hardware.

!!! warning "Needs a GPU"
    `score`, `serve`, `audit`, `study run`, and `organism train` load an 8B-class reward model and run a forward or backward pass. None of that happens on the operator layer. The commands name their dispatch and exit `3` rather than fabricate a result.

`score` is the representative case: load a signal, score a view, get `Evidence`.

```console
$ reward-lens score Skywork/Skywork-Reward-Llama-3.1-8B-v0.2
╭───────────────────────────────── GPU-gated ──────────────────────────────────╮
│ score needs a loaded reward model and a GPU.                                 │
│ It is GPU-gated on this torch-free operator layer.                           │
│                                                                              │
│ Dispatches to:                                                               │
│ reward_lens.signals.load_signal('Skywork/Skywork-Reward-Llama-3.1-8B-v0.2'). │
│ score(view) -> Evidence[Scores]                                              │
╰──────────────────────────────────────────────────────────────────────────────╯
```

The panel is the whole behavior. It names `load_signal(...).score(view)`, the call you would make yourself once the model is loaded on adequate hardware, and it returns exit `3`. No score is printed because none was computed.

`study run` shows the split most clearly. It freezes the spec first, which is real work and prints the `StudyID`, then gates on the analysis, which is not.

```console
$ reward-lens study run demo_study:SPEC
Frozen study study:length-bias-demo@v1#48f2cb59 (analysis:
studies.s12_hackability.analysis.analyze)
╭───────────────────────────────── GPU-gated ──────────────────────────────────╮
│ study run needs a loaded reward model and a GPU.                             │
│ It is GPU-gated on this torch-free operator layer.                           │
│                                                                              │
│ Dispatches to:                                                               │
│ reward_lens.studies.run_study(study:length-bias-demo@v1#48f2cb59)            │
│                                                                              │
│ The analysis resolves subjects and measures them; pass --execute to run it.  │
╰──────────────────────────────────────────────────────────────────────────────╯
```

The freeze happened. The `StudyID` is real and matches what `study freeze` produced. Only the analysis, which resolves subjects and measures them against a model, is gated.

`serve` is the honest edge of this design. Its panel names `reward_lens.signals.serve.serve`, and that module does not exist in the package.

```console
$ reward-lens serve Skywork/Skywork-Reward-Llama-3.1-8B-v0.2
╭───────────────────────────────── GPU-gated ──────────────────────────────────╮
│ serve needs a loaded reward model and a GPU.                                 │
│ It is GPU-gated on this torch-free operator layer.                           │
│                                                                              │
│ Dispatches to:                                                               │
│ reward_lens.signals.serve.serve(load_signal('Skywork/Skywork-Reward-Llama-3. │
│ 1-8B-v0.2')) (OpenRLHF/TRL/veRL proxy)                                       │
│                                                                              │
│ The reward server holds a model resident on the GPU; run it on hardware.     │
╰──────────────────────────────────────────────────────────────────────────────╯
```

`serve` refuses before it would import anything, so the missing module is never reached. The dispatch string is a pointer to the reward server this layer would run on hardware, not a promise that the code behind it ships today. Naming a call that is not yet built, and refusing rather than pretending, is the behavior the design asks for.

`audit` and `organism train` follow the same pattern: `audit` names `reward_lens.organisms.game.AuditingGame(signal, organism).run()`, `organism train` names `reward_lens.organisms.train.train(...)`, and both exit `3`.

## Where each command runs

| Command | Runs on | Exit |
| --- | --- | --- |
| `card` | CPU, view over the store | `0` |
| `scoreboard` | CPU, view over the store | `0` |
| `claims` | CPU, view over the store | `0`, or `1` if a claim is unbound |
| `atlas export` | CPU, view over the store | `0` |
| `study freeze` | CPU, gate 3 | `0` |
| `atlas sweep` | CPU to plan; `--execute` is gated | `0`, or `3` with `--execute` |
| `score`, `serve`, `audit` | GPU-gated | `3` |
| `study run`, `study report` | freeze on CPU, analysis gated | `3` without `--execute` |
| `organism make`, `train`, `score` | GPU-gated | `3` |

For the calls the gated commands name, the how-to guides show the real thing on a model you load yourself: [attribute a score](how-to/attribute-a-score.md), [freeze and run a study](how-to/freeze-and-run-a-study.md), and [build a card, check a manuscript](how-to/cards-and-claims.md). The [operate reference](reference/artifacts-operate.md) documents the underlying functions.

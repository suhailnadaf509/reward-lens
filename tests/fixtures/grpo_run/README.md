# Two real GRPO records

Written by a real `GRPOTrainer` and read by every in-run instrument. Regenerate with
`python tools/make_grpo_fixture.py`, which needs the `[trl]` extra.

| Fixture | Steps | Trajectories | Turns | Run id |
|---|---|---|---|---|
| `short/` | 12 | 96 | 192 | `run:8a8c7e29274db0a681313b48dbd1eb63` |
| `long/` | 200 | 1,600 | 3,200 | `run:f77bf75940ab982bbc35407af99cc094` |

Same configuration for both, and it is the TRL tap's, unchanged:
`trl-internal-testing/tiny-Qwen3ForCausalLM`, seed 1234, batch
8, `num_generations` 4, `max_completion_length` 12, CPU, with a length grader that returns `None`
on every seventh completion so the abstention channel is exercised on a real run rather than only
in a unit test.

Read them as records. **Nothing that consumes these should import `trl`**, which is the point of
having them on disk:

```python
from pathlib import Path
from reward_lens.record.reader import open_run

run = open_run(Path("tests/fixtures/grpo_run/long"), "run:f77bf75940ab982bbc35407af99cc094")
for step in run.steps:
    ...
```

## They replay exactly, and that is the point of having them

The record format asks how scores became advantages **exactly**, and these two are where
that claim is checked against a trainer rather than against a fixture. Replaying every group through
`record.scores.replay_advantages` and differencing against the advantages TRL itself wrote:

| Fixture | Groups | Refusing | max abs difference |
|---|---|---|---|
| `short/` | 24 | 0 | 1.17e-06 |
| `long/` | 400 | 0 | 2.97e-06 |

That residual is the float32 round-trip and nothing else. It took three fixes to get there, each of
which read as fine on its own: the policy-ratio clip was being applied as a bound on the advantage,
the divisor was the population standard deviation where TRL applies Bessel's correction, and
`std_ddof` was on the dataclass but in neither `__canonical__` nor `from_canonical`, so it did not
survive a write.

**So a replay that disagrees is now a finding rather than a known limitation**, which is what
`check_replay` was always supposed to mean.

## What these runs are not

A 2.45M-parameter model whose transformer is 26,664 parameters over two layers, the other 2.43M being an untied embedding and unembedding pair over a 151,669-token vocabulary, optimising against a length grader for 200 steps is a real optimisation
trace and **not a reward-hacking transition.** There is no hard reward gate in the composition, no
collapse, and no labelled hack rate.

So they are a good subject for asserting that an instrument computes what it claims, on real
advantages and real logprobs with real abstentions in them, and a bad subject for a claim about
lead time, transition width, or the onset of anything. An instrument whose claim needs a
phenomenon these runs do not contain should assert its mechanics here and say plainly what subject
the claim itself requires.

<div class="rl-chips">
  <span class="rl-chip rl-chip--works"><span class="rl-chip__k">reads</span> a live policy and a training record</span>
  <span class="rl-chip rl-chip--works"><span class="rl-chip__k">quantities</span> credit.by_turn, credit.by_tool_call, credit.conservation_error</span>
  <span class="rl-chip rl-chip--works"><span class="rl-chip__k">access</span> POLICY:BACKWARD</span>
</div>

# Credit geometry

**Which turn of that forty-turn episode actually got the reward, and can you prove your answer adds up?**

Most attempts at this question train something to predict where the credit lay. That is the wrong question. In a live run credit assignment is not inferred, it is executed: the optimizer computes it exactly and then throws it away. `reward_lens.policy.credit` catches it on the way past.

Define a signed measure on the token lattice by

\[ \mu(k, t) = A_k \cdot \nabla_\theta \log \pi(y_{k,t} \mid \cdot), \qquad \sum_{k,t} \mu(k,t) = \nabla_\theta J. \]

Because the total mass is the update itself, **a credit report either accounts for the whole step or it has a bug.** That is what makes this an audit rather than an attribution heuristic, and it is why `conservation_error` is a reported quantity on every reading rather than a test that runs once in CI.

## Disintegrations, not a per-token gradient

Nothing here ever materialises a per-token per-parameter gradient. That object has \(|\theta| \times T\) entries and computing it is not the way in.

What the module computes are *disintegrations*. Given a partition of the trained positions into \(m\) parts, mask the objective to each part and take one backward pass per part with `retain_graph=True`. That gives \(m\) parameter-space vectors whose sum is the full gradient, at \(m\) backward passes and three parameter-sized buffers of peak memory. A forty-turn episode is forty backward passes, which is feasible. The per-token form on a half-million-token episode is not, and that is exactly why the interesting partition is by turn.

```python
from reward_lens.policy.credit import batch_from_trajectories, by_rollout, by_segment, disintegrate

batch, segments = batch_from_trajectories(policy.tokenizer, group.trajectories)
report = disintegrate(policy, batch, by_rollout(batch))
print(report.render())
```

`by_rollout` and `by_segment` are the partitions that ship, and `turn_segments` builds the segment list a turn decomposition needs. `Partition` is an ordinary object, so a partition of your own is a few lines.

One detail in `batch_from_trajectories` is worth knowing because it is easy to get wrong elsewhere: it tokenises each turn's own text and concatenates, which is what a trainer does. Tokenising the concatenated string instead lets the tokeniser merge across a turn boundary, and a turn decomposition built on merged tokens attributes a token to whichever turn won the merge.

## Norms do not add; the projection does

This is the part most attribution write-ups get wrong, and it is worth a paragraph because the failure is silent.

\(\lVert g_S \rVert\) is not additive over parts. A report built out of norms cannot close, and one built out of normalised norms will happily claim a hundred and seventy per cent of a step. The functional that *is* exactly additive is the projection onto the full gradient, \(\langle g_S, g_{\text{full}}\rangle / \lVert g_{\text{full}} \rVert^2\), and those sum to one by linearity.

So `projected_share` is the share, `norm_share` sits beside it, and their disagreement is reported as `cancellation`. That disagreement is a real quantity rather than a diagnostic. On step 0 of the 200-step reference record it is `2.23`, which means `55%` of the per-rollout gradient mass cancels between rollouts before the optimizer sees any of it.

A projected share can be negative, and that is a finding rather than a defect: a part whose gradient opposes the direction the step went contributed negative credit to it.

## The audit closes at machine epsilon

Measured on step 0 of the 200-step record with the model that wrote it, over eight real rollouts and 107 trained positions:

| Check | Result |
|---|---|
| Per-rollout gradients against an independent full backward, float64 | `2.704e-16` |
| The same, float32 | `1.126e-07` |
| The same, the model's native bfloat16 | `1.56e-03` |
| One real `torch.optim.SGD` step against the summed gradient | `4.938e-09` |

`g_full` is taken by its own independent backward pass. Deriving it as the sum of the parts would make the number identically zero and the audit vacuous, so it is not done that way anywhere in the module.

The bfloat16 row is why the module casts to float32 by default, restores the original dtype afterwards, and reports which dtype produced the number. The gap between the native dtype and float32 is a factor of about fourteen thousand, so precision is not a footnote on this measurement.

**One trap is worth naming because it looks like a failure and is not.** `step_conservation` checks \(\sum \mu = \Delta\theta / \eta\) against an actual optimizer step, and that identity holds for plain gradient descent and for nothing else: momentum, Adam's second moment, weight decay and gradient clipping each break it. It also has a precision floor. At the reference record's own learning rate the parameter difference is tiny against the parameters themselves, and in float32 the smallest relative error the comparison can resolve is `7.9`. The naive check there returns `0.151`, which reads as catastrophic failure and is not one: it is a number below the floor of the instrument that produced it. `update_precision_floor` and `update_precision_limited` are on the report so no reader has to work that out.

## G3's kill condition fires, and that is the package closing

The third instrument in the series asks whether GRPO's outcome reward induces a usable process reward model. It does, in principle: GRPO with an outcome reward is equivalent to a PRM-aware objective whose process reward is the Monte-Carlo value \(q(s) = \mathbb{E}[R \mid \text{prefix } s]\). Extracting it needs rollouts that share prefixes, because a shared prefix is the only place a prefix has more than one sample.

On the reference record they never do. Measured across all 400 groups rather than argued from a few:

| | Across the 400 groups |
|---|---|
| Kill condition fires | on `400` of `400` |
| Mean divergence depth | `1.00` in every group |
| `informative_fraction` | `0.000`, and that is the maximum over all of them |

Every group has a longest common completion prefix of zero tokens. Every rollout is alone from its first token, every value past the root is that rollout's own outcome, the whole outcome advantage lands on token one, and `92.16%` of trained positions, averaged over the 400 groups, carry exactly zero process reward.

G3's own catalogue entry says "kill if the induced function is constant". This is the measured form of that, so **G3 is closed rather than failed**. The instrument is built, it runs, it returns a `Refusal`-shaped verdict with the numbers in it, and the verdict is that this quantity is not worth reading on a record whose sampler never revisits a prefix. The remedy on the reading names the estimator that does survive: a re-roll at prefixes the sampler never revisited, which is rung 1.

The other side of the same measurement is asserted too. Given four rollouts that branch, the induced function is non-degenerate and the instrument returns it, so the kill condition is a property of the subject rather than of the code.

## Reading another framework's tensor dump

The catalogue's rung 0 for this series points at SkyRL's existing dump, and the package consumes it rather than reimplementing it. Two things about that were found by reading the source rather than the documentation, and both are reproduced as tests rather than repeated as claims.

`dump_data_batch` is a boolean configuration field, not a function; the function is `dump_data`. And `rewards` is popped from the batch at `trainer.py:436` and `uids` at `:437`, five lines before the dump, so the one mechanism that is supposed to already write the interpretable tensor writes neither the reward it came from nor the group it belongs to. `read_skyrl_dump` consumes what does reach disk, recovers the grouping by hashing the prompt, and never imports `skyrl`: the read goes through a restricted unpickler, and the acceptance run asserts that `skyrl` is absent from `sys.modules` afterwards.

## What this cannot do

Three sentences, and they are in the module header as well as here.

The conservation identity is a claim about the optimizer, so `step_conservation` refuses to assert it for any optimizer outside the exact set. The disintegration is a statement about *this* gradient at *these* parameters, so it says where the step went and not what any part of the trajectory contributed to the run as a whole. And every number here is first-order: removing a turn does not remove its share, because removing it changes every downstream position.

The reference record also cannot support a turn-concentration claim, and the acceptance run says so by name rather than working around it. Its 1,600 trajectories are two turns each, one user and one assistant, with zero tool calls. So the turn decomposition is exercised on what is there, correctly putting the whole credit on the assistant turn and none on the masked user turn, the tool-call structure is exercised on a trajectory built through the record schema, and the subject a real claim would need is named in the test that would carry it.

Everything on this page is asserted in `tests/acceptance/test_w5_4_credit.py`.

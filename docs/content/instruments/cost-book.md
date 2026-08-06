<div class="rl-chips">
  <span class="rl-chip rl-chip--works"><span class="rl-chip__k">reads</span> a training record</span>
  <span class="rl-chip rl-chip--works"><span class="rl-chip__k">quantities</span> update.kl_min, update.kl_share, update.kl_spent, update.efficiency</span>
  <span class="rl-chip rl-chip--works"><span class="rl-chip__k">access</span> RECORD, plus POLICY:BACKWARD for the rung-2 metric</span>
</div>

# The cost book

**Your last optimizer step moved the policy. How many nats did it spend, and how many did the behaviour change actually require?**

Those are different numbers and the gap between them is the interesting one. A step buys some movement in the behaviour you care about, and it pays for that movement in policy divergence. If the step spent far more divergence than the movement needed, most of what it bought was somewhere you were not looking.

`reward_lens.measure.efficiency` is the book that keeps that account. It is series F3 of the catalogue, and it ships three things: an estimate of the behavioural metric `G`, a minimum cost per step, and the ratio between that minimum and what the step actually spent.

## The three numbers

The behavioural features are a vector \(z\), and a step moves them by \(\Delta z\). `G` is the covariance of those features that a parameter move can reach:

\[ G = J F^{-1} J^{\top}, \qquad J = \partial z / \partial\theta \]

with \(F\) the Fisher information of the policy. Given `G`, the cheapest policy divergence that could have produced the observed feature movement is a quadratic form:

\[ \mathrm{KL}_{\min} = \tfrac{1}{2}\,\Delta z^{\top} G^{-1} \Delta z \]

and the efficiency of the step is that minimum over what the step really spent, \(D_t = \mathrm{KL}(\pi_{t+1}\,\|\,\pi_t)\):

\[ \eta_t = \mathrm{KL}_{\min} / D_t \in [0, 1]. \]

The bound is a theorem, not a hope: \(\mathrm{KL}_{\min}\) is by construction the smallest divergence consistent with the movement, so it cannot exceed the divergence that happened. That makes the bound a self-check. An efficiency above one is the instrument telling you a premise failed, and the instrument is built so that it never hands you one.

`KL_min` also splits by feature. `shapley_shares` apportions the quadratic form across the named features with an exact Shapley decomposition, so "the step spent most of its budget lengthening responses" is a statement you can read off rather than infer.

## The headline is a refusal, and that is the package working

Run the cost book on either of the reference GRPO records and it declines to give you an efficiency.

```python
from reward_lens.measure.efficiency import cost_series, kl_min_series, metric_g
from reward_lens.measure.ledger.features import SurfaceFeatures
from reward_lens.measure.ledger.price import ledger_series, learning_rates, steps_from_run, whole_run
from reward_lens.record.reader import open_run

run = open_run("tests/fixtures/grpo_run/long", run_id)
window = whole_run(run)
samples = steps_from_run(run, SurfaceFeatures(), window=window)
ledgers = ledger_series(samples, eta_by_step=learning_rates(run, window))
g = metric_g(samples, method="covariance_bound")

book = cost_series(ledgers, g, run_=run)
print(book.render())
```

```text
cost_series  RECORD_INCOMPLETE
    199 step pair(s) of run run:f77bf759... carries no per-step KL (`D_t`) from record
    `kl_to_previous`
    Remedy: Log the per-step KL. In TRL set `beta` above zero, which makes the trainer
    compute and log `kl`; in veRL read `actor/kl`; any framework that penalises drift
    already has this number and the tap only has to carry it into
    `OptimizerTelemetry.kl_to_previous`. Failing that, pass `kl_spent={step: nats}` from
    wherever you do have it, or hand this function two consecutive checkpoints so `D_t`
    can be computed directly as `1/2 dtheta^T F dtheta`. What this will not do is
    substitute the clipped gradient norm times the learning rate: with AdamW the applied
    step is not the gradient times eta, the moment state that would relate them is not in
    the record, and the resulting number would be wrong by an unknown factor while still
    being called `kl_spent`. `KL_min` and its per-feature shares need no denominator and
    are returned by `kl_min_series` on this same record.
```

That remedy is long on purpose. It is the thing a reader acts on, so it names the framework field in two frameworks, the two ways round the problem, the proxy it is refusing to use and why, and the half of the book that still works. A remedy reading "envelope violated" would have been shorter and worth nothing.

Four fields would each have given \(D_t\) and all four are absent on every step of both records. `kl_to_previous` is the quantity itself. `kl_to_ref` is a different quantity and would not have done, but its absence rules out even the wrong answer. `update_norm` with the Fisher would have given \(\tfrac{1}{2}\Delta\theta^{\top} F \Delta\theta\), and `grad_norm_unclipped` with the learning rate would have given it for plain gradient descent, which this is not: the recorded optimizer is `adamw_torch_fused`, so the applied step is not the gradient times the learning rate, and the moment state that would relate them was never written. `beta` is `0.0` on every step, which is why the trainer never computed a KL in the first place.

The one field that *is* present is the clipped gradient norm, and that is exactly why the remedy names it. A proxy built from it would have carried the name `kl_spent` while measuring something else, and a downstream reader would have had no way to tell. The refusal is the honest return value and the remedy is the shopping list.

**The half that needs no denominator still runs.** `KL_min` and its per-feature shares depend on \(\Delta z\) and `G` and not on \(D_t\), so `kl_min_series` returns a real row for every consecutive step pair the record has: `199` rows on the 200-step record, with no refusal and no gap.

```python
rows = kl_min_series(ledgers, g)
print(len(rows), rows[0].kl_min >= 0.0)
# 199 True
```

That is worth stating plainly, because "the record is incomplete" is often read as "nothing can be measured here". Half the book is readable on a record that will never support the other half.

## What it measures when the denominator exists

A clause about a bounded ratio is worth nothing if the denominator is never present, so the acceptance run builds one. `trl-internal-testing/tiny-Qwen3ForCausalLM` is loaded at the checkpoint the 200-step record started from, one real policy-gradient step is taken on that record's own step-0 rollouts and advantages, and both checkpoints are sampled from. \(D_t\) is then computed exactly, as a full-vocabulary token-level KL over sequences drawn from the post-step policy, and `G` comes from the rung-2 Fisher kernel at the pre-step policy.

| Step size | \(D_t\) spent | \(\mathrm{KL}_{\min}\) | Reading |
|---|---|---|---|
| `1.0` | `4.07` nats per sequence | `2.31` | efficiency `0.566`, inside the bound |
| `0.5` | `0.035` nats per sequence | `2.80` | refusal, `ENVELOPE_VIOLATED` |

The second row is the more informative one. The ratio there is `81`, and the instrument refuses rather than reporting it. A value outside the unit interval is an instrument bug, so the instrument must never hand one out, and the way it never hands one out is by refusing when the premise fails. At the smaller step the failed premise is visible in the numbers: \(\Delta z\) on the leading feature is under half its own standard error, so the step's behavioural effect is below what sixteen samples resolve, and a quadratic form in a noisy \(\Delta z\) measures the noise.

Two things about that experiment are not the trainer's and are stated rather than smoothed over. The step is plain gradient descent rather than AdamW, so \(\Delta\theta\) is known exactly rather than through an optimizer's hidden state. And a step size of `1.0` is far larger than any real run would take. What the experiment establishes is that the bound holds on a real network with a real denominator, and that the self-check fires when it should.

The arithmetic is checked separately against hand-worked cases rather than against itself. For a correlated two-feature metric `G = [[2, 1], [1, 2]]`, whose determinant is `3`, a movement of one unit on the first feature costs one third of a nat, splitting into Shapley shares of `7/24` and `1/24`. Those fractions are worked out in the test body and compared at `1e-12`.

## Do not build `G` from a small-sample Fisher

The package refuses to compute \(D_t\) from an empirical Fisher, and the reason is a measurement rather than a preference. On the same subject, \(\tfrac{1}{2}\Delta\theta^{\top} F \Delta\theta\) from the eight-rollout empirical Fisher gives `1.3e-2` nats where the token-level KL gives `1.2e-4`, an overestimate of one hundred and ten times; at a step size of `1.0` it gives `13,179` against a true `3.9`. An empirical Fisher from eight samples in 2,453,368 dimensions puts all of its curvature in the six directions it can see, and the update direction lies inside that span, so it reads the whole step length there.

The same limit reaches the rung-2 metric itself, and it is stated three lines into `measure/efficiency/metric.py` rather than on a caveats page. With fewer rollouts than parameters, which is every real policy, the *undamped* kernel plug-in is not merely noisy: the feature matrix lies entirely inside the row space of the scores, the projection is the identity on it, and \(\hat G = \hat C\) exactly, so \(h^2 = 1\) for every feature no matter what is true. On the 200-step fixture with the damping taken to zero the three moving features come back at `0.99999`. The damping is therefore not a numerical convenience, it is the entire content of the estimate, which is why every `MetricG` carries its `damping` and a `damping_stable` flag, and why a reading with `damping_stable` false should be read as the covariance bound with a shrinkage applied rather than as a measurement of `G`.

The second limit is memory. The kernel form holds \(n \cdot |\theta|\) floats: 78 MB for eight rollouts of a 2.45M-parameter policy, and 224 GB for eight rollouts of a 7B one. Above roughly a hundred million parameters the score matrix has to be sketched or the Fisher solves done directly, and neither is implemented here.

## Where the numbers come from

Everything on this page is asserted in `tests/acceptance/test_w4_4_efficiency.py`, which runs the refusal half on the real records in CI and the two-checkpoint half behind the `whitebox` marker. The feature basis is the join key throughout and it is `measure.ledger.price.StepSample.names`, whole and in order: `G`, the ledger's \(\Delta z\) and this package's shares are vectors in that one basis, and a ledger paired with a metric in a different basis refuses with `UNIT_MISMATCH` rather than producing a quadratic form across two coordinate systems.

Next: [the four books](reconciliation.md), which reads this one alongside the effect and cause books and asks whether they agree.

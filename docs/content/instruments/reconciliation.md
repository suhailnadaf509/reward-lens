<div class="rl-chips">
  <span class="rl-chip rl-chip--works"><span class="rl-chip__k">reads</span> a training record and the cost book</span>
  <span class="rl-chip rl-chip--works"><span class="rl-chip__k">quantities</span> books.reconciliation_residual, books.budget_closure, selection.lande_slope</span>
  <span class="rl-chip rl-chip--works"><span class="rl-chip__k">access</span> RECORD, plus POLICY:BACKWARD for an independent test</span>
</div>

# The four books

**Two independent routes predict how far your policy's behaviour moved this step. Do they agree, and if not, by how much and why?**

The reconciliation keeps four books on a training run. The **effect** book measures the movement that happened, \(\Delta z_{\text{obs}}\). The **cause** book measures the selection differential \(S\), how much the advantage favoured each behavioural feature. The **capacity** book is `G`, what a parameter move can reach, and it comes from [the cost book](cost-book.md). Cause and capacity together give a prediction:

\[ \Delta z_{\text{pred}} = \eta \cdot G \cdot C^{-1} S. \]

The difference between the two is the reconciliation residual, and it is not noise. It is a budget with nine named terms, and the question this instrument answers is whether \(\mathrm{Var}(\rho)\) is accounted for by \(\sum u_i^2\). A closed budget characterises the instrument; an open one names an unmodelled term. Either outcome is publishable, which is why both are published here.

`reward_lens.measure.reconcile` is F4 and F6: the residual with its budget, and Lande's slope regressing the observed response on the predicted one.

## Both verdicts, and they differ

The instrument ran on both reference GRPO records, and the two runs disagree. That is the useful outcome, because two agreeing runs would have told you less about the instrument than one of each.

**The 200-step record closes.** All three varying surface features have a bootstrap interval on \(\mathrm{Var}(\rho) / \sum u_i^2\) that contains one:

| Feature | Ratio | 95% interval |
|---|---|---|
| `response_chars` | `1.082` | `[0.846, 1.341]` |
| `response_words` | `1.097` | `[0.887, 1.302]` |
| `mean_word_length` | `1.042` | `[0.769, 1.315]` |

over a cluster bootstrap on 199 step pairs. Welch-Satterthwaite gives `2,786` effective degrees of freedom, so the coverage factor stays at the conventional two.

**The 12-step record is over.** On `response_chars` the ratio is `0.555` with an interval of `[0.187, 0.921]`, which does not contain one: the budget composes to more scatter than the residual has. That is a defect in the budget rather than a property of the run, and the two mechanisms behind it are named in `measure.reconcile.residual`. `u_MC` is a root-mean-square over steps of a per-step standard error estimated from eight rollouts, and the root-mean-square of a noisy quantity overstates it. And consecutive step pairs share a step, which makes \(\rho\) negatively autocorrelated and biases the sample variance across eleven overlapping pairs downward by around a fifth.

## What "closed" means here, in the same breath as the number

The 200-step verdict is `closed`, `u_MC` is `100.0%` of the composed variance on every feature of both runs, and the test's own detection floor on that run is `854,000` times the first-order prediction it is arbitrating. Those three facts belong in one paragraph, because separating them is how a real result becomes an overstated one.

Read together they say this. The movement of these features is accounted for by the sampling noise of two batches of eight rollouts. The budget closing is not the four books reconciling; it is one term being large enough to absorb everything. On the 200-step record the test separates an extra term only above `3.76` in units of `response_chars`, and the prediction \(\eta G \beta\) it is arbitrating is `5.65e-06`. So the verdict `closed` is real and reproducible and could not have come out any other way at this scale.

`FeatureClosure.detectable_u` is on the reading so that no reader has to derive that for themselves. A verdict without it would be true and misleading at the same time.

## Nine terms, five computed, four named absent

The budget is itemised rather than summarised, and a term that could not be computed is named rather than dropped or silently zeroed.

| Term | On these records |
|---|---|
| `u_MC` | computed, and carries the whole budget |
| `u_batch` | computed |
| `u_basis` | computed |
| `u_KL` | exactly zero, because `beta` is `0.0` in the schedule at every step of both runs |
| `u_stale` | exactly zero, because every trajectory segment declares no staleness |
| `u_entropy` | absent: no trainer config in the record carries an entropy coefficient |
| `u_momentum` | absent: `grad_norm_unclipped` and `update_norm` are both missing and the optimizer is `adamw_torch_fused`, so this term is certainly non-zero and certainly not measurable here |
| `u_curv` | absent: no record carries a second derivative |
| `u_clip` | absent: `clip_fraction` is missing |

The two zeros are results rather than gaps: they are zero for a stated reason, not for want of a number. The four absences mean the composed uncertainty is a lower bound, so neither verdict can distinguish an unmodelled term from an unmeasured one. Closing that needs a record carrying `clip_fraction`, `grad_norm_unclipped` and `update_norm`, which is a change to the tap rather than to this instrument, plus one Hessian-vector product per step at `POLICY: BACKWARD`.

## F6, and why the rung-2 attempt is a finding about the estimator

Lande's equation holds at slope one, and a slope near zero would retire the whole first level of the story. That makes it the load-bearing assumption worth testing rather than assuming.

At rung 0, `G` is the covariance bound, so \(G = C\) and \(G\beta = S\) exactly: F4's residual reduces to F1's and F6's slope reduces to F2's effective learning rate. Both readings carry that as a field rather than as a caveat, because a fit reported as independent when it is not is a claimed distinction the estimator could not have found.

Getting a `G` that is not `C` needs the rung-2 Fisher solve, and it was run: a 24-step window of the 200-step record, the same `trl-internal-testing/tiny-Qwen3ForCausalLM` the record names, 192 rollouts against 2,453,368 parameters. It ran, and the result is about the estimator rather than about the run. Heritability came out between `0.99135` and `0.99143` on all three features against a damping of `0.0087442`, which is the same number. The empirical Fisher has rank at most the rollout count, so with 192 rollouts against 2,453,368 parameters every feature lies exactly in the span of the score vectors, \(N = C - G\) collapses onto the regulariser, and the reported heritability is \(1 - \lambda\) rather than a property of the policy.

`_is_degenerate` catches that case as well as the exact one, which is why the fit is reported as not independent at either rung. The slope over 23 step pairs was `-37624` with an interval of `[-279973, 232981]` at \(R^2\) `0.0018` and a permuted-step null at `p = 0.62`.

## Is two runs enough?

For the mechanics, yes, and two runs disagreeing is worth more than two agreeing. For a characterisation of the instrument, no, and the reason is the four unmeasured terms above rather than the count of runs.

Everything on this page is asserted in `tests/acceptance/test_w5_6_reconcile.py`, on the records under `tests/fixtures/grpo_run/`: a real `GRPOTrainer`, real weights, real sampling, real advantages, real abstentions. \(\Delta z\), \(S\), \(C\) and \(\eta\) come off the record and `G` comes from `measure.efficiency.metric_g`. Nothing here is synthetic.

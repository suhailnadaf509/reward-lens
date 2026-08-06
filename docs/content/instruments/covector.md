<div class="rl-chips">
  <span class="rl-chip rl-chip--works"><span class="rl-chip__k">reads</span> a live policy and its grader</span>
  <span class="rl-chip rl-chip--works"><span class="rl-chip__k">quantities</span> selection.covector, selection.dimensionality, selection.stable_rank</span>
  <span class="rl-chip rl-chip--works"><span class="rl-chip__k">gauge</span> needs a frame</span>
</div>

# The reward covector

**In which direction in the residual stream would a nudge raise the expected reward, and how many such directions are there?**

That is the frontier question, and it has two answers at different prices. The general form takes the derivative through the *policy*:

\[ s_l = \mathbb{E}\bigl[(r - b)\, \nabla_{h_l} \log \pi \bigr] \]

which needs nothing about the grader except that it returns a number. The grader can be a compiler, a test harness, a tournament or a wall-clock timer. The cheap form takes the derivative through the *reward*, \(\partial r/\partial h_l\), which needs the grader to be differentiable and to share a model with the policy.

The same breath that proposes the general form also records that its variance is the largest technical risk in the design, and that nothing may be built on it until a variance-versus-\(K\) curve has been measured.

## The spike was run and it returned no-go

That is what this page is mostly about, because a recorded no-go is a successful spike and it is worth reading as one.

Two thresholds were registered before the first number was produced: a relative standard error below `1.0`, and a mean cosine above `0.5` between two independent estimates of the same direction. With every standard reduction technique applied, group-mean baselines, antithetic sampling and control variates, the measured values at a group size of 64 were:

| Registered metric | Threshold | Measured | Verdict |
|---|---|---|---|
| Relative standard error | `< 1.0` | `1.075` | fails |
| Split-half direction cosine | `> 0.5` | `0.145` | fails |

Both fail, on every arm of the ablation. The kill criterion was the prediction's own negation and it fired, which is a plan change rather than a failure: **F5 ships as the differentiable-surrogate case at rung 3.**

The comparator makes the size of the gap concrete. On the same subject at the same group size, \(\partial r/\partial h_l\) has a relative standard error of `0.021` and a split-half cosine of `0.9998`. The surrogate is `51` times less noisy and its direction reproduces where the score-function form's does not. A prediction that named a comparator and lost to it is refuted rather than merely unconfirmed.

## The ablation ladder, and the technique that made things worse

The reductions were ablated one at a time rather than applied as a bundle, which is what makes the next result readable.

| Arm | Relative standard error at K=64 |
|---|---|
| No baseline | `4.99` |
| Group-mean baseline | `1.082` |
| Leave-one-out baseline | `1.099` |
| Leave-one-out plus control variates | `1.097` |
| All three, including antithetic pairing | `1.075` |
| Differentiable surrogate | `0.021` |

Only the group-mean baseline earned its place, and it earned it decisively: from `4.99` to `1.082`. Everything after it bought a two per cent improvement in the noise.

**And antithetic sampling actively hurt the thing that matters.** The split-half cosine at the same group size is `0.4515` with a leave-one-out baseline alone and `0.1451` once antithetic pairing is added. The direction got worse while the scalar noise got marginally better, which is the shape of a variance reduction that trades bias for variance in a place the headline metric does not look. That is structural rather than a tuning artefact, and it is recorded here because antithetic sampling is one of the three standard techniques the ablation ladder was built to apply.

## The score-function rungs still ship, with a floor

The spike measured a floor, not an impossibility, so rungs 0 to 2 are implemented rather than deleted.

At a group size of 64 over eight prompts the estimator pools 512 rollouts, and the noise follows a \(1/\sqrt{n}\) scaling. Extrapolating that scaling puts the point where the estimator's own dispersion falls to the size of the covector it is estimating at `592` pooled rollouts on the all-reduction arm and `616` on the arm without antithetic pairing. `POOLED_N_FLOOR` is `600`, rounded up from the measured value because quoting an extrapolated floor to four significant figures implies a precision it does not have. Above the floor the rungs run; below it they refuse with the number in the remedy.

So what the spike rules out is the rung-0 reading at the group sizes reinforcement learning actually uses, which is the reading somebody would reach for first.

## What this instrument cannot do

Three lines in, rather than on a caveats page.

It reports a direction in one model's residual-stream basis, so two models' covectors are not comparable without a shared frame, and the gauge gate enforces that rather than trusting the reader. It is a first-order object at the current parameters: it says what an infinitesimal constant offset at a layer does to the expected reward, not what removing a feature does. And on a differentiable surrogate it answers a different question from the score-function form. \(\partial r/\partial h_l\) is the direction that raises the reward's value at fixed text; \(s_l\) is the direction that raises the expected reward by moving the policy. They coincide only when the grader is a linear head on the model being studied.

## Dimensionality, and the four caveats that travel with it

`selection_second_moment`, `whitened_spectrum`, `stable_rank` and `participation_ratio` answer the second half of the question: how many directions there are. Four caveats travel with every number they produce, and one of them bit hard enough here to be worth stating with its measurement.

The participation ratio is linear, so it undercounts curvature: a spectrum spread over eight directions may still be one curved manifold. It depends on conditioning, so a value across prompts is not the value within a task. It is preprocessing-sensitive, and on this subject sum-pooling and mean-pooling over positions gave directions at cosine `0.9997` with relative standard errors of `1.097` and `1.125`, so the convention moved the noise and not the direction. And with the sample count near the dimension the spectrum is Marchenko-Pastur distorted, which is measured rather than cited: at a group size of 4 the stable rank of the second moment reads `2.686` out of `8`, and the same matrix estimated from the whole pool reads `6.05`. A reading taken at a realistic group size would have reported concentration in a spectrum that is nearly flat.

The spectrum's leading direction does not reproduce here either, and the comparison that says so is a chance level rather than an eyeball. Two independent uniform directions in eight dimensions have a mean absolute cosine of `0.290`; the measured top-eigenvector overlap between independent estimates is `0.357` at a group size of 64, and lower at every smaller group size tested.

The convention is on the payload as well as here, because a dimensionality quoted without it is not a number: the participation ratio is the moment-ratio form, \((\sum \lambda_i)^2 / \sum \lambda_i^2\). The other convention in circulation counts the modes needed to explain most of the variance and gives a different answer on the same matrix. Numerical rank is never reported, because a matrix is full numerical rank as soon as no singular value is exactly zero, which is generic for anything touched by floating-point arithmetic.

## What the subject is, and what it is not

The spike ran against the model that wrote the 200-step record: a 2.45M-parameter Qwen3 with two layers and a residual width of `8`, near-uniform at temperature `1.0`, graded on completion length. That is a real policy, a real sampler and a real grader, and the estimator's noise scaling is a property of the estimator, so the variance-versus-\(K\) curve transfers.

Two things do not. A stable rank out of `8` says almost nothing about a stable rank out of `4096`, so the shape claim in the catalogue's headline needs a frontier-scale residual stream. And the covector this subject supports is small in absolute terms: a constant offset of a tenth of the activation norm along it is predicted to move the expected reward by `1.67e-04` against a rollout-to-rollout reward standard deviation of `0.288`, so separating that from zero at two sigma would take about `1.19e+07` rollouts. Run at a quarter of the activation norm over 1,024 rollouts per arm, the plus-minus gap came back `0.0069` with a standard error of `0.0124` against a predicted `0.00042`: consistent with zero, and uninformative in exactly the way the arithmetic said it would be.

So the no-go is recorded as a no-go on the estimator's noise, which is what the prediction asked, and the effect-size arm sits beside it as a property of this subject rather than as part of the verdict. A go on a trained policy against a grader with real discrimination is not ruled out by this measurement. Running that needs a frontier-scale model at `POLICY: BACKWARD` and a few thousand rollouts per layer.

## The apparatus is on every reading

`nnsight` replaces `torch.Tensor.backward` at import, copying `__module__` and `__qualname__` onto the replacement so that no name-based check notices, and every backward call in that process routes through it afterwards. A gradient measured in a patched process is a measurement whose apparatus changed without anything in the reading saying so, so the reading records whether it was imported, along with the torch and transformers versions.

The pipeline's own correctness check passes and is worth knowing about: \(\mathbb{E}[g] = 0\) holds exactly for the position-summed gradient, so a pipeline computing something else fails there. Over 768 fresh rollouts the mean gradient had norm `3.35e-03` against an expected sampling noise of `2.79e-03`, with a largest per-coordinate z of `2.35`, which is what eight draws from a standard normal do.

The full resolution, every arm and every ablation, is the `P8_RESOLUTION` constant in `measure/frontier/covector.py`, and the acceptance run in `tests/acceptance/test_w5_2_covector.py` asserts the surrogate path and the recorded decision against it.

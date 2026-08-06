# Effective group size of a GRPO group

**Your group has sixteen rollouts. How many independent observations is the grader giving you?**

Fewer than sixteen, and the gap is what you are paying H100-hours for. A group of K rollouts costs K forward passes and K grader calls, but part of the spread between those K scores is the grader disagreeing with itself rather than the rollouts differing in quality. A group-relative estimator cannot tell the two apart, so the noise goes into the advantage and into the gradient exactly as signal does.

`grader.effective_group_size` is the number that survives:

\[
n_\text{eff} = K \times \rho
\]

where \(\rho\) is the reliability of a single observed score over the universe of grader draws you are willing to generalise over. It is one for a noiseless grader and falls toward zero as measurement error takes over the observed spread. This is Spearman's 1904 attenuation result, applied to a reward loop rather than to a correlation.

## Rung 0 is free and it can only tell you K

The cheapest rung reads the scores your run already logged. With one score per rollout there is no replication, nothing separates error from signal, and the honest answer is that all K observations count:

```python
import numpy as np
from reward_lens.measure.metrology.gstudy import GroupScores, effective_group_size

rng = np.random.default_rng(0)
scores = rng.normal(0, 1, (200, 8)) + rng.normal(0, 1, (200, 1))  # 200 groups of 8
groups = GroupScores.of(scores, grader="judge")

r0 = effective_group_size(groups, None, n_resamples=500, seed=0)
print(r0.n_eff, "of", r0.k_nominal)
print(r0.shape_factor)
# 8.0 of 8.0
# 0.701
```

That is not a disappointing instrument, it is the point. K is the number the field already quotes, the library will not pretend to have improved on it without data it does not have, and the reading says so in as many words: *"This is the ceiling, not a measurement of your grader."* Its declared bias direction is `upward` and its `invisible_terms` field names every error term it could not see.

The second number is the **Kish shape factor**, and it is a different question. It asks how evenly the group spends its gradient: one when every rollout sits the same distance from the group mean, falling toward \(1/K\) as the contrast concentrates on a single rollout. Gaussian rewards give \(2/\pi\), uniform ones give three quarters. It is a property of your reward distribution and your policy, not of your grader, and reading it as grader waste is the mistake this page exists to prevent.

## Rung 3 needs replication, and it is where the number moves

To learn what the grader costs you, score the same objects more than once. Two or more raters gives the fully crossed design that identifies the object-by-rater interaction, which is the term that pushes every advantage in a group the same way and is therefore the one a group-relative estimator cannot cancel:

```python
from reward_lens.measure.metrology.gstudy import (
    ReplicationDesign, effective_group_size, jackknife_reliability,
)

truth = rng.normal(0, 1, (500, 1))              # 500 objects, one true quality each
replicated = truth + rng.normal(0, 1, (500, 4))  # each scored by four grader draws
design = ReplicationDesign(scores=replicated, raters=("a", "b", "c", "d"))

_, se = jackknife_reliability(design)
r3 = effective_group_size(groups, design, n_resamples=500, seed=0, reliability_se=se)
print(r3.n_eff, r3.reliability, [r3.ci_low, r3.ci_high])
# 4.134 0.517 [3.937, 4.331]
print(r3.says())
# Your effective group size is 4.1, not 8. You are paying for 3.9 rollouts of grader
# noise. (rung 3; single-score reliability 0.517; 95% interval [3.94, 4.33]) The group's
# Kish shape factor is 0.70, which is how evenly it spends its gradient and is a property
# of the reward distribution rather than of the grader.
```

Half the group is grader noise. That is a synthetic grader built to have exactly that much noise, so the arithmetic is checkable; the same instrument run against eleven open reward models on a real shared bank is in `tests/acceptance/test_w3_2a_metrology.py`.

`reliability_se` is what gives the reading an interval. Without it the reliability is treated as known, and `has_interval` returns false rather than the reading quoting a width it did not earn. Pass `jackknife_reliability(design)[1]`, which leaves each rater out in turn: a coefficient estimated from four graders is a claim about the population those four were drawn from, and four is not many.

## The rungs

| Rung | What it reads | What it cannot see |
|---|---|---|
| 0 | the scores already in your log | every error term. Returns K |
| 1 | repeated calls to one grader | which grader you drew, and which rollouts that grader favours |
| 2 | one grader, two crossed facets | the same two rater-indexed terms |
| 3 | two or more raters, fully crossed | nothing this design can identify |

The rung is not a mode you select. It follows from what the design you pass actually contains, because a rung is a statement about what could be seen. Every rung below the top is biased upward, and the disagreement between two rungs on the same data is the cheap rung's transfer uncertainty, which `ladder_transfer` records as a calibration term.

## What changed, and why your old number was different

Until August 2026 this quantity was \(n_\text{eff} = \text{kish} \times \rho\), the Kish count of the observed spread times the reliability. **That product charged measurement noise twice.** The Kish count is computed on observed scores which already contain grader noise, and the reliability factor then discounts for the same noise: noise pushes the observed score distribution toward Gaussian, whose shape factor is \(2/\pi\), so the first factor fell for the same reason the second one did.

A two-point score distribution settles it, because its shape factor is exactly one by construction and noise is then the only thing that can move the reading. At K = 16 over 4,000 groups:

```text
sigma_err   reliability   observed shape   old n_eff    new n_eff
     0.00        1.0000           1.0000     16.0000      16.0000
     0.50        0.8000           0.8307     10.6329      12.8000
     1.00        0.5000           0.7011      5.6090       8.0000
     2.00        0.2000           0.6639      2.1246       3.2000
```

At a reliability of one half the group carries eight independent observations of sixteen, and the third row is where the old rule reported five and a half.

On real subjects the move is the same size. Eleven open reward models, fully crossed over a shared bank of 1,763 groups of four:

```text
              old rule                        new rule
rung 0   mean 2.9859, spread 0.0863     4.0000 for all eleven
rung 3   mean 1.9097, spread 0.0552     2.5582 for all eleven
```

The per-grader spread at rung 3 was entirely the shape factor. Eleven graders crossed in one design share one reliability, because that coefficient is a property of the universe a grader is drawn from rather than of any one member of it, so the corrected reading is one number for the panel. The variation did not disappear; it moved to the statistic it was always about, and the shape factors still spread across the eleven. A1's kill condition does not fire under either rule: none of the eleven has a rung-0 and rung-3 interval that overlap, and the margin is wider after the correction than before it.

If you have a stored reading from before the change, it is stamped with `observable_version` `1.0` and the corrected one with `2.0`. They are not the same quantity and must not be pooled. To recover roughly what a `1.0` reading would have said, multiply the new one by `shape_factor`, which is still on every reading; there is no way to go the other direction, because a `1.0` number does not record which of its two factors moved.

## What it still cannot do

The shape factor is itself measured on noisy scores, so it is pulled toward \(2/\pi\) from whichever side it starts. A concentrated reward distribution reads too low and a heavy-tailed one too high, and the direction is not signable without knowing which you have. Deconvolving the true-score shape from the observed one is an open research target. It is exactly why the shape factor is reported beside the effective size rather than multiplied into it: it would import an unsignable bias into a number whose job is to be an honest bound.

A fully crossed multi-rater design also gives **one** reliability for the whole panel, so every grader in one design receives the same reading. That is the coefficient the design estimates, the reliability of a score from a grader drawn at random from that universe. A genuinely per-grader effective size needs repeated calls to each grader, which is an occasion facet, and a design that has one does not need this caveat.

See also: [effective sample size of an eval set](effective-sample-size.md) for the clustering version of the same idea, [the trust ladder](../discipline/trust-ladder.md), and [`reward_lens.measure.metrology.gstudy`](../reference/measure.md).

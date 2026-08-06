# W6 rate package: what to run, in what order, and what it costs

Phase 6 is compute-gated. Nothing in this directory has been run and nothing in it has spent
anything. What ships is the instruments, their acceptance test on planted subjects, two study
specs frozen before any arm exists, a simulated power calculation at the realised arm sizes, and
this file. What does not ship is the arms.

**Buy them in this order, or buy only the first.** The ranking is by dollars per layer of the
library the result settles, computed in `price.py`:

| order | package | GPU-hours | at $2.15/GPU-h | in published campaigns | what it settles |
|---|---|---|---|---|---|
| 1 | **W6.1** H2, two-run rate test | 37.5 | **$80.62** | 8.4 | whether critical slowing down can fire at all |
| 2 | W6.2a H1 rung 1, perturb-and-hold | 75.0 | $161.25 | 16.8 | whether the free relaxation time is right |
| 3 | W6.2b H3, rate-extrapolated hysteresis | 180.0 | $387.00 | 40.3 | whether a hacked policy can be annealed back |
| | all three | 292.5 | $628.88 | 66 | |

Regenerate this table with `python -c "from studies.w6_rate.price import render_all; print(render_all())"`.
It prints at three price points, because the original costing's own three figures do not reconcile:
11,520 GPU-hours at its stated $2.15 preemptible floor is $24,768, above the top of its own
$17,000-to-$23,000 range, and that range implies $1.48 to $2.00 per GPU-hour instead. The GPU-hours
are the anchor here and the dollars are quoted at all three.

**W6.1 first is not a preference, it is the whole argument for this phase.** It costs a fifth of the
other two together, and a separation kills the early-warning layer, every lead-time claim in
`monitor/`, and the Level 0 equilibrium extrapolation that a dozen instruments assume. A collapse
licenses all of it. No other item in Phase 6 decides that much for $80.

---

## Before anything: freeze the studies

Both specs are already written. Freeze them and record the ids **before** the first arm starts, and
file the ids in the prediction register beside P7.

```
python -c "
from studies.w6_rate import freeze_all
for k, v in freeze_all().items():
    print(k, v.study_id, v.git_sha, v.frozen_at)
"
```

A `+dirty` suffix in the git sha means the tree had uncommitted changes when the freeze happened.
That is visible rather than blocked, on purpose, but for a freeze that is going to be cited, commit
first and freeze on a clean tree.

---

## W6.1 — H2, the two-run rate test

### The design

Two arms of a group-relative RL loop over **one** `lambda` interval, at anneal rates a factor of
four apart, sharing a seed. Everything else identical: the same base checkpoint, the same task
distribution, the same grader, the same group size, the same optimizer settings.

| | slow arm | fast arm |
|---|---|---|
| optimizer steps | 200 | 50 |
| `lambda` from | 0.05 | 0.05 |
| `lambda` to | 1.00 | 1.00 |
| schedule | exponential in `lambda` | exponential in `lambda` |
| `d log lambda / dt` | v | 4v |
| seeds | 3 | 3, matched to the slow arm's |

Exponential rather than linear, because `Ad` is defined on `d log lambda / dt` and an exponential
schedule makes that derivative constant. On a linear schedule the measured rate depends on which
pair of steps the window happened to contain.

`lambda` is whichever parameter is being annealed. On a run with a KL coefficient being annealed
down, that is the KL coefficient. On a run annealing a reward-shaping weight, it is that weight.
**It has to be the same parameter over the same interval on both arms**, or the arms swept different
experiments and the instrument refuses.

### What to log

Every step, both arms: the schedule parameter's value, and an order parameter. The order parameter
is whatever the transition is a transition in. A labelled hack rate if labels exist; otherwise the
gold reward on a held-out set. Log a second, unrelated channel as well: it costs nothing and it is
the noise control the early-warning statistics need.

### Run it

```
# arm 1, slow
reward-lens ... --anneal-steps 200 --lambda-from 0.05 --lambda-to 1.00 --seed 11 --out runs/w6_1/slow_s11
# arm 2, fast, same seed, same interval, quarter the steps
reward-lens ... --anneal-steps  50 --lambda-from 0.05 --lambda-to 1.00 --seed 11 --out runs/w6_1/fast_s11
# repeat for seeds 12 and 13
```

Wall-clock at the costed 45 seconds per step is 2.5 hours for a slow arm and 38 minutes for a fast
one, so one seed's pair is a little over three hours and all three seeds are about nine and a half
hours on one four-device node.

### Check after each arm

- **The schedule actually moved.** `RateArm.from_series` computes the rate from the log rather than
  taking it on trust, and an arm whose recorded `lambda` is constant refuses with "has no rate".
- **The rate ratio came out near four.** Print `arm.rate` for both. Below a ratio of two the
  instrument refuses, and that refusal is the one most likely to fire in practice, because arms
  sized to the same wall-clock budget drift together.
- **The order parameter moved.** If it did not, the arm did not cross a transition and there is
  nothing for two rates to disagree about.

### Score it

```
python -c "
from reward_lens.measure.rate.collapse import RateArm, two_run_rate_test
# load your two arms as RateArm.from_series(label, lambdas, observable, steps)
print(two_run_rate_test(fast, slow).render())
"
```

or through the frozen study, which is the form that produces REGISTERED Evidence:

```
from reward_lens.studies.runner import run_study
from studies.w6_rate import METRIC_ARCS, freeze_all
run_study(freeze_all()["w6-1-two-run-rate-test"],
          subjects={"fast": fast, "slow": slow}, metric_arcs=METRIC_ARCS)
```

### What each outcome means

- **`separated_fraction >= 0.05`**: rate-induced. P7 confirmed. Critical slowing down cannot fire on
  this schedule, and the early-warning layer is the wrong tool. Check the sign of
  `shift_in_widths`: positive means the faster arm transitions later in `lambda`, which is lag and
  is the expected signature. Negative is a real effect and is not lag, and it needs its own
  explanation before anything is concluded.
- **`separated_fraction < 0.05`**: bifurcation-induced over this range of rates. P7 refuted, the
  kill criterion `K-collapse-licenses-the-layer` fires, and that is good news reported as a fired
  kill because the registered prediction was that the layer fails.
- **A refusal.** Read the remedy. `ENVELOPE_VIOLATED` on the rate ratio means re-run one arm faster.
  `BELOW_LOD` means the bands are wider than the between-arm difference and the fix is more seeds,
  not more rates.

### A failed arm looks like

An arm whose order parameter never leaves its starting level, an arm whose `lambda` log is constant
because the schedule was configured but never applied, or two arms whose rates came out within a
factor of two because the step counts were set from a wall-clock budget rather than from this table.
All three refuse rather than returning a number, and all three are cheaper to catch by printing
`arm.rate` and `np.ptp(arm.observable)` after each arm than by discovering them at the analysis.

### Power at exactly these arm sizes

Simulated in `power.py` against a planted first-order tracker crossing a fixed critical point, at
200 and 50 steps and three seeds. Regenerate with
`python -c "from studies.w6_rate.power import power_two_run; print(power_two_run().render())"`.
The design detects a planted relaxation time of 20 steps against a null of 2 steps at power 1.00 with
a false-alarm rate of 0.00 over 25 simulations. The number that matters for sizing is where that
breaks down:

| planted `tau` | `Ad` on the fast arm | power | false-alarm rate |
|---|---|---|---|
| 5 | 0.302 | 0.60 | 0.00 |
| 8 | 0.484 | 1.00 | 0.00 |
| 12 | 0.726 | 1.00 | 0.00 |
| 20 | 1.21 | 1.00 | 0.00 |

So the 80 percent boundary sits between an adiabaticity number of **0.30 and 0.48** on the fast arm,
20 simulations per cell. **That is well below the `Ad` threshold of 1.0 at which `rate/regime.py`
licenses `QUASI_STATIC`**, so on this planted system the two-run test sees rate dependence in runs
the shipped regime reading calls quasi-static. If it carries over to a real policy, the 1.0 threshold
is permissive and that is a finding about the threshold. It is a property of a linear tracker until
someone runs the arms.

The false-alarm rate is 0.00 in every cell, so the design is not buying its power by firing on
everything. One seed reaches power 1.00 at a planted `tau` of 12 as well, which means the three seeds
are buying resolution near the boundary rather than the verdict far from it; if the budget is tight,
one seed per arm answers the coarse question for a third of the price and cannot resolve `Ad` near
0.3.

---

## W6.2a — H1 rung 1, `tau_relax` by perturb-and-hold

### The design

One arm per seed, 200 steps, with **five** perturb-and-hold episodes embedded in it. Five and not
three, and the two extra episodes are the reason this line item costs what it does: at three
episodes the linear-response check cannot reach significance at any level below 0.333, because
Kendall's tau over three pairs has only `3! = 6` orderings. At five it reaches 0.017.

At each of five chosen steps:

1. **Perturb the policy.** Add Gaussian noise to the weights, or apply a one-off reward shock, at a
   magnitude that moves the observable at least three pre-perturbation standard deviations. Below
   that the instrument refuses, because the decay being fitted is the noise's.
2. **Pin the schedule.** Freeze every annealed coefficient at its value at the moment of the
   perturbation. This is the part that is easy to skip and it is the whole protocol: a recovery
   measured while `lambda` anneals is the relaxation time and the drift added together, and nothing
   downstream separates them. Record how far `lambda` moved during the hold; above 2 percent of its
   range the instrument refuses.
3. **Hold until the observable returns**, not for a fixed step count. The budget assumes 60 steps
   and that is a guess: if the return takes longer, the fit is unstable rather than biased in a
   knowable direction, and the instrument returns a lower bound instead of a value.
4. **Release** and continue the schedule.

Use five magnitudes spanning about a factor of five, geometrically spaced. The spread is what makes
the linear-response check possible; five episodes at one magnitude test nothing.

### Check after each episode

- The observable moved at least three pre-perturbation standard deviations. Print
  `abs(episode.magnitude) / episode.pre_sd`.
- The observable came back. Print the mean of the last quarter of the hold against `pre_level`, in
  `pre_sd` units. Above 1.0 the hold was too short and the next episode should hold longer.
- `lambda` did not move. Print `episode.held_lambda_span`.

### Score it

```
python -c "
from reward_lens.measure.rate.perturb import relaxation_time_from_hold, rung_transfer
from reward_lens.measure.rate.adiabaticity import adiabaticity
out = relaxation_time_from_hold(episodes)
print(out.render())
print(rung_transfer(adiabaticity(run).tau.tau, out).note)
"
```

The rung transfer is the deliverable as much as the number is. Rung 0 is free and ships today; rung
1 is what the quantity is defined by; their disagreement measured on one run is the cost of the free
one, and nobody publishes it.

### What each outcome means

- **`tau_rung1_minus_rung0 > 0`**: the shipped free estimator is short, `Ad` as this library
  currently computes it is too small, and `QUASI_STATIC` has been licensing runs it should not.
- **`tau_rung1_minus_rung0 <= 0`**: the free estimator is not short, which is a genuine result and
  makes the free rung more trustworthy than its own docstring claims.
- **`ABOVE_LOD_BELOW_LOQ`**: every hold was truncated. The reading is a lower bound and the remedy
  is to end the hold on the observable returning rather than on a step count.

### A failed arm looks like

Episodes where the perturbation was applied but the schedule kept annealing, which shows up as
`held_lambda_span` above 0.02 and refuses. Episodes where the nudge was too gentle, which refuses
as `BELOW_LOD`. And the quiet one: five episodes at the same magnitude, which does not refuse and
produces a time constant whose defining assumption is untested. `magnitude_test_powered` on the
reading is what catches that.

### Power

The linear-response check's exact floor is `2 / n!`: 0.333 at three episodes, 0.083 at four, 0.017
at five, 0.003 at six. Confirmed against `scipy.stats.kendalltau` in the acceptance test.

The interval on the pooled time constant under-covers at small episode counts, measured on planted
single exponentials over 100 replicate protocols: a nominal 95 percent interval covers **0.84** at
five episodes, 0.90 at eight and 0.93 at twelve, at mean relative widths 0.107, 0.090 and 0.078. If
the number is going to be published rather than used as a gate, buy eight episodes.

---

## W6.2b — H3, rate-extrapolated hysteresis

### The design

Four up-and-back sweeps of one `lambda` interval, at rates spanning a factor of eight, three seeds
each. Each sweep visits its schedule points twice, so a sweep at `n` points costs `2n` optimizer
steps and the policy carries state from each point to the next.

| sweep | points up | points down | steps | rate |
|---|---|---|---|---|
| A | 40 | 40 | 80 | fastest |
| B | 80 | 80 | 160 | |
| C | 160 | 160 | 320 | |
| D | 320 | 320 | 640 | slowest |

The slowest sweep is 53 percent of this line item on its own. **Add seeds before you add rates**:
three rates leave one residual degree of freedom and a t-multiplier of 12.7, and seeds are what
supply the weights that replace it.

The down leg must continue from where the up leg ended. A protocol that re-initialises the policy
at each schedule point retraces its own path exactly, has no loop by construction, and looks
identical to a system with no hysteresis. The instrument refuses with that specific remedy when
every area comes back at zero.

### Score it

```
python -c "
from reward_lens.measure.rate.hysteresis import SweepArea, rate_extrapolated_area
sweeps = [SweepArea.from_seeds(rate, [area_seed0, area_seed1, area_seed2]) for rate, ... in ...]
print(rate_extrapolated_area(sweeps).render())
"
```

### What each outcome means

- **The interval on `area_zero` excludes zero**: genuine irreversibility. A hacked policy cannot be
  annealed back and KL-annealing is not a recovery tool.
- **The interval contains zero**: every loop measured at any finite rate is consistent with lag.
  `K-loop-is-lag` fires and it is a publishable negative. This is the outcome a single-rate protocol
  cannot distinguish from the first one, which is the whole reason for the extra sweeps: on the
  acceptance test's single-well control, which has no second stable state anywhere, the raw loop
  area at the fastest rate is **0.0514** and the extrapolated area is 0.00015 with an interval
  containing zero.
- **A large Birge ratio in the rendering** means the seed replicates do not explain the scatter
  about the rate model, and the interval has been widened to match. That is informative rather than
  a failure: it says `A0 + c v^alpha` is missing something, and the honest response is to add a
  fifth rate so the exponent can be fitted rather than fixed.

### Power

Simulated over 400 replicates at four rates and three seeds with a seed spread of 0.01: the smallest
extrapolated area detected at 80 percent power is **0.030** (power 0.858 at a false-alarm rate of
0.010; power 0.545 at 0.020). At one seed per rate it is **0.050** (power 0.843, false alarm 0.030).
The third seed buys a factor of 1.7 in the smallest detectable area and a third of the false-alarm
rate. Regenerate with
`python -c "from studies.w6_rate.power import power_hysteresis; print(power_hysteresis().render())"`.

---

## After any of them

Record the result in the prediction register against P7 for W6.1, and beside its own spec for
W6.2. A fired kill
criterion is a result and is recorded as prominently as a confirmation. A study whose metric could
not be computed is `VOID` and is a work item, never a pass.

If none of them is bought, that is a decision and not a gap. The instruments, the frozen specs, the
acceptance test on planted subjects, this runbook and the price are the deliverable of a
compute-gated phase, and the thing they are for is making the decision cheap.

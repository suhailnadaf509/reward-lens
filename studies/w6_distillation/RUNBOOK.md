# W6.3 / K1 runbook: measuring the distillation gap

Study `study:w6-3-distillation-gap@v1#5d34c1df`, frozen before the subject exists. The id depends only on the
spec, so if you edit a prediction it changes and the change is visible in every row the
study stamps.

## The decision this runbook is for

Nothing in `studies/w6_distillation/` has ever been run against a real pair of
checkpoints. The code, the frozen predictions, the acceptance test and this runbook exist
so that buying the compute is a decision with a bill in front of it. The bill:

```
K1 price: 1,323.1 GPU-hours, $1,985 to $2,659 at the H100 neocloud band ($1.50 to $2.01 per GPU-hour), $5,226 on Modal at $3.95.
    1,316.6 GPU-hours of it is producing the checkpoints and 6.5 is everything the instrument does.
    For scale: the entire published campaign metered $17.73 over 4.465 GPU-hours, so this is about 150 of them.
    A fourfold error in both throughput assumptions would still leave the inference arms at 1.9% of the total.

    A1.expert         1,152.00 GPU-h  $    2,316  3 group-relative RL runs on an 8B policy, one per seed
    A2.student          164.57 GPU-h  $      331  3 on-policy distillations of each expert back into the same base
    A0.rollouts           5.69 GPU-h  $       11  10 rollout draws of 1,000 prompts x 8 completions x 512 tokens (base, 3 blanks, and expert plus student per seed)
    A4.localisation       0.85 GPU-h  $        2  per-token gradients on the expert and the student over the audit rollouts
```

The multi-seed reference study the build spec prices at ten seeds by three conditions is
11,520 GPU-hours, so K1 at 3 seeds is 11% of that.

**What you cannot buy at any price.**

K1 needs a pre-RL base, an expert trained from it by group-relative RL, and a student distilled from that expert back into the same base. No lab releases the middle one: the consolidated model ships and the domain experts do not. The distilled checkpoints that are public are distilled across families, so they have no shared base to measure an installed shift against, and running K1 on them would measure cross-family transfer under K1's name.

## What has to be true before arm A1

1. **One base checkpoint**, and every later artifact descends from it. If the expert and
   the student have different bases there is no shared feature scale and no installed
   shift to divide by, and the reading answers a different question under K1's name.
2. **One prompt file**, held out of every training set used in A1 and A2. Every arm answers
   the same prompts, keyed by prompt id and not by position.
3. **One decoding configuration** for every draw: temperature, top_p, max_new_tokens, the
   stop set and the completions per prompt. The instrument refuses when the declared
   settings differ between arms, and it can only see what you declare, so declare them.
4. **8 completions per prompt; two is the hard minimum.** The
   survival slope is corrected for sampling error in its own regressor and the correction
   needs a within-prompt variance. With one completion there is none, the correction
   cannot run, and the uncorrected slope depends on how many completions you drew.

## The arms, in order, and what to check after each

### A0 - base rollouts, and the blanks (inference only, minutes)

Draw the base checkpoint's completions on the prompt file, then draw them again at two
more sampling seeds. That gives the base arm and 3 blank arms. Run this
**first**, before any training, because it is the only arm that can be checked without
anything to compare against and because a problem here invalidates everything after it.

```python
from reward_lens.policy.hf import HFPolicy  # or your own sampler
from reward_lens.policy.base import SampleSpec

policy = HFPolicy.load(BASE_CHECKPOINT)
spec = SampleSpec(max_new_tokens=512, temperature=1.0, top_p=1.0, group_size=8, seed=0)
rollouts = policy.sample(prompts, spec)  # then seed=1 and seed=2 for the blanks
```

**Check:** every prompt came back with the full group and non-empty assistant text. Then
run the detection floor on the blanks alone and read `blank_mean`. It should sit near
zero. A blank mean far from zero means the two draws differ systematically, which means
the decoding settings moved between them, and every later number would inherit it.

**A failed A0 looks like:** a `RECORD_INCOMPLETE` refusal naming how many prompts had no
readable rollout, or a blank mean of the same order as the shift you are trying to
measure. Fix the sampler before spending anything on A1.

### A1 - the RL expert (the expensive arm)

3 group-relative RL runs from the base against a real grader, one per seed,
differing in nothing but the seed. This is the arm this library exists to instrument, so
tap it: `reward_lens.tap.adapters.trl` records the run while it happens and F1, F2 and the
threshold instruments then have something to read besides K1.

```bash
accelerate launch train_grpo.py --seed 0 --output_dir experts/seed0   # then 1, 2
```

**Check after each run, before starting the next:** the reward trace rose and the entropy
trace did not collapse. That is void condition 1, `ARM_COLLAPSE`, and reading a study off
a collapsed arm is the failure `studies.void` exists to prevent. Also draw the expert's
rollouts now and confirm the installed shift clears the detection limit from A0: if RL
installed nothing this basis can see, K1 has no denominator and there is no point paying
for A2.

**A failed A1 looks like:** a `BELOW_LOD` refusal saying every feature's installed shift is
under the limit. That is not a failed measurement, it is the finding that this run left no
behavioural trace the basis can see. Widen the basis with `RecordedFeatures` over your own
features, or train longer, and re-check before A2.

### A2 - the distilled student

On-policy distillation from each expert back into **the same base**, one student per seed.
Per-token log-ratio teacher reward is the cheap version every lab but DeepSeek ships; if
you use full-vocabulary logit distillation instead, record which, because that is the one
named technical disagreement between labs on this step and nobody publishes the ablation.

**Check:** the student's aggregate score on the grader is within noise of the expert's.
That is what the labs report and it is also H4's premise. If the student is visibly worse
on the grader, the distillation run underconverged and K1 would be measuring that instead.

**A failed A2 looks like:** a student whose score is well below its teacher's, or a
survival fraction and a reliability that both sit at the very top of their range, which
usually means the student was initialised from the expert rather than from the base and
there was never a gap to measure.

### A3 - the audit draw

Draw expert and student rollouts on the same prompt file, same decoding, same group size.
Nothing else changes between them.

### A4 - localisation (optional, and it is nearly free)

Per-token gradients on the expert and the student over the audit rollouts, at
`POLICY: BACKWARD`. Compose `reward_lens.policy.credit` rather than writing a second
gradient disintegration; its conservation identity closes at 2.704e-16 and a second one
would not. This is the arm that answers the published audit's actual question, *where* the
teacher signal acts, and at under one GPU-hour it costs less than the electricity of
deciding whether to run it.

## Running the study

```python
from reward_lens.core.store import EvidenceStore
from reward_lens.studies import run_study, render_report
from studies.w6_distillation.analysis import build_spec
from studies.w6_distillation.survival import Arm

arms = {
    "base": Arm("base", rollouts_by_prompt(BASE_ROLLOUTS)),
    "expert": Arm("expert", rollouts_by_prompt(EXPERT_ROLLOUTS)),
    "student": Arm("student", rollouts_by_prompt(STUDENT_ROLLOUTS)),
    "blank0": Arm("blank0", rollouts_by_prompt(BASE_ROLLOUTS_SEED1)),
    "blank1": Arm("blank1", rollouts_by_prompt(BASE_ROLLOUTS_SEED2)),
    "hack_features": ("...",),  # the features you call reward-hacking-relevant
    "markers": ("...",),  # literals a hack is known to contain, for the string match
    "sampling": {"base": DECODE, "expert": DECODE, "student": DECODE},
}
frozen, result = run_study(build_spec(), subjects=arms, store=EvidenceStore(PATH))
print(render_report(frozen, result, EvidenceStore(PATH)))
```

`hack_features` is the one judgement call in the whole procedure and it is registered by
being written into the study's subjects before the run. Name them from the grader's own
structure, not from what the reading turns out to say.

## Reading the result

- **H1-something-survives**: a measurable fraction of the behaviour group-relative RL installed is still present in the distilled student. Registered as `survival_pp > 10`.
- **H2-hack-survives-better**: reward-hacking-relevant behaviour survives the distillation step at a higher rate than capability behaviour does. Registered as `hack_capability_contrast_pp > 0`.
- **H3-entry-survives-worse**: survival is lower at the entry of an assistant turn than in its body, which is the localisation arXiv:2607.07050 reports for a real distillation step. Registered as `entry_minus_body_pp < 0`.
- **H4-invisible-in-aggregate**: the expert and the student score within the detection floor of each other on the grader, so the behavioural shift is invisible from aggregate scores alone. Registered as `aggregate_score_z < 3.3`.
- **H5-beats-the-dumb-bank**: the projection onto the RL-installed feature direction adds something the six dumb baselines do not already have at telling an expert rollout from a student rollout. Registered as `increment_ci_low > 0`.

and three kill criteria:

- **K1-DEAD** fires on `survival_ci_high_pp < 10`: the upper bound on survival is below a tenth, so essentially nothing RL installed reaches the shipped model. Everything this library measures during a run is then measured on a model that never ships, and the in-run layer is for a lab's internal use rather than for the artifact anyone deploys.
- **K1-DUMB** fires on `increment_ci_high <= 0`: the six dumb baselines separate expert rollouts from student rollouts as well as the feature projection does, so the apparatus is decoration and the honest publication is the baseline.
- **K1-NO-DENOMINATOR** fires on `n_features_above_loq < 2`: fewer than two features have an RL-installed shift above the limit of quantitation, so there is almost nothing to divide by and any survival fraction is a ratio of noise. Not a finding about distillation: a finding that this expert and this feature basis cannot support the question.

## Power, at the n this runbook buys

Measured on the planted subject, which fixes the noise level at the plant's and not at a
real subject's. Read it as the scaling rule rather than as a promise, and recompute it
from your own A0 draw before committing to a prompt count.

- On 60 prompts at 4 completions each, the minimum detectable survival
  contrast is 22.0 pp at 80% power and alpha 0.05.
- Resolving a 10 pp contrast therefore needs about 291 prompts at that
  noise level. The priced design is 1,000 prompts at 8 completions, which is 3.4x that in
  prompts and halves the per-prompt sampling variance again, so the margin is real.
- The expert-versus-student detection arm reaches power 0.95 at the realised
  240 rollouts, with N* = 153 for 80%.
- The minimum detectable effect scales as one over the square root of the prompt count, so
  a pilot of 50 prompts after A1 tells you what the full draw will resolve.

## What this cannot tell you, however much you spend

K1 reads rollouts. A distillation step that preserves every measured behaviour by a
different mechanism reads here as full survival, and only A4 could tell those apart. The
survival fraction is conditional on the feature basis, so a basis that misses the axis
distillation moved reports high survival: the fitted feature names are on every reading
for exactly that reason. And the blank arm bounds sampling noise at fixed weights, not the
seed-to-seed variability of the RL run, which is why the price buys three seeds and not
one.

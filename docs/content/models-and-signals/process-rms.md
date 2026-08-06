<div class="rl-chips">
  <span class="rl-chip rl-chip--works"><span class="rl-chip__k">grader</span> process RM</span>
  <span class="rl-chip rl-chip--gauge"><span class="rl-chip__k">score gauge</span> invariant</span>
  <span class="rl-chip rl-chip--works"><span class="rl-chip__k">bring</span> a step-level classifier</span>
</div>

# Process reward models

**Which step did the reasoning go wrong at?**

A process reward model scores each reasoning step, not just the final answer, so this is a question you can actually put to it. Architecturally it is a sequence classifier read at every step boundary rather than only at the last token. That is the whole reason positions are first-class: the process RM reuses the classifier's reward direction \(w_r\) verbatim and changes only *where* it reads, from the final token to each step end.

## Load or wrap one

`ProcessRM.from_sequence_classifier(model, tokenizer, delimiter="\n")` wraps a reward head and reads it at step boundaries. The tiny CPU vehicle:

```python
from reward_lens.signals import ProcessRM

prm = ProcessRM.from_tiny(seed=0)
prm.caps    # SCORES|PREFIX_SCORES|ACTIVATIONS|GRADIENTS|HVP|LINEAR_READOUT|STEP_SCORES
```

The distinctive capability is `STEP_SCORES`. Everything the classifier exposes it exposes too, so `score` still returns the outcome scalar (the reward of the whole solution) and the linear-readout battery still attaches.

## Per-step scores

Give it a solution split into steps and ask for `step_scores`. The default delimiter is a newline:

```python
solution = "First, 2+2=4.\nThen, 4*3=12.\nFinally, 12-5=7."
ev = prm.step_scores([("Compute (2+2)*3-5.", solution)])
ev.value.step_counts    # [3]
ev.value.curves[0]      # array([-0.1091, -0.0984, -0.0929], dtype=float32)

outcome = prm.score([("Compute (2+2)*3-5.", solution)]).value.values[0]
outcome                 # -0.0929175
```

Three steps, three scores, one per boundary. The last step score equals the outcome score by construction, because the final step ends at the final token, and you can see it: the curve ends at `-0.0929` and the outcome is `-0.0929175`. That identity is not decoration. It is what lets you read the per-step curve as a decomposition of the whole-solution reward rather than three unrelated numbers. The scores themselves are meaningless on a random model; the segmentation and the boundary reads are exact.

Per-step scores are what the [verification index](../instruments/index-library.md) consumes to ask whether a grader rewards correct steps or merely correct-looking ones.

## The delimiter works; the learned detector is a stub

Step detection has two paths, and only one of them is real here. The delimiter path is implemented and exact: split on the configured marker, map each segment to token coordinates, read at each boundary. The learned fallback, a trained boundary classifier for solutions with no reliable delimiter, is a stub. reward-lens does not fake it. When a solution does not split into at least two steps, it treats the whole thing as one step and says so:

```python
one = prm.tokenize(("Compute.", "one line solution with no newline at all"))
one.meta["n_steps"]           # 1
one.meta["step_detection"]
# 'learned step-boundary detector is a stub: a trained boundary classifier over
#  the residual stream is the production fallback; here a solution with no delimiter is one step.'
```

You are never handed a wrong segmentation dressed up as a right one. If your solutions carry a delimiter, the process RM is fully working. If they do not, you get a single-step reading and a note telling you exactly why, so you can add a delimiter or wait for the learned detector rather than trust a silent guess.

## Honest caveats

The per-step scores inherit the classifier's gauge and trust: `INVARIANT`, and `EXPLORATORY` until calibrated. A process RM does not tell you a step is *wrong*, only that the reward at that boundary is lower; whether low reward tracks real error is exactly what an [organism](../discipline/calibration-and-organisms.md) with labeled error steps is for. The [score with a process reward model](../how-to/process-reward-model.md) guide runs the real-model path.

## Reference

[`ProcessRM`](../reference/signals.md#reward_lens.signals.process.ProcessRM), and its `StepScores` result type.

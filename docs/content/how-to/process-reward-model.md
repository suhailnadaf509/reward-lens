# Score with a process reward model

**You want a reward for each reasoning step, not one scalar for the whole answer.**

A process reward model is a sequence classifier read at every step boundary instead of only at the last token. `ProcessRM` keeps the head direction \( w_r \) verbatim and changes the position from `final` to `step_ends`, declaring the `STEP_SCORES` capability. Everything else, the readout and the projection, is the classifier path.

```python
from reward_lens.signals import ProcessRM

signal = ProcessRM.from_tiny(seed=0)
print(signal.caps)
# Capability.SCORES|PREFIX_SCORES|ACTIVATIONS|GRADIENTS|HVP|LINEAR_READOUT|STEP_SCORES

solution = ("First, list the primes under 6: 2, 3, 5.\n"
            "Then add them: 2+3+5 = 10.\n"
            "So the answer is 10.")
ev = signal.step_scores([("Sum the primes under six.", solution)])
print(ev.value.step_counts)                              # [3]
print([round(float(x), 3) for x in ev.value.curves[0]])  # [-0.106, -0.118, -0.12]
```

Three lines, three steps, three scores, one read at each step-end token. The last step's score equals the whole-solution outcome score, because the final step ends at the final token.

## The delimiter works; the learned detector is a stub

Step boundaries come from a delimiter, the common case being a model trained with `\n` between steps. The alternative, a trained boundary classifier over the residual stream, is a stub here. When the split yields fewer than two steps the whole response is treated as one step, and the fallback is recorded in the tokenization metadata so you are never handed a wrong segmentation silently:

```python
tok = signal.tokenize(("Sum the primes under six.", "The answer is 10."))
print(tok.meta["n_steps"])          # 1
print(tok.meta["step_detection"])
# learned step-boundary detector is a stub: a trained boundary classifier over
# the residual stream is the production fallback; here a solution with no delimiter is one step.
```

A delimited solution reports `step_detection` as `None`: nothing fell back. A one-liner reports the note above and collapses to a single step. The segmentation is honest about which path it took, every time.

For a real step-trained checkpoint, `ProcessRM.from_sequence_classifier(model, tokenizer, delimiter="\n")` reads the scalar head into a `step_ends` readout; the mechanics are identical to the tiny path.

See also: [Process reward models](../models-and-signals/process-rms.md). API: [`ProcessRM`](../reference/signals.md#reward_lens.signals.process.ProcessRM).

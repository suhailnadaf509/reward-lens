# Framework integrations

**How does reward-lens ride inside a TRL, OpenRLHF, or veRL run instead of analyzing it afterward?**

Through two framework-agnostic pieces that run today and one thin binding per framework that needs the framework. The reward function the loop calls to get its scores, and the logger that records the reward model's geometry every few steps, are the same shape no matter which trainer drives the loop. Only the wiring that registers them into a live trainer is framework-specific, and that wiring names its missing extra and refuses rather than pretending to run.

## The reward path is one call

Every one of these frameworks ultimately calls a function that turns a batch of prompts and responses into a list of scalar rewards. [`make_reward_fn`](../reference/dynamics-loops.md#reward_lens.loops.integrations.base.make_reward_fn) wraps any signal into exactly that shape, and it needs only the signal, no framework:

```python
from reward_lens.signals import from_tiny
from reward_lens.loops.integrations import make_reward_fn

signal = from_tiny(seed=0)                 # a real tiny reward model on CPU
reward_fn = make_reward_fn(signal)         # (prompts, responses) -> list[float]

scores = reward_fn(
    ["what is 2+2?", "why is the sky blue?"],
    ["4", "rayleigh scattering"],
)
print([round(x, 4) for x in scores])
# [0.0268, 0.1003]
print(type(scores).__name__, type(scores[0]).__name__)
# list float
```

Plain floats, the contract a PPO or GRPO reward hook expects. This is the real reward path. The three framework files add only the plumbing around it, which is what keeps them thin and keeps the reward identical across them.

## The logging shape is one object

The other half is the geometry log. A [`GeometryLogger`](../reference/dynamics-loops.md#reward_lens.loops.integrations.base.GeometryLogger) holds a fixed probe and, on the logging steps, records the model's scores on that probe, the reward-direction norm, and the concept doses against a feature bank. It runs on the tiny model with no framework installed, as the [training-loops overview](index.md) shows end to end. Because both the reward function and the logger are proven this way, the science of a run is independent of which trainer produced it.

## The three bindings

Each framework gets a reward entrypoint and a geometry entrypoint. The reward entrypoints and the logger are real now; the geometry entrypoints bind the logger into the live trainer and so need the framework.

| Framework | Reward function | Live geometry binding | Extra |
| --- | --- | --- | --- |
| TRL | `trl_reward_fn` | `trl_geometry_callback` | `reward-lens[trl]` |
| OpenRLHF | `openrlhf_reward_fn` | `openrlhf_geometry_hook` | `reward-lens[openrlhf]` |
| veRL | `verl_reward_score` | `verl_geometry_worker` | `reward-lens[verl]` |

The reward function is usable with no framework present. The geometry binding is not, and it says so:

```python
from reward_lens.loops.integrations import (
    trl_reward_fn,
    trl_geometry_callback,
    GeometryProbe,
    IntegrationUnavailableError,
)

# the reward function is real, with no TRL installed:
trl_fn = trl_reward_fn(signal)
print([round(x, 4) for x in trl_fn(
    ["what is 2+2?", "why is the sky blue?"],
    ["4", "rayleigh scattering"],
)])
# [0.0268, 0.1003]

# binding the logger into a live PPOTrainer / GRPOTrainer needs the framework:
probe = GeometryProbe(prompts=["what is 2+2?"], responses=["4"], k=50)
try:
    trl_geometry_callback(signal, probe)
except IntegrationUnavailableError as exc:
    print(exc)
# the 'trl' integration requires the optional 'trl' extra, which is not installed.
# Install reward-lens[trl] to wire the callback/worker into a live training run. The
# framework-agnostic reward function (make_reward_fn) and the geometry probe
# (probe_geometry) do not need 'trl' and are usable now.
```

The error is the honest half of the design. The binding does not fake a `TrainerCallback` and log nothing. It names the framework, names the extra that would make it work, and reminds you that the reward function and the geometry probe already do. The same holds for the OpenRLHF worker hook and the veRL worker: their reward-scoring half runs now, their live binding raises the same `IntegrationUnavailableError` until the extra is installed.

!!! warning "Needs the framework installed"
    `trl_geometry_callback`, `openrlhf_geometry_hook`, and `verl_geometry_worker` bind the logger into a live trainer, which requires that trainer. None of TRL, OpenRLHF, or veRL is installed here, so each names its extra (`reward-lens[trl]`, `reward-lens[openrlhf]`, `reward-lens[verl]`) and refuses. The structure below each refusal is what runs once the extra is present; the reward function and the `GeometryLogger` need none of it.

To wire this into a real run, see the how-to on [training-loop hooks](../how-to/training-loop-hooks.md). For the shape of a signal you would wrap, see [Models and signals](../models-and-signals/index.md).

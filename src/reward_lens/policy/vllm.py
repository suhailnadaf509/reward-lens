"""The vLLM policy: sampling and log-probabilities, and nothing that needs a graph.

A serving engine is a Plane A object. It is where the tokens come from, it is fast, and it is
closed: what you can get out of it is what its API returns, and no amount of configuration turns it
into a Plane B backend. So `ServingPolicy` implements the part of the policy contract an engine can
actually satisfy and **does not implement the rest**. It has no `capture`, no `grad_h` and no
`token_gradients`, its `caps` declares neither `ACTIVATIONS` nor `GRADIENTS`, and an instrument
that needs either gets a `Refusal` with a remedy from the capability gate rather than a slow path
or, worse, a fast path that silently returns numbers taken off a detached tensor.

That is deliberate, and it is the design point: the boundary between the two planes should be
impossible to cross rather than expensive to cross. A backend that offered a degraded
`capture` would be a backend where the degradation is invisible in the reading.

**The five things that are structurally unavailable in a paged-attention engine today.** These are
not performance notes and they are not a roadmap; each is a consequence of how the engine is built,
so a caller who plans around them will not be rescued by a later version.

1. **Gradients of any kind.** Inference runs under `torch.inference_mode()`. A tensor produced
   inside it carries no version counter and cannot be recorded by autograd at all, so there is no
   flag, hook or context manager that makes a backward pass work in-engine. Gradients live outside
   the engine, on an eager forward, which is what `policy.hf.HFPolicyRuntime` is.
2. **Attention patterns.** Paged attention computes attention over a block table inside a fused
   kernel and never materialises the `(heads, q, k)` matrix. There is nothing to read, rather than
   something expensive to read.
3. **Separable Q, K and V.** The projections are merged into one `qkv_proj` for throughput, so the
   three are one tensor with three slices whose boundaries the engine knows and the module tree
   does not expose. Anything defined on `q` alone is defined on a slice of something else.
4. **Arbitrary Python interventions together with CUDA graphs.** A captured graph has no room for a
   Python callback. Running eager instead is what makes an intervention possible, and eager costs
   between 2.7 and 6.3 times the captured-graph throughput, which is why the honest framing is that
   interventions and serving throughput are alternatives rather than a tuning knob.
5. **Full-depth whole-run capture at frontier scale.** Every layer at every position for a real run
   is not a storage problem that a better format fixes; it is more bytes than the run produces.

The consequence for an instrument author is short: if your instrument declares `ACTIVATIONS` or
`GRADIENTS`, it runs against `policy.hf.HFPolicy` on a checkpoint and never against a serving
engine, and the capability report says so before any money is spent. If it declares only `SCORES`
or `GENERATIVE`, it runs against either.

**This module does not import vLLM.** It defines the contract and the refusal; the engine call is
one method with a documented signature, and `vllm` is not a declared dependency of this package.
Wiring an actual client would take a new dependency, and that is deliberately not done here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.types import Capability
from reward_lens.policy.base import PolicyMeta, SampleSpec

#: What a serving engine offers. `PREFIX_SCORES` is present because an engine returns per-token
#: log-probabilities for a scored sequence; `ACTIVATIONS`, `GRADIENTS`, `HVP` and `LINEAR_READOUT`
#: are absent because each of them needs a graph or a hidden state the engine does not expose.
SERVING_CAPS = Capability.SCORES | Capability.PREFIX_SCORES | Capability.GENERATIVE

#: The engine facts an instrument may need to reason about, keyed for the capability report.
ENGINE_LIMITS: dict[str, str] = {
    "gradients": (
        "inference runs under torch.inference_mode(), which strips the version counter a tensor "
        "needs to enter an autograd graph. No configuration re-enables it in-engine."
    ),
    "attention_patterns": (
        "paged attention never materialises the (heads, q, k) matrix; it is computed inside a fused "
        "kernel over a block table."
    ),
    "separable_qkv": (
        "q, k and v share one merged projection, so a quantity defined on q alone is defined on a "
        "slice of a tensor the module tree does not expose separately."
    ),
    "python_interventions_with_cuda_graphs": (
        "a captured CUDA graph has no room for a Python callback; running eager to allow one costs "
        "2.7x to 6.3x throughput, so interventions and serving throughput are alternatives."
    ),
    "full_depth_whole_run_capture": (
        "every layer at every position for a frontier-scale run is more bytes than the run "
        "produces; this is not a storage format problem."
    ),
}


class EngineBoundary(RuntimeError, AttributeError):
    """An operation that a serving engine cannot perform, with why and where it can be performed.

    Raised rather than returned for the same reason `ArchitectureError` is: a caller asking a
    serving policy for a gradient has made a category error at wiring time, not encountered an
    anticipated condition at measurement time. An *instrument* that needs one is refused rather than
    raised at, and that refusal comes from the capability gate reading `caps`.

    Subclasses `AttributeError` as well as `RuntimeError`, the same trade `ExtraRequiredError`
    makes with `ImportError`: it is raised out of `__getattr__`, and the language treats an
    attribute that is absent as one whose lookup raises `AttributeError` specifically. Under a
    plain `RuntimeError` every `hasattr` on a serving policy propagated instead of answering
    False, which broke three things at once. `isinstance(serving, PolicySubject)` is documented on
    `ServingPolicy` to be False and was decided by `typing._get_protocol_attrs`, a **set**, so on
    Python 3.10 and 3.11 the protocol's members are walked in hash order and the answer depended
    on whether `capture` came up before some other absent name. That is per-process randomised, so
    the same commit passed and failed on the same interpreter. `policy/selection.py` and
    `geometry/hessian.py` both ask `hasattr(subject, ...)` for exactly these names and got an
    exception where they expected a boolean. Python 3.12 is immune to the first of those, having
    moved protocol checks to `inspect.getattr_static`, which is why only the 3.10 leg went red.

    Nothing about the message changes: a caller who reaches for `serving.grad_h` still gets the
    sentence naming the boundary and the backend that can take the reading.
    """


def refuse_plane(instrument: str, needed: Capability) -> Refusal:
    """The refusal an instrument gets when it asks a serving engine for Plane B access.

    A separate constructor because the remedy is specific and is not "get more access": the access
    exists, on a different backend, and the sentence a user needs is which one and what it costs.
    """
    names = ", ".join(
        sorted(c.name or "" for c in Capability if c is not Capability.NONE and c & needed)
    )
    return Refusal(
        instrument=instrument,
        reason=RefusalReason.ACCESS_INSUFFICIENT,
        detail=(
            f"a serving engine offers {SERVING_CAPS!r} and this instrument needs {names}. "
            f"{ENGINE_LIMITS['gradients']}"
        ),
        remedy=(
            "load the same checkpoint through `reward_lens.policy.hf.from_pretrained` and point "
            "this instrument at that. The engine stays where it is and keeps producing the run; "
            "the white-box reading is taken beside it on an eager forward, which is the Plane A "
            "versus Plane B split and not a workaround for it."
        ),
        statistics={"needed": names, "engine_offers": SERVING_CAPS.name or str(SERVING_CAPS.value)},
    )


@dataclass
class ServingPolicy:
    """A policy behind a serving engine: draws completions and scores them, and stops there.

    Deliberately **not** a `PolicySubject`. `isinstance(serving, PolicySubject)` is False, because
    the protocol declares `capture`, `grad_h` and `token_gradients` and this class does not have
    them. An instrument handed one is refused at the capability gate with the remedy above; an
    instrument that only needs scores runs against it unchanged, which is the whole point of
    keeping it in the same package.

    ``call`` is the seam. It takes ``(prompts, spec)`` and returns the engine's response; a vLLM
    client, an SGLang client and a hosted endpoint all fit it. Nothing here imports an engine.
    """

    meta: PolicyMeta
    call: Callable[[Sequence[str], SampleSpec], Any]
    engine: str = "vllm"
    caps: Capability = SERVING_CAPS
    limits: dict[str, str] = field(default_factory=lambda: dict(ENGINE_LIMITS))

    def sample(self, prompts: Sequence[str], spec: SampleSpec) -> Any:
        """Draw completions through the engine. The one thing the engine is for."""
        return self.call(list(prompts), spec)

    def capabilities_note(self) -> str:
        """One line per structural limit, for the capability report."""
        return "\n".join(f"{name}: {why}" for name, why in sorted(self.limits.items()))

    def __getattr__(self, name: str) -> Any:
        """Fail with the boundary, by name, for anything Plane B.

        A bare `AttributeError` would be correct and useless: the caller would learn that a method
        is missing and not that it is missing on purpose and where the same measurement can be
        taken. `EngineBoundary` *is* an `AttributeError`, so the message is carried without
        breaking the one thing the language asks of `__getattr__`, which is that a name it will not
        supply raises `AttributeError` and is therefore invisible to `hasattr`. This intercepts
        only the five names that matter and lets everything else raise normally.
        """
        if name in ("capture", "grad_h", "token_gradients", "hvp", "with_interventions"):
            raise EngineBoundary(
                f"`{name}` is not available behind a serving engine. "
                f"{ENGINE_LIMITS['gradients'] if name in ('grad_h', 'token_gradients', 'hvp') else ENGINE_LIMITS['full_depth_whole_run_capture']} "
                f"Load the same checkpoint through `reward_lens.policy.hf.from_pretrained` and "
                f"take the reading there."
            )
        raise AttributeError(name)


__all__ = [
    "ENGINE_LIMITS",
    "SERVING_CAPS",
    "EngineBoundary",
    "ServingPolicy",
    "refuse_plane",
]

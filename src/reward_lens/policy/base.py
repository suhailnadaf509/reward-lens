"""The policy as an openable object, symmetric with `signals/`.

One structural claim, and this package is what it costs to believe it: **the grader and the policy
are the same kind of object.** Both are usually networks with weights, activations and gradients;
sometimes the grader is a program instead; sometimes the grader *is* the policy. An instrument that
reads internals should not care which side of the loop it points at. So `policy/` sits beside
`signals/` as a peer rather than under it, and a `PolicySubject` carries the same `Runtime` a
`RewardSignal` carries, exposes readouts the same way, and captures activations through the same
mount. What the package has to deliver is that the shipped lens and attribution instruments run
against a policy with no change beyond the argument, and the way that is met is by not writing a
second implementation of anything.

Four methods are the policy's own and have no grader analogue, which is why this is a peer and not
an alias:

``sample`` draws completions, because the policy is the only node in the loop that produces them.
``score_under`` returns the log-probability a policy assigns to text somebody else produced, which
is what every off-policy correction, every KL term and every staleness check is built out of.
``grad_h`` differentiates a readout with respect to an activation site. ``token_gradients``
differentiates it with respect to the input embeddings, one row per token, which is the per-token
attribution the credit measure consumes.

**Where gradients work, and where they cannot.** Everything here runs outside a serving engine, and
that is not a limitation to be engineered away later, it is the line between Plane A and Plane B.
Inference engines run under `torch.inference_mode()`, which is a documented hard block on autograd
rather than a configuration: a tensor produced inside it carries no version counter and cannot
enter a graph, so there is no flag that makes a backward pass work in-engine. Four more
things are structurally unavailable in a paged-attention engine today and are listed in
`reward_lens.policy.vllm` rather than here, because that module is where a caller who wants them
will look. The design consequence is that `ServingPolicy` does not implement `PolicySubject`: it
has no `capture` and no `grad_h`, so an instrument that needs them refuses at the capability gate
with a remedy instead of running slowly or, worse, running on a silently detached tensor.

**One provenance note that costs nothing and is easy to lose.** `nnsight` patches
`torch.Tensor.backward` at import and preserves `__module__` and `__qualname__` on the replacement,
so no name-based check notices, and every `.backward()` in the process routes through it
thereafter. This package does not depend on `nnsight`, and `runtime_provenance` records whether it
is nonetheless present in the interpreter, because a gradient measured in a patched process is a
measurement whose apparatus changed without anything in the reading saying so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, Sequence, runtime_checkable

from reward_lens.core.evidence import register_payload
from reward_lens.core.types import Capability, ModelFP, Site
from reward_lens.runtime.backend import SiteMap

# The subject-neutral vocabulary. `Readout`, `PositionSpec` and `TokenizedInput` describe *where and
# how to read a network*, which is not a grader-side idea and is not duplicated here. They live in
# `signals/base.py` because that is the package that needed them first; the import direction is the
# one thing in this file that contradicts the peer claim, and moving them to a neutral module is a
# change this package does not make.
from reward_lens.signals.base import (  # noqa: F401  (re-exported as this package's vocabulary)
    PositionSpec,
    Readout,
    ReadoutKind,
    Scores,
    TokenCurves,
    TokenizedInput,
)

if TYPE_CHECKING:
    import numpy as np
    import torch

    from reward_lens.core.evidence import Evidence
    from reward_lens.interventions.base import Intervention
    from reward_lens.runtime.backend import CaptureHandle, CaptureSpec, Runtime


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


@dataclass
class PolicyMeta:
    """Policy identity, lineage and numerics.

    Field-for-field compatible with `signals.base.SignalMeta` where the two overlap, deliberately:
    every instrument that reads `subject.meta.n_layers` or `subject.meta.fingerprint` reads it off
    either object without knowing which it has, which is the peer claim expressed as a shape rather
    than as a comment.

    ``sampling`` records the decoding policy the completions in a record were drawn under, because
    a log-probability recomputed under different decoding settings is a different number and the
    difference is the whole of `policy.train_infer_logprob_mismatch`.
    """

    fingerprint: ModelFP
    adapter: str = "ArchitectureView"
    architecture: str = ""
    lineage: dict[str, Any] = field(default_factory=dict)
    template: dict[str, Any] = field(default_factory=dict)
    numerics_policy: str = "default"
    soft_cap: float | None = None
    d_model: int | None = None
    n_layers: int | None = None
    n_heads: int | None = None
    vocab_size: int | None = None
    sampling: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SampleSpec:
    """What to draw, and under what decoding policy.

    ``group_size`` is K, the number of completions per prompt, because group-relative estimation is
    the setting this library exists for and drawing K rollouts for one prompt is one call rather
    than K. ``seed`` is required to be explicit: a sample nobody can redraw is not a measurement,
    and defaulting it to None would let that happen without anybody choosing it.
    """

    max_new_tokens: int = 32
    temperature: float = 1.0
    top_p: float = 1.0
    group_size: int = 1
    seed: int = 0
    stop: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "group_size": self.group_size,
            "seed": self.seed,
            "stop": list(self.stop),
        }


@register_payload
@dataclass
class Rollouts:
    """Completions drawn from a policy, grouped by prompt (the payload of ``sample``).

    ``texts[i]`` holds the ``group_size`` completions for ``prompts[i]``, and ``logprobs[i][k]`` is
    the summed log-probability the policy assigned its own k-th completion **at sampling time**.
    Keeping the sampling-time value rather than recomputing it is the point: the difference between
    it and a later `score_under` on the same tokens is a real quantity
    (`policy.train_infer_logprob_mismatch`), and a payload that recomputes it silently destroys the
    thing worth measuring.
    """

    prompts: list[str]
    texts: list[list[str]]
    token_ids: list[list[list[int]]]
    logprobs: list[list[float]]
    spec: dict[str, Any] = field(default_factory=dict)

    @property
    def n_prompts(self) -> int:
        return len(self.prompts)

    @property
    def n_completions(self) -> int:
        return sum(len(g) for g in self.texts)


@register_payload
@dataclass
class TokenGradients:
    """Per-token gradients of a readout scalar with respect to the input embeddings.

    ``norms[i]`` is the ragged per-token gradient norm for item ``i``, one entry per valid token in
    that item's own coordinates (padding removed), and ``dotted[i]`` is the same gradient contracted
    with the token's own embedding, which is the first-order effect of removing the token. Two
    numbers rather than one because they disagree in the case that matters: a token whose embedding
    is nearly orthogonal to the gradient has a large norm and almost no first-order effect, and a
    per-token attribution that reports only the norm calls that token important.

    What this cannot do. It is a first-order quantity at the current parameters, so it says what
    happens to the readout under an infinitesimal change to one token's embedding and says nothing
    about deleting the token, which moves every downstream position. `deviations` on any instrument
    consuming it has to carry that sentence.
    """

    norms: list["np.ndarray"]
    dotted: list["np.ndarray"]
    readout: str = "decision"
    wrt: str = "embeddings"


# ---------------------------------------------------------------------------
# The weight contract: what the reach-through should have been going through
# ---------------------------------------------------------------------------


@runtime_checkable
class SiteWeights(Protocol):
    """Read the parameters behind a `Site`, without reaching past the runtime.

    This exists because of one line. `measure/battery/path.py` needed the attention output
    projection for one head, and the only way to get it was
    ``signal.runtime.adapter.get_attn_o_proj(signal.runtime.adapter.get_layers(signal.runtime.model)[layer])``:
    four attribute hops past the last protocol call, through an adapter ABC the instrument has no
    business knowing about, into a module tree the runtime exists to hide. It worked, and it meant
    the instrument could only ever run against one backend.

    Two methods. ``sites`` is already on `Runtime`, so any runtime satisfies half of this for free.
    ``weight_at`` is the new one, and a runtime that implements it makes every head-level instrument
    portable to it. `HFPolicyRuntime` implements it natively; `HFRuntime` does not yet, and
    `site_weights` adapts it so the instrument does not have to care which it was handed.
    """

    def sites(self) -> SiteMap: ...

    def weight_at(self, site: Site) -> "torch.Tensor": ...


class WeightsUnavailable(RuntimeError):
    """A runtime that exposes neither `weight_at` nor a module tree to read one from.

    Raised with the method to implement, because the fix is on the runtime and naming it is the
    difference between a blocked instrument and a blocked afternoon.
    """


def site_weights(runtime: Any) -> SiteWeights:
    """Return something that can read a weight at a site, for any runtime.

    Three cases, in order. A runtime that implements `SiteWeights` is returned unchanged, which is
    the case this contract exists to make normal. A runtime that carries a torch module and a site
    map is wrapped in the one adapter in this library that walks a module tree. Anything else raises
    with the method to add.

    The wrap is the compatibility path and it is deliberately narrow: it reads the site map the
    runtime already publishes and resolves a dotted path against the model with `getattr`, and it
    never touches an architecture adapter. That is what lets `path.py` stop importing anything about
    `o_proj` and lets the same instrument run against a policy whose runtime answers natively.
    """
    if isinstance(runtime, SiteWeights):
        return runtime
    model = getattr(runtime, "model", None)
    sites = getattr(runtime, "sites", None)
    if model is not None and callable(sites):
        return _ModuleTreeWeights(model=model, site_map=sites())
    raise WeightsUnavailable(
        f"{type(runtime).__name__} cannot read a weight at a site. Implement "
        f"`weight_at(self, site: Site) -> torch.Tensor` on it (the `policy.base.SiteWeights` "
        f"protocol), or expose `model` and `sites()` so the module-tree adapter can resolve one. "
        f"An instrument that needs a head's output projection has no other way in that does not "
        f"reach through the runtime into the architecture."
    )


@dataclass(frozen=True)
class _ModuleTreeWeights:
    """`SiteWeights` over a torch module tree addressed by a `SiteMap`.

    The one place in this library that walks a module tree to find a parameter. It uses the site
    map the runtime already publishes, so the addressing is the runtime's own and this class adds
    only the walk and the per-head slice.
    """

    model: Any
    site_map: SiteMap

    def sites(self) -> SiteMap:
        return self.site_map

    def weight_at(self, site: Site) -> "torch.Tensor":
        import torch

        key = Site(site.layer, site.point, None) if site.point == "head_out" else site
        path = self.site_map.module_paths.get(key)
        if path is None:
            raise WeightsUnavailable(
                f"the site map exposes no module at {key}; it has "
                f"{sorted(str(s) for s in self.site_map.module_paths)}. A model whose attention "
                f"output projection was not resolved at load has no head-level weights to read, "
                f"and a head-level instrument should refuse rather than guess a slice."
            )
        module: Any = self.model
        for part in path.split("."):
            module = module[int(part)] if part.isdigit() else getattr(module, part)
        weight = getattr(module, "weight", None)
        if weight is None:
            raise WeightsUnavailable(
                f"the module at {key} ({type(module).__name__}) has no `weight`, so there is no "
                f"projection to slice."
            )
        weight = weight.detach().to(torch.float32)
        if site.point != "head_out" or site.head is None:
            return weight  # type: ignore[no-any-return]
        n_heads = max(int(self.site_map.n_heads), 1)
        d_head = int(weight.shape[1]) // n_heads
        lo = site.head * d_head
        return weight[:, lo : lo + d_head]  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Provenance of the differentiation apparatus
# ---------------------------------------------------------------------------


def runtime_provenance() -> dict[str, Any]:
    """What is patched in this interpreter that a gradient reading depends on.

    Only one entry so far and it is the one that hides. `nnsight` replaces
    `torch.Tensor.backward` at import while copying `__module__` and `__qualname__` onto the
    replacement, so every name-based check reports the original and every `.backward()` in the
    process goes through the patch. This package does not import `nnsight`; another package in the
    same process might, and a gradient measured afterwards was taken on a different apparatus. This
    reports presence in `sys.modules`, which is the fact that matters, rather than trying to detect
    the patch, which by construction cannot be done by name.
    """
    import sys

    nnsight = sys.modules.get("nnsight")
    return {
        "nnsight_imported": nnsight is not None,
        "nnsight_version": getattr(nnsight, "__version__", None) if nnsight else None,
        "backward_patched_by_name": False,
        "note": (
            "nnsight preserves __module__ and __qualname__ on its replacement for "
            "torch.Tensor.backward, so `backward_patched_by_name` is False whether or not the "
            "patch is installed. Presence in sys.modules is the checkable fact."
        ),
    }


# ---------------------------------------------------------------------------
# The protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class PolicySubject(Protocol):
    """A policy with white-box access, the peer of `RewardSignal`.

    The first block of members is identical to `RewardSignal`'s and that is the whole point: an
    instrument written against `meta`, `caps`, `runtime`, `readout`, `score`, `capture` and
    `tokenize` runs against either object with the subject as the only thing that changed. The
    second block is the policy's own.

    ``score`` is the `RewardSignal` spelling and ``score_under`` is the honest one. They return the
    same numbers. A grader's score is a judgement about the text; a policy's is the log-probability
    the policy assigns it, which is a different quantity wearing the same interface, and an
    instrument that reads either through `score` is reading "this subject's scalar for this item"
    and is correct to be indifferent. `Unit` on the declared quantity is what stops the two being
    compared, not the method name.
    """

    meta: PolicyMeta
    caps: Capability
    runtime: "Runtime"

    # -- the shared surface -------------------------------------------------

    def readouts(self) -> list[Readout]: ...

    def readout(self, name: str = "decision") -> Readout: ...

    def score(self, view: Any, readout: str = "decision") -> Any: ...

    def score_prefixes(self, view: Any, readout: str = "decision") -> Any: ...

    def capture(self, view: Any, spec: "CaptureSpec") -> "CaptureHandle": ...

    def with_interventions(self, *ivs: "Intervention") -> "PolicySubject": ...

    def tokenize(self, item: Any) -> TokenizedInput: ...

    # -- the policy's own ---------------------------------------------------

    def sample(self, prompts: Sequence[str], spec: SampleSpec) -> "Evidence[Rollouts]": ...

    def score_under(self, view: Any, readout: str = "logprob") -> "Evidence[Scores]": ...

    def grad_h(self, view: Any, at: Site, readout: str = "decision") -> "torch.Tensor": ...

    def token_gradients(
        self, view: Any, readout: str = "decision"
    ) -> "Evidence[TokenGradients]": ...


__all__ = [
    "PolicyMeta",
    "PolicySubject",
    "PositionSpec",
    "Readout",
    "ReadoutKind",
    "Rollouts",
    "SampleSpec",
    "Scores",
    "SiteWeights",
    "TokenCurves",
    "TokenGradients",
    "TokenizedInput",
    "WeightsUnavailable",
    "runtime_provenance",
    "site_weights",
]

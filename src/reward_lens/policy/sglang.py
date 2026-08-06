"""The SGLang policy: the same Plane A boundary, with the one difference that matters.

SGLang is a second serving engine and it sits on the same side of the line as vLLM: sampling and
log-probabilities out, no activations, no gradients. The five structural limits enumerated in
`reward_lens.policy.vllm` apply here for the same reasons and are not restated; that module is the
one to read.

The difference worth recording is RadixAttention, the prefix cache SGLang shares across requests. It
does not change what is readable and it does change what a measurement means: two requests with a
common prefix do not recompute that prefix, so a per-request latency, a token count and anything
derived from either are a function of what else the server has seen. An instrument timing an engine
or attributing cost to a request has to say whether the cache was cold, and there is no way to read
that off the response. `cache_state` carries the caller's declaration and defaults to `"unknown"`,
which is honest and is what an instrument should refuse on rather than assume.

This module imports no engine and declares no dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Sequence

from reward_lens.core.types import Capability
from reward_lens.policy.base import PolicyMeta, SampleSpec
from reward_lens.policy.vllm import ENGINE_LIMITS, SERVING_CAPS, ServingPolicy

CacheState = Literal["cold", "warm", "unknown"]

#: The vLLM limits plus the one that is SGLang's own.
SGLANG_LIMITS: dict[str, str] = {
    **ENGINE_LIMITS,
    "radix_prefix_cache": (
        "RadixAttention shares a prefix cache across requests, so a request's latency and its "
        "prefill token count depend on what the server saw before it. Neither is a property of the "
        "request alone, and the response does not say which."
    ),
}


@dataclass
class SGLangPolicy(ServingPolicy):
    """A policy behind an SGLang server. `ServingPolicy` with the cache declaration attached.

    Not a `PolicySubject`, for the reasons in `reward_lens.policy.vllm`. The extra field exists so
    that an instrument reading a cost or a latency off this engine can refuse when nobody said
    whether the cache was cold, instead of reporting a number that is partly about the server's
    history.
    """

    engine: str = "sglang"
    caps: Capability = SERVING_CAPS
    limits: dict[str, str] = field(default_factory=lambda: dict(SGLANG_LIMITS))
    #: Whether the prefix cache was cold for these requests. `"unknown"` is the default and is the
    #: value a timing instrument should refuse on.
    cache_state: CacheState = "unknown"

    def timing_is_attributable(self) -> bool:
        """Whether a per-request latency from this engine is a property of the request.

        False under `"unknown"`, which is the default, because a shared prefix cache makes the
        number depend on the server's history and no field in the response records it.
        """
        return self.cache_state == "cold"


def sglang_policy(
    meta: PolicyMeta,
    call: Callable[[Sequence[str], SampleSpec], Any],
    *,
    cache_state: CacheState = "unknown",
) -> SGLangPolicy:
    """Build an `SGLangPolicy` around a client callable."""
    return SGLangPolicy(meta=meta, call=call, cache_state=cache_state)


__all__ = ["SGLANG_LIMITS", "CacheState", "SGLangPolicy", "sglang_policy"]

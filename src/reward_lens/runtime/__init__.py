"""``reward_lens.runtime`` — backends, hooks, activation store, precision, fingerprints, samplers.

The runtime is the layer that actually touches ``transformers``: it owns the forward pass, the hook
mechanics, the activation cache, the numerics policies, the model fingerprint, and the sampler
bridge. Nothing above it imports ``transformers`` directly; everything goes through
the ``Runtime`` protocol in ``backend.py`` (the frozen contract). Importing this package pulls torch,
so the pure epistemics layers (``core``, ``stats``, ``data``) never import it.

``backend`` is the frozen protocol surface (import it for the types even without torch, since it
annotates torch only under ``TYPE_CHECKING``). The concrete ``HFRuntime`` and the batteries of
utilities around it live in the other modules and are re-exported here for convenience.
"""

from __future__ import annotations

from reward_lens.core.extras import require_extra

require_extra("white-box", subsystem="reward_lens.runtime")

from reward_lens.runtime.backend import (
    Capture,
    CaptureHandle,
    CaptureSpec,
    RawOutput,
    Runtime,
    ScalarFn,
    SiteMap,
    TokenBatch,
)
from reward_lens.runtime.fingerprint import fingerprint, fingerprint_bytes
from reward_lens.runtime.precision import (
    NumericsPolicy,
    register_policy,
    resolve_policy,
    safe_cosine,
)
from reward_lens.runtime.store import (
    ActivationStore,
    V1Cache,
    read_v1_cache,
)

__all__ = [
    # frozen protocol surface
    "Runtime",
    "ScalarFn",
    "TokenBatch",
    "RawOutput",
    "CaptureSpec",
    "Capture",
    "CaptureHandle",
    "SiteMap",
    # precision
    "NumericsPolicy",
    "resolve_policy",
    "register_policy",
    "safe_cosine",
    # fingerprints
    "fingerprint",
    "fingerprint_bytes",
    # activation store + v1 cache read adapter
    "ActivationStore",
    "V1Cache",
    "read_v1_cache",
    # concrete backend + sampler are imported lazily to keep this module's import cheap
]


def __getattr__(name: str):  # PEP 562: lazy access to the heavy concrete backend and sampler
    if name in ("HFRuntime", "auto_batch_size"):
        from reward_lens.runtime import hf

        return getattr(hf, name)
    if name in ("SamplerBridge", "SampleRecord", "SamplerUnavailableError"):
        from reward_lens.runtime import sampling

        return getattr(sampling, name)
    raise AttributeError(f"module 'reward_lens.runtime' has no attribute {name!r}")

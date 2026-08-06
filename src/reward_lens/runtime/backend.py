"""The Runtime protocol: the backend contract.

A deliberately thin contract so backends can vary (HF eager now; nnsight or a compiled runtime
later) without touching anything above it. The whole point is that interventions and captures
share one mounting path, so any Observable can run under any Intervention. In v1, patching and
caching were separate code paths, which is why an Observable could not be measured under an
arbitrary intervention; that pain is designed out here.

``grad`` and ``hvp`` are the capabilities v1 lacked. ``hvp`` is double-backprop on the readout
scalar; both must work under bf16 trunks with fp32 accumulation for the scalar head. These two
methods unlock reward field theory (Hessian spectroscopy), gradient-ascent hack generation,
incentive Jacobians, and second-order attribution: four sciences through one contract.

Frozen interface. torch is referenced in annotations only, under ``TYPE_CHECKING``,
so importing the protocol surface does not import torch; the concrete ``hf`` backend imports torch
for real.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ContextManager,
    Literal,
    Protocol,
    Sequence,
    runtime_checkable,
)

from reward_lens.core.types import Site

if TYPE_CHECKING:
    import torch

    from reward_lens.interventions.base import CompiledIntervention
    from reward_lens.signals.base import PositionSpec

# A scalar function of a forward output: extracts the readout scalar per item, for grad/hvp.
ScalarFn = Callable[["RawOutput"], "torch.Tensor"]


@dataclass
class TokenBatch:
    """A left-padded batch of tokenized inputs ready for the model.

    Left padding is deliberate: it aligns the final positions across a batch so a final-token
    readout reads the same relative location for every item. ``meta`` carries the per-item
    metadata (lengths, span maps) the capture path needs to resolve positions.
    """

    input_ids: "torch.Tensor"
    attention_mask: "torch.Tensor"
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])


@dataclass
class RawOutput:
    """The result of a forward pass.

    Wraps whatever the backend produced: the reward-relevant tensors (a per-item scalar or a
    multi-row head output), and, when a capture was requested, the hidden states. Kept thin on
    purpose; readouts extract what they need. ``reward`` is the primary scalar(s); ``logits`` is
    present for generative judges; ``hidden`` is populated only under capture.
    """

    reward: "torch.Tensor | None" = None
    logits: "torch.Tensor | None" = None
    hidden: "dict[Site, torch.Tensor] | None" = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class CaptureSpec:
    """What to capture in a forward pass.

    ``sites`` are the locations to read; ``position`` is a PositionSpec resolved per input;
    ``full_sequence`` keeps every position rather than the resolved ones; ``dtype`` is the storage
    dtype (fp16 for activations by default, fp32 for anything entering covariance or whitening,
    since frames refuse fp16 inputs); ``keep_on_device`` versus ``stream_to_store`` decides whether
    the capture stays in VRAM or is written to the ActivationStore.
    """

    sites: tuple[Site, ...]
    position: "PositionSpec | None" = None
    full_sequence: bool = False
    dtype: str = "float16"
    keep_on_device: bool = False
    stream_to_store: bool = False


@dataclass
class Capture:
    """Captured activations from one forward pass.

    ``tensors`` maps each requested Site to its activation tensor (position-resolved unless
    ``full_sequence`` was set). ``positions`` records the token indices actually read per item, so
    a downstream Observable knows exactly where each number came from.
    """

    tensors: "dict[Site, torch.Tensor]"
    positions: list[list[int]] = field(default_factory=list)
    dtype: str = "float16"


class CaptureHandle(Protocol):
    """A handle to a capture that may be in memory or memory-mapped from the ActivationStore.

    Iterating a handle yields per-batch ``Capture`` objects co-batched with the DataView, so a
    large sweep never materializes every activation at once. A handle backed by the store supports
    ``get(site)`` for a memory-mapped read. This is the abstraction that lets a chi spectrum and a
    BoN ladder computed on the same draw be exactly comparable (they read the same cached capture).
    """

    def __iter__(self) -> Any: ...

    def get(self, site: Site) -> "torch.Tensor": ...


@dataclass
class SiteMap:
    """What sites an architecture exposes, resolved by the adapter.

    Maps a logical ``Site`` to the concrete module path the backend hooks. Adapters populate this
    once at load; the runtime consults it when mounting captures and interventions, so no
    Observable ever hardcodes a module name.
    """

    module_paths: dict[Site, str] = field(default_factory=dict)
    n_layers: int = 0
    d_model: int = 0
    n_heads: int = 0

    def resolve(self, site: Site) -> str:
        if site not in self.module_paths:
            raise KeyError(f"architecture does not expose site {site}")
        return self.module_paths[site]


@runtime_checkable
class Runtime(Protocol):
    """The backend contract.

    Implemented by ``runtime.hf.HFRuntime`` now; a compiled or nnsight backend later implements the
    same six methods and nothing above ``runtime`` knows. ``forward_with_capture`` and ``mounted``
    share one hook path so any capture composes with any intervention. ``grad`` and ``hvp`` operate
    on the readout scalar with fp32 accumulation regardless of trunk dtype.
    """

    def forward(self, batch: TokenBatch) -> RawOutput: ...

    def forward_with_capture(
        self, batch: TokenBatch, spec: CaptureSpec
    ) -> tuple[RawOutput, Capture]: ...

    def mounted(
        self, interventions: Sequence["CompiledIntervention"]
    ) -> ContextManager["Runtime"]: ...

    def grad(
        self, batch: TokenBatch, scalar_fn: ScalarFn, wrt: Site | Literal["embeddings"]
    ) -> "torch.Tensor": ...

    def hvp(
        self, batch: TokenBatch, scalar_fn: ScalarFn, at: Site, vecs: "torch.Tensor"
    ) -> "torch.Tensor": ...

    def sites(self) -> SiteMap: ...


__all__ = [
    "ScalarFn",
    "TokenBatch",
    "RawOutput",
    "CaptureSpec",
    "Capture",
    "CaptureHandle",
    "SiteMap",
    "Runtime",
]

"""The RewardSignal protocol and its first-class readouts.

This module is the answer to substrate lock-in and the enabling layer for half the
corpus. v1 funnelled everything through one scalar ``w_r`` at the final token; ArmoRM's nineteen
heads were force-collapsed to a row mean, and judges, PRMs, implicit rewards, and trajectories
were simply unreachable. The fix is one move: positions and readouts are first-class. Every
measurement in the kernel is parameterized by ``(signal, readout)``, and a
``PositionSpec`` resolves "where to read" per input. Crystallization depth of a judge's verdict
is then the same Observable as crystallization depth of a scalar head, called with a different
readout.

This is a frozen interface. The protocol here is what the whole battery, every
index, and every science compile against; changing it takes a dated ADR. It imports torch only
under ``TYPE_CHECKING`` so the type surface is available without importing torch, which keeps the
pure layers that reference signal types (data, some stats) torch-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from reward_lens.core.evidence import register_payload
from reward_lens.core.types import Capability, ModelFP, Site, Span

if TYPE_CHECKING:  # torch and the runtime are referenced only in annotations
    import numpy as np

    from reward_lens.data.schema import DataView
    from reward_lens.interventions.base import Intervention
    from reward_lens.runtime.backend import CaptureHandle, CaptureSpec, Runtime


# ---------------------------------------------------------------------------
# Tokenized input: the carrier of span-level structure
# ---------------------------------------------------------------------------


@dataclass
class TokenizedInput:
    """A tokenized data item with the structure the kernel needs to read it precisely.

    ``token_offsets`` maps each token to its ``(char_start, char_end)`` in the source text, and
    ``spans`` carries the typed spans (receipt, error step, critique, verdict) resolved into
    token coordinates. This is the unglamorous, load-bearing part: without exact
    character-to-token maps, span-level patching and attribution silently misalign, which is the
    quiet killer of every pairwise causal method. ``tokenize`` on a signal produces this.
    """

    input_ids: list[int]
    attention_mask: list[int]
    text: str = ""
    token_offsets: tuple[tuple[int, int], ...] = ()
    spans: tuple[Span, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.input_ids)

    def valid_positions(self) -> list[int]:
        """Token indices with a nonzero attention mask (the non-pad positions)."""
        return [i for i, m in enumerate(self.attention_mask) if m]


# ---------------------------------------------------------------------------
# PositionSpec: where a readout reads
# ---------------------------------------------------------------------------

PositionKind = Literal["final", "judgment", "step_ends", "span_ends", "all", "explicit"]


@dataclass(frozen=True)
class PositionSpec:
    """A resolvable specification of which token positions a readout reads.

    Nothing in the kernel hardcodes "final token". A classifier head reads at ``final``; a
    process reward model reads at ``step_ends``; a generative judge reads at ``judgment`` (the
    verdict token, whose location the signal detects and validates); a dense reward reads at
    ``all``. ``detail`` carries the kind-specific configuration (the span kind for ``span_ends``,
    the explicit indices for ``explicit``, the delimiter config or detected index for
    ``judgment``).
    """

    kind: PositionKind = "final"
    detail: Any = None

    def resolve(self, tokens: "TokenizedInput") -> list[int]:
        """Resolve to concrete token indices for a given tokenized input.

        The position-only kinds (``final``, ``all``, ``explicit``, and span-derived kinds when
        the spans are present) resolve here with no model knowledge. ``judgment`` is resolved by
        the signal, which detects the verdict position from the chat template and validates it;
        when a signal has done that detection it passes the resolved index through ``detail``, so
        this method honours an explicit ``detail`` index list for ``judgment`` as well.
        """
        valid = tokens.valid_positions()
        if not valid:
            return []
        if self.kind == "final":
            return [valid[-1]]
        if self.kind == "all":
            return valid
        if self.kind == "explicit":
            idx = self.detail if isinstance(self.detail, (list, tuple)) else [self.detail]
            return [int(i) for i in idx]
        if self.kind == "judgment":
            if self.detail is None:
                raise ValueError(
                    "judgment PositionSpec must be resolved by the signal (verdict-token "
                    "detection); the detected index is passed through detail"
                )
            idx = self.detail if isinstance(self.detail, (list, tuple)) else [self.detail]
            return [int(i) for i in idx]
        if self.kind in ("step_ends", "span_ends"):
            want = self.detail if self.kind == "span_ends" else "step"
            ends = [min(s.end, valid[-1]) for s in tokens.spans if want is None or s.kind == want]
            return ends or [valid[-1]]
        raise ValueError(f"unknown PositionSpec kind: {self.kind}")


# ---------------------------------------------------------------------------
# Readout: the first-class reward direction + kind
# ---------------------------------------------------------------------------

ReadoutKind = Literal["linear", "logit_diff", "simplex", "token_value"]


@dataclass(frozen=True)
class Readout:
    """A first-class readout: what to read, where, and how.

    ``name`` is the human key ("reward", "verdict", "criterion:coherence", "quantile:0.9"). For
    ``linear`` and ``logit_diff`` readouts, ``vector`` is the direction ``w`` (fp32) the scalar is
    projected onto; for ``simplex`` (Likert over score tokens) and ``token_value`` (per-token
    value) the meaning is carried in ``meta``. ``site`` is where the readout reads, usually the
    final residual stream; ``position`` says at which token(s). This object is the pivot of the
    design: it is architecturally resolved by adapters at load time (w_r is read off the
    checkpoint, not probe-extracted), so there is no "is this feature real" regress at the readout
    (I1).
    """

    name: str
    kind: ReadoutKind
    site: Site
    position: PositionSpec
    vector: Any = None  # torch.Tensor for linear/logit_diff; None otherwise
    meta: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# SignalMeta: identity, lineage, numerics
# ---------------------------------------------------------------------------


@dataclass
class SignalMeta:
    """Signal identity, lineage, and numerics.

    ``lineage`` records the declared base model, declared training data (free text plus dataset
    ids where known), release date, and provenance tier (weights-verified vs card-claimed). It
    costs almost nothing to collect at load and is impossible to reconstruct afterwards, and it
    is what makes kinship, the Atlas population axes, and the monoculture index computable later
    (RK9). ``numerics_policy`` names the per-family policy (head-in-fp32, Gemma soft-cap
    handling); ``soft_cap`` records whether a soft cap was disabled on the reward path so lens
    observables can annotate it.
    """

    fingerprint: ModelFP
    adapter: str
    architecture: str = ""
    lineage: dict[str, Any] = field(default_factory=dict)
    template: dict[str, Any] = field(default_factory=dict)
    numerics_policy: str = "default"
    soft_cap: float | None = None
    d_model: int | None = None
    n_layers: int | None = None
    n_heads: int | None = None


# ---------------------------------------------------------------------------
# Score payloads
# ---------------------------------------------------------------------------


@register_payload
@dataclass
class Scores:
    """Per-item reward scores from a readout (the payload of ``score``)."""

    values: "np.ndarray"
    readout: str = "reward"
    n_items: int = 0


@register_payload
@dataclass
class TokenCurves:
    """Per-item, per-token value curves r(y_{1:t}) from ``score_prefixes``.

    A ragged collection (one curve per item, variable length) stored as a list of arrays plus the
    readout name. At least five sciences consume this (verification, dense rewards, thermodynamics
    diagnostics, the recorder, PRM comparison), which is why it is a kernel method, not a
    science-side hack.
    """

    curves: list["np.ndarray"]
    readout: str = "reward"


# ---------------------------------------------------------------------------
# The protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class RewardSignal(Protocol):
    """The substrate abstraction: any reward signal with white-box access.

    Eight adapters implement this (classifier, judge, process, implicit, rubric, trajectory,
    dense, ensemble). An Observable written against this protocol runs on all of them; a new
    grader paradigm the field invents becomes a new adapter passing the conformance suite, and
    the whole battery, every index, and every science become available to it for free (the
    extensibility contract).

    ``caps`` declares capabilities; ``readouts`` lists the readouts the signal exposes;
    ``score`` and ``score_prefixes`` return Evidence; ``capture`` returns a handle to activations
    (possibly cached); ``with_interventions`` returns a wrapped signal any Observable accepts
    unchanged; ``tokenize`` owns span carry-through.
    """

    meta: SignalMeta
    caps: Capability
    runtime: "Runtime"

    def readouts(self) -> list[Readout]: ...

    def score(self, view: "DataView", readout: str = "reward") -> Any: ...

    def score_prefixes(self, view: "DataView", readout: str = "reward") -> Any: ...

    def capture(self, view: "DataView", spec: "CaptureSpec") -> "CaptureHandle": ...

    def with_interventions(self, *ivs: "Intervention") -> "RewardSignal": ...

    def tokenize(self, item: Any) -> TokenizedInput: ...


__all__ = [
    "TokenizedInput",
    "PositionSpec",
    "PositionKind",
    "Readout",
    "ReadoutKind",
    "SignalMeta",
    "Scores",
    "TokenCurves",
    "RewardSignal",
]

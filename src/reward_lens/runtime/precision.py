"""Per-family numerics policies.

This module is the structural cure for two whole classes of v1 failure: the QRM dtype-mismatch
crash and the Gemma-2 soft-cap NaN. v1 fixed both by mutating the loaded model (coerce the reward
head to bf16; null the soft-cap on the config). That worked but left the fix implicit and per-load.
v3 states the policy as data.

The one load-bearing decision here is that **the reward head always computes in fp32**, and the
cast happens at the head boundary (``head_project``), never by down-casting the head to the trunk
dtype. This supersedes v1's ``_coerce_reward_head_dtype`` (which pushed the fp32 head down to bf16
to satisfy the GEMM kernels) and removes the QRM failure class at the root: a bf16 trunk feeds a
bf16 hidden state into an fp32 projection, and the projection upcasts its input rather than the
head being downcast. The scalar reward is worth the handful of extra flops.

Gemma-2's ``final_logit_softcapping``/``attn_logit_softcapping`` are disabled on the reward path
(the reward head reads the hidden state, not the LM logits, so the soft cap is dead weight that
only flattens late-layer differentials into a tanh plateau and drives the lens to NaN, the E09
lesson). The disabling is recorded so ``SignalMeta.soft_cap`` can carry it and lens observables can
annotate it.

Policies are resolved by architecture family and are conformance-tested: score
parity within ``tol`` across trunk-dtype configurations, and no NaN on the cosine path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # torch only in annotations so importing the policy table is torch-free
    import torch

# The dtype names the kernel speaks. Kept as strings (not torch.dtype) so the policy table is a
# plain, hashable, torch-free data structure; ``torch_dtype`` resolves them on demand.
_DTYPES: dict[str, str] = {
    "float32": "float32",
    "float16": "float16",
    "bfloat16": "bfloat16",
}


def torch_dtype(name: str) -> "torch.Tensor":
    """Resolve a dtype name to the ``torch.dtype`` object (imported lazily)."""
    import torch

    try:
        return getattr(torch, name)
    except AttributeError as exc:  # pragma: no cover - guards a typo in a registered policy
        raise ValueError(f"unknown dtype name {name!r}") from exc


@dataclass(frozen=True)
class NumericsPolicy:
    """The numerics contract for one model family.

    ``trunk_dtype`` is the default dtype the backbone loads and runs in (bf16 on the 8B campaign
    GPUs, fp32 on the CPU test vehicle). ``head_dtype`` is always ``"float32"``: the reward
    projection upcasts its input and computes the scalar in fp32 regardless of the trunk. The two
    ``softcap_*`` attrs name the config fields to null on the reward path (Gemma-2 only), and
    ``disables_soft_cap`` says whether this policy touches them at all so ``SignalMeta`` can record
    it. ``attn_implementation`` and ``allow_tf32`` are the load-time knobs; ``tol`` is the score
    parity tolerance the conformance suite holds this family to across dtype configurations.
    """

    name: str
    family: str
    trunk_dtype: str = "float32"
    head_dtype: str = "float32"
    softcap_fields: tuple[str, ...] = ()
    attn_implementation: str = "eager"
    allow_tf32: bool = False
    tol: float = 1e-4
    notes: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def disables_soft_cap(self) -> bool:
        """Whether this policy nulls any soft-cap field on the reward path."""
        return bool(self.softcap_fields)

    def torch_head_dtype(self) -> "torch.Tensor":
        """The head compute dtype as a ``torch.dtype`` (always fp32)."""
        return torch_dtype(self.head_dtype)

    def torch_trunk_dtype(self) -> "torch.Tensor":
        """The trunk dtype as a ``torch.dtype``."""
        return torch_dtype(self.trunk_dtype)

    def with_trunk(self, trunk_dtype: str) -> "NumericsPolicy":
        """Return a copy with a different trunk dtype (the conformance dtype matrix uses this).

        The head stays fp32; only the trunk changes. This is exactly the axis the conformance
        tests sweep (bf16/fp16/fp32 trunk x fp32 head) to prove score parity and NaN-freedom.
        """
        from dataclasses import replace

        if trunk_dtype not in _DTYPES:
            raise ValueError(f"unknown trunk dtype {trunk_dtype!r}; known: {sorted(_DTYPES)}")
        return replace(self, trunk_dtype=trunk_dtype)

    def apply_to_config(self, config: object) -> dict[str, float | None]:
        """Null the soft-cap fields on a model config, returning what was disabled.

        Called once at load. The returned mapping (field name to the value that was there) is
        recorded so ``SignalMeta.soft_cap`` carries the original cap and a lens observable can note
        that the reward path ran with it off. A field that is absent or already ``None`` is skipped.
        Idempotent and best-effort: a config that refuses assignment is left alone.
        """
        disabled: dict[str, float | None] = {}
        if config is None:
            return disabled
        for field_name in self.softcap_fields:
            old = getattr(config, field_name, None)
            if old is None:
                continue
            try:
                setattr(config, field_name, None)
                disabled[field_name] = old
            except (AttributeError, TypeError):  # frozen or computed config field
                continue
        return disabled

    def head_project(
        self, hidden: "torch.Tensor", weight: "torch.Tensor", bias: float = 0.0
    ) -> "torch.Tensor":
        """Project a hidden state onto a reward direction in fp32 (the head boundary).

        ``hidden`` is ``(..., d_model)`` in whatever the trunk produced (bf16/fp16/fp32); ``weight``
        is ``(d_model,)``. The input is upcast to fp32 and the scalar is accumulated in fp32, which
        is the whole point of the policy: the head never runs in the trunk dtype. Returns ``(...)``.
        """
        h32 = hidden.to(dtype=self.torch_head_dtype())
        w32 = weight.to(dtype=self.torch_head_dtype(), device=hidden.device)
        return (h32 @ w32) + bias


def safe_cosine(
    a: "torch.Tensor", b: "torch.Tensor", eps: float = 1e-12, dim: int = -1
) -> "torch.Tensor":
    """Cosine similarity with an fp32 upcast and a denominator floor (the E09 NaN lesson).

    E09 produced an all-NaN table because a near-zero late-layer differential divided by its own
    vanishing norm. The cure is not a threshold on the science side (that was the operationalization
    drift the audit flagged); it is a numerics guard here: upcast to fp32, floor the norms at
    ``eps``, and never emit a NaN from a zero vector (a zero vector has cosine 0 with everything,
    which is the honest answer). Conformance asserts this path stays finite.
    """
    import torch

    a32 = a.to(dtype=torch.float32)
    b32 = b.to(dtype=torch.float32)
    num = (a32 * b32).sum(dim=dim)
    denom = a32.norm(dim=dim).clamp_min(eps) * b32.norm(dim=dim).clamp_min(eps)
    return num / denom


# ---------------------------------------------------------------------------
# The policy registry: one entry per family, keyed for resolution.
# ---------------------------------------------------------------------------

_DEFAULT = NumericsPolicy(
    name="default",
    family="generic",
    notes="fp32 head, eager attention, no soft cap; the safe baseline for any unknown family.",
)

_REGISTRY: dict[str, NumericsPolicy] = {
    "llama": NumericsPolicy(
        name="llama",
        family="llama",
        notes="Skywork/FsfairX-class Llama reward models; standard resid stream, o_proj heads.",
    ),
    "mistral": NumericsPolicy(
        name="mistral",
        family="mistral",
        notes="Architecturally Llama-like for the reward path.",
    ),
    "gemma2": NumericsPolicy(
        name="gemma2",
        family="gemma2",
        softcap_fields=("final_logit_softcapping", "attn_logit_softcapping"),
        notes="Soft cap disabled on the reward path (E09); recorded in SignalMeta.soft_cap.",
    ),
    "armorm": NumericsPolicy(
        name="armorm",
        family="armorm",
        notes="Multi-objective Llama backbone; 19-row head projected in fp32, not row-meaned.",
    ),
    "internlm2": NumericsPolicy(
        name="internlm2",
        family="internlm2",
        notes="Custom InternLM2ForRewardModel; per-token v_head, last-valid-token pooling.",
    ),
    "qrm": NumericsPolicy(
        name="qrm",
        family="llama",
        notes="QRM regression_layer; the fp32-head policy is exactly the fix for the QRM crash.",
    ),
    "default": _DEFAULT,
}

# Keyword table for resolving a free-text architecture string to a family. Order matters: the more
# specific families (gemma2, internlm2, armorm) are tried before the generic llama/mistral keys.
_ARCH_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("armorm", "armorm"),
    ("gemma2", "gemma2"),
    ("gemma", "gemma2"),
    ("internlm2", "internlm2"),
    ("internlm", "internlm2"),
    ("qrm", "qrm"),
    ("mistral", "mistral"),
    ("llama", "llama"),
)


def register_policy(policy: NumericsPolicy) -> None:
    """Register a numerics policy under its ``name`` (extension point for new families)."""
    _REGISTRY[policy.name] = policy


def resolve_policy(architecture: str | None) -> NumericsPolicy:
    """Resolve an architecture string or ``model_type`` to a ``NumericsPolicy``.

    Matching is keyword-based over a lower-cased architecture string, most-specific family first
    (so ``"Gemma2ForSequenceClassification"`` resolves to the gemma2 policy, not a generic one, and
    picks up the soft-cap disabling). Unknown architectures resolve to the fp32-head default, which
    is always safe: it just declines the family-specific knobs. Never raises; an unrecognised family
    is a lower-trust default, not a crash.
    """
    if not architecture:
        return _DEFAULT
    key = architecture.lower()
    if key in _REGISTRY:
        return _REGISTRY[key]
    for needle, family_key in _ARCH_KEYWORDS:
        if needle in key:
            return _REGISTRY[family_key]
    return _DEFAULT


__all__ = [
    "NumericsPolicy",
    "resolve_policy",
    "register_policy",
    "safe_cosine",
    "torch_dtype",
]

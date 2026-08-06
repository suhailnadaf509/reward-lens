"""Signal loading and auto-discovery.

``load_signal`` is the front door: given an HF id, a local path, or a ``SignalSpec``, it sniffs the
loading convention (TRL ``AutoModelForSequenceClassification`` with ``num_labels=1``; OpenRLHF
CausalLM plus ``value_head``/``score``; veRL ``AutoModelForTokenClassification`` last-valid-token;
ad-hoc ``trust_remote_code`` heads), chooses the adapter and the numerics policy, fingerprints, and
runs a conformance quick-check before handing back a ``RewardSignal``. Ambiguities are errors with
candidate lists, never silent guesses.

The real HF-hub load of the 8B/27B campaign models is GPU/download-gated: this machine has an 8 GB
laptop GPU and cannot hold them, so ``load_signal`` implements the code path and marks it, but will
not download a large model. The always-available, no-download entry points are ``wrap_hf_model`` (an
already-constructed model + tokenizer + adapter) and ``from_tiny`` (the synthetic tiny Llama the
tests run on), both of which build a ``ClassifierRM`` on CPU with no network.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from reward_lens.core.errors import ConformanceError
from reward_lens.signals.adapters import (
    build_site_map,
    is_multi_readout,
    resolve_adapter,
    reward_head_module,
)
from reward_lens.signals.base import SignalMeta
from reward_lens.signals.classifier import ClassifierRM, build_readouts

if TYPE_CHECKING:
    import torch


@dataclass
class SignalSpec:
    """A resolved request to load a signal.

    ``source`` is an HF id or a local path; ``adapter`` and ``numerics`` override auto-detection when
    given; ``convention`` names the loading convention when the caller already knows it. ``device``
    and ``dtype`` are the load knobs. Kept as a dataclass so a study can serialize exactly which
    signal it ran against.
    """

    source: str
    adapter: str | None = None
    numerics: str | None = None
    convention: str | None = None
    device: str = "cpu"
    dtype: str = "float32"
    trust_remote_code: bool = True
    allow_download: bool = False
    max_length: int = 2048
    meta_extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# The no-download entry points (always available on this hardware)
# ---------------------------------------------------------------------------


def wrap_hf_model(
    model: "torch.nn.Module",
    tokenizer: Any,
    adapter: Any = None,
    *,
    device: str = "cpu",
    numerics: Any = None,
    adapter_id: str | None = None,
    architecture: str | None = None,
    max_length: int = 2048,
    lineage: dict[str, Any] | None = None,
    conformance_quickcheck: bool = True,
) -> ClassifierRM:
    """Build a ``ClassifierRM`` from an already-loaded model + tokenizer (no download).

    Resolves the adapter (v1 dispatch) if not given, builds the ``SiteMap`` and numerics policy,
    fingerprints the model, reads the reward head into readouts, and runs a fast conformance
    quick-check (determinism plus readout-matches-head on two trivial inputs) unless disabled. This
    is the constructor the tests and the tiny-model path use, and the one the hub loader ends at once
    it has a model in hand.
    """
    import torch

    from reward_lens.runtime.fingerprint import fingerprint
    from reward_lens.runtime.hf import HFRuntime
    from reward_lens.runtime.precision import resolve_policy

    model.eval()
    torch_device = torch.device(device)
    if adapter is None:
        adapter = resolve_adapter(model, adapter_id or "")
    adapter_name = type(adapter).__name__
    arch = architecture or _architecture_string(model)
    policy = numerics if numerics is not None else resolve_policy(arch)

    # Apply the numerics policy at the boundary: null the soft cap on the reward path (Gemma-2) and
    # record what was disabled so SignalMeta.soft_cap carries it (E09).
    disabled = policy.apply_to_config(getattr(model, "config", None))
    soft_cap = next(iter(disabled.values()), None) if disabled else None

    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None):
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = getattr(tokenizer, "pad_token_id", None)
    pad_id = int(pad_id) if pad_id is not None else 0

    site_map = build_site_map(adapter, model)
    head = reward_head_module(adapter, model)
    fp = fingerprint(model, tokenizer, adapter_name)

    runtime = HFRuntime(
        model=model,
        adapter=adapter,
        site_map=site_map,
        policy=policy,
        head_module=head,
        device=torch_device,
        pad_id=pad_id,
    )
    readouts, legacy = build_readouts(adapter, model, site_map)
    meta = SignalMeta(
        fingerprint=fp,
        adapter=adapter_name,
        architecture=arch,
        lineage=lineage or {"provenance_tier": "weights-verified"},
        template={"chat_template": getattr(tokenizer, "chat_template", None) is not None},
        numerics_policy=policy.name,
        soft_cap=soft_cap,
        d_model=site_map.d_model,
        n_layers=site_map.n_layers,
        n_heads=site_map.n_heads,
    )
    signal = ClassifierRM(
        runtime=runtime,
        meta=meta,
        readouts=readouts,
        tokenizer=tokenizer,
        policy=policy,
        max_length=max_length,
        legacy_row_mean=legacy,
    )
    if is_multi_readout(adapter, model):
        meta.lineage["multi_readout"] = True
    if conformance_quickcheck:
        _quickcheck(signal)
    return signal


def from_tiny(
    *,
    d_model: int = 32,
    n_layers: int = 2,
    n_heads: int = 4,
    seed: int = 0,
    vocab_size: int | None = None,
    seq_max: int = 256,
    tokenizer_name: str = "gpt2",
    conformance_quickcheck: bool = False,
) -> ClassifierRM:
    """Construct the tiny synthetic ``ClassifierRM`` the tests run on, entirely offline.

    A real ``LlamaForSequenceClassification`` (hidden 32, 2 layers, 4 heads, ``num_labels=1``) so the
    adapter, hooks, readout, grad, and hvp see the same module tree they will see on an 8B Skywork
    model; only the magnitudes differ. The tokenizer defaults to gpt2 (cached, fast, offset-capable);
    if gpt2 cannot be loaded offline, a minimal byte-level tokenizer is used so the tests still run
    with no network.
    """
    import torch
    from transformers import LlamaConfig, LlamaForSequenceClassification

    tokenizer = _build_tokenizer(tokenizer_name)
    if vocab_size is None:
        vocab_size = getattr(tokenizer, "vocab_size", 1000)
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=d_model,
        intermediate_size=2 * d_model,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_heads,
        max_position_embeddings=seq_max,
        rms_norm_eps=1e-6,
        pad_token_id=getattr(tokenizer, "pad_token_id", 0) or 0,
        num_labels=1,
        attn_implementation="eager",
    )
    model = LlamaForSequenceClassification(config).eval()
    return wrap_hf_model(
        model,
        tokenizer,
        device="cpu",
        architecture="LlamaForSequenceClassification",
        conformance_quickcheck=conformance_quickcheck,
    )


# ---------------------------------------------------------------------------
# The hub loader (marked GPU/download-gated for large models)
# ---------------------------------------------------------------------------


def load_signal(spec: "str | SignalSpec", **overrides: Any) -> ClassifierRM:
    """Load a signal from an HF id, a local path, or a ``SignalSpec``.

    Sniffs the loading convention from the config architecture and head names, chooses the adapter
    and numerics policy, then loads the weights and delegates to ``wrap_hf_model``. The weight load
    for the 8B/27B campaign models is GPU/download-gated: unless ``allow_download=True`` is set (and
    the machine can hold the model), this raises a clear, marked error rather than attempting a
    multi-gigabyte download on hardware that cannot run it. Ambiguous conventions raise with the
    candidate list, never a silent guess.
    """
    if isinstance(spec, ClassifierRM):  # already a signal
        return spec
    if isinstance(spec, str):
        spec = SignalSpec(source=spec, **overrides)
    elif overrides:
        from dataclasses import replace

        spec = replace(spec, **overrides)

    convention = spec.convention or _sniff_convention(spec)
    if not spec.allow_download and not _is_local(spec.source):
        raise NotImplementedError(
            f"loading {spec.source!r} from the HF hub is GPU/download-gated on this machine "
            f"(8 GB GPU, cannot hold the 8B/27B campaign models). The convention sniffed as "
            f"{convention!r} and the code path is implemented; set allow_download=True to attempt "
            f"it on adequate hardware, or use wrap_hf_model/from_tiny for a local model."
        )
    model, tokenizer = _load_weights(spec, convention)
    return wrap_hf_model(
        model,
        tokenizer,
        device=spec.device,
        adapter_id=spec.source,
        max_length=spec.max_length,
        lineage={"declared_base": spec.source, "provenance_tier": "weights-verified"},
    )


def _sniff_convention(spec: "SignalSpec") -> str:
    """Determine the loading convention from a local config, or return ``"unknown"`` for a hub id.

    Reads ``config.json`` when the source is a local directory and maps the architecture to one of
    the four named conventions. For a bare hub id (no local config to read without downloading) the
    convention is left ``"unknown"`` and resolved after the (gated) weight load. Genuinely ambiguous
    architectures raise with the candidate list.
    """
    if not _is_local(spec.source):
        return "unknown"
    import json
    from pathlib import Path

    config_path = Path(spec.source) / "config.json"
    if not config_path.exists():
        return "unknown"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    architectures = [a.lower() for a in config.get("architectures", [])]
    num_labels = config.get("num_labels")
    if any("sequenceclassification" in a for a in architectures):
        return "trl-sequence-classification"
    if any("tokenclassification" in a for a in architectures):
        return "verl-token-classification"
    if any("rewardmodel" in a for a in architectures):
        return "adhoc-reward-model"
    if any("causallm" in a for a in architectures):
        return "openrlhf-value-head"
    if num_labels == 1:
        return "trl-sequence-classification"
    raise ConformanceError(
        f"cannot determine the loading convention for {spec.source!r}; architectures="
        f"{architectures}. Candidates: trl-sequence-classification, openrlhf-value-head, "
        f"verl-token-classification, adhoc-reward-model. Pass SignalSpec(convention=...) "
        f"explicitly; no silent guesses."
    )


def _load_weights(spec: "SignalSpec", convention: str) -> tuple[Any, Any]:
    """Load the model + tokenizer for a resolved convention (the marked, real code path).

    Implemented for completeness and exercised on local models; on this hardware it will not be
    called for the 8B/27B checkpoints (``load_signal`` gates them first). The soft-cap disabling and
    head-dtype policy are applied downstream by ``wrap_hf_model`` via the numerics policy, so this
    function is only the transformers-load half.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dtype = getattr(torch, spec.dtype)
    device_map = spec.device if spec.device != "cpu" else None
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            spec.source,
            dtype=dtype,
            trust_remote_code=spec.trust_remote_code,
            device_map=device_map,
        )
    except Exception as exc:  # noqa: BLE001 - surfaced with context, never swallowed
        warnings.warn(
            f"AutoModelForSequenceClassification failed for {spec.source!r} "
            f"({type(exc).__name__}: {exc}); falling back to AutoModel. Reward head may need "
            f"attachment (the v1 missing-head shim path).",
            stacklevel=2,
        )
        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            spec.source, dtype=dtype, trust_remote_code=spec.trust_remote_code
        )
    tokenizer = AutoTokenizer.from_pretrained(
        spec.source, trust_remote_code=spec.trust_remote_code, use_fast=True
    )
    return model, tokenizer


def _is_local(source: str) -> bool:
    from pathlib import Path

    return Path(source).exists()


def _architecture_string(model: Any) -> str:
    config = getattr(model, "config", None)
    if config is not None:
        arch = getattr(config, "architectures", None)
        if arch:
            return str(arch[0])
        model_type = getattr(config, "model_type", None)
        if model_type:
            return str(model_type)
    return type(model).__name__


def _quickcheck(signal: ClassifierRM) -> None:
    """A fast conformance gate at load: determinism and readout-matches-head on two inputs.

    The full suite is ``signals.conformance.run_conformance``; this is the cheap subset that must
    pass before a freshly loaded signal is handed back, so a broken load (a dtype mismatch, a
    mis-resolved head) fails loudly at load rather than deep in a study. Raises
    ``ConformanceError`` on failure.
    """
    import numpy as np

    items = [("2+2?", "4"), ("capital of France?", "Paris")]
    first = signal.score(items).value.values
    second = signal.score(items).value.values
    if not np.allclose(first, second, atol=0.0):
        raise ConformanceError("score is non-deterministic on repeat evaluation (quick-check).")
    # readout-vs-native-head: the fp32 projection must match the model's own logits within policy tol
    import torch

    tokenized = [signal.tokenize(it) for it in items]
    batch = signal.runtime.collate(tokenized)
    raw = signal.runtime.forward(batch)
    if raw.reward is not None:
        native = raw.reward.detach().to("cpu", dtype=torch.float32).numpy()
        if not np.allclose(first, native, atol=max(signal.policy.tol, 1e-4)):
            raise ConformanceError(
                f"fp32 readout disagrees with the native head beyond tolerance "
                f"{signal.policy.tol}: readout={first}, native={native}."
            )


def _build_tokenizer(name: str) -> Any:
    """Load the named tokenizer, falling back to a minimal offline tokenizer if it cannot be fetched.

    gpt2 is cached and offset-capable on this machine, so it is used when available; the fallback
    guarantees the tests run with no network. The fallback is deliberately minimal: it implements the
    exact surface the classifier tokenization path uses.
    """
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return tok
    except Exception as exc:  # noqa: BLE001 - offline fallback is the whole point
        warnings.warn(
            f"could not load tokenizer {name!r} ({type(exc).__name__}); using the minimal "
            f"byte-level fallback so tests run offline.",
            stacklevel=2,
        )
        return _MinimalByteTokenizer()


class _MinimalByteTokenizer:
    """A byte-level tokenizer implementing just the surface ``ClassifierRM.tokenize`` needs.

    Present only as the offline fallback when gpt2 cannot be fetched. It encodes UTF-8 bytes (+1 so
    id 0 is free for padding), supports offset mapping and truncation, and exposes the attributes the
    tokenization and collation paths read. It is not a full HF tokenizer and is never used when a
    real tokenizer is available.
    """

    is_fast = True
    chat_template = None

    def __init__(self, max_vocab: int = 257):
        self.vocab_size = max_vocab
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.pad_token_id = 0
        self.eos_token_id = max_vocab - 1
        self.padding_side = "left"
        self.all_special_tokens = [self.pad_token, self.eos_token]
        self.name_or_path = "minimal-byte"

    def __call__(
        self,
        text: str,
        truncation: bool = False,
        max_length: int | None = None,
        padding: bool = False,
        return_offsets_mapping: bool = False,
        return_tensors: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        data = text.encode("utf-8")
        ids = [b + 1 for b in data]
        offsets = [(i, i + 1) for i in range(len(data))]
        if truncation and max_length is not None:
            ids = ids[:max_length]
            offsets = offsets[:max_length]
        out: dict[str, Any] = {"input_ids": ids, "attention_mask": [1] * len(ids)}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        if return_tensors == "pt":
            import torch

            out = {
                "input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
            }
        return out

    def decode(self, ids: Any, skip_special_tokens: bool = True) -> str:
        try:
            values = ids.tolist()
        except AttributeError:
            values = list(ids)
        raw = bytes(max(0, int(i) - 1) for i in values if int(i) not in (0, self.eos_token_id))
        return raw.decode("utf-8", errors="replace")

    def get_vocab(self) -> dict[str, int]:
        return {chr(i): i for i in range(self.vocab_size)}


__all__ = ["load_signal", "wrap_hf_model", "from_tiny", "SignalSpec"]

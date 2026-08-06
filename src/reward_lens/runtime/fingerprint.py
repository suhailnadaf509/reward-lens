"""Model fingerprints and lineage.

``fingerprint`` produces the ``ModelFP`` that identifies a signal everywhere in the store: it is
the ``mfp:`` prefix on every ``SubjectRef``, the key that stops a patched-run cache from aliasing a
clean-run cache, and the axis the Atlas, kinship, and monoculture measurements are computed over.
It costs almost nothing to collect at load and is impossible to reconstruct afterwards, which is
why it is mandatory rather than optional (RK9).

The digest hashes four things: the weight content, the normalized config JSON, the tokenizer
identity, and the adapter id. Weight content is streamed, never materialized whole: for a model
whose safetensors live on disk (the 8B campaign case) the files are hashed block by block off disk;
for a tiny in-memory model with no files, the ``state_dict`` is serialized one tensor at a time.
Both paths use ``reward_lens.core.hash_bytes``/``content_hash`` so the id format matches the rest of
the kernel.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from reward_lens.core.types import ModelFP, content_hash, hash_bytes

if TYPE_CHECKING:
    import torch

# Config keys that vary run to run or machine to machine and must not enter the identity. Dropping
# them is what lets two loads of the same checkpoint on different boxes fingerprint identically.
_VOLATILE_CONFIG_KEYS = frozenset(
    {
        "_name_or_path",
        "transformers_version",
        "torch_dtype",
        "use_cache",
        "_attn_implementation",
        "device_map",
        "name_or_path",
    }
)

# Read weight files in 8 MiB blocks so an 8B checkpoint never lands in RAM to be hashed.
_STREAM_BLOCK = 8 * 1024 * 1024


def _hash_weights_from_disk(local_dir: Path) -> str | None:
    """Stream-hash the safetensors shards under ``local_dir`` in sorted order.

    Returns a hex digest, or ``None`` if the directory holds no safetensors (the caller then falls
    back to the state-dict path). Shards are read in filename order and in fixed-size blocks, so the
    peak memory is one block regardless of checkpoint size. This is the 8B path; it reads the full
    weight bytes off disk (gigabytes) but holds only a block at a time.
    """
    shards = sorted(local_dir.glob("*.safetensors"))
    if not shards:
        return None
    h = hashlib.blake2b(digest_size=16)
    for shard in shards:
        h.update(shard.name.encode("utf-8"))
        with shard.open("rb") as fh:
            while True:
                block = fh.read(_STREAM_BLOCK)
                if not block:
                    break
                h.update(block)
    return h.hexdigest()


def _hash_weights_from_state_dict(model: "torch.nn.Module") -> str:
    """Serialize and hash a model's ``state_dict`` one tensor at a time (the in-memory path).

    Used for the tiny synthetic models that have no files on disk. Each parameter is moved to CPU,
    made contiguous, and serialized with safetensors (which handles bf16/fp16 losslessly, unlike
    ``numpy().tobytes()``), then folded into a running digest along with its name, dtype, and shape.
    Serializing per tensor rather than the whole dict keeps the transient allocation to one tensor.
    """
    import safetensors.torch as st
    import torch

    h = hashlib.blake2b(digest_size=16)
    sd = model.state_dict()
    for name in sorted(sd.keys()):
        tensor = sd[name]
        if not hasattr(tensor, "detach"):
            continue
        cpu = tensor.detach().to("cpu").contiguous()
        header = f"{name}|{cpu.dtype}|{tuple(cpu.shape)}".encode("utf-8")
        h.update(header)
        # safetensors requires at least one tensor; save this one to a byte buffer and fold it in.
        try:
            h.update(st.save({"t": cpu}))
        except (ValueError, RuntimeError):
            # A rare dtype safetensors will not serialize (e.g. an integer buffer view); fall back
            # to the raw storage bytes, which are still deterministic for identity purposes.
            h.update(bytes(cpu.flatten().to("cpu").view(torch.uint8).numpy().tobytes()))
    return h.hexdigest()


def _resolve_local_dir(model: "torch.nn.Module") -> Path | None:
    """Best-effort resolution of a local weights directory from the model config."""
    cfg = getattr(model, "config", None)
    for attr in ("_name_or_path", "name_or_path"):
        candidate = getattr(cfg, attr, None) if cfg is not None else None
        if candidate:
            path = Path(str(candidate))
            if path.is_dir():
                return path
    return None


def _hash_config(model: "torch.nn.Module") -> str:
    """Hash the model config as normalized JSON, dropping volatile keys."""
    cfg = getattr(model, "config", None)
    if cfg is None:
        return "no-config"
    try:
        raw = cfg.to_dict()
    except (AttributeError, TypeError):
        raw = {k: v for k, v in vars(cfg).items() if not k.startswith("_")}
    normalized = {k: v for k, v in raw.items() if k not in _VOLATILE_CONFIG_KEYS}
    # json with a string fallback for anything not natively serializable (e.g. nested config
    # objects) so the hash is stable and never raises on an exotic config value.
    blob = json.dumps(normalized, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.blake2b(blob.encode("utf-8"), digest_size=16).hexdigest()


def _hash_tokenizer(tokenizer: Any) -> str:
    """Hash the tokenizer identity: class, vocab size, specials, and chat template.

    Full tokenizer-file hashing is available when the files are on disk, but the identity that
    actually matters for a reward signal is the vocabulary and the template that turns a pair into
    tokens; those are what change a score. Hashing them (plus the class name) is stable and cheap.
    """
    if tokenizer is None:
        return "no-tokenizer"
    parts: dict[str, Any] = {"class": type(tokenizer).__name__}
    for attr in ("vocab_size", "name_or_path", "padding_side"):
        parts[attr] = getattr(tokenizer, attr, None)
    specials = getattr(tokenizer, "all_special_tokens", None)
    if specials is not None:
        parts["special_tokens"] = sorted(str(t) for t in specials)
    template = getattr(tokenizer, "chat_template", None)
    parts["chat_template"] = template if isinstance(template, str) else None
    blob = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.blake2b(blob.encode("utf-8"), digest_size=16).hexdigest()


def fingerprint(model: "torch.nn.Module", tokenizer: Any = None, adapter_id: str = "") -> ModelFP:
    """Compute the content-derived ``ModelFP`` for a loaded model.

    The digest folds the weight-content hash (streamed from disk when the safetensors are present,
    else serialized from the ``state_dict``), the normalized config hash, the tokenizer-identity
    hash, and the adapter id into one ``mfp:`` id via ``content_hash``. Two loads of the same
    checkpoint with the same tokenizer and adapter produce the same id on any machine; a changed
    weight, config field, tokenizer, or adapter changes it. Never raises on a well-formed model.
    """
    local_dir = _resolve_local_dir(model)
    weights = _hash_weights_from_disk(local_dir) if local_dir is not None else None
    if weights is None:
        weights = _hash_weights_from_state_dict(model)
    material = {
        "weights": weights,
        "config": _hash_config(model),
        "tokenizer": _hash_tokenizer(tokenizer),
        "adapter": adapter_id or "",
    }
    return ModelFP(content_hash(material, "mfp"))


def fingerprint_bytes(data: bytes, prefix: str = "mfp") -> ModelFP:
    """Fingerprint raw bytes (used by tests and by callers that already have a serialized blob)."""
    return ModelFP(hash_bytes(data, prefix))


__all__ = ["fingerprint", "fingerprint_bytes"]

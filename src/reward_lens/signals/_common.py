"""Shared machinery for the v3 signal adapters.

``ClassifierRM`` is the template. The seven remaining adapters (judge, process, implicit,
rubric, trajectory, dense, ensemble) implement the same ``RewardSignal`` protocol and must reuse the
same runtime, the same fp32 head-projection numerics, the same left-padded batching, the same
span carry-through, and the same Evidence assembly. Rather than copy ``ClassifierRM``'s hardening
into each file, that hardening lives here once.

``SignalImplBase`` owns everything an adapter shares: readout lookup, tokenization with
character-to-token span mapping, capture, intervention wrapping, and Evidence assembly. A subclass
overrides ``_render`` to control how its item type becomes text (a judge builds a verdict prompt, a
trajectory renders its steps) and implements ``score`` / ``score_prefixes`` with its own readout
semantics. ``build_hf_runtime`` is the loader half: it builds an ``HFRuntime`` whose head-capture is
pointed at an arbitrary head module (a ``score`` head for the classifier-style adapters, the
``lm_head`` for the generative ones), which is the one thing ``signals.loaders.wrap_hf_model`` does
not parameterize because it only ever wraps a sequence classifier.

This is an internal module (leading underscore): the public surface is the adapters themselves and
``signals.loaders``. Nothing here is a frozen interface.
"""

from __future__ import annotations

import contextlib
import copy
import time
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np

from reward_lens.core.evidence import Evidence, Uncertainty, make_evidence
from reward_lens.core.gates import CalibrationRef
from reward_lens.core.provenance import Cost, capture_provenance
from reward_lens.core.types import (
    Capability,
    GaugeStatus,
    ModelFP,
    Site,
    Span,
    SubjectRef,
)
from reward_lens.signals.base import (
    Readout,
    SignalMeta,
    TokenizedInput,
)

if TYPE_CHECKING:
    import torch

    from reward_lens.runtime.backend import CaptureHandle, CaptureSpec
    from reward_lens.runtime.hf import HFRuntime
    from reward_lens.runtime.precision import NumericsPolicy

_OBS_VERSION = "1"


# ---------------------------------------------------------------------------
# Item normalization
# ---------------------------------------------------------------------------


def split_item(item: Any) -> tuple[str, str, bool]:
    """Normalize a data item into ``(prompt, response, raw)`` (mirrors ``classifier._split_item``).

    Accepts a ``(prompt, response)`` pair, a mapping with ``prompt``/``response`` (or ``text``), a
    raw string, or an object exposing ``.prompt``/``.chosen``/``.response``/``.text``. ``raw`` is
    True when the text is tokenized as-is (no template). Kept here so every adapter normalizes items
    the same way before the typed ``DataView`` is universal.
    """
    if isinstance(item, str):
        return "", item, True
    if isinstance(item, (tuple, list)) and len(item) == 2:
        return str(item[0]), str(item[1]), False
    if isinstance(item, dict):
        if "text" in item and "response" not in item:
            return str(item.get("prompt", "")), str(item["text"]), bool(item.get("raw", True))
        return str(item.get("prompt", "")), str(item.get("response", "")), False
    prompt = getattr(item, "prompt", "")
    response = getattr(item, "response", None)
    if response is None:
        response = getattr(item, "chosen", None) or getattr(item, "text", "")
    response = getattr(response, "text", response)
    return str(prompt), str(response), False


def item_char_spans(item: Any) -> tuple[tuple[int, int, str], ...]:
    """Extract explicit ``(char_start, char_end, kind)`` spans from an item, if any."""
    if isinstance(item, dict) and "spans" in item:
        out = []
        for span in item["spans"]:
            if isinstance(span, (tuple, list)) and len(span) >= 2:
                kind = span[2] if len(span) > 2 else "text"
                out.append((int(span[0]), int(span[1]), str(kind)))
        return tuple(out)
    return ()


def single_token_id(tokenizer: Any, word: str) -> int:
    """Resolve a verdict word ("Yes", "A", "5") to a single vocabulary id.

    Encodes the word and returns the last content id, which drops a leading BOS if the tokenizer
    adds one. Used to read the ``W_U[Yes] - W_U[No]`` direction off the unembedding for a generative
    judge. Raises if the word does not encode to at least one token.
    """
    ids = tokenizer.encode(word) if hasattr(tokenizer, "encode") else tokenizer(word)["input_ids"]
    ids = list(ids)
    if not ids:
        raise ValueError(f"verdict token {word!r} did not encode to any id")
    return int(ids[-1])


# ---------------------------------------------------------------------------
# Runtime construction (the head-module-agnostic loader half)
# ---------------------------------------------------------------------------


def build_site_map(adapter: Any, model: "torch.nn.Module", d_model: int) -> Any:
    """Resolve every logical ``Site`` an architecture exposes to a module path.

    This mirrors ``signals.adapters.build_site_map`` but takes ``d_model`` explicitly rather than
    reading it off a reward-head weight, because a generative signal's model has an ``lm_head``, not
    a scalar ``score`` head, so the reward-head probe returns nothing. The module walk is identical:
    the adapter's own navigation, resolved to qualified names by identity against ``named_modules``.
    """
    from reward_lens.runtime.backend import SiteMap

    name_by_id = {id(module): name for name, module in model.named_modules()}

    def path_of(module: Any) -> str | None:
        return None if module is None else name_by_id.get(id(module))

    paths: dict[Site, str] = {}
    layers = adapter.get_layers(model)
    n_layers = len(layers)
    for layer_idx, layer in enumerate(layers):
        layer_path = path_of(layer)
        if layer_path is not None:
            paths[Site(layer_idx, "resid_post")] = layer_path
            paths[Site(layer_idx, "resid_pre")] = layer_path
        attn_path = path_of(adapter.get_attn_module(layer))
        if attn_path is not None:
            paths[Site(layer_idx, "attn_out")] = attn_path
        mlp_path = path_of(adapter.get_mlp_module(layer))
        if mlp_path is not None:
            paths[Site(layer_idx, "mlp_out")] = mlp_path
        o_proj_path = path_of(adapter.get_attn_o_proj(layer))
        if o_proj_path is not None:
            paths[Site(layer_idx, "head_out", None)] = o_proj_path
    embed_path = path_of(adapter.get_embedding(model))
    if embed_path is not None:
        paths[Site(-1, "embed")] = embed_path
    return SiteMap(
        module_paths=paths,
        n_layers=n_layers,
        d_model=int(d_model),
        n_heads=int(adapter.n_heads(model)),
    )


def build_hf_runtime(
    model: "torch.nn.Module",
    tokenizer: Any,
    head_module: "torch.nn.Module",
    *,
    adapter: Any = None,
    adapter_id: str = "",
    architecture: str | None = None,
    numerics: Any = None,
    device: str = "cpu",
    lineage: dict[str, Any] | None = None,
) -> tuple["HFRuntime", SignalMeta, "NumericsPolicy"]:
    """Build an ``HFRuntime`` + ``SignalMeta`` + policy for a model with an explicit head module.

    The one thing this parameterizes that ``wrap_hf_model`` does not is ``head_module``: the runtime
    installs its forward pre-hook on exactly this module to capture the tensor it consumes (the
    ``score`` head input for a classifier-style adapter, the ``lm_head`` input for a generative one),
    which is what the fp32 readout projects. Everything else (adapter resolution, numerics
    policy, soft-cap disabling, fingerprint, pad token) is the same boundary work ``wrap_hf_model``
    does. Returns the pieces an adapter's constructor assembles into a signal.
    """
    import torch

    from reward_lens.runtime.fingerprint import fingerprint
    from reward_lens.runtime.hf import HFRuntime
    from reward_lens.runtime.precision import resolve_policy
    from reward_lens.signals.adapters import resolve_adapter

    model.eval()
    torch_device = torch.device(device)
    if adapter is None:
        adapter = resolve_adapter(model, adapter_id)
    adapter_name = type(adapter).__name__
    arch = architecture or _architecture_string(model)
    policy = numerics if numerics is not None else resolve_policy(arch)

    disabled = policy.apply_to_config(getattr(model, "config", None))
    soft_cap = next(iter(disabled.values()), None) if disabled else None

    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None):
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = getattr(tokenizer, "pad_token_id", None)
    pad_id = int(pad_id) if pad_id is not None else 0

    d_model = int(head_module.weight.shape[-1])
    site_map = build_site_map(adapter, model, d_model)
    fp = fingerprint(model, tokenizer, adapter_name)

    runtime = HFRuntime(
        model=model,
        adapter=adapter,
        site_map=site_map,
        policy=policy,
        head_module=head_module,
        device=torch_device,
        pad_id=pad_id,
    )
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
    return runtime, meta, policy


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


# ---------------------------------------------------------------------------
# The shared protocol implementation base
# ---------------------------------------------------------------------------


class SignalImplBase:
    """Shared implementation of the ``RewardSignal`` protocol machinery.

    Holds the runtime, meta, numerics policy, tokenizer, readouts, declared capabilities, and any
    mounted interventions. Provides the protocol methods every adapter shares (``readouts``,
    ``tokenize``, ``capture``, ``with_interventions``) plus the fp32 pooling helpers and Evidence
    assembly. A subclass supplies its ``observable`` prefix, overrides ``_render`` for its item type,
    and implements ``score`` / ``score_prefixes``.
    """

    #: Observable-name prefix, e.g. "signals.judge"; subclasses override.
    observable_prefix: str = "signals.adapter"

    def __init__(
        self,
        *,
        runtime: "HFRuntime",
        meta: SignalMeta,
        policy: "NumericsPolicy",
        tokenizer: Any,
        readouts: Sequence[Readout],
        caps: Capability,
        max_length: int = 2048,
        default_batch_size: int = 16,
        interventions: tuple[Any, ...] = (),
        add_generation_prompt: bool = False,
    ) -> None:
        self.runtime = runtime
        self.meta = meta
        self.policy = policy
        self.tokenizer = tokenizer
        self.caps = caps
        self.max_length = max_length
        self.default_batch_size = default_batch_size
        self._readouts = list(readouts)
        self._readout_by_name = {r.name: r for r in self._readouts}
        self._interventions = interventions
        self._add_generation_prompt = add_generation_prompt

    # -- readouts -----------------------------------------------------------

    def readouts(self) -> list[Readout]:
        """The readouts this signal exposes."""
        return list(self._readouts)

    def readout(self, name: str) -> Readout:
        """Look up a readout by name, with a helpful error listing the candidates."""
        if name in self._readout_by_name:
            return self._readout_by_name[name]
        raise KeyError(f"unknown readout {name!r}; available: {sorted(self._readout_by_name)}")

    def default_readout_name(self) -> str:
        """The name a bare ``score(view)`` call resolves to (the first declared readout)."""
        return self._readouts[0].name

    # -- tokenization (span carry-through) ---------------------------------

    def _render(self, item: Any) -> tuple[str, tuple[tuple[int, int, str], ...], dict[str, Any]]:
        """Render an item to ``(text, char_spans, meta)``.

        The default is the classifier rendering: a ``user``/``assistant`` chat turn, or the raw text
        for a bare string. A subclass overrides this to build its own text (a judge's verdict prompt,
        a trajectory's rendered steps) while inheriting the encoding and span-mapping below.
        """
        prompt, response, raw = split_item(item)
        text = response if raw else self._chat(prompt, response)
        meta = {"prompt": prompt, "response": response}
        return text, item_char_spans(item), meta

    def _chat(
        self,
        prompt: str,
        response: str | None,
        *,
        add_generation_prompt: bool | None = None,
    ) -> str:
        """Apply the chat template, or a plain ``User:``/``Assistant:`` fallback.

        ``response is None`` with ``add_generation_prompt=True`` produces a prompt that ends exactly
        where the model would begin its answer, which is the judgment position a generative judge
        reads. Mirrors ``ClassifierRM._template`` for the response-present case so a
        classifier and a judge tokenize a shared prefix identically.
        """
        gen = (
            self._add_generation_prompt if add_generation_prompt is None else add_generation_prompt
        )
        if self.tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            if response is not None:
                messages.append({"role": "assistant", "content": response})
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=gen and response is None
            )
        if response is None:
            tail = "\nAssistant:" if gen else "\nAssistant"
            return f"User: {prompt}{tail}"
        return f"User: {prompt}\nAssistant: {response}"

    def tokenize(self, item: Any) -> TokenizedInput:
        """Tokenize an item, carrying character-to-token offsets and typed spans.

        Delegates rendering to ``_render`` (adapter-specific) and owns the load-bearing part: request
        offset mapping from the fast tokenizer, then map any character spans on the item into token
        coordinates so span-level patching and attribution stay exact. Identical in spirit to
        ``ClassifierRM.tokenize``; the only difference is the pluggable renderer.
        """
        text, char_spans, meta = self._render(item)
        enc = self._encode(text)
        input_ids = list(enc["input_ids"])
        attention_mask = list(enc.get("attention_mask", [1] * len(input_ids)))
        offsets: tuple[tuple[int, int], ...] = ()
        raw_offsets = enc.get("offset_mapping")
        if raw_offsets is not None:
            offsets = tuple((int(a), int(b)) for a, b in raw_offsets)
        spans = map_char_spans(offsets, char_spans)
        return TokenizedInput(
            input_ids=input_ids,
            attention_mask=attention_mask,
            text=text,
            token_offsets=offsets,
            spans=spans,
            meta=meta,
        )

    def _encode(self, text: str) -> dict[str, Any]:
        want_offsets = bool(getattr(self.tokenizer, "is_fast", False))
        try:
            enc = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_offsets_mapping=want_offsets,
            )
        except (TypeError, NotImplementedError):
            enc = self.tokenizer(text, truncation=True, max_length=self.max_length, padding=False)
        return dict(enc)

    # -- fp32 pooling helpers (used by the linear/logit_diff adapters) ------

    def _final_pooled(self, tokenized: Sequence[TokenizedInput]) -> "torch.Tensor":
        """Head-input hidden states at the final valid token, one row per item (trunk dtype)."""
        with self._mounted():
            return self.runtime.final_head_inputs(tokenized, batch_size=self.default_batch_size)

    def project_final(
        self, tokenized: Sequence[TokenizedInput], vector: "torch.Tensor", bias: float = 0.0
    ) -> np.ndarray:
        """Project the final head-input onto a readout direction in fp32, as a numpy array."""
        import torch

        pooled = self._final_pooled(tokenized)
        values = self.policy.head_project(pooled, vector, bias)
        return values.detach().to("cpu", dtype=torch.float32).numpy()

    def linear_prefix_curves(
        self, tokenized: Sequence[TokenizedInput], vector: "torch.Tensor", bias: float = 0.0
    ) -> tuple[list[np.ndarray], int]:
        """Per-token reward curves for a linear/logit_diff readout.

        Mirrors ``ClassifierRM.score_prefixes``: the head input at every valid position is projected
        onto the readout vector in one forward, and because a causal model pools under causal
        attention the value at position ``t`` is the score of the prefix ending at ``t``. The final
        entry of each curve therefore equals the scalar ``score`` for that item, which conformance
        asserts. Returns ``(curves, n_tokens)``.
        """
        import torch

        curves: list[np.ndarray] = [np.empty(0)] * len(tokenized)
        n_tokens = 0
        batch = self.default_batch_size
        items = list(tokenized)
        with self._mounted():
            for start in range(0, len(items), batch):
                sub = items[start : start + batch]
                token_batch = self.runtime.collate(sub)
                head_input, valid_per_item = self.runtime.full_head_inputs(token_batch)
                for local_i, valid in enumerate(valid_per_item):
                    idx = torch.tensor(valid, device=head_input.device, dtype=torch.long)
                    rows = head_input[local_i].index_select(0, idx)
                    curve = self.policy.head_project(rows, vector, bias)
                    curves[start + local_i] = curve.detach().to("cpu", dtype=torch.float32).numpy()
                    n_tokens += len(valid)
        return curves, n_tokens

    def full_logits(
        self, tokenized: Sequence[TokenizedInput]
    ) -> tuple["torch.Tensor", list[list[int]]]:
        """The full-sequence LM logits and per-item valid positions (for simplex/log-ratio readouts).

        Runs one forward over a left-padded batch and returns ``(B, T, vocab)`` logits plus the valid
        (non-pad) token indices per row. Used by readouts that need a softmax over vocabulary (a
        Likert simplex, an implicit-reward log-ratio), which a single projection cannot express.
        """
        batch = self.runtime.collate(list(tokenized))
        with self._mounted():
            raw = self.runtime.forward(batch)
        if raw.logits is None:
            raise RuntimeError(
                "the model produced no logits; a simplex/log-ratio readout needs LM logits. This "
                "signal wraps a model without an lm_head."
            )
        valid = _valid_positions(batch)
        return raw.logits, valid

    # -- capture ------------------------------------------------------------

    def capture(self, view: Any, spec: "CaptureSpec") -> "CaptureHandle":
        """Capture activations at the spec's sites, returning a ``CaptureHandle``.

        Identical to ``ClassifierRM.capture``: collate the view into one left-padded batch and run
        ``forward_with_capture`` under any mounted interventions. The store-backed streaming path is
        ``runtime.store.ActivationStore.get_or_compute``; this is the direct single-batch path.
        """
        from reward_lens.runtime.store import InMemoryCaptureHandle

        tokenized = [self.tokenize(it) for it in view]
        token_batch = self.runtime.collate(tokenized)
        with self._mounted():
            _raw, capture = self.runtime.forward_with_capture(token_batch, spec)
        return InMemoryCaptureHandle(capture)

    # -- interventions ------------------------------------------------------

    def with_interventions(self, *ivs: Any) -> "SignalImplBase":
        """Return a shallow clone with additional interventions mounted.

        Each intervention is compiled against this signal and appended to the mount list; the
        intervention fingerprints become part of every Evidence subject, so an intervened score can
        never masquerade as a clean one. A shallow copy shares the runtime, readouts, and tokenizer
        (all immutable for scoring), so this is cheap and every subclass inherits it unchanged.
        """
        compiled = tuple(iv.compile(self) if hasattr(iv, "compile") else iv for iv in ivs)
        clone = copy.copy(self)
        clone._interventions = self._interventions + compiled
        return clone

    def _mounted(self):
        if not self._interventions:
            return contextlib.nullcontext()
        return self.runtime.mounted(self._interventions)

    def _intervention_fingerprints(self) -> tuple[str, ...]:
        out: list[str] = []
        for iv in self._interventions:
            fp = getattr(iv, "fingerprint", None)
            out.append(fp() if callable(fp) else str(fp) if fp is not None else str(iv))
        return tuple(out)

    # -- evidence assembly --------------------------------------------------

    def _evidence(
        self,
        *,
        observable: str,
        value: Any,
        readout: str,
        n: int,
        n_tokens: int,
        wall: float,
        gauge: GaugeStatus = GaugeStatus.INVARIANT,
        calibration: CalibrationRef | None = None,
        signals: tuple[ModelFP, ...] | None = None,
        subject_extra: dict[str, Any] | None = None,
    ) -> Evidence[Any]:
        """Assemble an ``Evidence`` with a signal-fingerprinted subject and metered provenance.

        Trust is never passed: with ``calibration=None`` the gates cap it at EXPLORATORY (gate 1),
        which is the correct level for every raw adapter score until a scorecard entry exists. The
        subject names the signal(s) by fingerprint and carries the readout and any interventions.
        """
        subject = SubjectRef(
            signals=signals if signals is not None else (self.meta.fingerprint,),
            readout=readout,
            interventions=self._intervention_fingerprints(),
            extra=subject_extra or {},
        )
        provenance = capture_provenance(cost=Cost(tokens=n_tokens, wall_seconds=wall))
        return make_evidence(
            observable=observable,
            observable_version=_OBS_VERSION,
            subject=subject,
            value=value,
            uncertainty=Uncertainty(n=n, method="none"),
            gauge=gauge,
            calibration=calibration,
            provenance=provenance,
        )

    def _timed_evidence(
        self,
        observable_suffix: str,
        value: Any,
        readout: str,
        n: int,
        n_tokens: int,
        started: float,
        **kw: Any,
    ) -> Evidence[Any]:
        """Convenience: assemble Evidence with the adapter's observable prefix and a wall time."""
        return self._evidence(
            observable=f"{self.observable_prefix}.{observable_suffix}",
            value=value,
            readout=readout,
            n=n,
            n_tokens=n_tokens,
            wall=time.perf_counter() - started,
            **kw,
        )

    # -- grad support -------------------------------------------------------

    def reward_scalar_fn(self, readout: str) -> Callable[[Any], "torch.Tensor"]:
        """A ``ScalarFn`` projecting the head input onto a linear/logit_diff readout (for grad/hvp).

        Mirrors ``ClassifierRM.reward_scalar_fn``: reads the grad-attached head input from
        ``RawOutput.extra`` and projects it onto the readout vector in fp32, pooled at the resolved
        final positions. Only defined for ``linear`` and ``logit_diff`` readouts (a simplex or a
        log-ratio has no single direction); raises for the others.
        """
        import torch

        read = self.readout(readout)
        if read.kind not in ("linear", "logit_diff") or read.vector is None:
            raise ValueError(
                f"reward_scalar_fn is only defined for linear/logit_diff readouts with a vector; "
                f"readout {readout!r} is kind {read.kind!r}."
            )
        weight = read.vector
        bias = float(read.meta.get("bias", 0.0))

        def scalar_fn(raw: Any) -> "torch.Tensor":
            head_input = raw.extra["head_input"]
            final_pos = raw.extra["final_pos"]
            batch_idx = torch.arange(head_input.shape[0], device=head_input.device)
            pooled = head_input[batch_idx, final_pos]
            return self.policy.head_project(pooled, weight, bias)

        return scalar_fn


# ---------------------------------------------------------------------------
# span mapping and batch helpers (module-level so ImplicitRM can reuse them)
# ---------------------------------------------------------------------------


def map_char_spans(
    offsets: tuple[tuple[int, int], ...],
    char_spans: tuple[tuple[int, int, str], ...],
) -> tuple[Span, ...]:
    """Map character spans to token ``Span`` objects using the offset mapping."""
    if not offsets or not char_spans:
        return ()
    out: list[Span] = []
    for c0, c1, kind in char_spans:
        token_idxs = [
            i
            for i, (a, b) in enumerate(offsets)
            if b > a and a < c1 and b > c0  # any overlap
        ]
        if token_idxs:
            out.append(Span(start=token_idxs[0], end=token_idxs[-1] + 1, kind=kind))
    return tuple(out)


def _valid_positions(batch: Any) -> list[list[int]]:
    """Per-row valid (non-pad) token indices in the padded coordinates of a ``TokenBatch``."""
    tokenized = batch.meta.get("tokenized")
    offsets = batch.meta.get("offsets")
    if tokenized is not None and offsets is not None:
        return [[off + i for i in tok.valid_positions()] for tok, off in zip(tokenized, offsets)]
    return [[i for i, m in enumerate(row) if m] for row in batch.attention_mask.tolist()]


__all__ = [
    "SignalImplBase",
    "build_hf_runtime",
    "build_site_map",
    "split_item",
    "item_char_spans",
    "single_token_id",
    "map_char_spans",
]

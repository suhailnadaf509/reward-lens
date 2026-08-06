"""``ClassifierRM``: the v1 ``RewardModel`` rebuilt behind the ``RewardSignal`` protocol.

This is the first and most common signal adapter: a sequence-classification reward model whose head
is a single linear map from the final hidden state to a scalar. It keeps v1's hardening (the adapter
navigation, left-padded batching, missing-head attachment upstream in the loader) and moves the
readout to a first-class ``Readout`` object: the reward direction ``w_r`` is read off the
checkpoint into a ``linear`` readout at the final residual, and every score is the fp32 projection of
the head-input hidden state onto that direction, not the trunk-dtype head output.

Three protocol methods carry the weight. ``score`` returns ``Evidence[Scores]`` (INVARIANT gauge,
EXPLORATORY trust: raw scores are gauge-free, and nothing is calibrated yet). ``score_prefixes``
returns the per-token reward curve ``r(y_{1:t})`` in a single forward, exploiting that a classifier
pools the last token under causal attention so the head input at position ``t`` is the score of the
prefix ending at ``t``. ``capture`` returns a handle to the residual/attention/MLP activations.
Multi-row heads (ArmoRM's nineteen objectives, QRM's quantiles) are surfaced as separate criterion
readouts, never collapsed to a row mean by default (the row mean is a documented legacy option).
"""

from __future__ import annotations

import contextlib
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from reward_lens.core.evidence import Evidence, Uncertainty, make_evidence
from reward_lens.core.provenance import Cost, capture_provenance
from reward_lens.core.types import Capability, GaugeStatus, Site, SubjectRef
from reward_lens.signals.adapters import (
    capabilities_for,
    reward_head_module,
)
from reward_lens.signals.base import (
    PositionSpec,
    Readout,
    Scores,
    SignalMeta,
    TokenCurves,
    TokenizedInput,
)

if TYPE_CHECKING:
    import torch

    from reward_lens.runtime.backend import CaptureHandle, CaptureSpec
    from reward_lens.runtime.hf import HFRuntime
    from reward_lens.runtime.precision import NumericsPolicy

_OBS_VERSION = "1"


def _split_item(item: Any) -> tuple[str, str, bool]:
    """Normalize a data item into ``(prompt, response, raw)``.

    Accepts a ``(prompt, response)`` pair, a mapping with ``prompt``/``response`` (or ``text``), a
    raw string, or an object exposing ``.prompt``/``.chosen``/``.response``/``.text``. ``raw`` is
    True when the text should be tokenized as-is (no chat template), which is how a single string is
    treated. This keeps signals usable before the data plane lands the typed ``DataView``.
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
    # a Response-like object with `.text`
    response = getattr(response, "text", response)
    return str(prompt), str(response), False


def _item_spans(item: Any) -> tuple[tuple[int, int, str], ...]:
    """Extract explicit character spans (over the templated text) from an item, if any."""
    if isinstance(item, dict) and "spans" in item:
        out = []
        for span in item["spans"]:
            if isinstance(span, (tuple, list)) and len(span) >= 2:
                kind = span[2] if len(span) > 2 else "text"
                out.append((int(span[0]), int(span[1]), str(kind)))
        return tuple(out)
    return ()


class ClassifierRM:
    """A sequence-classification reward model as a ``RewardSignal``.

    Build it through ``signals.loaders.wrap_hf_model`` / ``from_tiny`` (which resolve the adapter,
    site map, head, policy, and fingerprint); direct construction is for the loader and tests. The
    signal holds a ``HFRuntime``, the numerics policy, the readouts read off the head, and the
    tokenizer. ``caps`` declares its capabilities; ``meta`` carries the fingerprint, lineage,
    template, and numerics policy.
    """

    def __init__(
        self,
        *,
        runtime: "HFRuntime",
        meta: SignalMeta,
        readouts: list[Readout],
        tokenizer: Any,
        policy: "NumericsPolicy",
        max_length: int = 2048,
        default_batch_size: int = 16,
        interventions: tuple[Any, ...] = (),
        legacy_row_mean: Readout | None = None,
    ):
        self.runtime = runtime
        self.meta = meta
        self.caps: Capability = capabilities_for(runtime.adapter)
        self.tokenizer = tokenizer
        self.policy = policy
        self.max_length = max_length
        self.default_batch_size = default_batch_size
        self._readouts = readouts
        self._readout_by_name = {r.name: r for r in readouts}
        self._interventions = interventions
        self._legacy_row_mean = legacy_row_mean

    # -- readouts -----------------------------------------------------------

    def readouts(self) -> list[Readout]:
        """The readouts this signal exposes.

        A single-row head exposes one ``reward`` readout. A multi-row head exposes one
        ``criterion:k`` readout per row plus a ``reward`` composite; the row-mean aggregate is
        available separately as a documented legacy readout, never the default.
        """
        return list(self._readouts)

    def readout(self, name: str = "reward") -> Readout:
        """Look up a readout by name, with a helpful error listing the candidates."""
        if name in self._readout_by_name:
            return self._readout_by_name[name]
        if self._legacy_row_mean is not None and name == self._legacy_row_mean.name:
            return self._legacy_row_mean
        raise KeyError(
            f"unknown readout {name!r}; available: {sorted(self._readout_by_name)}"
            + (
                f" (legacy: {self._legacy_row_mean.name})"
                if self._legacy_row_mean is not None
                else ""
            )
        )

    def reward_scalar_fn(self, readout: str = "reward"):
        """Return a ``ScalarFn`` computing the readout scalar from a ``RawOutput`` (for grad/hvp).

        The closure reads the grad-attached head input from ``RawOutput.extra`` and projects it onto
        the readout vector in fp32, pooling at the resolved final positions. Passed straight into
        ``runtime.grad``/``runtime.hvp``; that is how the Hessian test differentiates the reward.
        """
        import torch

        read = self.readout(readout)
        weight = read.vector
        bias = float(read.meta.get("bias", 0.0))

        def scalar_fn(raw: Any) -> "torch.Tensor":
            head_input = raw.extra["head_input"]
            final_pos = raw.extra["final_pos"]
            batch_idx = torch.arange(head_input.shape[0], device=head_input.device)
            pooled = head_input[batch_idx, final_pos]
            return self.policy.head_project(pooled, weight, bias)

        return scalar_fn

    # -- tokenization (span carry-through) ---------------------------------

    def tokenize(self, item: Any) -> TokenizedInput:
        """Tokenize a data item, carrying character-to-token offsets and typed spans.

        Applies the model's chat template when it has one (else a plain ``User:``/``Assistant:``
        fallback, which is what the gpt2-tokenizer tiny model uses), and requests offset mapping from
        the fast tokenizer so span-level patching and attribution stay exact. Explicit character
        spans on the item (over the templated text) are mapped into token coordinates and carried on
        the ``TokenizedInput``; this is the load-bearing, unglamorous part of the protocol.
        """
        prompt, response, raw = _split_item(item)
        text = response if raw else self._template(prompt, response)

        offsets: tuple[tuple[int, int], ...] = ()
        enc = self._encode(text)
        input_ids = list(enc["input_ids"])
        attention_mask = list(enc.get("attention_mask", [1] * len(input_ids)))
        raw_offsets = enc.get("offset_mapping")
        if raw_offsets is not None:
            offsets = tuple((int(a), int(b)) for a, b in raw_offsets)

        spans = self._map_char_spans(offsets, _item_spans(item))
        return TokenizedInput(
            input_ids=input_ids,
            attention_mask=attention_mask,
            text=text,
            token_offsets=offsets,
            spans=spans,
            meta={"prompt": prompt, "response": response},
        )

    def _template(self, prompt: str, response: str) -> str:
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        if self.tokenizer.chat_template is not None:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        return f"User: {prompt}\nAssistant: {response}"

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

    @staticmethod
    def _map_char_spans(
        offsets: tuple[tuple[int, int], ...],
        char_spans: tuple[tuple[int, int, str], ...],
    ) -> tuple[Any, ...]:
        """Map character spans to token ``Span`` objects using the offset mapping."""
        from reward_lens.core.types import Span

        if not offsets or not char_spans:
            return ()
        out: list[Span] = []
        for c0, c1, kind in char_spans:
            token_idxs = [
                i
                for i, (a, b) in enumerate(offsets)
                if b > a and a < c1 and b > c0  # any overlap with [c0, c1)
            ]
            if token_idxs:
                out.append(Span(start=token_idxs[0], end=token_idxs[-1] + 1, kind=kind))
        return tuple(out)

    # -- scoring ------------------------------------------------------------

    def score(self, view: Any, readout: str = "reward") -> Evidence[Scores]:
        """Score every item under a readout, returning ``Evidence[Scores]``.

        Each score is the fp32 projection of the item's final head-input hidden state onto the
        readout vector. Gauge is INVARIANT (a raw score is gauge-free); trust is EXPLORATORY
        (no scorecard entry yet, gate 1). Provenance records the token cost and the wall time.
        """
        import torch

        read = self.readout(readout)
        bias = float(read.meta.get("bias", 0.0))
        items = list(view)
        started = time.perf_counter()
        tokenized = [self.tokenize(it) for it in items]
        with self._mounted():
            pooled = self.runtime.final_head_inputs(tokenized, batch_size=self.default_batch_size)
        values = self.policy.head_project(pooled, read.vector, bias)
        arr = values.detach().to("cpu", dtype=torch.float32).numpy()
        n_tokens = int(sum(len(t.input_ids) for t in tokenized))
        payload = Scores(values=arr, readout=readout, n_items=len(items))
        return self._evidence(
            observable="signals.classifier.score",
            value=payload,
            readout=readout,
            n=len(items),
            n_tokens=n_tokens,
            wall=time.perf_counter() - started,
        )

    def score_prefixes(self, view: Any, readout: str = "reward") -> Evidence[TokenCurves]:
        """Per-token reward curves ``r(y_{1:t})`` for every item.

        One forward per batch: the head input at every valid position is projected onto the readout
        vector, and because the classifier pools the last token under causal attention, the value at
        position ``t`` is exactly the reward the model assigns the prefix ending at ``t``. The final
        entry of each curve therefore equals ``score(...)`` for that item, which the conformance
        suite asserts.
        """
        import torch

        read = self.readout(readout)
        bias = float(read.meta.get("bias", 0.0))
        items = list(view)
        started = time.perf_counter()
        tokenized = [self.tokenize(it) for it in items]
        curves: list[np.ndarray] = [np.empty(0)] * len(items)
        n_tokens = 0
        batch = self.default_batch_size
        with self._mounted():
            for start in range(0, len(tokenized), batch):
                sub = tokenized[start : start + batch]
                token_batch = self.runtime.collate(sub)
                head_input, valid_per_item = self.runtime.full_head_inputs(token_batch)
                for local_i, valid in enumerate(valid_per_item):
                    idx = torch.tensor(valid, device=head_input.device, dtype=torch.long)
                    rows = head_input[local_i].index_select(0, idx)
                    curve = self.policy.head_project(rows, read.vector, bias)
                    arr = curve.detach().to("cpu", dtype=torch.float32).numpy()
                    curves[start + local_i] = arr
                    n_tokens += len(valid)
        payload = TokenCurves(curves=curves, readout=readout)
        return self._evidence(
            observable="signals.classifier.score_prefixes",
            value=payload,
            readout=readout,
            n=len(items),
            n_tokens=n_tokens,
            wall=time.perf_counter() - started,
        )

    # -- capture ------------------------------------------------------------

    def capture(self, view: Any, spec: "CaptureSpec") -> "CaptureHandle":
        """Capture activations at the spec's sites, returning a ``CaptureHandle``.

        Collates the whole view into one left-padded batch and runs ``forward_with_capture`` under
        any mounted interventions, returning an in-memory handle. Population-scale, store-backed
        streaming is ``runtime.store.ActivationStore.get_or_compute``; this method is the direct,
        single-batch path the store calls on a miss.
        """
        from reward_lens.runtime.store import InMemoryCaptureHandle

        items = list(view)
        tokenized = [self.tokenize(it) for it in items]
        token_batch = self.runtime.collate(tokenized)
        with self._mounted():
            _raw, capture = self.runtime.forward_with_capture(token_batch, spec)
        return InMemoryCaptureHandle(capture)

    # -- interventions ------------------------------------------------------

    def with_interventions(self, *ivs: Any) -> "ClassifierRM":
        """Return a signal wrapped in interventions; any Observable accepts it unchanged.

        Each intervention is compiled against this signal (if it exposes ``compile``) and mounted on
        the runtime's shared hook path during scoring and capture. The intervention fingerprints
        become part of the wrapped signal's subject, so an intervened Evidence can never masquerade
        as a clean one. Interventions are a full subsystem of their own; this is the signal-side
        wiring.
        """
        compiled = tuple(iv.compile(self) if hasattr(iv, "compile") else iv for iv in ivs)
        clone = ClassifierRM(
            runtime=self.runtime,
            meta=self.meta,
            readouts=self._readouts,
            tokenizer=self.tokenizer,
            policy=self.policy,
            max_length=self.max_length,
            default_batch_size=self.default_batch_size,
            interventions=self._interventions + compiled,
            legacy_row_mean=self._legacy_row_mean,
        )
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
    ) -> Evidence[Any]:
        subject = SubjectRef(
            signals=(self.meta.fingerprint,),
            readout=readout,
            interventions=self._intervention_fingerprints(),
        )
        provenance = capture_provenance(cost=Cost(tokens=n_tokens, wall_seconds=wall))
        return make_evidence(
            observable=observable,
            observable_version=_OBS_VERSION,
            subject=subject,
            value=value,
            uncertainty=Uncertainty(n=n, method="none"),
            gauge=GaugeStatus.INVARIANT,
            calibration=None,
            provenance=provenance,
        )


def build_readouts(
    adapter: Any,
    model: Any,
    site_map: Any,
) -> tuple[list[Readout], Readout | None]:
    """Read the reward head off the checkpoint into ``Readout`` objects.

    Returns ``(readouts, legacy_row_mean)``. A single-row head yields one ``reward`` readout. A
    multi-row head yields one ``criterion:k`` readout per row plus a ``reward`` composite whose
    vector is the row mean (marked as the legacy aggregate; the true gated composite is the model's
    own input-dependent head, reachable via the native score), and a separate legacy ``reward:mean``
    readout. All vectors are fp32; the readout site is the final residual, ``PositionSpec("final")``.
    """
    import torch

    head = reward_head_module(adapter, model)
    if head is None:
        raise ValueError(
            "no linear reward head found (score/regression_layer/v_head); this model is not a "
            "classifier-style reward signal."
        )
    weight = head.weight.data.detach().to(torch.float32)
    bias = (
        head.bias.data.detach().to(torch.float32)
        if getattr(head, "bias", None) is not None
        else None
    )
    n_layers = site_map.n_layers
    site = Site(max(n_layers - 1, 0), "resid_post")
    position = PositionSpec("final")

    def make(name: str, vec: "torch.Tensor", b: float, meta: dict) -> Readout:
        meta = {**meta, "bias": float(b)}
        return Readout(
            name=name, kind="linear", site=site, position=position, vector=vec, meta=meta
        )

    if weight.ndim == 1 or weight.shape[0] == 1:
        vec = weight if weight.ndim == 1 else weight[0]
        b = 0.0 if bias is None else float(bias.reshape(-1)[0])
        return [make("reward", vec.contiguous(), b, {"aggregate": "single"})], None

    # Multi-row head: nineteen ArmoRM objectives, or QRM quantiles.
    readouts: list[Readout] = []
    for k in range(weight.shape[0]):
        b = 0.0 if bias is None else float(bias[k])
        readouts.append(make(f"criterion:{k}", weight[k].contiguous(), b, {"row": k}))
    mean_vec = weight.mean(dim=0).contiguous()
    mean_bias = 0.0 if bias is None else float(bias.mean())
    composite = make("reward", mean_vec, mean_bias, {"aggregate": "row_mean", "legacy": True})
    readouts.insert(0, composite)
    legacy = make("reward:mean", mean_vec, mean_bias, {"aggregate": "row_mean", "legacy": True})
    return readouts, legacy


__all__ = ["ClassifierRM", "build_readouts"]

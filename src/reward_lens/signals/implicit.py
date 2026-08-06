"""``ImplicitRM``: the DPO implicit reward as a ``RewardSignal`` (adapter 4).

A model trained with DPO (or any log-ratio objective) is a reward model without ever having a reward
head: its implicit reward is r-hat(y) = beta * sum_t log( pi_policy(y_t | y_<t) / pi_ref(y_t | y_<t) ),
summed over the response tokens. This adapter makes that a first-class signal. Two properties fall out
for free and are the reason the design lists it as its own adapter rather than a special case:

  - the reward decomposes per token natively (r_t = beta * (log pi_policy(y_t) - log pi_ref(y_t))), so
    ``score_prefixes`` is exact, not an attribution; the increments sum to the sequence score by
    construction, which the conformance suite and the M-tests both assert; and
  - it is a *paired-model* signal (``PAIRED_MODELS``): capture and grad route to the policy model by
    default, with the reference reachable through an explicit namespace, because "the activation at
    layer L" is ambiguous across two models and the adapter refuses to guess.

Both models share one tokenizer and one templating, so the response token region is identical for the
two forwards and the log-ratio is well defined token by token.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Literal, Sequence

import numpy as np

from reward_lens.core.types import Capability, ModelFP, Site
from reward_lens.signals._common import SignalImplBase, build_hf_runtime, split_item
from reward_lens.signals.base import PositionSpec, Readout, Scores, TokenCurves, TokenizedInput

if TYPE_CHECKING:
    import torch

    from reward_lens.runtime.hf import HFRuntime

_IMPLICIT_CAPS = (
    Capability.SCORES | Capability.PREFIX_SCORES | Capability.ACTIVATIONS | Capability.PAIRED_MODELS
)


class ImplicitRM(SignalImplBase):
    """The DPO implicit reward r-hat = beta * sum log-ratio as a ``RewardSignal`` (adapter 4).

    Constructed from a policy model and a reference model (``from_models`` / ``from_tiny``), both
    ``CausalLM`` with a shared tokenizer. ``score`` returns r-hat; ``score_prefixes`` returns the
    cumulative per-token reward whose last entry equals r-hat; ``per_token_rewards`` returns the raw
    per-token decomposition (the increments). ``self.runtime`` is the policy runtime; the reference
    runtime is ``self.reference_runtime`` and ``capture(..., namespace="ref")`` routes to it.
    """

    observable_prefix = "signals.implicit"

    def __init__(
        self,
        *,
        policy_runtime: "HFRuntime",
        reference_runtime: "HFRuntime",
        meta: Any,
        reference_meta: Any,
        policy: Any,
        tokenizer: Any,
        beta: float = 0.1,
        max_length: int = 2048,
        default_batch_size: int = 16,
        interventions: tuple[Any, ...] = (),
    ) -> None:
        implicit_readout = Readout(
            name="implicit_reward",
            kind="token_value",
            site=Site(max(meta.n_layers - 1, 0), "resid_post"),
            position=PositionSpec("all"),
            vector=None,
            meta={"beta": beta, "reference_fingerprint": str(reference_meta.fingerprint)},
        )
        super().__init__(
            runtime=policy_runtime,
            meta=meta,
            policy=policy,
            tokenizer=tokenizer,
            readouts=[implicit_readout],
            caps=_IMPLICIT_CAPS,
            max_length=max_length,
            default_batch_size=default_batch_size,
            interventions=interventions,
        )
        self.reference_runtime = reference_runtime
        self.reference_meta = reference_meta
        self.beta = beta
        self._ref_fp: ModelFP = reference_meta.fingerprint

    def _signals(self) -> tuple[ModelFP, ...]:
        """Both model fingerprints, policy first (a paired-model subject)."""
        return (self.meta.fingerprint, self._ref_fp)

    # -- rendering: type the response region -------------------------------

    def _render(self, item: Any) -> tuple[str, tuple[tuple[int, int, str], ...], dict[str, Any]]:
        """Render ``(prompt, response)`` and type the response text as a ``response`` span.

        The log-ratio is summed over the response tokens only, so the response character range is
        typed and mapped into token coordinates by ``tokenize``. The prompt tokens carry no reward.
        """
        prompt, response, raw = split_item(item)
        text = response if raw else self._chat(prompt, response)
        start = text.find(response) if response else -1
        spans = ((start, start + len(response), "response"),) if start >= 0 else ()
        return text, spans, {"prompt": prompt, "response": response}

    # -- scoring ------------------------------------------------------------

    def score(self, view: Any, readout: str | None = None) -> Any:
        """r-hat per item: beta * sum over response tokens of the policy/reference log-ratio."""
        name = readout or self.default_readout_name()
        items = list(view)
        started = time.perf_counter()
        tokenized = [self.tokenize(it) for it in items]
        decomp, n_tokens = self._per_token_decomposition(tokenized)
        values = np.array([float(d.sum()) for d in decomp], dtype=np.float32)
        payload = Scores(values=values, readout=name, n_items=len(items))
        return self._timed_evidence(
            "score", payload, name, len(items), n_tokens, started, signals=self._signals()
        )

    def score_prefixes(self, view: Any, readout: str | None = None) -> Any:
        """Cumulative per-token reward curves; the last entry of each equals ``score``.

        The curve is the running sum of the per-token log-ratio rewards over the response, so
        ``curve[-1] == r-hat``, which is the prefix-consistency invariant conformance checks. The raw
        increments (the native per-token decomposition) are ``per_token_rewards``.
        """
        name = readout or self.default_readout_name()
        items = list(view)
        started = time.perf_counter()
        tokenized = [self.tokenize(it) for it in items]
        decomp, n_tokens = self._per_token_decomposition(tokenized)
        curves = [np.cumsum(d).astype(np.float32) for d in decomp]
        payload = TokenCurves(curves=curves, readout=name)
        return self._timed_evidence(
            "score_prefixes", payload, name, len(items), n_tokens, started, signals=self._signals()
        )

    def per_token_rewards(self, view: Any, readout: str | None = None) -> Any:
        """The native per-token reward decomposition r_t (increments).

        Each item's curve is ``beta * (log pi_policy(y_t) - log pi_ref(y_t))`` over the response
        tokens; the increments sum to the sequence score. This is the decomposition the verification
        and dense-reward sciences consume directly rather than reconstructing by attribution.
        """
        name = readout or self.default_readout_name()
        items = list(view)
        started = time.perf_counter()
        tokenized = [self.tokenize(it) for it in items]
        decomp, n_tokens = self._per_token_decomposition(tokenized)
        payload = TokenCurves(curves=[d.astype(np.float32) for d in decomp], readout=name)
        return self._timed_evidence(
            "per_token_rewards",
            payload,
            name,
            len(items),
            n_tokens,
            started,
            signals=self._signals(),
        )

    def _per_token_decomposition(
        self, tokenized: Sequence[TokenizedInput]
    ) -> tuple[list[np.ndarray], int]:
        """Per-item, per-response-token reward r_t = beta * (lp_policy - lp_ref) and total tokens."""
        pol = self._response_logprobs(self.runtime, tokenized)
        ref = self._response_logprobs(self.reference_runtime, tokenized)
        decomp = [self.beta * (p - r) for p, r in zip(pol, ref)]
        n_tokens = int(sum(len(t.input_ids) for t in tokenized)) * 2  # two forwards
        return decomp, n_tokens

    def _response_logprobs(
        self, runtime: "HFRuntime", tokenized: Sequence[TokenizedInput]
    ) -> list[np.ndarray]:
        """Per-item next-token log-probabilities of the actual response tokens under one model.

        One left-padded forward; log-softmax in fp32; for each response token at padded position
        ``p`` the log-prob is ``log_softmax(logits[p-1])[input_ids[p]]``. Batches per
        ``default_batch_size``; the response token region is read off the ``response`` span carried
        on each tokenized input (the whole sequence past the prompt when no span survived).
        """
        import torch

        out: list[np.ndarray] = [np.empty(0)] * len(tokenized)
        batch = self.default_batch_size
        items = list(tokenized)
        with self._mounted_on(runtime):
            for start in range(0, len(items), batch):
                sub = items[start : start + batch]
                token_batch = runtime.collate(sub)
                raw = runtime.forward(token_batch)
                if raw.logits is None:
                    raise RuntimeError("implicit RM needs LM logits; the model produced none.")
                logprobs = torch.log_softmax(raw.logits.to(torch.float32), dim=-1)
                offsets = token_batch.meta["offsets"]
                ids = token_batch.input_ids
                for local_i, tok in enumerate(sub):
                    s, e = _response_range(tok)
                    off = offsets[local_i]
                    vals = np.empty(max(e - s, 0), dtype=np.float32)
                    for k, t in enumerate(range(s, e)):
                        padded = off + t
                        target = int(ids[local_i, padded].item())
                        vals[k] = float(logprobs[local_i, padded - 1, target])
                    out[start + local_i] = vals
        return out

    def _mounted_on(self, runtime: "HFRuntime"):
        """Mount interventions on a specific runtime (policy by default; capture may pass the ref)."""
        import contextlib

        if not self._interventions:
            return contextlib.nullcontext()
        return runtime.mounted(self._interventions)

    # -- capture routing ----------------------------------------------------

    def capture(self, view: Any, spec: Any, namespace: Literal["policy", "ref"] = "policy") -> Any:
        """Capture activations, routing to the policy model by default (PAIRED_MODELS).

        ``namespace="ref"`` captures from the reference model instead. The two runtimes are distinct
        models, so a caller must say which one it means; there is no shared "layer L". Interventions
        mount on the chosen runtime.
        """
        from reward_lens.runtime.store import InMemoryCaptureHandle

        runtime = self.runtime if namespace == "policy" else self.reference_runtime
        tokenized = [self.tokenize(it) for it in view]
        token_batch = runtime.collate(tokenized)
        with self._mounted_on(runtime):
            _raw, capture = runtime.forward_with_capture(token_batch, spec)
        return InMemoryCaptureHandle(capture)

    # -- constructors -------------------------------------------------------

    @classmethod
    def from_models(
        cls,
        policy: "torch.nn.Module",
        reference: "torch.nn.Module",
        tokenizer: Any,
        *,
        beta: float = 0.1,
        device: str = "cpu",
        architecture: str | None = None,
    ) -> "ImplicitRM":
        """Wrap a policy + reference ``CausalLM`` (shared tokenizer) as an implicit RM (no download)."""
        from reward_lens.signals.judge import _lm_head_module

        pol_rt, pol_meta, num_policy = build_hf_runtime(
            policy,
            tokenizer,
            _lm_head_module(policy),
            architecture=architecture,
            device=device,
            lineage={"provenance_tier": "weights-verified", "role": "policy"},
        )
        ref_rt, ref_meta, _ = build_hf_runtime(
            reference,
            tokenizer,
            _lm_head_module(reference),
            architecture=architecture,
            device=device,
            lineage={"provenance_tier": "weights-verified", "role": "reference"},
        )
        return cls(
            policy_runtime=pol_rt,
            reference_runtime=ref_rt,
            meta=pol_meta,
            reference_meta=ref_meta,
            policy=num_policy,
            tokenizer=tokenizer,
            beta=beta,
        )

    @classmethod
    def from_tiny(
        cls, *, policy_seed: int = 1, reference_seed: int = 2, beta: float = 0.1, **kw: Any
    ) -> "ImplicitRM":
        """Two tiny offline ``LlamaForCausalLM`` (distinct seeds) sharing a tokenizer (adapter 4)."""
        from reward_lens.signals.judge import _tiny_causal_lm

        policy_model, tokenizer = _tiny_causal_lm(seed=policy_seed, **kw)
        reference_model, _ = _tiny_causal_lm(seed=reference_seed, tokenizer=tokenizer, **kw)
        return cls.from_models(
            policy_model,
            reference_model,
            tokenizer,
            beta=beta,
            architecture="LlamaForCausalLM",
        )


def _response_range(tok: TokenizedInput) -> tuple[int, int]:
    """The ``[start, end)`` response token range: the ``response`` span, else all but the first token."""
    for span in tok.spans:
        if span.kind == "response":
            start = max(span.start, 1)  # need a predecessor position for the next-token log-prob
            return start, max(span.end, start)
    length = len(tok.input_ids)
    return (1, length) if length > 1 else (0, 0)


__all__ = ["ImplicitRM"]

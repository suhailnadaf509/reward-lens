"""The HuggingFace policy: a `PolicySubject` over a loaded causal language model.

`HFPolicyRuntime` implements the six-method `Runtime` protocol against a decoder stack, so
everything above `runtime` (every capture, every intervention mount, every instrument in the
battery) works against a policy exactly as it works against a grader. It adds `weight_at`, the
`SiteWeights` method that removes the last reach-through in the library, and `generate`, which a
grader has no need for.

**What the readout is, for a policy.** A classifier reward model has a linear head and the reward
direction is read off the checkpoint. A causal language model has an unembedding matrix instead,
and its rows are the same kind of object: a direction in the residual stream whose inner product
with a hidden state is a scalar the model computes. So a policy's `Readout` is a row of `W_U`, or a
difference of two rows, and projecting the residual stream at each layer onto it is the logit lens.
That is not an analogy. `LensCrystallization` projects the residual at every layer onto the readout
vector and reports the depth at which half the final differential has formed; run against a grader
with its reward head it is the reward lens, and run against a policy with a two-token contrast it is
the logit lens, and it is the same code with the subject as the only argument that changed. Section
2.1's claim that the pivot between the two prior designs is an argument value rather than a pivot is
either true here or nowhere.

**Gradients work here and cannot work in-engine.** `grad` and `hvp` run under normal autograd on an
eager forward, and both are exercised on CPU. They are unavailable inside a paged-attention serving
engine and that is structural rather than an optimisation gap; `reward_lens.policy.vllm` carries the
list and the reasons.

Everything in this module runs on CPU on a 2.45M-parameter Qwen3. The GPU paths (bf16 trunks,
batched population forwards) are the same code and are not exercised here.
"""

from __future__ import annotations

import contextlib
import time
from typing import TYPE_CHECKING, Any, Callable, Literal, Sequence

import numpy as np

from reward_lens.core.evidence import Evidence, Uncertainty, make_evidence
from reward_lens.core.provenance import Cost, capture_provenance
from reward_lens.core.types import Capability, GaugeStatus, Site, SubjectRef
from reward_lens.policy.arch import ArchitectureView, describe
from reward_lens.policy.base import (
    PolicyMeta,
    PositionSpec,
    Readout,
    Rollouts,
    SampleSpec,
    Scores,
    TokenCurves,
    TokenGradients,
    TokenizedInput,
    runtime_provenance,
)
from reward_lens.runtime.backend import (
    Capture,
    CaptureSpec,
    RawOutput,
    SiteMap,
    TokenBatch,
)
from reward_lens.runtime.hooks import CaptureMount, LeafCutMount, mounted_interventions

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

    from reward_lens.runtime.backend import CaptureHandle
    from reward_lens.runtime.precision import NumericsPolicy

_SUBJECT_VERSION = "1"

#: The capabilities a decoder-stack policy offers with an eager backend. `LINEAR_READOUT` is here
#: because an unembedding row is a linear readout in exactly the sense the flag means: a fixed
#: direction in the residual stream that the subject's own scalar is a projection onto.
POLICY_CAPS = (
    Capability.SCORES
    | Capability.PREFIX_SCORES
    | Capability.ACTIVATIONS
    | Capability.GRADIENTS
    | Capability.HVP
    | Capability.LINEAR_READOUT
    | Capability.GENERATIVE
)


def _split_item(item: Any) -> tuple[str, str]:
    """Normalise a data item into ``(prompt, completion)``.

    Accepts the same shapes the grader side accepts, because the battery hands both subjects the
    same items: a ``(prompt, completion)`` pair, a mapping, a bare string (all completion, no
    prompt), or an object with ``.prompt`` and one of ``.completion``/``.response``/``.text``.
    """
    if isinstance(item, str):
        return "", item
    if isinstance(item, (tuple, list)) and len(item) == 2:
        return str(item[0]), str(item[1])
    if isinstance(item, dict):
        if "completion" in item:
            return str(item.get("prompt", "")), str(item["completion"])
        if "response" in item:
            return str(item.get("prompt", "")), str(item["response"])
        return str(item.get("prompt", "")), str(item.get("text", ""))
    prompt = getattr(item, "prompt", "")
    completion = getattr(item, "completion", None)
    if completion is None:
        completion = getattr(item, "response", None) or getattr(item, "text", "")
    completion = getattr(completion, "text", completion)
    return str(prompt), str(completion)


# ---------------------------------------------------------------------------
# The runtime
# ---------------------------------------------------------------------------


class HFPolicyRuntime:
    """A `Runtime` over a loaded causal LM, plus `weight_at` and `generate`.

    Construct through `wrap_hf_policy`, which navigates the architecture and resolves the site map.
    The runtime is readout-agnostic: it captures the input to the unembedding (the post-final-norm
    residual, which is the tensor `lm_head` consumes) into ``RawOutput.extra["head_input"]``, and
    the subject projects it onto whichever readout direction it was asked for, in fp32.
    """

    def __init__(
        self,
        model: "nn.Module",
        arch: ArchitectureView,
        site_map: SiteMap,
        policy: "NumericsPolicy",
        head_module: "nn.Module | None",
        device: "torch.device",
        pad_id: int = 0,
    ):
        self.model = model
        self.arch = arch
        #: `runtime/hooks.py` names its second argument `adapter` and calls exactly three methods on
        #: it. `ArchitectureView` provides those three, so it is what gets passed. The parameter
        #: name is the last trace of `ModelAdapter` in the capture path.
        self.adapter = arch
        self.site_map = site_map
        self.policy = policy
        self.head_module = head_module
        self.device = device
        self.pad_id = pad_id

    # -- protocol: sites ----------------------------------------------------

    def sites(self) -> SiteMap:
        return self.site_map

    # -- SiteWeights --------------------------------------------------------

    def weight_at(self, site: Site) -> "torch.Tensor":
        """The fp32 weight behind a site, sliced to one head where the site names one.

        This is the method `measure/battery/path.py` now calls instead of walking
        ``runtime.adapter.get_layers(runtime.model)[layer]``. A backend that implements it makes
        every head-level instrument portable to it without the instrument changing.
        """
        return self.arch.weight_at(self.model, site)

    # -- batching -----------------------------------------------------------

    def _final_positions(self, attention_mask: "torch.Tensor") -> "torch.Tensor":
        import torch

        seq_len = attention_mask.shape[1]
        idx = torch.arange(seq_len, device=attention_mask.device)
        masked = idx.unsqueeze(0) * attention_mask.to(torch.long)
        return masked.argmax(dim=1)

    def collate(self, tokenized: Sequence[TokenizedInput]) -> TokenBatch:
        """Left-pad tokenized inputs into a `TokenBatch`.

        Left padding for the same reason the grader side uses it: it puts every row's last real
        token at column ``T-1``, so a final-position readout reads the same relative location for
        the whole batch. ``meta['offsets']`` carries the per-row pad width, which is what maps an
        item-local token index into padded coordinates.
        """
        import torch

        n = len(tokenized)
        max_len = max(len(t.input_ids) for t in tokenized)
        input_ids = torch.full((n, max_len), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((n, max_len), dtype=torch.long)
        offsets: list[int] = []
        for i, tok in enumerate(tokenized):
            length = len(tok.input_ids)
            pad = max_len - length
            input_ids[i, pad:] = torch.tensor(tok.input_ids, dtype=torch.long)
            mask = tok.attention_mask if tok.attention_mask else [1] * length
            attention_mask[i, pad:] = torch.tensor(mask, dtype=torch.long)
            offsets.append(pad)
        return TokenBatch(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            meta={"tokenized": list(tokenized), "offsets": offsets, "pad_side": "left"},
        )

    # -- protocol: forward --------------------------------------------------

    def _install_head_capture(self, sink: dict[str, Any]) -> list[Any]:
        if self.head_module is None:
            return []

        def pre_hook(_module: Any, args: Any) -> None:
            sink["head_input"] = args[0] if isinstance(args, tuple) else args

        return [self.head_module.register_forward_pre_hook(pre_hook)]

    def forward(self, batch: TokenBatch) -> RawOutput:
        """One forward pass, capturing the unembedding's input.

        ``extra['head_input']`` is the ``(B, T, d)`` post-final-norm residual the unembedding
        consumes, which is what a readout direction is projected onto, and ``extra['final_pos']``
        the last valid index per row. ``logits`` is the model's own output, kept because the
        log-probability path needs the full vocabulary and a rank-8 projection cannot reconstruct
        it.
        """
        import torch

        ids = batch.input_ids.to(self.device)
        mask = batch.attention_mask.to(self.device)
        captured: dict[str, Any] = {}
        handles = self._install_head_capture(captured)
        try:
            with torch.no_grad():
                out = self.model(input_ids=ids, attention_mask=mask, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        return RawOutput(
            reward=None,
            logits=getattr(out, "logits", None),
            hidden=None,
            extra={
                "head_input": captured.get("head_input"),
                "final_pos": self._final_positions(mask),
            },
        )

    # -- protocol: capture --------------------------------------------------

    def forward_with_capture(
        self, batch: TokenBatch, spec: CaptureSpec
    ) -> tuple[RawOutput, Capture]:
        """Forward once, capturing the requested sites through the shared mount.

        The same `CaptureMount` the grader path uses, so any intervention composes with any capture
        on this side too. Position resolution follows the grader path exactly: the default `final`
        position stores ``(B, d)`` per site, anything else stores the whole sequence.
        """
        import torch

        ids = batch.input_ids.to(self.device)
        mask = batch.attention_mask.to(self.device)
        final_pos = self._final_positions(mask)
        single_position = self._is_final_position(spec.position) and not spec.full_sequence
        positions = final_pos if single_position else None
        mount = CaptureMount(
            self.model,
            self.arch,  # type: ignore[arg-type]  # hooks.py types this ModelAdapter; it calls three methods
            self.site_map,
            spec.sites,
            positions=positions,
            full_sequence=spec.full_sequence or not single_position,
            dtype=spec.dtype,
        )
        captured: dict[str, Any] = {}
        handles = self._install_head_capture(captured)
        try:
            with mount:
                with torch.no_grad():
                    out = self.model(input_ids=ids, attention_mask=mask, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        positions_list = [[int(p)] for p in final_pos.tolist()] if single_position else []
        capture = Capture(tensors=mount.tensors, positions=positions_list, dtype=spec.dtype)
        raw = RawOutput(
            reward=None,
            logits=getattr(out, "logits", None),
            hidden=mount.tensors,
            extra={"head_input": captured.get("head_input"), "final_pos": final_pos},
        )
        return raw, capture

    @staticmethod
    def _is_final_position(position: "PositionSpec | None") -> bool:
        return position is None or getattr(position, "kind", "final") == "final"

    # -- protocol: interventions -------------------------------------------

    @contextlib.contextmanager
    def mounted(self, interventions: Sequence[Any]) -> Any:
        with mounted_interventions(
            self.model,
            self.arch,  # type: ignore[arg-type]  # hooks.py types this ModelAdapter; it calls three methods
            self.site_map,
            interventions,
        ):
            yield self

    # -- protocol: grad and hvp --------------------------------------------

    def grad(
        self,
        batch: TokenBatch,
        scalar_fn: Callable[[RawOutput], "torch.Tensor"],
        wrt: "Site | Literal['embeddings']",
    ) -> "torch.Tensor":
        """Autograd of the summed readout scalar with respect to a site or the embeddings.

        The scalar is summed over the batch, which is exact rather than an approximation: each
        item's readout depends only on its own activations under causal attention, so the gradient
        of the sum at row ``i`` is the gradient of row ``i``'s own scalar.
        """
        import torch

        ids = batch.input_ids.to(self.device)
        mask = batch.attention_mask.to(self.device)
        if wrt == "embeddings":
            embed_layer = self.model.get_input_embeddings()
            embeds = embed_layer(ids).detach().clone().requires_grad_(True)
            raw, _ = self._graph_forward(inputs_embeds=embeds, attention_mask=mask)
            scalar = scalar_fn(raw).to(torch.float32).sum()
            (grad,) = torch.autograd.grad(scalar, embeds)
            return grad
        raw, leaf = self._graph_forward(input_ids=ids, attention_mask=mask, leaf_site=wrt)
        scalar = scalar_fn(raw).to(torch.float32).sum()
        (grad,) = torch.autograd.grad(scalar, leaf)
        return grad

    def hvp(
        self,
        batch: TokenBatch,
        scalar_fn: Callable[[RawOutput], "torch.Tensor"],
        at: Site,
        vecs: "torch.Tensor",
    ) -> "torch.Tensor":
        """Hessian-vector products of the readout scalar at a site, at the final positions.

        Double backprop, exactly as the grader runtime does it: cut a leaf, take the first gradient
        with ``create_graph=True``, then differentiate ``(grad . v)`` for each direction. Returns
        ``(B, K, d)``.
        """
        import torch

        ids = batch.input_ids.to(self.device)
        mask = batch.attention_mask.to(self.device)
        raw, leaf = self._graph_forward(input_ids=ids, attention_mask=mask, leaf_site=at)
        scalar = scalar_fn(raw).to(torch.float32).sum()
        (first_grad,) = torch.autograd.grad(scalar, leaf, create_graph=True)
        final_pos = raw.extra["final_pos"]
        batch_size, _seq, d_model = leaf.shape
        batch_idx = torch.arange(batch_size, device=leaf.device)
        vecs2d = vecs.reshape(-1, d_model).to(device=leaf.device)
        products: list["torch.Tensor"] = []
        for vec in vecs2d:
            placed = torch.zeros_like(leaf)
            placed[batch_idx, final_pos] = vec.to(leaf.dtype)
            (hv,) = torch.autograd.grad((first_grad * placed).sum(), leaf, retain_graph=True)
            products.append(hv[batch_idx, final_pos].to(torch.float32))
        return torch.stack(products, dim=1)

    def _graph_forward(
        self,
        *,
        input_ids: "torch.Tensor | None" = None,
        inputs_embeds: "torch.Tensor | None" = None,
        attention_mask: "torch.Tensor",
        leaf_site: Site | None = None,
    ) -> tuple[RawOutput, "torch.Tensor | None"]:
        """Forward with grad enabled, capturing the head input and optionally cutting a leaf."""
        captured: dict[str, Any] = {}
        handles = self._install_head_capture(captured)
        mount = (
            LeafCutMount(
                self.model,
                self.arch,  # type: ignore[arg-type]  # hooks.py types this ModelAdapter; it calls three methods
                self.site_map,
                leaf_site,
            )
            if leaf_site is not None
            else None
        )
        try:
            kwargs: dict[str, Any] = {"attention_mask": attention_mask, "use_cache": False}
            if inputs_embeds is not None:
                kwargs["inputs_embeds"] = inputs_embeds
            else:
                kwargs["input_ids"] = input_ids
            if mount is not None:
                with mount:
                    out = self.model(**kwargs)
            else:
                out = self.model(**kwargs)
        finally:
            for handle in handles:
                handle.remove()
        raw = RawOutput(
            reward=None,
            logits=getattr(out, "logits", None),
            hidden=None,
            extra={
                "head_input": captured.get("head_input"),
                "final_pos": self._final_positions(attention_mask),
            },
        )
        return raw, (mount.leaf if mount is not None else None)

    # -- generation ---------------------------------------------------------

    def generate(self, batch: TokenBatch, spec: SampleSpec) -> "torch.Tensor":
        """Sample continuations, returning the full ``(B*K, T + max_new)`` id tensor.

        Seeded explicitly on a local generator state rather than the global one, so drawing a sample
        does not move any other stream in the process. `SampleSpec.seed` has no default for the same
        reason.
        """
        import torch

        ids = batch.input_ids.to(self.device)
        mask = batch.attention_mask.to(self.device)
        torch.manual_seed(spec.seed)
        # `generate` is a `GenerationMixin` method; the torch stubs resolve it through
        # `nn.Module.__getattr__` and type it as a tensor. The Any binding ends that here.
        hf_model: Any = self.model
        with torch.no_grad():
            return hf_model.generate(  # type: ignore[no-any-return]
                input_ids=ids,
                attention_mask=mask,
                do_sample=spec.temperature > 0,
                temperature=max(spec.temperature, 1e-6),
                top_p=spec.top_p,
                max_new_tokens=spec.max_new_tokens,
                num_return_sequences=spec.group_size,
                pad_token_id=self.pad_id,
            )


# ---------------------------------------------------------------------------
# The subject
# ---------------------------------------------------------------------------


class HFPolicy:
    """A causal language model as a `PolicySubject`.

    Build through `wrap_hf_policy` or `from_pretrained`. ``contrast`` names the two tokens whose
    logit difference is the policy's primary scalar readout; the readout is called ``decision``
    because that is what it reads, and naming it ``reward`` to make a grader instrument accept it
    would have been a lie the type system was helping to tell.

    What this cannot do. The `decision` readout is one fixed direction, so a lens curve read through
    it traces the formation of one contrast and nothing else: on items where neither contrast token
    is what the policy was going to emit, the curve is the formation of an irrelevant preference and
    it still returns a number. The contrast is part of the claim and belongs in the subject
    description of anything that reports one.
    """

    def __init__(
        self,
        *,
        runtime: HFPolicyRuntime,
        meta: PolicyMeta,
        readouts: list[Readout],
        tokenizer: Any,
        policy: "NumericsPolicy",
        max_length: int = 2048,
        default_batch_size: int = 16,
        interventions: tuple[Any, ...] = (),
    ):
        self.runtime = runtime
        self.meta = meta
        self.caps: Capability = POLICY_CAPS
        self.tokenizer = tokenizer
        self.policy = policy
        self.max_length = max_length
        self.default_batch_size = default_batch_size
        self._readouts = readouts
        self._readout_by_name = {r.name: r for r in readouts}
        self._interventions = interventions

    # -- readouts -----------------------------------------------------------

    def readouts(self) -> list[Readout]:
        return list(self._readouts)

    def readout(self, name: str = "decision") -> Readout:
        if name in self._readout_by_name:
            return self._readout_by_name[name]
        raise KeyError(
            f"unknown readout {name!r}; this policy exposes {sorted(self._readout_by_name)}. A "
            f"policy's readouts are unembedding directions, so a new one is a token or a token "
            f"contrast: build it with `logit_readout` or `contrast_readout`."
        )

    def readout_scalar_fn(self, readout: str = "decision") -> Callable[[RawOutput], "torch.Tensor"]:
        """A `ScalarFn` computing the readout scalar from a `RawOutput`, for grad and hvp.

        Reads the grad-attached unembedding input out of ``RawOutput.extra`` and projects it onto
        the readout vector in fp32 at the resolved final positions. The projection stays in fp32
        whatever the trunk dtype, which is the numerics policy's whole job.
        """
        import torch

        read = self.readout(readout)
        weight = read.vector
        bias = float(read.meta.get("bias", 0.0))
        if weight is None:
            raise ValueError(
                f"readout {readout!r} has no direction vector, so there is no scalar to "
                f"differentiate. Gradients need a `linear` or `logit_diff` readout; "
                f"{readout!r} is {read.kind!r}."
            )

        def scalar_fn(raw: Any) -> "torch.Tensor":
            head_input = raw.extra["head_input"]
            final_pos = raw.extra["final_pos"]
            batch_idx = torch.arange(head_input.shape[0], device=head_input.device)
            pooled = head_input[batch_idx, final_pos]
            return self.policy.head_project(pooled, weight, bias)

        return scalar_fn

    # -- tokenization -------------------------------------------------------

    def tokenize(self, item: Any) -> TokenizedInput:
        """Tokenize a ``(prompt, completion)`` item, recording where the completion starts.

        The prompt is rendered through the model's chat template with a generation prompt when it
        has one, so the tokens the policy sees are the tokens it would have seen while sampling.
        ``meta['n_prompt_tokens']`` is the boundary `score_under` masks at: a sequence
        log-probability that includes the prompt is a number about the prompt, and on a 20-token
        completion after a 12-token prompt it is dominated by it.
        """
        prompt, completion = _split_item(item)
        prompt_text = self._render_prompt(prompt)
        full = prompt_text + completion

        enc = self._encode(full)
        input_ids = list(enc["input_ids"])
        attention_mask = list(enc.get("attention_mask", [1] * len(input_ids)))
        offsets = enc.get("offset_mapping")
        token_offsets = tuple((int(a), int(b)) for a, b in offsets) if offsets is not None else ()

        n_prompt_tokens = len(self._encode(prompt_text)["input_ids"]) if prompt_text else 0
        n_prompt_tokens = min(n_prompt_tokens, max(len(input_ids) - 1, 0))

        return TokenizedInput(
            input_ids=input_ids,
            attention_mask=attention_mask,
            text=full,
            token_offsets=token_offsets,
            spans=(),
            meta={
                "prompt": prompt,
                "completion": completion,
                "n_prompt_tokens": n_prompt_tokens,
            },
        )

    def _render_prompt(self, prompt: str) -> str:
        if not prompt:
            return ""
        template = getattr(self.tokenizer, "chat_template", None)
        if template:
            return str(
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        return f"User: {prompt}\nAssistant:"

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

    # -- scoring ------------------------------------------------------------

    def score(self, view: Any, readout: str = "decision") -> Evidence[Scores]:
        """The subject's scalar for each item, under a readout.

        The `RewardSignal` spelling, and the reason a grader instrument runs against a policy
        unchanged. For a direction readout this is the fp32 projection of the final residual onto
        that direction, which is a logit or a logit difference. For ``logprob`` it delegates to
        `score_under`, because the log-probability of a completion is not a projection of one
        position and pretending otherwise would return the wrong number silently.
        """
        import torch

        if self.readout(readout).kind == "token_value":
            return self.score_under(view, readout)

        read = self.readout(readout)
        bias = float(read.meta.get("bias", 0.0))
        items = list(view)
        started = time.perf_counter()
        tokenized = [self.tokenize(it) for it in items]
        pooled_rows: list["torch.Tensor"] = []
        n_tokens = 0
        with self._mounted():
            for start in range(0, len(tokenized), self.default_batch_size):
                sub = tokenized[start : start + self.default_batch_size]
                raw = self.runtime.forward(self.runtime.collate(sub))
                head_input = raw.extra["head_input"]
                final_pos = raw.extra["final_pos"]
                idx = torch.arange(head_input.shape[0], device=head_input.device)
                pooled_rows.append(head_input[idx, final_pos])
                n_tokens += int(sum(len(t.input_ids) for t in sub))
        pooled = torch.cat(pooled_rows, dim=0)
        values = self.policy.head_project(pooled, read.vector, bias)
        arr = values.detach().to("cpu", dtype=torch.float32).numpy()
        payload = Scores(values=arr, readout=readout, n_items=len(items))
        return self._evidence(
            observable="policy.hf.score",
            value=payload,
            readout=readout,
            n=len(items),
            n_tokens=n_tokens,
            wall=time.perf_counter() - started,
        )

    def score_under(self, view: Any, readout: str = "logprob") -> Evidence[Scores]:
        """The log-probability this policy assigns each item's completion.

        Summed over completion tokens only, with the prompt masked out at the boundary `tokenize`
        recorded. This is the number every off-policy correction, every KL term and every staleness
        check is built out of, and it is the policy's own scalar in the sense a reward is a grader's.

        What it cannot do. It is the log-probability under *these* weights with *this* tokenization,
        not under the weights that produced the text. Comparing it to a sampling-time log-probability
        is a real quantity (`policy.train_infer_logprob_mismatch`) and comparing it to nothing is a
        number about a policy that may never have emitted the text.
        """
        import torch

        items = list(view)
        started = time.perf_counter()
        tokenized = [self.tokenize(it) for it in items]
        out = np.zeros(len(items), dtype=np.float64)
        n_tokens = 0
        with self._mounted():
            for start in range(0, len(tokenized), self.default_batch_size):
                sub = tokenized[start : start + self.default_batch_size]
                batch = self.runtime.collate(sub)
                raw = self.runtime.forward(batch)
                logits = raw.logits
                if logits is None:
                    raise RuntimeError(
                        "the model returned no logits, so there is no distribution to read a "
                        "log-probability out of. A policy without an unembedding is not a policy."
                    )
                log_probs = torch.log_softmax(logits.to(torch.float32), dim=-1)
                offsets = batch.meta["offsets"]
                for local, (tok, pad) in enumerate(zip(sub, offsets)):
                    lo = pad + int(tok.meta.get("n_prompt_tokens", 0))
                    hi = pad + len(tok.input_ids)
                    if hi - lo < 1:
                        continue
                    target = batch.input_ids[local, lo:hi]
                    # position t predicts token t+1, so the logits for target token j sit at j-1.
                    picked = log_probs[local, lo - 1 : hi - 1].gather(1, target.unsqueeze(1))
                    out[start + local] = float(picked.sum().item())
                    n_tokens += int(hi - lo)
        payload = Scores(values=out, readout=readout, n_items=len(items))
        return self._evidence(
            observable="policy.hf.score_under",
            value=payload,
            readout=readout,
            n=len(items),
            n_tokens=n_tokens,
            wall=time.perf_counter() - started,
        )

    def score_prefixes(self, view: Any, readout: str = "decision") -> Evidence[TokenCurves]:
        """The readout scalar at every valid position, one curve per item.

        For a direction readout this is the logit-lens curve along the sequence in a single forward.
        For ``logprob`` it is the running per-token log-probability of the item's own tokens, which
        is the curve `credit.implicit_prm` reads.
        """
        import torch

        read = self.readout(readout)
        items = list(view)
        started = time.perf_counter()
        tokenized = [self.tokenize(it) for it in items]
        curves: list[np.ndarray] = [np.empty(0)] * len(items)
        n_tokens = 0
        with self._mounted():
            for start in range(0, len(tokenized), self.default_batch_size):
                sub = tokenized[start : start + self.default_batch_size]
                batch = self.runtime.collate(sub)
                raw = self.runtime.forward(batch)
                offsets = batch.meta["offsets"]
                if read.kind == "token_value":
                    logits = raw.logits
                    if logits is None:
                        raise RuntimeError(
                            "the model returned no logits, so there is no distribution to read a "
                            "per-token log-probability curve out of."
                        )
                    log_probs = torch.log_softmax(logits.to(torch.float32), dim=-1)
                    for local, (tok, pad) in enumerate(zip(sub, offsets)):
                        lo, hi = pad + 1, pad + len(tok.input_ids)
                        target = batch.input_ids[local, lo:hi]
                        picked = log_probs[local, lo - 1 : hi - 1].gather(1, target.unsqueeze(1))
                        curves[start + local] = (
                            picked.squeeze(1).detach().to("cpu").numpy().astype(np.float64)
                        )
                        n_tokens += int(hi - lo)
                else:
                    bias = float(read.meta.get("bias", 0.0))
                    head_input = raw.extra["head_input"]
                    for local, (tok, pad) in enumerate(zip(sub, offsets)):
                        rows = head_input[local, pad : pad + len(tok.input_ids)]
                        curve = self.policy.head_project(rows, read.vector, bias)
                        curves[start + local] = (
                            curve.detach().to("cpu", dtype=torch.float32).numpy()
                        )
                        n_tokens += len(tok.input_ids)
        payload = TokenCurves(curves=curves, readout=readout)
        return self._evidence(
            observable="policy.hf.score_prefixes",
            value=payload,
            readout=readout,
            n=len(items),
            n_tokens=n_tokens,
            wall=time.perf_counter() - started,
        )

    # -- capture ------------------------------------------------------------

    def capture(self, view: Any, spec: CaptureSpec) -> "CaptureHandle":
        """Capture activations at the spec's sites, through the shared mount.

        The same shape and the same handle type the grader side returns, which is what lets
        `measure/battery/_common.capture_sites` take either subject.
        """
        from reward_lens.runtime.store import InMemoryCaptureHandle

        items = list(view)
        tokenized = [self.tokenize(it) for it in items]
        token_batch = self.runtime.collate(tokenized)
        with self._mounted():
            _raw, capture = self.runtime.forward_with_capture(token_batch, spec)
        return InMemoryCaptureHandle(capture)

    # -- gradients ----------------------------------------------------------

    def grad_h(self, view: Any, at: Site, readout: str = "decision") -> "torch.Tensor":
        """The gradient of the readout scalar with respect to the activation at a site.

        Returns ``(B, T, d)``. This is where the policy stops being a black box: it is the direction
        in activation space that most increases the scalar the readout reads, at the current
        parameters, which is the object the selection covector (F5) and the incentive Jacobian are
        built from.

        Structurally unavailable in a serving engine. See `reward_lens.policy.vllm`.
        """
        items = list(view)
        tokenized = [self.tokenize(it) for it in items]
        batch = self.runtime.collate(tokenized)
        with self._mounted():
            return self.runtime.grad(batch, self.readout_scalar_fn(readout), at)

    def token_gradients(self, view: Any, readout: str = "decision") -> Evidence[TokenGradients]:
        """Per-token gradients of the readout with respect to the input embeddings.

        Two numbers per token, and they disagree in the case that matters: the gradient norm, and
        the gradient contracted with the token's own embedding. A token whose embedding is nearly
        orthogonal to the gradient has a large norm and almost no first-order effect, so a per-token
        attribution reporting only the norm calls it important. Both are returned; neither is
        called the answer.

        This is a first-order quantity at the current parameters. It says what an infinitesimal
        change to one token's embedding does to the readout and says nothing about deleting the
        token, which moves every downstream position.
        """
        import torch

        items = list(view)
        started = time.perf_counter()
        tokenized = [self.tokenize(it) for it in items]
        norms: list[np.ndarray] = []
        dotted: list[np.ndarray] = []
        n_tokens = 0
        hf_model: Any = self.runtime.model
        embed_layer = hf_model.get_input_embeddings()
        with self._mounted():
            for start in range(0, len(tokenized), self.default_batch_size):
                sub = tokenized[start : start + self.default_batch_size]
                batch = self.runtime.collate(sub)
                grad = self.runtime.grad(batch, self.readout_scalar_fn(readout), "embeddings")
                with torch.no_grad():
                    embeds = embed_layer(batch.input_ids)
                    contracted = (grad * embeds).sum(dim=-1)
                    row_norms = grad.norm(dim=-1)
                for local, (tok, pad) in enumerate(zip(sub, batch.meta["offsets"])):
                    hi = pad + len(tok.input_ids)
                    norms.append(row_norms[local, pad:hi].detach().to("cpu", torch.float32).numpy())
                    dotted.append(
                        contracted[local, pad:hi].detach().to("cpu", torch.float32).numpy()
                    )
                    n_tokens += len(tok.input_ids)
        payload = TokenGradients(norms=norms, dotted=dotted, readout=readout, wrt="embeddings")
        return self._evidence(
            observable="policy.hf.token_gradients",
            value=payload,
            readout=readout,
            n=len(items),
            n_tokens=n_tokens,
            wall=time.perf_counter() - started,
            extra={"apparatus": runtime_provenance()},
        )

    # -- sampling -----------------------------------------------------------

    def sample(self, prompts: Sequence[str], spec: SampleSpec) -> Evidence[Rollouts]:
        """Draw ``spec.group_size`` completions per prompt, with their sampling-time log-probs.

        The log-probability recorded is the one computed on the sampled tokens in this same call.
        It is deliberately not recomputed later: the difference between a sampling-time log-prob and
        a recomputed one is `policy.train_infer_logprob_mismatch`, and a payload that recomputes it
        destroys the quantity.
        """
        import torch

        started = time.perf_counter()
        prompt_list = [str(p) for p in prompts]
        tokenized = [self.tokenize((p, "")) for p in prompt_list]
        batch = self.runtime.collate(tokenized)
        prompt_len = int(batch.input_ids.shape[1])
        with self._mounted():
            sequences = self.runtime.generate(batch, spec)
            completions = sequences[:, prompt_len:]
            mask = torch.ones_like(sequences)
            raw = self.runtime.forward(TokenBatch(input_ids=sequences, attention_mask=mask))
        if raw.logits is None:
            raise RuntimeError(
                "the model returned no logits after generation, so the sampled tokens have no "
                "log-probabilities to record."
            )
        log_probs = torch.log_softmax(raw.logits.to(torch.float32), dim=-1)
        picked = log_probs[:, prompt_len - 1 : -1].gather(2, completions.unsqueeze(2)).squeeze(2)
        totals = picked.sum(dim=1).detach().to("cpu").numpy()

        k = spec.group_size
        texts: list[list[str]] = []
        ids: list[list[list[int]]] = []
        lps: list[list[float]] = []
        for i in range(len(prompt_list)):
            rows = completions[i * k : (i + 1) * k]
            texts.append([self.tokenizer.decode(r, skip_special_tokens=True) for r in rows])
            ids.append([[int(t) for t in r.tolist()] for r in rows])
            lps.append([float(x) for x in totals[i * k : (i + 1) * k]])
        payload = Rollouts(
            prompts=prompt_list, texts=texts, token_ids=ids, logprobs=lps, spec=spec.as_dict()
        )
        return self._evidence(
            observable="policy.hf.sample",
            value=payload,
            readout="logprob",
            n=len(prompt_list) * k,
            n_tokens=int(completions.numel()),
            wall=time.perf_counter() - started,
        )

    # -- interventions ------------------------------------------------------

    def with_interventions(self, *ivs: Any) -> "HFPolicy":
        """Return a policy wrapped in interventions; any instrument accepts it unchanged."""
        compiled = tuple(iv.compile(self) if hasattr(iv, "compile") else iv for iv in ivs)
        return HFPolicy(
            runtime=self.runtime,
            meta=self.meta,
            readouts=self._readouts,
            tokenizer=self.tokenizer,
            policy=self.policy,
            max_length=self.max_length,
            default_batch_size=self.default_batch_size,
            interventions=self._interventions + compiled,
        )

    def _mounted(self) -> Any:
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
        extra: dict[str, Any] | None = None,
    ) -> Evidence[Any]:
        subject = SubjectRef(
            signals=(self.meta.fingerprint,),
            readout=readout,
            interventions=self._intervention_fingerprints(),
            extra=extra or {},
        )
        provenance = capture_provenance(cost=Cost(tokens=n_tokens, wall_seconds=wall))
        return make_evidence(
            observable=observable,
            observable_version=_SUBJECT_VERSION,
            subject=subject,
            value=value,
            uncertainty=Uncertainty(n=n, method="none"),
            gauge=GaugeStatus.INVARIANT,
            calibration=None,
            provenance=provenance,
        )


# ---------------------------------------------------------------------------
# Readout construction
# ---------------------------------------------------------------------------


def _unembedding(model: "nn.Module") -> "torch.Tensor":
    """The `(vocab, d_model)` unembedding matrix, fp32.

    `get_output_embeddings()` is `transformers`' own accessor and it is correct under weight tying,
    where the unembedding is the input embedding and reading `lm_head.weight` off a model that has
    no `lm_head` attribute would fail. Falls back to the input embedding, which is what tying means.
    """
    import torch

    # `get_output_embeddings` and `get_input_embeddings` are `PreTrainedModel` methods and the torch
    # stubs resolve them through `nn.Module.__getattr__`, which is typed as returning a tensor. The
    # local Any binding is where that ends, rather than a `type: ignore` on each call.
    hf_model: Any = model
    head = hf_model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    if head is None and hasattr(model, "get_input_embeddings"):
        head = hf_model.get_input_embeddings()
    if head is None or not hasattr(head, "weight"):
        raise ValueError(
            "this model exposes no output or input embedding, so it has no unembedding directions "
            "to build a readout from."
        )
    return head.weight.detach().to(torch.float32)  # type: ignore[no-any-return]


def logit_readout(model: "nn.Module", token_id: int, site: Site, *, name: str = "") -> Readout:
    """A readout that reads one token's logit: the unembedding row for that token."""
    weight = _unembedding(model)
    return Readout(
        name=name or f"logit:{token_id}",
        kind="linear",
        site=site,
        position=PositionSpec("final"),
        vector=weight[token_id].contiguous(),
        meta={"token_id": int(token_id), "bias": 0.0},
    )


def contrast_readout(
    model: "nn.Module", positive: int, negative: int, site: Site, *, name: str = "decision"
) -> Readout:
    """A readout that reads the logit difference between two tokens.

    The policy-side analogue of a reward direction, and the direction a logit lens is read through.
    The difference rather than a single row on purpose: a single logit moves with everything that
    moves the residual norm, and the difference of two cancels that to first order, which is the
    same reason a preference pair is read as a differential on the grader side.
    """
    weight = _unembedding(model)
    return Readout(
        name=name,
        kind="logit_diff",
        site=site,
        position=PositionSpec("final"),
        vector=(weight[positive] - weight[negative]).contiguous(),
        meta={"positive_id": int(positive), "negative_id": int(negative), "bias": 0.0},
    )


def logprob_readout(site: Site) -> Readout:
    """The per-token log-probability readout: a value per position, no fixed direction."""
    return Readout(
        name="logprob",
        kind="token_value",
        site=site,
        position=PositionSpec("all"),
        vector=None,
        meta={"aggregate": "sum_over_completion"},
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def wrap_hf_policy(
    model: "nn.Module",
    tokenizer: Any,
    *,
    contrast: tuple[str, str] | tuple[int, int] | None = None,
    device: str = "cpu",
    numerics: Any = None,
    architecture: str | None = None,
    max_length: int = 2048,
    lineage: dict[str, Any] | None = None,
) -> HFPolicy:
    """Build an `HFPolicy` from an already-loaded causal LM and tokenizer. No download.

    ``contrast`` names the two tokens whose logit difference is the ``decision`` readout, as text or
    as ids. Absent, the policy exposes ``logprob`` only, which is enough for `score_under`,
    `sample` and `score_prefixes` and is not enough for a lens: an instrument that needs a direction
    then fails at readout lookup with the two constructors named, which is better than a default
    contrast nobody chose appearing in a published curve.
    """
    import torch

    from reward_lens.runtime.fingerprint import fingerprint
    from reward_lens.runtime.precision import resolve_policy

    model.eval()
    torch_device = torch.device(device)
    arch = describe(model)
    site_map = arch.site_map()
    arch_string = architecture or type(model).__name__
    numerics_policy = numerics if numerics is not None else resolve_policy(arch_string)

    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None):
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = getattr(tokenizer, "pad_token_id", None)
    pad_id = int(pad_id) if pad_id is not None else 0

    hf_model: Any = model
    head_module = (
        hf_model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    )
    runtime = HFPolicyRuntime(
        model=model,
        arch=arch,
        site_map=site_map,
        policy=numerics_policy,
        head_module=head_module,
        device=torch_device,
        pad_id=pad_id,
    )

    last = Site(max(arch.n_layers - 1, 0), "resid_post")
    readouts: list[Readout] = []
    if contrast is not None:
        pos, neg = _resolve_contrast(tokenizer, contrast)
        readouts.append(contrast_readout(model, pos, neg, last))
        readouts.append(logit_readout(model, pos, last))
        readouts.append(logit_readout(model, neg, last))
    readouts.append(logprob_readout(last))

    meta = PolicyMeta(
        fingerprint=fingerprint(model, tokenizer, "ArchitectureView"),
        adapter="ArchitectureView",
        architecture=arch_string,
        lineage=lineage or {"provenance_tier": "weights-verified"},
        template={"chat_template": getattr(tokenizer, "chat_template", None) is not None},
        numerics_policy=numerics_policy.name,
        d_model=site_map.d_model,
        n_layers=site_map.n_layers,
        n_heads=site_map.n_heads,
        vocab_size=int(getattr(getattr(model, "config", None), "vocab_size", 0) or 0),
    )
    return HFPolicy(
        runtime=runtime,
        meta=meta,
        readouts=readouts,
        tokenizer=tokenizer,
        policy=numerics_policy,
        max_length=max_length,
    )


def _resolve_contrast(tokenizer: Any, contrast: tuple[Any, Any]) -> tuple[int, int]:
    """Turn a text or id contrast into two token ids, refusing a multi-token side.

    A contrast whose positive side tokenizes to three tokens is not a contrast between two tokens
    and reading the first of the three is the silent wrong answer. Raising names which side and what
    it produced.
    """
    out: list[int] = []
    for side, value in zip(("positive", "negative"), contrast):
        if isinstance(value, int):
            out.append(value)
            continue
        ids = tokenizer(str(value), add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise ValueError(
                f"the {side} side of the contrast, {value!r}, tokenizes to {len(ids)} tokens "
                f"({ids}). A logit contrast is between two single tokens; pass ids directly, or "
                f"choose surface forms this tokenizer keeps whole."
            )
        out.append(int(ids[0]))
    return out[0], out[1]


def from_pretrained(
    model_id: str,
    *,
    contrast: tuple[str, str] | tuple[int, int] | None = None,
    device: str = "cpu",
    revision: str | None = None,
    **overrides: Any,
) -> HFPolicy:
    """Load a causal LM by id and wrap it as a policy.

    The tiny vehicle this package is developed against is
    ``trl-internal-testing/tiny-Qwen3ForCausalLM``: a real Qwen3 decoder stack, small enough to run
    a lens, a gradient and a Hessian-vector product on CPU in seconds.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs: dict[str, Any] = {}
    if revision:
        kwargs["revision"] = revision
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
    return wrap_hf_policy(
        model,
        tokenizer,
        contrast=contrast,
        device=device,
        lineage={"model_id": model_id, "revision": revision, "provenance_tier": "weights-verified"},
        **overrides,
    )


__all__ = [
    "POLICY_CAPS",
    "HFPolicy",
    "HFPolicyRuntime",
    "contrast_readout",
    "from_pretrained",
    "logit_readout",
    "logprob_readout",
    "wrap_hf_policy",
]

"""The HuggingFace runtime backend.

``HFRuntime`` implements the six-method ``Runtime`` protocol against a loaded ``transformers`` model
plus an adapter and a numerics policy. It ports v1's proven mechanics (the left-padded batched
forward, the per-layer hooks, OOM halve-and-retry) and adds the two capabilities v1 lacked: ``grad``
(autograd of a readout scalar w.r.t. a site's activation or the embeddings) and ``hvp`` (a
double-backprop Hessian-vector product on the readout scalar). Those two unlock Hessian
spectroscopy, gradient-ascent hack generation, incentive Jacobians, and second-order attribution.

Two design decisions make the readout exact and cheap. First, the reward is read by capturing the
**input to the reward head** with a forward pre-hook (the exact tensor the head consumes) and
projecting it onto the readout vector in fp32, rather than trusting the model's own head
output in the trunk dtype. On the tiny fp32 model this is bit-identical to the native logits; on a
bf16 8B model it is the more correct value and supersedes v1's coerce-head-to-bf16 hack. Second,
because a classifier pools the last token under causal attention, the head input at position ``t``
of a full-sequence forward equals the head input of a forward over the prefix ``y[:t+1]``; that is
what lets ``score_prefixes`` compute the whole per-token curve in one forward (verified to 1e-7).

Everything here runs on CPU for the tiny test vehicle. The real-model paths (bf16 trunks, CUDA,
length-bucketed population forwards, OOM recovery) are coded and exercised structurally, but the 8B
campaign models are GPU/download-gated per the hardware reality and are not loaded here.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, Callable, Literal, Sequence

from reward_lens.core.types import Site
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

    from reward_lens.runtime.precision import NumericsPolicy
    from reward_lens.signals.adapters import GraderAdapter
    from reward_lens.signals.base import PositionSpec


def _is_oom(exc: Exception) -> bool:
    """Whether an exception is a CUDA out-of-memory error (drives halve-and-retry)."""
    if type(exc).__name__ == "OutOfMemoryError":
        return True
    return "out of memory" in str(exc).lower()


def auto_batch_size(
    *,
    d_model: int,
    n_layers: int,
    seq_len: int = 2048,
    weight_gb: float = 16.0,
    headroom_gb: float = 8.0,
    bytes_per_activation: int = 2,
    fudge: float = 4.0,
) -> int:
    """Pick a forward batch size that should fit in free CUDA memory (ported from v1).

    Peak activation memory is roughly ``6 * B * seq_len * d_model * bytes`` per layer (residual +
    attention + MLP working set); the ``fudge`` factor covers hooks, workspace, and the wrapper's
    own bookkeeping. On CPU or without CUDA this returns 32, the safe preflight default. Snaps to a
    multiple of 16 in ``[16, 512]`` for tensor-core friendliness.
    """
    import torch

    if not torch.cuda.is_available():
        return 32
    free, _total = torch.cuda.mem_get_info()
    free_gb = free / (1024**3)
    available = max(0.5, free_gb - weight_gb - headroom_gb)
    bytes_per_pair = fudge * 6 * seq_len * d_model * bytes_per_activation
    raw = int((available * (1024**3)) // bytes_per_pair)
    return max(16, min(512, (raw // 16) * 16))


class HFRuntime:
    """A ``Runtime`` backed by a loaded HF model, an adapter, and a numerics policy.

    Construct via ``signals.loaders.wrap_hf_model`` / ``from_tiny``, which resolve the adapter, the
    site map, the head module, and the policy. The runtime is signal-agnostic: it captures the
    head-input hidden state and exposes it in ``RawOutput.extra["head_input"]`` (with the resolved
    final positions in ``extra["final_pos"]``), and the signal projects it onto its own readout
    vector in fp32. ``grad`` and ``hvp`` take a ``scalar_fn`` from the signal so the runtime never
    needs to know the readout direction.
    """

    def __init__(
        self,
        model: "nn.Module",
        adapter: "GraderAdapter",
        site_map: SiteMap,
        policy: "NumericsPolicy",
        head_module: "nn.Module | None",
        device: "torch.device",
        pad_id: int = 0,
    ):
        self.model = model
        self.adapter = adapter
        self.site_map = site_map
        self.policy = policy
        self.head_module = head_module
        self.device = device
        self.pad_id = pad_id

    # -- protocol: sites ----------------------------------------------------

    def sites(self) -> SiteMap:
        """The ``SiteMap`` this architecture exposes (adapter-resolved)."""
        return self.site_map

    # -- position + batching helpers ---------------------------------------

    def _final_positions(self, attention_mask: "torch.Tensor") -> "torch.Tensor":
        """The last valid (non-pad) token index per row, for left- or right-padding alike.

        One implementation, in `signals.adapters.final_positions`, because the per-token reward
        convention needs the same answer and two of these drift apart silently: the wrong form
        (`mask.sum(dim=1) - 1`) returns a plausible index rather than an error.
        """
        from reward_lens.signals.adapters import final_positions

        return final_positions(attention_mask)

    def collate(self, tokenized: Sequence[Any]) -> TokenBatch:
        """Left-pad a list of ``TokenizedInput`` into a ``TokenBatch``.

        Left padding aligns the final (response-end) token at column ``T-1`` for every row, so a
        final-token readout reads the same relative position for the whole batch. ``meta`` carries
        the per-row left-pad amounts and the tokenized inputs, which the score and capture paths use
        to map a per-item token index into padded coordinates.
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

    def forward(self, batch: TokenBatch) -> RawOutput:
        """Run a forward pass, capturing the head-input hidden state.

        Returns a ``RawOutput`` whose ``extra["head_input"]`` is the ``(B, T, d)`` tensor the reward
        head consumes and ``extra["final_pos"]`` the ``(B,)`` last-valid indices. ``reward`` carries
        the model's native scalar (adapter-extracted, trunk dtype) for reference and parity checks;
        the fp32 readout is the signal's job, computed from ``head_input``.
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
        reward = None
        try:
            reward = self.adapter.extract_reward_batch(
                out, {"input_ids": ids, "attention_mask": mask}
            )
        except (ValueError, AttributeError, IndexError):
            reward = None
        return RawOutput(
            reward=reward,
            logits=getattr(out, "logits", None),
            hidden=None,
            extra={
                "head_input": captured.get("head_input"),
                "final_pos": self._final_positions(mask),
            },
        )

    def _install_head_capture(self, sink: dict[str, Any]) -> list[Any]:
        """Register a pre-hook that stores the reward head's input tensor into ``sink``."""
        if self.head_module is None:
            return []

        def pre_hook(_module: Any, args: Any) -> None:
            sink["head_input"] = args[0] if isinstance(args, tuple) else args

        return [self.head_module.register_forward_pre_hook(pre_hook)]

    # -- protocol: capture --------------------------------------------------

    def forward_with_capture(
        self, batch: TokenBatch, spec: CaptureSpec
    ) -> tuple[RawOutput, Capture]:
        """Forward once, capturing the requested sites.

        Position resolution: for the default ``final`` (or ``None``) position the mount gathers the
        last-valid token and stores ``(B, d)`` per site; for any other position kind, or when
        ``spec.full_sequence`` is set, it stores the whole sequence. Captures and interventions share
        this mounting path, so a captured Observable composes with a mounted intervention unchanged.
        """
        import torch

        ids = batch.input_ids.to(self.device)
        mask = batch.attention_mask.to(self.device)
        final_pos = self._final_positions(mask)
        single_position = self._is_final_position(spec.position) and not spec.full_sequence
        positions = final_pos if single_position else None
        mount = CaptureMount(
            self.model,
            self.adapter,
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
        """Mount interventions via the shared hook path; remove them on exit."""
        with mounted_interventions(self.model, self.adapter, self.site_map, interventions):
            yield self

    # -- protocol: grad -----------------------------------------------------

    def grad(
        self,
        batch: TokenBatch,
        scalar_fn: Callable[[RawOutput], "torch.Tensor"],
        wrt: "Site | Literal['embeddings']",
    ) -> "torch.Tensor":
        """Autograd of the summed readout scalar w.r.t. a site's activation or the embeddings.

        With ``wrt="embeddings"`` the input embeddings are made a leaf and the gradient is returned
        as ``(B, T, d)``. With ``wrt`` a ``Site`` the activation at that site is cut into a leaf and
        the gradient w.r.t. it is returned as ``(B, T, d)``; a caller wanting the final-token
        gradient gathers it. The scalar is summed over the batch (per-item gradients do not
        interfere because each item's readout depends only on its own activations).
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

    # -- protocol: hvp ------------------------------------------------------

    def hvp(
        self,
        batch: TokenBatch,
        scalar_fn: Callable[[RawOutput], "torch.Tensor"],
        at: Site,
        vecs: "torch.Tensor",
    ) -> "torch.Tensor":
        """Hessian-vector products of the readout scalar w.r.t. a site, at the final positions.

        Double-backprop: cut a leaf at ``at``, take the first gradient with ``create_graph=True``,
        then differentiate ``(grad . v)`` again for each direction ``v``. ``vecs`` is ``(K, d)`` (a
        single ``(d,)`` vector is accepted and treated as ``K=1``); the return is ``(B, K, d)`` where
        entry ``[b, k]`` is ``H_b @ vecs[k]`` for item ``b``'s ``d x d`` reward Hessian at its final
        token. Passing ``vecs = I_d`` therefore materializes the dense Hessian for a single item,
        which is exactly how the acceptance test checks this method against a finite-difference
        reference. The head scalar is accumulated in fp32 regardless of trunk dtype.
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

    # -- internal: the differentiable forward ------------------------------

    def _graph_forward(
        self,
        *,
        input_ids: "torch.Tensor | None" = None,
        inputs_embeds: "torch.Tensor | None" = None,
        attention_mask: "torch.Tensor",
        leaf_site: Site | None = None,
    ) -> tuple[RawOutput, "torch.Tensor | None"]:
        """Forward with grad enabled, capturing the head input and (optionally) cutting a leaf.

        Shared by ``grad`` and ``hvp``. Runs the model without ``no_grad`` so the autograd graph
        from the leaf (embeddings or a cut site) through the reward head survives for a first and,
        for hvp, a second backward pass. Returns the ``RawOutput`` (with the grad-attached head
        input in ``extra``) and the leaf tensor when a site was cut.
        """
        captured: dict[str, Any] = {}
        handles = self._install_head_capture(captured)
        mount = (
            LeafCutMount(self.model, self.adapter, self.site_map, leaf_site)
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

    # -- scoring support (used by signals) ---------------------------------

    def final_head_inputs(
        self, tokenized: Sequence[Any], batch_size: int | None = None
    ) -> "torch.Tensor":
        """Pooled head-input hidden states at the final token, one row per input.

        Returns ``(n, d_model)`` in the input order (trunk dtype; the signal upcasts to fp32 for the
        projection). Chunks the inputs at ``batch_size`` and recovers from CUDA OOM by halving the
        chunk and retrying (the v1 mechanic), which never fires on CPU but keeps the population-scale
        path honest. The concatenation is over the uniform ``d_model`` axis, so halves recombine
        exactly.
        """
        import torch

        chunk = batch_size or len(tokenized)
        outs: list["torch.Tensor"] = []
        start = 0
        items = list(tokenized)
        while start < len(items):
            outs.append(self._final_head_inputs_chunk(items[start : start + chunk]))
            start += chunk
        return torch.cat(outs, dim=0)

    def _final_head_inputs_chunk(self, sub: Sequence[Any]) -> "torch.Tensor":
        import torch

        try:
            out = self.forward(self.collate(sub))
        except RuntimeError as exc:
            if not _is_oom(exc) or len(sub) <= 1:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            mid = len(sub) // 2
            first = self._final_head_inputs_chunk(list(sub)[:mid])
            second = self._final_head_inputs_chunk(list(sub)[mid:])
            return torch.cat([first, second], dim=0)
        head_input = out.extra.get("head_input")
        if head_input is None:
            raise RuntimeError(
                "no reward head module captured; cannot compute the fp32 readout. The adapter did "
                "not expose a linear head (score/regression_layer/v_head)."
            )
        final_pos = out.extra["final_pos"]
        batch_idx = torch.arange(head_input.shape[0], device=head_input.device)
        return head_input[batch_idx, final_pos]

    def full_head_inputs(self, batch: TokenBatch) -> tuple["torch.Tensor", list[list[int]]]:
        """The head input at every valid position, per item (drives ``score_prefixes``).

        Returns the ``(B, T, d)`` head-input tensor and, per row, the list of valid (non-pad) token
        indices in order. Because the classifier pools the last token under causal attention, the
        head input at position ``t`` equals the score of the prefix ending at ``t``; projecting it
        gives the whole per-token reward curve from one forward.
        """
        out = self.forward(batch)
        head_input = out.extra.get("head_input")
        if head_input is None:
            raise RuntimeError("no reward head module captured; cannot compute prefix curves.")
        tokenized = batch.meta.get("tokenized")
        offsets = batch.meta.get("offsets")
        valid_per_item: list[list[int]] = []
        if tokenized is not None and offsets is not None:
            for tok, offset in zip(tokenized, offsets):
                valid_per_item.append([offset + i for i in tok.valid_positions()])
        else:
            mask = batch.attention_mask
            for row in mask.tolist():
                valid_per_item.append([i for i, m in enumerate(row) if m])
        return head_input, valid_per_item


__all__ = ["HFRuntime", "auto_batch_size"]

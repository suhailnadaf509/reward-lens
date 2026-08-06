"""``ProcessRM``: a process (step-level) reward model behind the protocol (adapter 3).

A process reward model scores each reasoning step, not just the final answer. Architecturally it is a
sequence classifier whose scalar head is read at every step boundary rather than only at the last token,
which is precisely why positions are first-class: the ``ProcessRM`` reuses the classifier
head direction ``w_r`` verbatim and changes only the ``PositionSpec`` from ``final`` to ``step_ends``.

Step-boundary detection has two paths: an explicit delimiter config (the common case: a
model trained with ``\n`` or an explicit step marker between steps) and a learned fallback for solutions
with no reliable delimiter. The delimiter path is implemented and exact. The learned fallback (a trained
boundary classifier over the residual stream) is a STUB here: when the delimiter yields fewer than two
steps the whole response is treated as a single step and the fallback is recorded as unavailable, so a
caller is never silently handed a wrong segmentation. ``STEP_SCORES`` is the declared capability.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from reward_lens.core.evidence import register_payload
from reward_lens.core.types import Capability, Site
from reward_lens.signals._common import SignalImplBase, build_hf_runtime, split_item
from reward_lens.signals.base import PositionSpec, Readout, Scores, TokenCurves, TokenizedInput

if TYPE_CHECKING:
    import torch

    from reward_lens.runtime.hf import HFRuntime

_PROCESS_CAPS = (
    Capability.SCORES
    | Capability.PREFIX_SCORES
    | Capability.ACTIVATIONS
    | Capability.GRADIENTS
    | Capability.HVP
    | Capability.LINEAR_READOUT
    | Capability.STEP_SCORES
)

_LEARNED_FALLBACK_NOTE = (
    "learned step-boundary detector is a stub: a trained boundary classifier over "
    "the residual stream is the production fallback; here a solution with no delimiter is one step."
)


@register_payload
@dataclass
class StepScores:
    """Per-item, per-step reward scores from a process reward model (the payload of ``step_scores``).

    ``curves`` is a ragged collection: one array of per-step scores per item, in step order.
    ``step_counts`` records the number of detected steps per item so a consumer can index steps
    without re-detecting boundaries. This is the ``STEP_SCORES`` analogue of ``TokenCurves`` and is a
    registered Evidence payload so it round-trips through the store exactly.
    """

    curves: list["np.ndarray"]
    step_counts: list[int]
    readout: str = "reward"


class ProcessRM(SignalImplBase):
    """A step-level reward model as a ``RewardSignal`` (adapter 3).

    Build it through ``from_tiny`` or ``from_sequence_classifier``. ``score`` returns the outcome
    scalar (the final-token reward of the whole solution, identical to a classifier); ``step_scores``
    returns the per-step reward vector read at the detected step boundaries. The step delimiter is
    configurable; the default is a newline.
    """

    observable_prefix = "signals.process"

    def __init__(
        self,
        *,
        runtime: "HFRuntime",
        meta: Any,
        policy: Any,
        tokenizer: Any,
        readouts: Sequence[Readout],
        delimiter: str = "\n",
        max_length: int = 2048,
        default_batch_size: int = 16,
        interventions: tuple[Any, ...] = (),
    ) -> None:
        super().__init__(
            runtime=runtime,
            meta=meta,
            policy=policy,
            tokenizer=tokenizer,
            readouts=readouts,
            caps=_PROCESS_CAPS,
            max_length=max_length,
            default_batch_size=default_batch_size,
            interventions=interventions,
        )
        self.delimiter = delimiter

    # -- rendering with step spans -----------------------------------------

    def _render(self, item: Any) -> tuple[str, tuple[tuple[int, int, str], ...], dict[str, Any]]:
        """Render a ``(prompt, solution)`` item and type each reasoning step as a ``step`` span.

        The solution is split on the configured delimiter; each non-empty segment becomes a ``step``
        character span over the templated text, which ``tokenize`` maps into token coordinates. When
        the split yields fewer than two steps the whole solution is one step and the learned-fallback
        stub is noted in the returned meta.
        """
        prompt, solution, raw = split_item(item)
        text = solution if raw else self._chat(prompt, solution)
        spans, count, fell_back = _step_char_spans(text, solution, self.delimiter)
        meta: dict[str, Any] = {"prompt": prompt, "response": solution, "n_steps": count}
        if fell_back:
            meta["step_detection"] = _LEARNED_FALLBACK_NOTE
        return text, spans, meta

    # -- scoring ------------------------------------------------------------

    def score(self, view: Any, readout: str | None = None) -> Any:
        """The outcome scalar per item: the final-token reward of the whole solution."""
        name = readout or self.default_readout_name()
        read = self.readout(name)
        items = list(view)
        started = time.perf_counter()
        tokenized = [self.tokenize(it) for it in items]
        n_tokens = int(sum(len(t.input_ids) for t in tokenized))
        values = self.project_final(tokenized, read.vector, float(read.meta.get("bias", 0.0)))
        payload = Scores(values=values, readout=name, n_items=len(items))
        return self._timed_evidence("score", payload, name, len(items), n_tokens, started)

    def score_prefixes(self, view: Any, readout: str | None = None) -> Any:
        """Per-token reward curve for each item (the classifier prefix curve)."""
        name = readout or self.default_readout_name()
        read = self.readout(name)
        items = list(view)
        started = time.perf_counter()
        tokenized = [self.tokenize(it) for it in items]
        curves, n_tokens = self.linear_prefix_curves(
            tokenized, read.vector, float(read.meta.get("bias", 0.0))
        )
        payload = TokenCurves(curves=curves, readout=name)
        return self._timed_evidence("score_prefixes", payload, name, len(items), n_tokens, started)

    def step_scores(self, view: Any, readout: str | None = None) -> Any:
        """Per-step reward scores read at the detected step boundaries (STEP_SCORES).

        For each item the head input at each step-end token is projected onto the reward direction in
        fp32, giving one score per reasoning step. Returns ``Evidence[StepScores]``; the last step's
        score equals the outcome ``score`` because the final step ends at the final token.
        """
        import torch

        name = readout or self.default_readout_name()
        read = self.readout(name)
        bias = float(read.meta.get("bias", 0.0))
        items = list(view)
        started = time.perf_counter()
        tokenized = [self.tokenize(it) for it in items]
        curves: list[np.ndarray] = [np.empty(0)] * len(items)
        counts: list[int] = [0] * len(items)
        n_tokens = 0
        batch = self.default_batch_size
        with self._mounted():
            for start in range(0, len(tokenized), batch):
                sub = tokenized[start : start + batch]
                token_batch = self.runtime.collate(sub)
                head_input, valid_per_item = self.runtime.full_head_inputs(token_batch)
                for local_i, valid in enumerate(valid_per_item):
                    tok = sub[local_i]
                    step_ends_local = _step_end_positions(tok)
                    # Map per-item token index -> padded coordinate via the row's valid list.
                    padded = [valid[j] for j in step_ends_local]
                    idx = torch.tensor(padded, device=head_input.device, dtype=torch.long)
                    rows = head_input[local_i].index_select(0, idx)
                    step_curve = self.policy.head_project(rows, read.vector, bias)
                    curves[start + local_i] = step_curve.detach().to("cpu", torch.float32).numpy()
                    counts[start + local_i] = len(padded)
                    n_tokens += len(valid)
        payload = StepScores(curves=curves, step_counts=counts, readout=name)
        return self._timed_evidence("step_scores", payload, name, len(items), n_tokens, started)

    # -- constructors -------------------------------------------------------

    @classmethod
    def from_sequence_classifier(
        cls,
        model: "torch.nn.Module",
        tokenizer: Any,
        *,
        delimiter: str = "\n",
        device: str = "cpu",
        architecture: str | None = None,
        lineage: dict[str, Any] | None = None,
    ) -> "ProcessRM":
        """Wrap an already-loaded sequence-classifier reward model as a process RM (no download).

        Reads the scalar reward head (``score``) direction into a single ``reward`` readout whose
        position is ``step_ends`` rather than ``final``. A production PRM would be a checkpoint
        trained with a step-level objective; the adapter's mechanics are identical either way.
        """
        import torch

        from reward_lens.signals.adapters import reward_head_module

        head = reward_head_module(None, model)
        if head is None:
            raise ValueError(
                "ProcessRM needs a scalar reward head (score/regression_layer/v_head)."
            )
        runtime, meta, policy = build_hf_runtime(
            model, tokenizer, head, architecture=architecture, device=device, lineage=lineage
        )
        weight = head.weight.data.detach().to(torch.float32)
        vec = (weight if weight.ndim == 1 else weight[0]).contiguous()
        bias = 0.0 if head.bias is None else float(head.bias.data.reshape(-1)[0])
        site = Site(max(meta.n_layers - 1, 0), "resid_post")
        readout = Readout(
            name="reward",
            kind="linear",
            site=site,
            position=PositionSpec("step_ends", detail="step"),
            vector=vec,
            meta={"bias": bias, "aggregate": "single"},
        )
        return cls(
            runtime=runtime,
            meta=meta,
            policy=policy,
            tokenizer=tokenizer,
            readouts=[readout],
            delimiter=delimiter,
        )

    @classmethod
    def from_tiny(cls, *, seed: int = 0, delimiter: str = "\n", **kw: Any) -> "ProcessRM":
        """Construct the tiny offline process RM the tests run on (a tiny sequence classifier)."""
        model, tokenizer = _tiny_sequence_classifier(seed=seed, num_labels=1, **kw)
        return cls.from_sequence_classifier(
            model,
            tokenizer,
            delimiter=delimiter,
            architecture="LlamaForSequenceClassification",
            lineage={"provenance_tier": "weights-verified", "tiny": True},
        )


# ---------------------------------------------------------------------------
# step detection helpers
# ---------------------------------------------------------------------------


def _step_char_spans(
    text: str, solution: str, delimiter: str
) -> tuple[tuple[tuple[int, int, str], ...], int, bool]:
    """Character spans (one per step) over ``text``, plus the step count and a fallback flag.

    Splits ``solution`` on ``delimiter``, locating each step's character range inside ``text``. A
    solution that does not split into at least two steps falls back to a single whole-solution step
    and the flag is True so the caller can record the learned-fallback stub.
    """
    base = text.find(solution)
    if base < 0:
        base = 0
    segments = _segments(solution, delimiter)
    fell_back = len(segments) < 2
    spans: list[tuple[int, int, str]] = []
    for seg_start, seg_end in segments:
        spans.append((base + seg_start, base + seg_end, "step"))
    if not spans:
        spans.append((base, base + len(solution), "step"))
    return tuple(spans), len(spans), fell_back


def _segments(solution: str, delimiter: str) -> list[tuple[int, int]]:
    """Non-empty ``(start, end)`` character ranges of the delimiter-separated steps in ``solution``."""
    out: list[tuple[int, int]] = []
    cursor = 0
    if not delimiter:
        return [(0, len(solution))] if solution.strip() else []
    for part in solution.split(delimiter):
        start = cursor
        end = cursor + len(part)
        if part.strip():
            out.append((start, end))
        cursor = end + len(delimiter)
    return out


def _step_end_positions(tok: TokenizedInput) -> list[int]:
    """The per-item token index (into the row's valid positions) of each step's last token.

    Reads the ``step`` spans carried on the tokenized input and returns, for each step, the index of
    its final valid token *relative to the row's valid-position list*, so it composes with the padded
    batch coordinates the caller resolves. Falls back to the final valid token when no step spans
    survived tokenization.
    """
    valid = tok.valid_positions()
    valid_rank = {p: i for i, p in enumerate(valid)}
    ends: list[int] = []
    for span in tok.spans:
        if span.kind != "step":
            continue
        last_tok = min(span.end - 1, valid[-1]) if valid else 0
        ends.append(valid_rank.get(last_tok, len(valid) - 1))
    if not ends:
        ends = [len(valid) - 1]
    # Ensure the final step ends at the final token so step_scores[-1] == score.
    if ends[-1] != len(valid) - 1:
        ends.append(len(valid) - 1)
    return ends


def _tiny_sequence_classifier(
    *,
    seed: int = 0,
    num_labels: int = 1,
    d_model: int = 32,
    n_layers: int = 2,
    n_heads: int = 4,
    seq_max: int = 256,
    tokenizer_name: str = "gpt2",
) -> tuple[Any, Any]:
    """Build a tiny ``LlamaForSequenceClassification`` + tokenizer (shared by process/rubric/traj)."""
    import torch
    from transformers import LlamaConfig, LlamaForSequenceClassification

    from reward_lens.signals.loaders import _build_tokenizer

    tokenizer = _build_tokenizer(tokenizer_name)
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
        num_labels=num_labels,
        attn_implementation="eager",
    )
    model = LlamaForSequenceClassification(config).eval()
    return model, tokenizer


__all__ = ["ProcessRM", "StepScores"]

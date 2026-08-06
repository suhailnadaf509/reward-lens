"""``GenerativeJudge``: an LLM-as-judge behind the ``RewardSignal`` protocol (adapter 2).

A generative judge does not have a scalar reward head. It emits a verdict *token* ("Yes"/"No", "A"/"B",
a rating), and its reward is read off the unembedding: the pointwise verdict is the logit difference
``W_U[Yes] - W_U[No]`` at the judgment position, which is a first-class ``logit_diff`` ``Readout``
whose vector is that difference of two rows of the LM head. This is exactly the payoff:
crystallization depth of a judge's verdict is the same Observable as crystallization depth of a
scalar head, called with a different readout. The adapter wraps a ``CausalLM`` (not a sequence classifier), so its runtime
captures the ``lm_head`` input rather than a ``score`` head input.

Three readouts:

  - ``verdict`` (``logit_diff``): ``W_U[Yes] - W_U[No]`` for pointwise judging.
  - ``verdict_ab`` (``logit_diff``): ``W_U[A] - W_U[B]`` for pairwise A/B judging.
  - ``likert`` (``simplex``): the expected score under the softmax over the score tokens 1..K.

The judgment position is detected from the chat template (the sequence is templated with the generation
prompt appended, so the verdict token is emitted at the final valid position) and *validated*: a forward
over k calibration prompts checks how often the model's greedy next token at that position is actually a
verdict token, and the detection confidence is recorded in ``meta.lineage["judgment_detection"]``. On the
random tiny model that confidence is near chance; the structure is real and the production path (a real
instruct judge) is marked. Nothing here fabricates a verdict from a model that was not run: every score is
the genuine fp32 projection of the model's own final hidden state onto the readout direction.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from reward_lens.core.types import Capability, Site
from reward_lens.signals._common import (
    SignalImplBase,
    build_hf_runtime,
    single_token_id,
    split_item,
)
from reward_lens.signals.base import PositionSpec, Readout, Scores, TokenCurves, TokenizedInput

if TYPE_CHECKING:
    import torch

    from reward_lens.runtime.hf import HFRuntime

# The judge's declared capabilities: it scores, exposes per-token curves, captures activations,
# supports autograd on the logit-diff direction, exposes a linear (logit_diff) readout, and is
# GENERATIVE. It has no scalar reward head, so it does not claim MULTI_READOUT the way ArmoRM does.
_JUDGE_CAPS = (
    Capability.SCORES
    | Capability.PREFIX_SCORES
    | Capability.ACTIVATIONS
    | Capability.GRADIENTS
    | Capability.HVP
    | Capability.LINEAR_READOUT
    | Capability.GENERATIVE
)


class GenerativeJudge(SignalImplBase):
    """An LLM-as-judge as a ``RewardSignal`` (adapter 2).

    Build it through ``from_tiny`` (the offline test vehicle) or ``from_causal_lm`` (an
    already-loaded instruct model). The verdict readouts are ``logit_diff`` directions read off the
    LM head; the Likert readout is a ``simplex`` over score tokens. ``score`` dispatches on the
    readout kind: a projection for the logit-diff verdicts, an expected value under the score-token
    softmax for the Likert.
    """

    observable_prefix = "signals.judge"

    def __init__(
        self,
        *,
        runtime: "HFRuntime",
        meta: Any,
        policy: Any,
        tokenizer: Any,
        readouts: Sequence[Readout],
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
            caps=_JUDGE_CAPS,
            max_length=max_length,
            default_batch_size=default_batch_size,
            interventions=interventions,
            add_generation_prompt=True,
        )

    # -- rendering: the judging prompt -------------------------------------

    def _render(self, item: Any) -> tuple[str, tuple[tuple[int, int, str], ...], dict[str, Any]]:
        """Render an item into a judging prompt that ends at the verdict position.

        A 2-item ``(question, answer)`` becomes a pointwise judging prompt; a 3-item
        ``(question, answer_a, answer_b)`` becomes a pairwise prompt. In both cases the generation
        prompt is appended so the final valid token is the judgment position. The evaluated answer(s)
        are typed as ``critique_target`` spans so a later mediation experiment can patch exactly the
        content the verdict is about.
        """
        pair = _as_pairwise(item)
        if pair is not None:
            question, ans_a, ans_b = pair
            body = (
                f"You are comparing two answers.\nQuestion: {question}\n"
                f"Answer A: {ans_a}\nAnswer B: {ans_b}\nWhich answer is better? Reply A or B."
            )
            full = self._chat(body, None, add_generation_prompt=True)
            spans = _find_spans(full, [(ans_a, "critique_target"), (ans_b, "critique_target")])
            return full, spans, {"kind": "pairwise", "question": question}
        question, answer, _raw = split_item(item)
        body = (
            f"You are grading an answer.\nQuestion: {question}\nAnswer: {answer}\n"
            f"Is the answer correct? Reply Yes or No."
        )
        full = self._chat(body, None, add_generation_prompt=True)
        spans = _find_spans(full, [(answer, "critique_target")])
        return full, spans, {"kind": "pointwise", "question": question}

    # -- scoring ------------------------------------------------------------

    def score(self, view: Any, readout: str | None = None) -> Any:
        """Score every item under a readout, returning ``Evidence[Scores]``.

        ``logit_diff`` readouts (``verdict``, ``verdict_ab``) project the final hidden state onto the
        readout direction in fp32: ``h . (W_U[a] - W_U[b])`` is exactly ``logit(a) - logit(b)`` for
        the token the model is about to emit. The ``simplex`` readout (``likert``) returns the
        expected score under the softmax over the score tokens. Gauge INVARIANT, trust EXPLORATORY.
        """
        name = readout or self.default_readout_name()
        read = self.readout(name)
        items = list(view)
        started = time.perf_counter()
        tokenized = [self.tokenize(it) for it in items]
        n_tokens = int(sum(len(t.input_ids) for t in tokenized))
        if read.kind == "simplex":
            values = self._likert_values(tokenized, read)
        else:
            bias = float(read.meta.get("bias", 0.0))
            values = self.project_final(tokenized, read.vector, bias)
        payload = Scores(values=values, readout=name, n_items=len(items))
        return self._timed_evidence("score", payload, name, len(items), n_tokens, started)

    def score_prefixes(self, view: Any, readout: str | None = None) -> Any:
        """Per-token verdict curves ``verdict(y_{1:t})`` for every item.

        For a ``logit_diff`` readout the curve at position ``t`` is the verdict the model would emit
        if the sequence ended at ``t``; its final entry equals ``score``. Only defined for the
        logit-diff verdicts (a simplex has no single direction to trace); raises otherwise.
        """
        name = readout or self.default_readout_name()
        read = self.readout(name)
        if read.kind != "logit_diff" or read.vector is None:
            raise ValueError(
                f"score_prefixes is defined for the logit_diff verdict readouts; {name!r} is kind "
                f"{read.kind!r}."
            )
        items = list(view)
        started = time.perf_counter()
        tokenized = [self.tokenize(it) for it in items]
        bias = float(read.meta.get("bias", 0.0))
        curves, n_tokens = self.linear_prefix_curves(tokenized, read.vector, bias)
        payload = TokenCurves(curves=curves, readout=name)
        return self._timed_evidence("score_prefixes", payload, name, len(items), n_tokens, started)

    def _likert_values(self, tokenized: Sequence[TokenizedInput], read: Readout) -> np.ndarray:
        """Expected Likert score under the softmax over the score tokens (the simplex readout)."""
        import torch

        score_ids = list(read.meta["score_ids"])
        values = torch.tensor(read.meta["values"], dtype=torch.float32)
        logits, valid = self.full_logits(tokenized)
        out = np.empty(len(tokenized), dtype=np.float32)
        for i, positions in enumerate(valid):
            final = positions[-1]
            row = logits[i, final, score_ids].to(torch.float32)
            probs = torch.softmax(row, dim=-1)
            out[i] = float((probs * values).sum())
        return out

    # -- judgment-position detection + validation --------------------------

    def validate_judgment_position(
        self, calibration_items: Sequence[Any], k: int = 4
    ) -> dict[str, Any]:
        """Validate that the verdict token lands at the detected judgment position.

        Detection is structural: templating with the generation prompt makes the final valid token
        the judgment position. Validation runs a forward over up to ``k`` calibration prompts and
        checks how often the model's greedy next token there is one of the verdict tokens; the
        fraction is the detection confidence. On the random tiny model this is near chance and is
        recorded honestly. The production path samples a real instruct model, which this structure is
        ready for but does not stand in for. Returns the detection record stored in ``meta.lineage``.
        """
        import torch

        verdict_ids: set[int] = set()
        for name in ("verdict", "verdict_ab", "likert"):
            if name not in self._readout_by_name:
                continue
            read = self._readout_by_name[name]
            for key in ("a_id", "b_id"):
                if key in read.meta:
                    verdict_ids.add(int(read.meta[key]))
            for tid in read.meta.get("score_ids", ()):
                verdict_ids.add(int(tid))

        items = list(calibration_items)[:k]
        hits = 0
        checked = 0
        if items and verdict_ids:
            tokenized = [self.tokenize(it) for it in items]
            logits, valid = self.full_logits(tokenized)
            for i, positions in enumerate(valid):
                greedy = int(torch.argmax(logits[i, positions[-1]]).item())
                hits += 1 if greedy in verdict_ids else 0
                checked += 1
        confidence = (hits / checked) if checked else 0.0
        record = {
            "position": "final-valid-token(add_generation_prompt)",
            "validated": checked > 0,
            "confidence": confidence,
            "k": checked,
            "verdict_token_ids": sorted(verdict_ids),
            "note": (
                "confidence is measured on the wrapped model; on a random tiny model it is near "
                "chance. A production judge is a real instruct model that emits genuine verdicts."
            ),
        }
        self.meta.lineage["judgment_detection"] = record
        return record

    # -- constructors -------------------------------------------------------

    @classmethod
    def from_causal_lm(
        cls,
        model: "torch.nn.Module",
        tokenizer: Any,
        *,
        yes: str = "Yes",
        no: str = "No",
        option_a: str = "A",
        option_b: str = "B",
        likert: Sequence[str] = ("1", "2", "3", "4", "5"),
        device: str = "cpu",
        architecture: str | None = None,
        lineage: dict[str, Any] | None = None,
        validate_with: Sequence[Any] | None = None,
    ) -> "GenerativeJudge":
        """Wrap an already-loaded ``CausalLM`` + tokenizer as a judge (no download).

        Reads the verdict directions off the LM head: ``W_U[yes] - W_U[no]`` for the pointwise
        verdict, ``W_U[A] - W_U[B]`` for the pairwise, and the score-token ids for the Likert
        simplex. Runs the judgment-position validation on ``validate_with`` if given (else a small
        built-in calibration set). This is the entry point a real instruct judge uses on adequate
        hardware; ``from_tiny`` calls it on the synthetic model.
        """
        import torch

        lm_head = _lm_head_module(model)
        runtime, meta, policy = build_hf_runtime(
            model,
            tokenizer,
            lm_head,
            architecture=architecture,
            device=device,
            lineage=lineage,
        )
        w_u = lm_head.weight.data.detach().to(torch.float32)  # (vocab, d_model)
        site = Site(max(meta.n_layers - 1, 0), "resid_post")
        jpos = PositionSpec("judgment")

        def logit_diff(name: str, plus: str, minus: str) -> Readout:
            pid, mid = single_token_id(tokenizer, plus), single_token_id(tokenizer, minus)
            vec = (w_u[pid] - w_u[mid]).contiguous()
            return Readout(
                name=name,
                kind="logit_diff",
                site=site,
                position=jpos,
                vector=vec,
                meta={"a_id": pid, "b_id": mid, "a": plus, "b": minus, "bias": 0.0},
            )

        score_ids = [single_token_id(tokenizer, s) for s in likert]
        likert_readout = Readout(
            name="likert",
            kind="simplex",
            site=site,
            position=jpos,
            vector=None,
            meta={"score_ids": score_ids, "values": [float(i + 1) for i in range(len(score_ids))]},
        )
        readouts = [
            logit_diff("verdict", yes, no),
            logit_diff("verdict_ab", option_a, option_b),
            likert_readout,
        ]
        judge = cls(
            runtime=runtime, meta=meta, policy=policy, tokenizer=tokenizer, readouts=readouts
        )
        judge.validate_judgment_position(
            validate_with if validate_with is not None else _DEFAULT_CALIBRATION
        )
        return judge

    @classmethod
    def from_tiny(
        cls,
        *,
        d_model: int = 32,
        n_layers: int = 2,
        n_heads: int = 4,
        seed: int = 0,
        seq_max: int = 256,
        tokenizer_name: str = "gpt2",
    ) -> "GenerativeJudge":
        """Construct the tiny offline judge the tests run on (tiny vehicles).

        A real ``LlamaForCausalLM`` (hidden 32, 2 layers) with a real tokenizer, so the adapter,
        hooks, LM-head capture, and logit-diff readout see the same module tree a production judge
        would; only the magnitudes differ, and the weights are random so no verdict is meaningful.
        The verdict *mechanism* (reading ``W_U[Yes] - W_U[No]`` off the head) is exact and real.
        """
        model, tokenizer = _tiny_causal_lm(
            seed=seed,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            seq_max=seq_max,
            tokenizer_name=tokenizer_name,
        )
        return cls.from_causal_lm(
            model,
            tokenizer,
            architecture="LlamaForCausalLM",
            lineage={"provenance_tier": "weights-verified", "tiny": True},
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_DEFAULT_CALIBRATION: tuple[tuple[str, str], ...] = (
    ("What is 2+2?", "4"),
    ("Capital of France?", "Paris"),
    ("Is the sky green?", "Yes"),
    ("Name a fruit.", "Apple"),
)


def _as_pairwise(item: Any) -> tuple[str, str, str] | None:
    """Detect a pairwise ``(question, answer_a, answer_b)`` item; return None for pointwise."""
    if isinstance(item, (tuple, list)) and len(item) == 3:
        return str(item[0]), str(item[1]), str(item[2])
    if isinstance(item, dict) and "a" in item and "b" in item:
        return str(item.get("prompt", item.get("question", ""))), str(item["a"]), str(item["b"])
    return None


def _find_spans(text: str, needles: list[tuple[str, str]]) -> tuple[tuple[int, int, str], ...]:
    """Locate each ``(substring, kind)`` in ``text`` and return character spans for those found."""
    out: list[tuple[int, int, str]] = []
    for needle, kind in needles:
        if not needle:
            continue
        start = text.find(needle)
        if start >= 0:
            out.append((start, start + len(needle), kind))
    return tuple(out)


def _tiny_causal_lm(
    *,
    seed: int = 0,
    tokenizer: Any = None,
    d_model: int = 32,
    n_layers: int = 2,
    n_heads: int = 4,
    seq_max: int = 256,
    tokenizer_name: str = "gpt2",
) -> tuple["torch.nn.Module", Any]:
    """Build a tiny ``LlamaForCausalLM`` + tokenizer (shared by the judge and the implicit RM).

    A real ``LlamaForCausalLM`` with the given seed, so two calls with different seeds give distinct
    models over one tokenizer (which is what ``ImplicitRM`` needs for a nonzero log-ratio). Reuses the
    loaders' offline-capable tokenizer builder so the tests never touch the network.
    """
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    from reward_lens.signals.loaders import _build_tokenizer

    if tokenizer is None:
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
        attn_implementation="eager",
    )
    model = LlamaForCausalLM(config).eval()
    return model, tokenizer


def _lm_head_module(model: "torch.nn.Module") -> "torch.nn.Module":
    """Return the model's unembedding (``lm_head``) linear module, or raise a clear error."""
    import torch.nn as nn

    head = getattr(model, "lm_head", None)
    if isinstance(head, nn.Linear):
        return head
    out = getattr(model, "get_output_embeddings", None)
    if callable(out):
        head = out()
        if isinstance(head, nn.Linear):
            return head
    raise ValueError(
        "GenerativeJudge wraps a CausalLM with a linear lm_head; this model exposes none. "
        "A tied-embedding model still returns its output embedding via get_output_embeddings()."
    )


__all__ = ["GenerativeJudge"]

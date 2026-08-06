"""``TrajectoryRM``: a trajectory-level reward model behind the protocol (adapter 6).

An agent's reward is over a whole episode, not a single response, and the episode has structure the
receipt/narrative sciences read: a *receipt* is the evidence a step produced (a tool result), a
*narrative* is the agent's own account of it, and an *action* is what it did. This adapter consumes
``Trajectory`` items (``reward_lens.data.schema``), renders them to text while carrying those typed spans
into token coordinates, and scores at the trajectory scoring position (the end of the episode). Declaring
``SPAN_TYPES`` is what lets a receipt-falsification or narrative-patching experiment address exactly the
right tokens; without the span carry-through those experiments silently misalign.

The scorer itself is a sequence classifier, so it reuses the classifier's fp32 projection. The work
specific to this adapter is the rendering: turning a structured episode into text and mapping each step's
receipt/narrative/action spans through the concatenation into spans over the rendered whole.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Sequence

from reward_lens.core.types import Capability, Site
from reward_lens.signals._common import SignalImplBase, build_hf_runtime
from reward_lens.signals.base import PositionSpec, Readout, Scores, TokenCurves

if TYPE_CHECKING:
    import torch

    from reward_lens.runtime.hf import HFRuntime

_TRAJECTORY_CAPS = (
    Capability.SCORES
    | Capability.PREFIX_SCORES
    | Capability.ACTIVATIONS
    | Capability.GRADIENTS
    | Capability.HVP
    | Capability.LINEAR_READOUT
    | Capability.SPAN_TYPES
)


class TrajectoryRM(SignalImplBase):
    """A trajectory-level reward model as a ``RewardSignal`` (adapter 6).

    Build it through ``from_tiny`` or ``from_sequence_classifier``. ``tokenize`` accepts a
    ``Trajectory`` (rendering its steps with receipt/narrative/action span typing) or a plain
    ``(prompt, response)`` item (the classifier rendering, so the conformance suite's generic checks
    apply). ``score`` reads the reward at the trajectory scoring position (the final token).
    """

    observable_prefix = "signals.trajectory"

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
            caps=_TRAJECTORY_CAPS,
            max_length=max_length,
            default_batch_size=default_batch_size,
            interventions=interventions,
        )

    # -- rendering a trajectory with typed spans ---------------------------

    def _render(self, item: Any) -> tuple[str, tuple[tuple[int, int, str], ...], dict[str, Any]]:
        """Render a ``Trajectory`` to text with receipt/narrative/action spans.

        Each step is rendered as ``Action: <action>\\n<step text>\\n``. The action string is typed as
        an ``action`` span; each step's ``receipts`` and ``narrative`` spans are interpreted as
        character ranges into that step's text and shifted into the rendered whole, so a span typed on
        a step survives into the full-sequence token coordinates. A non-``Trajectory`` item falls back
        to the classifier rendering so the generic conformance checks still exercise this adapter.
        """
        if not _is_trajectory(item):
            return super()._render(item)

        parts: list[str] = []
        char_spans: list[tuple[int, int, str]] = []
        cursor = 0

        def emit(chunk: str) -> int:
            nonlocal cursor
            start = cursor
            parts.append(chunk)
            cursor += len(chunk)
            return start

        if item.prompt_text:
            emit(f"Task: {item.prompt_text}\n")
        for step in item.steps:
            action_start = emit("Action: ") + 0
            action_text_start = cursor
            emit(f"{step.action}\n")
            char_spans.append((action_text_start, action_text_start + len(step.action), "action"))
            step_text_start = cursor
            emit(f"{step.text}\n")
            for span in step.receipts:
                char_spans.append(
                    (step_text_start + span.start, step_text_start + span.end, "receipt")
                )
            for span in step.narrative:
                char_spans.append(
                    (step_text_start + span.start, step_text_start + span.end, "narrative")
                )
            _ = action_start  # rendered prefix already emitted; kept explicit for clarity
        text = "".join(parts)
        meta = {"n_steps": len(item.steps), "outcome": dict(item.outcome)}
        return text, tuple(char_spans), meta

    # -- scoring at the trajectory position --------------------------------

    def score(self, view: Any, readout: str | None = None) -> Any:
        """Score each trajectory at its scoring position (the final token)."""
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
        """Per-token reward curve over the rendered trajectory."""
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

    # -- constructors -------------------------------------------------------

    @classmethod
    def from_sequence_classifier(
        cls,
        model: "torch.nn.Module",
        tokenizer: Any,
        *,
        device: str = "cpu",
        architecture: str | None = None,
        lineage: dict[str, Any] | None = None,
    ) -> "TrajectoryRM":
        """Wrap an already-loaded sequence-classifier reward model as a trajectory RM (no download)."""
        import torch

        from reward_lens.signals.adapters import reward_head_module

        head = reward_head_module(None, model)
        if head is None:
            raise ValueError("TrajectoryRM needs a scalar reward head (score/regression_layer).")
        runtime, meta, policy = build_hf_runtime(
            model, tokenizer, head, architecture=architecture, device=device, lineage=lineage
        )
        weight = head.weight.data.detach().to(torch.float32)
        vec = (weight if weight.ndim == 1 else weight[0]).contiguous()
        bias = 0.0 if head.bias is None else float(head.bias.data.reshape(-1)[0])
        readout = Readout(
            name="reward",
            kind="linear",
            site=Site(max(meta.n_layers - 1, 0), "resid_post"),
            position=PositionSpec("final"),
            vector=vec,
            meta={"bias": bias, "aggregate": "single", "scoring_position": "trajectory_end"},
        )
        return cls(
            runtime=runtime, meta=meta, policy=policy, tokenizer=tokenizer, readouts=[readout]
        )

    @classmethod
    def from_tiny(cls, *, seed: int = 0, **kw: Any) -> "TrajectoryRM":
        """Construct the tiny offline trajectory RM the tests run on (a tiny sequence classifier)."""
        from reward_lens.signals.process import _tiny_sequence_classifier

        model, tokenizer = _tiny_sequence_classifier(seed=seed, num_labels=1, **kw)
        return cls.from_sequence_classifier(
            model,
            tokenizer,
            architecture="LlamaForSequenceClassification",
            lineage={"provenance_tier": "weights-verified", "tiny": True},
        )


def _is_trajectory(item: Any) -> bool:
    """Duck-typed ``Trajectory`` detection (has steps and an outcome), import-light."""
    try:
        from reward_lens.data.schema import Trajectory

        return isinstance(item, Trajectory)
    except ImportError:  # pragma: no cover - data plane always present in this repo
        return hasattr(item, "steps") and hasattr(item, "outcome")


__all__ = ["TrajectoryRM"]

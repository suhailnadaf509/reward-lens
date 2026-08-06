"""``RubricRM``: a rubric grader behind the protocol (adapter 5).

A rubric grader scores a response against a set of named criteria (coherence, correctness, safety, ...)
and combines them. The design's discipline here is that the criterion set is *data*, not code: a
``RubricSpec`` names the criteria and their weights, and the adapter reads one per-criterion ``Readout``
off the multi-row reward head plus a weighted-sum aggregate. Changing the rubric is changing a spec, never
editing the adapter, which is what makes a new rubric a new dataset rather than a new code path.

Architecturally this is a multi-row sequence classifier (one head row per criterion), so it reuses the
classifier's fp32 projection unchanged and declares ``MULTI_READOUT``. Each ``score`` call selects a
criterion readout or the aggregate; both are ordinary linear projections of the final hidden state onto a
direction (a single head row for a criterion, the weighted row-sum for the aggregate).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

from reward_lens.core.types import Capability, Site
from reward_lens.signals._common import SignalImplBase, build_hf_runtime
from reward_lens.signals.base import PositionSpec, Readout, Scores, TokenCurves

if TYPE_CHECKING:
    import torch

    from reward_lens.runtime.hf import HFRuntime

_RUBRIC_CAPS = (
    Capability.SCORES
    | Capability.PREFIX_SCORES
    | Capability.ACTIVATIONS
    | Capability.GRADIENTS
    | Capability.HVP
    | Capability.LINEAR_READOUT
    | Capability.MULTI_READOUT
)


@dataclass(frozen=True)
class RubricSpec:
    """A rubric grader's criteria and their aggregate weights (the DATA).

    ``criteria`` names the criteria in head-row order; ``weights`` gives the aggregate weight of each
    (defaulting to equal weights). This is what a study serializes to record exactly which rubric was
    graded, and swapping it is how a new rubric is expressed without touching the adapter.
    """

    criteria: tuple[str, ...]
    weights: tuple[float, ...] = ()

    def resolved_weights(self) -> tuple[float, ...]:
        """The aggregate weights, defaulting to uniform when none were given."""
        if self.weights:
            if len(self.weights) != len(self.criteria):
                raise ValueError(
                    f"rubric has {len(self.criteria)} criteria but {len(self.weights)} weights"
                )
            return self.weights
        n = len(self.criteria)
        return tuple(1.0 / n for _ in range(n)) if n else ()


class RubricRM(SignalImplBase):
    """A rubric grader as a ``RewardSignal`` (adapter 5).

    Build it through ``from_tiny`` (a tiny multi-label classifier) or ``from_sequence_classifier``
    (a real multi-objective head). ``readouts`` exposes one ``criterion:<name>`` per criterion plus a
    weighted ``reward`` aggregate; ``score`` projects the final hidden state onto whichever direction
    the readout names. The rubric spec is on ``self.spec``.
    """

    observable_prefix = "signals.rubric"

    def __init__(
        self,
        *,
        runtime: "HFRuntime",
        meta: Any,
        policy: Any,
        tokenizer: Any,
        readouts: Sequence[Readout],
        spec: RubricSpec,
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
            caps=_RUBRIC_CAPS,
            max_length=max_length,
            default_batch_size=default_batch_size,
            interventions=interventions,
        )
        self.spec = spec

    def default_readout_name(self) -> str:
        """A bare ``score`` resolves to the aggregate ``reward`` readout."""
        return "reward"

    # -- scoring (ordinary linear projections) -----------------------------

    def score(self, view: Any, readout: str | None = None) -> Any:
        """Score every item under a criterion or the aggregate readout."""
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
        """Per-token reward curve under a criterion or aggregate readout."""
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

    def criterion_scores(self, view: Any) -> dict[str, Any]:
        """Convenience: a mapping ``criterion name -> Evidence[Scores]`` for the whole rubric."""
        return {name: self.score(view, f"criterion:{name}") for name in self.spec.criteria}

    # -- constructors -------------------------------------------------------

    @classmethod
    def from_sequence_classifier(
        cls,
        model: "torch.nn.Module",
        tokenizer: Any,
        spec: RubricSpec,
        *,
        device: str = "cpu",
        architecture: str | None = None,
        lineage: dict[str, Any] | None = None,
    ) -> "RubricRM":
        """Wrap a multi-row sequence-classifier head as a rubric grader (no download).

        Requires the head to have one row per criterion. Reads each row into a ``criterion:<name>``
        readout and builds the weighted-sum ``reward`` aggregate whose vector is
        ``sum_k weight_k * row_k`` (so the aggregate is itself a single fp32 projection).
        """
        import torch

        from reward_lens.signals.adapters import reward_head_module

        head = reward_head_module(None, model)
        if head is None:
            raise ValueError("RubricRM needs a linear reward head with one row per criterion.")
        weight = head.weight.data.detach().to(torch.float32)
        if weight.ndim != 2 or weight.shape[0] != len(spec.criteria):
            raise ValueError(
                f"rubric head has {tuple(weight.shape)} rows but the spec names "
                f"{len(spec.criteria)} criteria; they must match one-to-one."
            )
        bias = (
            head.bias.data.detach().to(torch.float32)
            if getattr(head, "bias", None) is not None
            else None
        )
        runtime, meta, policy = build_hf_runtime(
            model, tokenizer, head, architecture=architecture, device=device, lineage=lineage
        )
        site = Site(max(meta.n_layers - 1, 0), "resid_post")
        readouts: list[Readout] = []
        for k, name in enumerate(spec.criteria):
            b = 0.0 if bias is None else float(bias[k])
            readouts.append(
                Readout(
                    name=f"criterion:{name}",
                    kind="linear",
                    site=site,
                    position=PositionSpec("final"),
                    vector=weight[k].contiguous(),
                    meta={"bias": b, "criterion": name, "row": k},
                )
            )
        weights = spec.resolved_weights()
        agg_vec = sum(w * weight[k] for k, w in enumerate(weights)).contiguous()
        agg_bias = 0.0 if bias is None else float(sum(w * bias[k] for k, w in enumerate(weights)))
        readouts.insert(
            0,
            Readout(
                name="reward",
                kind="linear",
                site=site,
                position=PositionSpec("final"),
                vector=agg_vec,
                meta={"bias": agg_bias, "aggregate": "rubric_weighted", "weights": list(weights)},
            ),
        )
        meta.lineage["rubric"] = {"criteria": list(spec.criteria), "weights": list(weights)}
        return cls(
            runtime=runtime,
            meta=meta,
            policy=policy,
            tokenizer=tokenizer,
            readouts=readouts,
            spec=spec,
        )

    @classmethod
    def from_tiny(
        cls,
        *,
        criteria: Sequence[str] = ("coherence", "correctness", "safety"),
        weights: Sequence[float] = (),
        seed: int = 0,
        **kw: Any,
    ) -> "RubricRM":
        """Construct the tiny offline rubric grader the tests run on (a k-label tiny classifier)."""
        from reward_lens.signals.process import _tiny_sequence_classifier

        spec = RubricSpec(criteria=tuple(criteria), weights=tuple(weights))
        model, tokenizer = _tiny_sequence_classifier(seed=seed, num_labels=len(spec.criteria), **kw)
        return cls.from_sequence_classifier(
            model,
            tokenizer,
            spec,
            architecture="LlamaForSequenceClassification",
            lineage={"provenance_tier": "weights-verified", "tiny": True},
        )


__all__ = ["RubricRM", "RubricSpec"]

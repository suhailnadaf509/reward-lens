"""Per-rollout score vectors `∇_θ log π(y | x)`, which is what the rung-2 `G` is built from.

This is the one place in the cost book that needs a checkpoint and a backward pass. Everything else
reads a record. It is kept in its own module so that importing `measure.efficiency` does not import
torch, which is the property `tests/acceptance/test_w0_3_base_install.py` asserts for the package as
a whole.

**Why this is not `policy.hf`'s `grad`.** `HFPolicyRuntime.grad` and `hvp` differentiate a readout
scalar with respect to an *activation at a site*, which is `d_model`-dimensional and is what the
lens and the reward-Hessian instruments want. `F = E[∇_θ log π ∇_θ log πᵀ]` is over *parameters* and
is `|θ|`-dimensional, so it is a different object and there was no existing implementation of it to
reuse. Nothing here duplicates the runtime's `hvp`; it never forms a Hessian and never takes a
second derivative. It arguably belongs on `PolicySubject` rather than in `measure/`, and moving it
there is left open.

**The tokenisation caveat, because it is real and it is measurable.** The GRPO record carries turn
text and `Turn.token_ids` is `None`, so the completion has to be re-tokenised from its string. On
the 200-step fixture that recovers 12 tokens for most completions and 15 to 17 for some, against a
`max_completion_length` of 12: re-tokenising a string does not in general reproduce the token
boundaries the sampler produced, because a decoded string can retokenise into a different sequence.
The score of a sequence is a sum over its own token boundaries, so a different segmentation gives a
different score vector for the same text. The effect on `G` is not signable a priori. A record that
writes `token_ids` removes this entirely, and that is the remedy the refusal names.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass


def sequence_scores(
    model: Any,
    tokenizer: Any,
    pairs: Sequence[tuple[str, str]],
    *,
    dtype: Any = np.float64,
) -> np.ndarray:
    """`(n, |θ|)` of `∇_θ Σ_i log π(y_i | y_<i, x)`, one row per `(prompt, completion)` pair.

    The sequence log-probability, not the per-token mean: `F` and `J` are expectations of
    per-*sequence* quantities, and a length-normalised score would be the gradient of a different
    functional. `update.kl_spent` and `update.kl_min` are registered per sequence for the same
    reason, and mixing the two is the unit error that is the commonest silent failure here.

    Memory is `n · |θ| · 8` bytes for the returned array, so 157 MB for eight rollouts of a
    2.45M-parameter policy in float64 and 448 GB for eight rollouts of a 7B one. Pass
    ``dtype=np.float32`` to halve it. Above roughly a hundred million parameters this construction
    does not fit and the score matrix has to be sketched, which is not implemented here.
    """
    import torch

    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError(
            "the model exposes no parameter with `requires_grad`, so there is no score to take. "
            "Load the checkpoint without `torch.no_grad()` and without freezing, or pass the "
            "policy rather than an inference-only copy of it."
        )
    width = sum(int(p.numel()) for p in parameters)
    out = np.empty((len(pairs), width), dtype=dtype)
    was_training = model.training
    model.eval()
    try:
        for row, (prompt, completion) in enumerate(pairs):
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
            if not completion_ids:
                raise ValueError(
                    f"pair {row} has an empty completion after tokenisation, so its "
                    f"log-probability is the empty sum and its score is zero. Drop the rollout "
                    f"upstream (the featuriser's `on_empty` decides this for the ledger) rather "
                    f"than letting a zero row into the Gram matrix, where it reads as a rollout "
                    f"the policy could not have moved."
                )
            ids = torch.tensor([list(prompt_ids) + list(completion_ids)])
            model.zero_grad(set_to_none=True)
            logits = model(input_ids=ids).logits.float()
            start = max(len(prompt_ids) - 1, 0)
            log_probs = torch.log_softmax(logits[0, start : ids.shape[1] - 1], dim=-1)
            targets = torch.tensor(list(completion_ids))
            total = log_probs.gather(1, targets.unsqueeze(1)).sum()
            total.backward()
            offset = 0
            for parameter in parameters:
                size = int(parameter.numel())
                grad = parameter.grad
                if grad is None:
                    out[row, offset : offset + size] = 0.0
                else:
                    out[row, offset : offset + size] = (
                        grad.detach().reshape(-1).to(torch.float64).cpu().numpy()
                    )
                offset += size
        model.zero_grad(set_to_none=True)
    finally:
        if was_training:
            model.train()
    return out


def pairs_from_steps(steps: Sequence[Any], featuriser: Any) -> list[tuple[str, str]]:
    """`(prompt, completion)` per rollout, in exactly `pooled_rollouts` row order.

    The row order is the contract: `metric_g` pairs row `a` of the score matrix with row `a` of the
    pooled feature matrix, and a permutation between them puts one rollout's gradient against
    another's features. Both orders come from walking steps in index order and groups in record
    order, and this function exists so that there is one place where that is written down.

    The same featuriser is applied here as there, so the rollouts dropped here are the rollouts
    `steps_from_run` drops there. A trajectory the featuriser declines has no feature row, so it
    must have no score row either.
    """
    from reward_lens.measure.ledger.features import assistant_text

    pairs: list[tuple[str, str]] = []
    for step in sorted(steps, key=lambda s: s.index):
        for group in step.groups:
            for trajectory in group.trajectories:
                if featuriser.featurise(trajectory) is None:
                    continue
                prompt = "\n".join(
                    turn.text for turn in trajectory.turns if turn.role == "user" and turn.text
                )
                pairs.append((prompt, assistant_text(trajectory)))
    return pairs


__all__ = ["pairs_from_steps", "sequence_scores"]

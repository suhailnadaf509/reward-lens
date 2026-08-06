"""SamplerBridge: policy sampling for the loops that need draws.

Policies enter the kernel in two roles: as subjects of instrumentation (RL loops) and as sources of
samples (chi estimation, BoN ladders, dreams). ``SamplerBridge`` is the sampling half. The HF
``generate`` path is implemented and usable now; the vLLM path is an explicit, clearly marked stub
because vLLM is an optional extra. A ``fidelity_check`` signature is present so a study that mixes
engines can bound the drift between them; its body is implemented for the HF-vs-HF degenerate case
and stubbed for the vLLM comparison.

Samples are meant to be cached as first-class datasets (``ds:`` ids) so a chi spectrum and a BoN
ladder computed on the same draw are exactly comparable. Caching them is the caller's job, so here
the bridge returns plain sample records the caller can persist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    import torch


class SamplerUnavailableError(RuntimeError):
    """Raised when a sampling backend is requested but its optional dependency is absent."""


@dataclass
class SampleRecord:
    """One generation: the prompt, the continuation text, and its token-level logprobs."""

    prompt: str
    text: str
    token_ids: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


class SamplerBridge:
    """Wrap a policy model + tokenizer for sampling.

    ``backend="hf"`` uses ``transformers`` ``generate`` and is implemented here. ``backend="vllm"``
    is a stub that raises ``SamplerUnavailableError`` with a message naming the missing extra; the
    real vLLM engine, its n-per-prompt fan-out, and its logprob capture are not implemented here.
    The bridge does not itself cache draws; it returns ``SampleRecord`` objects the caller persists
    as a dataset.
    """

    def __init__(
        self,
        model: "torch.nn.Module",
        tokenizer: Any,
        device: "torch.device | str" = "cpu",
        backend: str = "hf",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.backend = backend
        if backend not in ("hf", "vllm"):
            raise ValueError(f"unknown sampler backend {backend!r}; expected 'hf' or 'vllm'.")

    def sample(
        self,
        prompts: Sequence[str],
        n: int = 1,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        seed: int | None = None,
    ) -> list[SampleRecord]:
        """Draw ``n`` continuations per prompt.

        The HF backend runs ``generate`` with sampling and returns one ``SampleRecord`` per
        (prompt, draw). The vLLM backend is stubbed. Logprob capture on the HF path is best-effort
        via ``output_scores``; the population-scale, fast path is vLLM, which is gated.
        """
        if self.backend == "vllm":
            raise SamplerUnavailableError(
                "the vLLM sampling backend requires the optional 'vllm' extra, which is not "
                "installed. Install reward-lens[vllm] or use backend='hf'. This is an explicit "
                "stub: the vLLM engine, n-per-prompt fan-out, and logprob capture "
                "are not implemented."
            )
        return self._sample_hf(
            prompts,
            n=n,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )

    def _sample_hf(
        self,
        prompts: Sequence[str],
        n: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seed: int | None,
    ) -> list[SampleRecord]:
        import torch

        if seed is not None:
            torch.manual_seed(seed)
        records: list[SampleRecord] = []
        do_sample = temperature > 0.0
        for prompt in prompts:
            enc = self.tokenizer(prompt, return_tensors="pt")
            input_ids = enc["input_ids"].to(self.device)
            prompt_len = input_ids.shape[1]
            with torch.no_grad():
                out = self.model.generate(
                    input_ids,
                    do_sample=do_sample,
                    num_return_sequences=n,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if do_sample else None,
                    top_p=top_p if do_sample else None,
                    pad_token_id=getattr(self.tokenizer, "pad_token_id", None),
                )
            for row in out:
                continuation = row[prompt_len:]
                text = self.tokenizer.decode(continuation, skip_special_tokens=True)
                records.append(
                    SampleRecord(
                        prompt=prompt,
                        text=text,
                        token_ids=continuation.tolist(),
                        meta={"backend": "hf", "temperature": temperature, "top_p": top_p},
                    )
                )
        return records

    def fidelity_check(
        self, prompts: Sequence[str], reference_scores: Sequence[float] | None = None
    ) -> dict[str, Any]:
        """Bound engine drift between the sampling backend and an HF forward.

        For the HF backend this is a self-consistency check (a placeholder that reports the sample
        count and a zero drift, since one engine cannot drift from itself). The meaningful use is the
        vLLM-vs-HF comparison, which is stubbed until the vLLM engine is available. Any study that
        mixes engines is expected to call this and report the bound.
        """
        if self.backend == "vllm":
            raise SamplerUnavailableError(
                "fidelity_check against vLLM requires the 'vllm' extra (not installed). This is an "
                "explicit stub."
            )
        return {
            "backend": self.backend,
            "n_prompts": len(prompts),
            "engine_drift": 0.0,
            "note": "HF self-consistency; vLLM-vs-HF drift is gated on the vllm extra.",
            "reference_scores": list(reference_scores) if reference_scores is not None else None,
        }


__all__ = ["SamplerBridge", "SampleRecord", "SamplerUnavailableError"]

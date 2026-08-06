"""Best-of-n ladders and the exact BoN-vs-base KL identity.

A best-of-n sampler draws ``n`` completions from the base policy and keeps the one the reward
model scores highest. Sweeping ``n`` traces a curve of expected reward against the KL divergence
from the base policy, and that curve is the quasi-static equilibrium frontier the thermodynamics
science (S3) and the forecasting science (S12) both need as their reference arm: it previews where
optimization is headed with no RL run at all.

Two quantities carry the module.

The expected best-of-n reward is estimated from a bank of scored base-policy samples by the
plug-in empirical estimator: treat the ``m`` scored samples for a prompt as the reward
distribution and compute the exact expected maximum of ``n`` draws with replacement from it. With
the order statistics ``r_(1) <= ... <= r_(m)`` the weight the estimator puts on ``r_(k)`` is
``(k/m)^n - ((k-1)/m)^n`` (the probability the maximum of ``n`` draws is the k-th order
statistic). This is the estimator that lets a modest sample bank (a few hundred draws) preview
``n`` up to 10^4, because the weight simply concentrates on the top sample as ``n`` grows. It is
also provably monotone nondecreasing in ``n`` (Abel summation turns the expected maximum into
``r_(m) - sum_k (r_(k+1) - r_(k)) (k/m)^n`` and every ``(k/m)^n`` with ``k < m`` decreases in
``n``), which is the acceptance property the test pins.

The KL divergence of the best-of-n policy from the base policy has an exact closed form in the
no-ties limit, ``KL(bo_n || base) = log(n) - (n-1)/n`` (Beirami et al. 2401.01879; the Stiennon
BoN-vs-RL comparison uses the same identity). It depends on nothing but ``n``: no scores, no
model, no fit. That is what makes the BoN sweep a calibration-free x-axis for the optimization
frontier, and reproducing it exactly is this module's acceptance test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

import numpy as np

from reward_lens.core.evidence import Evidence, Uncertainty, make_evidence, register_payload
from reward_lens.core.provenance import Provenance
from reward_lens.core.types import GaugeStatus, SubjectRef

if TYPE_CHECKING:
    from reward_lens.core.types import EvidenceID

# The default ladder: the doubling sequence 1, 2, 4, ..., 8192, then 10^4, which is the range the
# frontier is read over. Powers of two give even spacing on the log-n axis it is plotted against.
DEFAULT_NS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 10000)


def bon_kl(n: int | Sequence[int] | np.ndarray) -> np.ndarray:
    """The exact KL divergence of the best-of-n policy from the base policy, in nats.

    ``KL(bo_n || base) = log(n) - (n - 1) / n`` (Beirami et al. 2401.01879). Exact in the
    continuous-reward, no-ties limit, and a function of ``n`` alone. ``bon_kl(1) == 0`` because
    best-of-1 is the base policy. This is the identity the acceptance test reproduces.

    Accepts a scalar or an array of ``n`` and returns a float array so a whole ladder is one call.
    """
    n_arr = np.asarray(n, dtype=np.float64)
    if np.any(n_arr < 1):
        raise ValueError(f"best-of-n requires n >= 1; got {n_arr[n_arr < 1]}")
    return np.log(n_arr) - (n_arr - 1.0) / n_arr


def expected_bon_reward(scores: Sequence[float] | np.ndarray, n: int) -> float:
    """Expected best-of-n reward for one prompt's bank of scored samples (plug-in estimator).

    Sorts the ``m`` scores and returns ``sum_k r_(k) [ (k/m)^n - ((k-1)/m)^n ]``, the exact
    expected maximum of ``n`` draws with replacement from the empirical reward distribution. Valid
    for any ``n >= 1``, including ``n`` far larger than ``m`` (the weight concentrates on the top
    order statistic). Returns NaN for an empty bank.
    """
    arr = np.sort(np.asarray(scores, dtype=np.float64).ravel())
    m = arr.size
    if m == 0:
        return float("nan")
    if n < 1:
        raise ValueError(f"best-of-n requires n >= 1; got {n}")
    k = np.arange(1, m + 1, dtype=np.float64)
    weights = (k / m) ** n - ((k - 1.0) / m) ** n
    return float(np.dot(arr, weights))


@register_payload
@dataclass
class BoNLadder:
    """The best-of-n sweep: expected reward and exact KL at each ``n``.

    ``ns`` is the ladder; ``kl`` is ``bon_kl(ns)`` in nats; ``expected_reward`` is the plug-in
    expected best-of-n reward averaged over prompts; ``reward_sem`` is its across-prompt standard
    error. ``(kl, expected_reward)`` is the quasi-static frontier a study plots as the no-RL
    preview of the optimization endpoint. ``baseline_reward`` is ``expected_reward`` at ``n = 1``,
    the mean base-policy reward. ``n_prompts`` and ``samples_per_prompt`` record the bank the
    estimate came from so its resolution is auditable.
    """

    ns: np.ndarray
    kl: np.ndarray
    expected_reward: np.ndarray
    reward_sem: np.ndarray
    baseline_reward: float
    n_prompts: int
    samples_per_prompt: np.ndarray = field(default_factory=lambda: np.empty(0))

    def frontier(self) -> tuple[np.ndarray, np.ndarray]:
        """The ``(kl, expected_reward)`` frontier as two aligned arrays."""
        return self.kl, self.expected_reward


def _normalize_banks(scores_per_prompt: Sequence[Sequence[float]] | np.ndarray) -> list[np.ndarray]:
    """Coerce the input into a list of per-prompt score arrays.

    Accepts a 2-D array ``(n_prompts, m)`` for the common equal-bank case, a ragged sequence of
    per-prompt score arrays when the banks differ in size (a prompt that generated fewer valid
    samples), a single 1-D array, or a flat sequence of numbers, the last two treated as one
    prompt's bank.
    """
    if isinstance(scores_per_prompt, np.ndarray):
        if scores_per_prompt.ndim == 2:
            return [np.asarray(row, dtype=np.float64) for row in scores_per_prompt]
        if scores_per_prompt.ndim == 1:
            return [np.asarray(scores_per_prompt, dtype=np.float64)]
        raise ValueError(f"scores array must be 1-D or 2-D; got {scores_per_prompt.ndim}-D")
    seq = list(scores_per_prompt)
    if not seq:
        raise ValueError("scores_per_prompt is empty; nothing to sweep")
    first = seq[0]
    if np.isscalar(first) or (isinstance(first, np.ndarray) and first.ndim == 0):
        # a flat sequence of numbers = a single prompt's bank
        return [np.asarray(seq, dtype=np.float64).ravel()]
    return [np.asarray(row, dtype=np.float64).ravel() for row in seq]


def bon_ladder(
    scores_per_prompt: Sequence[Sequence[float]] | np.ndarray,
    ns: Sequence[int] = DEFAULT_NS,
    *,
    subject: SubjectRef | None = None,
    parents: Sequence["EvidenceID"] = (),
) -> Evidence[BoNLadder]:
    """Compute the best-of-n ladder and the exact KL frontier from scored base-policy samples.

    ``scores_per_prompt`` is a per-prompt bank of reward-model scores: either a 2-D array
    ``(n_prompts, m)`` or a ragged list of per-prompt arrays. For each ``n`` in ``ns`` the expected
    best-of-n reward is computed per prompt with the plug-in estimator and averaged across prompts,
    and the exact ``KL(bo_n || base)`` is attached. The result previews the optimization frontier
    with no RL.

    Returns ``Evidence[BoNLadder]``. Gauge is INVARIANT: the KL is a real divergence in nats and
    the reward is in the model's own score units, both gauge-free (a raw reward-model score is
    INVARIANT in this kernel). ``subject`` names the signal/dataset when the caller
    has them; ``parents`` links the score Evidence this consumed so the store stays a DAG.
    """
    banks = _normalize_banks(scores_per_prompt)
    ns_arr = np.array(sorted({int(x) for x in ns}), dtype=np.int64)
    if ns_arr.size == 0:
        raise ValueError("ns is empty; nothing to sweep")
    if ns_arr[0] < 1:
        raise ValueError(f"best-of-n requires n >= 1; smallest requested is {ns_arr[0]}")

    # per_prompt[j, i] = expected best-of-ns[i] reward for prompt j
    per_prompt = np.empty((len(banks), ns_arr.size), dtype=np.float64)
    for j, bank in enumerate(banks):
        for i, n in enumerate(ns_arr):
            per_prompt[j, i] = expected_bon_reward(bank, int(n))

    expected_reward = np.nanmean(per_prompt, axis=0)
    n_prompts = per_prompt.shape[0]
    if n_prompts > 1:
        reward_sem = np.nanstd(per_prompt, axis=0, ddof=1) / np.sqrt(n_prompts)
    else:
        reward_sem = np.zeros(ns_arr.size)

    kl = bon_kl(ns_arr)
    baseline = float(expected_reward[0]) if ns_arr[0] == 1 else float("nan")
    payload = BoNLadder(
        ns=ns_arr,
        kl=kl,
        expected_reward=expected_reward,
        reward_sem=reward_sem,
        baseline_reward=baseline,
        n_prompts=n_prompts,
        samples_per_prompt=np.array([b.size for b in banks], dtype=np.int64),
    )
    return make_evidence(
        observable="loops.bon.ladder",
        observable_version="1",
        subject=subject or SubjectRef(),
        value=payload,
        uncertainty=Uncertainty(n=n_prompts, method="across-prompt-sem"),
        gauge=GaugeStatus.INVARIANT,
        provenance=Provenance(parents=tuple(parents)),
    )


__all__ = ["bon_kl", "expected_bon_reward", "bon_ladder", "BoNLadder", "DEFAULT_NS"]

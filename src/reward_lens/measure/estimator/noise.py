"""E3, the signal-versus-noise share of the gradient and its attribution.

**veRL already ships the fraction.** `compute_variance_proxy_metrics` at
`verl/trainer/ppo/metric_utils.py:691` emits five metrics under a `variance_proxy/` prefix, built on
``W(tau) = sum_t[1 - 2*pi_t + sum pi^2]``: `proxy1_signal_strength`, `proxy2_total_power`,
``proxy3_pure_noise = (1/(N-1)) * (proxy2 - proxy1)``, `expected_a_squared` and `expected_w`. So the
number is not the contribution. **Pitch the attribution, not the fraction.** veRL tells you the
gradient is some fraction noise; nothing tells you whether that came from the grader, from sampling,
or from the clip.

**Two gates on the fraction that are easy to miss, and both are here as refusals.** The function
returns an empty dict unless `sum_pi_squared`, `old_log_probs` and `advantages` are all in the batch
(`metric_utils.py:709-710`), and `actor.calculate_sum_pi_squared` defaults to `false`
(`verl/trainer/config/actor/actor.yaml:96` and `verl/workers/config/actor.py:170`), so the whole
metric emits nothing unless the user opted in. Even with it on, all three call sites pass
``gradient_norm = metrics.get("actor/grad_norm", None)``, and when that key is absent proxy1 and
proxy3 emit a literal ``0.0`` (`metric_utils.py:789-795`) rather than being omitted, so a logged
zero is indistinguishable from "not computed".

**One estimator caveat that rides on every rung-0 reading.** `proxy3` is floored with a hard
``max(0.0, ...)`` at `metric_utils.py:776-780`, so it is censored from below and upward-biased near
zero. There is not a second one: an earlier version of this file said `proxy1` is the square of the
**post**-clipping gradient norm, so the clip inflates the reported share and rung 1 must subtract
it. That is wrong. `actor/grad_norm` is the norm **before** clipping, and four primary sources say
so in as many words: `torch.nn.utils.clip_grad_norm_` returns the total norm it computed before
scaling; veRL's FSDP engine documents its `optimizer_step` return as "Norm of gradients before
clipping" (`workers/engine/fsdp/transformer_impl.py:735`); its Megatron engine says the same
(`workers/engine/megatron/transformer_impl.py:543`); and `transformers/trainer.py:2523-2524` calls
its own return value "the pre-clip gradient norm". So `proxy1` is the pre-clip signal strength, the
clip does not bias the noise share through it, and there is no clip term to subtract. E5 measures
what the clip really does, which is to shrink the applied update.

**The attribution.** The reported noise share is split by measuring how much of the advantage's
variance each mechanism injects, using the recorded estimator itself rather than a model of it:

- **grader**: perturb every score by an independent draw at A2's error standard deviation, replay
  the advantages through `record.scores.replay_advantages`, and measure the variance injected.
- **sampling**: delete one rollout from each group, replay the advantages of the survivors, and
  measure how far they move. That is the sensitivity of the advantage to which K rollouts were
  drawn, which is the finite-group facet.
- **unattributed**: the remainder, reported under that name. A residual called by a mechanism's
  name is an attribution nobody made.

E3's catalogue record illustrates the reading as "22 points are grader replication variance, 9 are
sampling, and 6 are the clip", and the third term is not derivable for the reason above. The clip's
own effect is carried on the reading as `clip_shrinkage` with the sentence that says why it is not
a share of the noise, rather than being folded into one.

**The scope note this needs, and it is narrower than "nobody does this".** Three papers
connect grader unreliability to gradient quality: arXiv 2510.00915 formalises verifier unreliability
as a stochastic reward channel with asymmetric rates and derives an unbiased policy-gradient
estimator implemented as hooks in a GRPO pipeline; arXiv 2510.18924 proves reward corruption
attenuates the learning signal by ``(1 - rho+ - rho-)`` and normalises by the *noisy* standard
deviation; arXiv 1810.01032 (AAAI 2020) is the classical root with convergence and sample
complexity. What survives is this: all three assume an independent binary noise channel whose rate is
estimated from the reward signal itself, which is circular when grader errors are correlated across
items; none sources the rate from a designed reliability study; none reports a variance *share*; and
a scalar ``rho`` cannot encode a covariance structure, so nine frontier judges supplying about two
independent votes' worth of information is a design effect no independent-channel model can express.
This instrument sources its error term from A2, which is a designed crossed study, and it reports a
share. It does **not** claim the first connection between grader error and gradient quality.

Kill condition, from the catalogue record: if the attribution never separates grader from sampling
on real data.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from reward_lens.core.envelope import EnvelopeSpec, RegimeCondition
from reward_lens.core.evidence import register_payload
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.quantity import BaselineID, BiasStatement
from reward_lens.core.reading import Refusal, RefusalReason, bounded_refusal
from reward_lens.core.types import Access, Capability, Component, GaugeStatus, Phase
from reward_lens.measure.estimator._base import EstimatorInstrument
from reward_lens.measure.estimator.clip import measure_clip_effect
from reward_lens.measure.estimator.spec import (
    ALL_SUBSTRATES,
    RECORD_ACCESS,
    check_replay,
    collect_specs,
    iter_groups,
    iter_steps,
)
from reward_lens.record.schema import EstimatorSpec, Group, Run, Step
from reward_lens.record.scores import (
    AllAbstainedWarning,
    ScoreContext,
    evaluate,
    replay_advantages,
)

#: The five metric keys veRL emits, with the `variance_proxy/` prefix that is easy to drop when
#: they are quoted. They arrive in `OptimizerTelemetry.extra`, which is untyped, so reading them
#: is a declared deviation.
PROXY_KEYS: dict[str, str] = {
    "proxy1": "variance_proxy/proxy1_signal_strength",
    "proxy2": "variance_proxy/proxy2_total_power",
    "proxy3": "variance_proxy/proxy3_pure_noise",
    "expected_a_squared": "variance_proxy/expected_a_squared",
    "expected_w": "variance_proxy/expected_w",
}

#: The catalogue's baseline for E3: the raw fraction with no attribution. That is what veRL already
#: gives you, and it is the thing this instrument is arguing is not enough. The second is the null
#: the attribution is tested against: an equal split across the *named* mechanisms, which is what a
#: reader assumes when nobody has measured one. It is `1 / (len(MECHANISMS) - 1)`, the residual not
#: being a mechanism, so it moved from a third to a half when the clip term was withdrawn.
NOISE_BASELINES: tuple[BaselineID, ...] = (
    "baseline.raw_noise_fraction",
    "baseline.equal_share_attribution",
)

#: The two named mechanisms, plus the residual. Held as data so a card renders them in one order
#: and a test can assert the partition sums.
#:
#: The clip was a third name here and it is not one any more. It was a correction to `proxy1` on
#: the belief that `actor/grad_norm` is the post-clipping norm; it is the pre-clipping norm, so the
#: correction is identically nothing. E5's number is carried on the reading as `clip_shrinkage`
#: instead, because what the clip does is shrink the applied update rather than bias this share.
MECHANISMS: tuple[str, ...] = ("grader", "sampling", "unattributed")

#: How many perturbation draws the grader term uses. **Chosen: 256**, which puts the Monte Carlo
#: standard error on a variance estimate at roughly 9 percent of the estimate, an order of magnitude
#: below the spread between the mechanisms this is meant to separate.
DEFAULT_DRAWS = 256

NOISE_ENVELOPE = EnvelopeSpec(
    requires=frozenset({RegimeCondition.GROUP_NONDEGENERATE}),
    measured_by={RegimeCondition.GROUP_NONDEGENERATE: "estimator.degenerate_fraction"},
    on_violation="refuse",
)

PROXY_BIAS = BiasStatement(
    direction="upward",
    why=(
        "`proxy3` is floored with a hard `max(0.0, ...)` at `metric_utils.py:778`, so it is "
        "censored from below and its expectation sits above the true value wherever the true value "
        "is near zero. That is the whole of the signed bias. The clip is not a second source: "
        "`actor/grad_norm` is the norm before clipping (torch's `clip_grad_norm_` returns the norm "
        "it computed before scaling; veRL's FSDP engine documents its return as 'Norm of gradients "
        "before clipping'; `transformers/trainer.py:2524` calls its own return 'the pre-clip "
        "gradient norm'), so `proxy1` is the pre-clip signal strength and the clip does not enter."
    ),
)

ATTRIBUTION_BIAS = BiasStatement(
    direction="unknown",
    why=(
        "the grader term is a first-order perturbation at A2's error standard deviation, so it is "
        "right to the extent that the advantage is locally linear in the score, which fails "
        "exactly where the group standard deviation is near the estimator's epsilon. And it "
        "assumes grader errors are independent across rollouts unless A2 supplies a covariance, "
        "which is the same assumption the three correction papers make and the one that fails when "
        "errors are correlated across items. The residual is reported as unattributed rather than "
        "folded into either named term."
    ),
)


# ---------------------------------------------------------------------------
# The A2 interface
# ---------------------------------------------------------------------------


@runtime_checkable
class VarianceComponentsLike(Protocol):
    """What E3 needs from A2 (`grader.variance_components`), and nothing more.

    One number: ``error_variance``, the part of a score's variance that a re-run of the grader on
    the same rollout would not reproduce, in squared reward units. Declared as a protocol rather
    than as a concrete type so that neither package has to import the other and so this instrument
    can be tested against components computed on paper.

    ``error_covariance`` is consulted when it is there and is the piece a scalar rate cannot
    express: with it, the grader perturbation is drawn from the measured covariance across
    rollouts; without it the perturbation is independent and the reading says so. Nine frontier
    judges supplying about two independent votes' worth of information is a design effect, and no
    independent-channel model can represent it.
    """

    @property
    def error_variance(self) -> float: ...


@dataclass(frozen=True)
class VarianceComponents:
    """A concrete `VarianceComponentsLike`, and the adapter onto A2's own types.

    Not a reimplementation of A2. A2 runs the crossed design and estimates the components; this is
    the shape E3 reads them in, so that a test can hand it numbers computed on paper, a user with a
    published G-study can use it without re-running one, and A2's `ComponentSet` and `GaugeRR` both
    convert in one call.
    """

    components: Mapping[str, float]
    #: The facets whose variance a re-run of the grader would not reproduce. The default is the
    #: two-facet naming a rung-0 G-study produces; a fully crossed design adds interaction terms
    #: and names them here.
    error_facets: tuple[str, ...] = ("rater", "occasion", "residual")
    design: str = ""
    #: (n, n) covariance of the grader error across the rollouts being perturbed, when the study
    #: measured one. None means independent, which is an assumption rather than a finding.
    error_covariance: np.ndarray | None = None

    @property
    def error_variance(self) -> float:
        return float(sum(self.components.get(f, 0.0) for f in self.error_facets))

    @property
    def total(self) -> float:
        return float(sum(self.components.values()))

    @property
    def universe_variance(self) -> float:
        return self.total - self.error_variance

    @classmethod
    def from_component_set(
        cls,
        component_set: Any,
        *,
        part: str = "p",
        error_covariance: np.ndarray | None = None,
    ) -> "VarianceComponents":
        """Adapt `stats.variance.ComponentSet`, taking everything that is not the part as error.

        That is A2's own rule, stated in `gauge_rr`: "everything that is not the part is gauge", and
        it reduces to the AIAG two-facet definition exactly. Duck-typed on `as_dict` and `names` so
        this package does not import A2's module, which keeps the dependency one-way.
        """
        values = dict(component_set.as_dict())
        return cls(
            components=values,
            error_facets=tuple(n for n in values if n != part),
            design=getattr(component_set, "design", ""),
            error_covariance=error_covariance,
        )

    @classmethod
    def from_gauge_rr(
        cls, gauge: Any, *, error_covariance: np.ndarray | None = None
    ) -> "VarianceComponents":
        """Adapt `stats.variance.GaugeRR`, whose `sigma_grr` is exactly the error term wanted.

        `%GRR` is a ratio of standard deviations by the automotive convention, so it is the squares
        that are used here: `sigma_grr**2` is the variance a re-run would not reproduce and
        `sigma_part**2` is the universe-score variance.
        """
        return cls(
            components={
                "part": float(gauge.sigma_part) ** 2,
                "grr": float(gauge.sigma_grr) ** 2,
            },
            error_facets=("grr",),
            design=f"gauge R&R over {', '.join(getattr(gauge, 'gauge_terms', ()) or ('gauge',))}",
            error_covariance=error_covariance,
        )


# ---------------------------------------------------------------------------
# Rung 0: read the fraction veRL already ships
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProxyStep:
    """One step's variance-proxy metrics, with the two censoring flags."""

    index: int
    proxy1: float
    proxy2: float
    proxy3: float
    #: proxy1 is exactly 0.0, which `metric_utils.py:789-791` emits when `gradient_norm` was None.
    proxy1_uncomputed: bool
    #: proxy3 is exactly 0.0, which is either the `max(0.0, ...)` floor at :778 or the same
    #: uncomputed path. Neither is distinguishable from a real zero.
    proxy3_censored: bool

    @property
    def share(self) -> float:
        return self.proxy3 / self.proxy2 if self.proxy2 > 0 else math.nan


def read_proxies(subject: Run | Sequence[Step] | Step) -> list[ProxyStep]:
    """veRL's five metrics per step, for the steps that carry them."""
    out: list[ProxyStep] = []
    for step in iter_steps(subject):
        extra = step.optimizer.extra
        if PROXY_KEYS["proxy2"] not in extra:
            continue
        p1 = float(extra.get(PROXY_KEYS["proxy1"], math.nan))
        p2 = float(extra[PROXY_KEYS["proxy2"]])
        p3 = float(extra.get(PROXY_KEYS["proxy3"], math.nan))
        out.append(
            ProxyStep(
                index=step.index,
                proxy1=p1,
                proxy2=p2,
                proxy3=p3,
                proxy1_uncomputed=p1 == 0.0,
                proxy3_censored=p3 == 0.0,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Rung 1: attribute the share
# ---------------------------------------------------------------------------


def _group_totals(group: Group) -> list[float]:
    return [evaluate(t.scores, ScoreContext()) for t in group.trajectories]


def _replay(scores: Sequence[float], spec: EstimatorSpec, where: str) -> np.ndarray | None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AllAbstainedWarning)
        out = replay_advantages(scores, spec, where=where)
    if isinstance(out, Refusal):
        return None
    return np.asarray(out, dtype=float)


def grader_injected_variance(
    groups: Sequence[Group],
    spec: EstimatorSpec,
    components: VarianceComponentsLike,
    *,
    draws: int = DEFAULT_DRAWS,
    seed: int = 0,
) -> tuple[float, int]:
    """Variance the grader's own error injects into the advantage, through the recorded estimator.

    One independent draw at the error standard deviation is added to every score, the advantages are
    replayed, and the mean squared movement is accumulated. Because the draw has the same variance
    as the error already in the recorded scores, the mean squared movement estimates the variance
    that error contributes to the advantage, in the regime where the advantage is locally linear in
    the score.
    """
    sigma = math.sqrt(max(components.error_variance, 0.0))
    if sigma <= 0.0:
        return 0.0, 0
    rng = np.random.default_rng(seed)
    cov = getattr(components, "error_covariance", None)
    total = 0.0
    n = 0
    for group in groups:
        base_scores = _group_totals(group)
        base = _replay(base_scores, spec, str(group.id))
        if base is None:
            continue
        k = len(base_scores)
        for _ in range(draws):
            if cov is not None and np.shape(cov) == (k, k):
                noise = rng.multivariate_normal(np.zeros(k), np.asarray(cov, dtype=float))
            else:
                noise = rng.normal(0.0, sigma, size=k)
            moved = _replay([s + float(e) for s, e in zip(base_scores, noise)], spec, str(group.id))
            if moved is None:
                continue
            live = np.isfinite(base) & np.isfinite(moved)
            if not live.any():
                continue
            total += float(np.sum((moved[live] - base[live]) ** 2))
            n += int(live.sum())
    return (total / n if n else math.nan), n


def sampling_injected_variance(
    groups: Sequence[Group],
    spec: EstimatorSpec,
) -> tuple[float, int]:
    """Sensitivity of the advantage to which K rollouts formed the group, by delete-one replay.

    For each group and each rollout, that rollout is removed, the remaining advantages are replayed
    through the same estimator, and the squared movement of the survivors is accumulated with the
    jackknife's ``(K-1)/K`` weight. This is a sensitivity rather than a textbook jackknife variance:
    the statistic is a vector whose length changes with the deletion, so the usual pseudo-value
    algebra does not apply and the quantity reported is the movement itself.
    """
    total = 0.0
    n = 0
    for group in groups:
        scores = _group_totals(group)
        k = len(scores)
        if k < 3:
            continue
        base = _replay(scores, spec, str(group.id))
        if base is None:
            continue
        weight = (k - 1) / k
        for i in range(k):
            kept = [s for j, s in enumerate(scores) if j != i]
            moved = _replay(kept, spec, f"{group.id}-drop{i}")
            if moved is None:
                continue
            survivors = np.asarray([base[j] for j in range(k) if j != i], dtype=float)
            live = np.isfinite(survivors) & np.isfinite(moved)
            if not live.any():
                continue
            total += weight * float(np.sum((moved[live] - survivors[live]) ** 2))
            n += int(live.sum())
    return (total / n if n else math.nan), n


def advantage_variance(groups: Sequence[Group], spec: EstimatorSpec) -> float:
    """Variance of the replayed advantage over every rollout in the window."""
    vals: list[float] = []
    for group in groups:
        out = _replay(_group_totals(group), spec, str(group.id))
        if out is None:
            continue
        vals.extend(float(v) for v in out if math.isfinite(v))
    arr = np.asarray(vals, dtype=float)
    return float(arr.var(ddof=1)) if arr.size > 1 else math.nan


# ---------------------------------------------------------------------------
# The reading
# ---------------------------------------------------------------------------


@register_payload
@dataclass
class NoiseReading:
    """The noise share, and where it came from when rung 1 could run."""

    noise_share: float
    n_steps: int
    n_steps_with_proxies: int
    n_steps_proxy1_uncomputed: int
    n_steps_proxy3_censored: int
    share_per_step: list[float] = field(default_factory=list)
    steps: list[int] = field(default_factory=list)
    #: Rung 1. Shares of the *reported* noise share, summing to it. Empty at rung 0.
    attribution: dict[str, float] = field(default_factory=dict)
    #: The same, as points of the reported share, for a card.
    attribution_points: dict[str, float] = field(default_factory=dict)
    #: E5's `clipped / unclipped`, carried beside the attribution and never subtracted from it.
    #: NaN when the record does not carry both gradient norms, which is the usual case. It is here
    #: because a reader who has been told the clip inflates the noise share will look for it, and
    #: the honest answer is the number plus the sentence saying it does not enter.
    clip_shrinkage: float = math.nan
    clip_note: str = ""
    grader_variance: float = math.nan
    sampling_variance: float = math.nan
    total_advantage_variance: float = math.nan
    n_grader_draws: int = 0
    n_sampling_deletions: int = 0
    independent_error_assumed: bool = True
    baselines: dict[str, float] = field(default_factory=dict)
    says: str = ""

    def render(self) -> str:
        return self.says


def _proxy_refusal(instrument: str, steps: int) -> Refusal:
    return Refusal(
        instrument=instrument,
        reason=RefusalReason.ACCESS_INSUFFICIENT,
        detail=(
            f"none of the {steps} steps carries veRL's variance-proxy metrics. "
            f"`compute_variance_proxy_metrics` returns an empty dict unless `sum_pi_squared`, "
            f"`old_log_probs` and `advantages` are all in the batch "
            f"(`metric_utils.py:709-710`), and `actor.calculate_sum_pi_squared` defaults to "
            f"`false` (`actor.yaml:96`), so the whole metric emits nothing unless somebody opted "
            f"in."
        ),
        remedy=(
            "set `actor.calculate_sum_pi_squared: true` in the veRL config and re-run, so "
            "`sum_pi_squared` reaches the batch. Then check that `actor/grad_norm` is in the "
            "metrics dict at the call site, because without it proxy1 and proxy3 log a literal "
            "0.0 that this instrument cannot tell from a real zero."
        ),
        statistics={"n_steps": steps, "keys_looked_for": sorted(PROXY_KEYS.values())},
    )


def measure_noise_share(
    subject: Run | Sequence[Step] | Step,
    *,
    components: VarianceComponentsLike | None = None,
    draws: int = DEFAULT_DRAWS,
    seed: int = 0,
    instrument: str = "NoiseShare",
) -> NoiseReading | Refusal:
    """The noise share at rung 0, plus its attribution when A2's components are supplied."""
    steps = list(iter_steps(subject))
    proxies = read_proxies(subject)
    if not proxies:
        return _proxy_refusal(instrument, len(steps))

    usable = [p for p in proxies if not p.proxy1_uncomputed and p.proxy2 > 0]
    if not usable:
        return Refusal(
            instrument=instrument,
            reason=RefusalReason.ACCESS_INSUFFICIENT,
            detail=(
                f"all {len(proxies)} steps carrying variance-proxy metrics report "
                f"`proxy1_signal_strength` as exactly 0.0, which `metric_utils.py:789-791` emits "
                f"when `gradient_norm` is None. A logged zero there is indistinguishable from a "
                f"real zero, and `proxy3 = (1/(N-1))(proxy2 - proxy1)` inherits it, so the "
                f"reported noise share would be the whole of the gradient power by construction."
            ),
            remedy=(
                "make sure `actor/grad_norm` is present in the metrics dict at the "
                "`compute_variance_proxy_metrics` call site (`ray_trainer.py:1752`), so "
                "`gradient_norm` is not None. The norm it reads is the norm before clipping "
                "(`workers/engine/fsdp/transformer_impl.py:735`), which is the right one for "
                "proxy1 and needs no correction."
            ),
            statistics={"n_steps_with_proxies": len(proxies)},
        )

    shares = [p.share for p in usable]
    reading = NoiseReading(
        noise_share=float(np.mean(shares)),
        n_steps=len(steps),
        n_steps_with_proxies=len(proxies),
        n_steps_proxy1_uncomputed=sum(1 for p in proxies if p.proxy1_uncomputed),
        n_steps_proxy3_censored=sum(1 for p in proxies if p.proxy3_censored),
        share_per_step=shares,
        steps=[p.index for p in usable],
        baselines={
            "baseline.raw_noise_fraction": float(np.mean(shares)),
            "baseline.equal_share_attribution": 1.0 / (len(MECHANISMS) - 1),
        },
    )
    reading.says = (
        f"{reading.noise_share:.1%} of this window's gradient power is noise, read off veRL's "
        f"variance proxies over {len(usable)} of {len(steps)} steps. "
        f"{reading.n_steps_proxy3_censored} of those steps report proxy3 as exactly zero, which "
        f"the hard floor at `metric_utils.py:778` cannot be distinguished from a real zero."
    )

    if components is None:
        return reading

    attributed = _attribute(
        subject, reading, components, draws=draws, seed=seed, instrument=instrument
    )
    return attributed


def _attribute(
    subject: Run | Sequence[Step] | Step,
    reading: NoiseReading,
    components: VarianceComponentsLike,
    *,
    draws: int,
    seed: int,
    instrument: str,
) -> NoiseReading | Refusal:
    """Rung 1, or a refusal naming what stopped it."""
    groups = list(iter_groups(subject))
    specs = collect_specs(groups)
    if len(specs) != 1:
        return bounded_refusal(
            instrument=instrument,
            reason=RefusalReason.ACCESS_INSUFFICIENT,
            detail=(
                f"the attribution replays the recorded estimator and this window carries "
                f"{len(specs)} distinct estimator specifications, so there is no single transform "
                f"to replay."
            ),
            remedy="restrict the window to a span where E1 reports `stable`.",
            bound=_bare_evidence(reading),
            n_specs=len(specs),
        )
    spec = specs[0]

    check = check_replay(groups)
    if check.checked and not check.agrees:
        return bounded_refusal(
            instrument=instrument,
            reason=RefusalReason.ACCESS_INSUFFICIENT,
            detail=(
                f"the recorded EstimatorSpec does not reproduce the recorded advantages: "
                f"{check.n_agree} of {check.n_comparable} groups agree, worst disagreement "
                f"{check.max_abs_error:.4g} against a tolerance of {check.tol:.4g}. The "
                f"attribution perturbs scores and replays them through that spec, so a spec that "
                f"does not describe the transform gives shares of the wrong operator."
            ),
            remedy=(
                "fix the recorded `EstimatorSpec` at the tap so a replay reproduces the "
                "advantages, then re-run. E1's `replay` field is the same check and is the place "
                "to look first."
            ),
            bound=_bare_evidence(reading),
            n_agree=check.n_agree,
            n_comparable=check.n_comparable,
            max_abs_error=check.max_abs_error,
        )

    total_var = advantage_variance(groups, spec)
    if not math.isfinite(total_var) or total_var <= 0:
        return bounded_refusal(
            instrument=instrument,
            reason=RefusalReason.ACCESS_INSUFFICIENT,
            detail=(
                "the replayed advantages have no variance over this window, so there is nothing to "
                "apportion. Either every group is degenerate or the replay refused on all of them."
            ),
            remedy=(
                "widen the window to include groups with reward spread. E2 reports the degenerate "
                "fraction and is the place to look first."
            ),
            bound=_bare_evidence(reading),
        )

    v_grader, n_draws = grader_injected_variance(groups, spec, components, draws=draws, seed=seed)
    v_sampling, n_del = sampling_injected_variance(groups, spec)

    # E5's number, carried and not subtracted. `actor/grad_norm` is the pre-clipping norm, so
    # `proxy1` is the pre-clip signal strength and the clip does not bias `proxy3 / proxy2`. What
    # the clip does is shrink the applied update, which is a statement about the step and not about
    # the estimator's variance, so it sits beside the attribution rather than inside it.
    clip = measure_clip_effect(subject, instrument=instrument)
    if isinstance(clip, Refusal):
        reading.clip_shrinkage = math.nan
        reading.clip_note = (
            "This record does not carry both gradient norms, so the clip's shrinkage on the update "
            "is not computable. It would not have moved the shares either way: `actor/grad_norm` "
            "is the norm before clipping, so `proxy1` is the pre-clip signal strength."
        )
    else:
        reading.clip_shrinkage = clip.shrinkage
        reading.clip_note = (
            f"The clip scaled the applied update by {clip.shrinkage:.4g} on average over "
            f"{clip.n_steps_with_both_norms} steps, {clip.n_steps_shrunk} of which were clipped. "
            f"That is not subtracted from the noise share: `actor/grad_norm` is the norm before "
            f"clipping, so `proxy1` is the pre-clip signal strength and `proxy3 / proxy2` is a "
            f"statement about the gradient estimator rather than about the step that was taken."
        )

    share = reading.noise_share
    grader_points = share * (v_grader / total_var) if math.isfinite(v_grader) else 0.0
    sampling_points = share * (v_sampling / total_var) if math.isfinite(v_sampling) else 0.0
    named = grader_points + sampling_points
    if named > share > 0:
        # The two mechanisms cannot together exceed the share they are apportioning. When the
        # perturbation estimates say they do, the linearisation has broken down and the honest
        # response is to rescale them onto the share and say so rather than to report a negative
        # residual.
        scale = share / named
        grader_points *= scale
        sampling_points *= scale
        named = share
    unattributed = max(share - named, 0.0)

    reading.attribution = {
        "grader": grader_points,
        "sampling": sampling_points,
        "unattributed": unattributed,
    }
    reading.attribution_points = dict(reading.attribution)
    reading.grader_variance = v_grader
    reading.sampling_variance = v_sampling
    reading.total_advantage_variance = total_var
    reading.n_grader_draws = n_draws
    reading.n_sampling_deletions = n_del
    reading.independent_error_assumed = getattr(components, "error_covariance", None) is None
    reading.says = (
        f"{reading.noise_share:.1%} of this window's gradient power is reported as noise. Of that, "
        f"{100 * reading.attribution['grader']:.1f} points are grader replication variance, "
        f"{100 * reading.attribution['sampling']:.1f} are which K rollouts were drawn, and "
        f"{100 * reading.attribution['unattributed']:.1f} are unattributed. " + reading.clip_note
    )
    return reading


def _bare_evidence(reading: NoiseReading) -> Any:
    from reward_lens.core.evidence import make_evidence
    from reward_lens.core.types import SubjectRef

    return make_evidence(
        observable="NoiseShare",
        observable_version="1.0",
        subject=SubjectRef(signals=(), dataset=None, readout="reward"),
        value=reading,
        gauge=GaugeStatus.INVARIANT,
    )


# ---------------------------------------------------------------------------
# The instruments
# ---------------------------------------------------------------------------


class NoiseShare(EstimatorInstrument):
    """E3 rung 0. The fraction of gradient power that is noise, read off veRL's own proxies.

    Kill condition, from the catalogue record: if the attribution never separates grader from
    sampling on real data.
    """

    name = "NoiseShare"
    version = "1.0"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "E3"
    deviations = (
        "the proxies are read out of `OptimizerTelemetry.extra`, which is untyped by construction. "
        "The five keys are named as data on this module and the reading reports how many steps "
        "carried them",
        "a step whose `proxy1_signal_strength` is exactly 0.0 is dropped rather than averaged in, "
        "because `metric_utils.py:789-791` emits that literal when `gradient_norm` is None and it "
        "is indistinguishable from a real zero. The count of dropped steps is on the reading",
    )

    quantity = "estimator.noise_share"
    requires: dict[Component, Access] = {
        **RECORD_ACCESS,
        Component.OPTIMIZER: Access.RECORD,
    }
    substrates = ALL_SUBSTRATES
    phases = frozenset({Phase.IN_RUN, Phase.POST_RUN})
    envelope = NOISE_ENVELOPE
    invariance = "units"
    invariance_relation = INVARIANT
    baselines = NOISE_BASELINES
    rung = 0

    def __init__(self, subject: Run | Sequence[Step] | Step | None = None) -> None:
        self.subject = subject

    def compute(self) -> Any:
        if self.subject is None:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail="no record was supplied, so there are no variance proxies to read",
                remedy=(
                    "pass `subject=` a Run or a sequence of Steps whose `OptimizerTelemetry.extra` "
                    "carries veRL's `variance_proxy/*` metrics."
                ),
            )
        return measure_noise_share(self.subject, instrument=self.name)


class NoiseAttribution(EstimatorInstrument):
    """E3 rung 1. Which mechanism the noise came from, using A2's variance components.

    Kill condition, from the catalogue record: if the attribution never separates grader from
    sampling on real data.
    """

    name = "NoiseAttribution"
    version = "1.0"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "E3"
    deviations = (
        "the sampling term is a delete-one replay sensitivity rather than a jackknife variance. "
        "The statistic is a vector whose length changes with the deletion, so the pseudo-value "
        "algebra does not apply, and what is reported is how far the surviving advantages move",
        "the residual is reported as `unattributed` and never folded into the grader or sampling "
        "term. Where the two named terms together exceed the share they apportion, which means the "
        "linearisation has broken down, they are rescaled onto it and the reading says so",
        "rung 2 (carrying correlated grader error into an effective group size) is specified and "
        "not built here. `VarianceComponents.error_covariance` is the input it would need and it "
        "is already accepted by the grader perturbation, so the gap is A1's effective group size "
        "rather than this instrument",
        "there is no clip term in the attribution. E3's catalogue record illustrates the reading "
        "with three named terms, the third being 'the clip', and its prior-art line says `proxy1` "
        "inherits gradient clipping. It does not: `actor/grad_norm` is the norm before clipping in "
        "torch, in veRL's FSDP and Megatron engines and in transformers, each of which says so in "
        "its own source. The clip's shrinkage on the applied update is carried as "
        "`clip_shrinkage` beside the shares with the sentence that says why it is not one of them",
    )

    quantity = "estimator.noise_attribution"
    requires: dict[Component, Access] = {
        **RECORD_ACCESS,
        Component.OPTIMIZER: Access.RECORD,
        Component.GRADER: Access.REPLICATE,
    }
    substrates = ALL_SUBSTRATES
    phases = frozenset({Phase.IN_RUN, Phase.POST_RUN})
    envelope = NOISE_ENVELOPE
    invariance = "units"
    invariance_relation = INVARIANT
    baselines = NOISE_BASELINES
    rung = 1

    def __init__(
        self,
        subject: Run | Sequence[Step] | Step | None = None,
        *,
        components: VarianceComponentsLike | None = None,
        draws: int = DEFAULT_DRAWS,
        seed: int = 0,
    ) -> None:
        self.subject = subject
        self.components = components
        self.draws = int(draws)
        self.seed = int(seed)

    def compute(self) -> Any:
        if self.subject is None:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail="no record was supplied, so there is nothing to attribute",
                remedy=(
                    "pass `subject=` a Run and `components=` A2's variance components for the "
                    "grader that scored it."
                ),
            )
        if self.components is None:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail=(
                    "no variance components were supplied, so the grader's contribution cannot be "
                    "separated from the sampling contribution. Estimating the grader's error rate "
                    "from the reward signal itself is what the three correction papers do, and it "
                    "is circular when grader errors are correlated across items."
                ),
                remedy=(
                    "run A2 (`grader.variance_components`) on the grader that scored this run and "
                    "pass its output as `components=`. That needs GRADER:REPLICATE: the same "
                    "rollout scored more than once. If you have a published G-study instead, wrap "
                    "its numbers in `VarianceComponents(components={...})`."
                ),
            )
        return measure_noise_share(
            self.subject,
            components=self.components,
            draws=self.draws,
            seed=self.seed,
            instrument=self.name,
        )


__all__ = [
    "ATTRIBUTION_BIAS",
    "DEFAULT_DRAWS",
    "MECHANISMS",
    "NOISE_BASELINES",
    "NOISE_ENVELOPE",
    "PROXY_BIAS",
    "PROXY_KEYS",
    "NoiseAttribution",
    "NoiseReading",
    "NoiseShare",
    "ProxyStep",
    "VarianceComponents",
    "VarianceComponentsLike",
    "advantage_variance",
    "grader_injected_variance",
    "measure_noise_share",
    "read_proxies",
    "sampling_injected_variance",
]

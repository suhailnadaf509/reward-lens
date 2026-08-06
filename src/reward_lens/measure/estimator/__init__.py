"""Series E, the estimator: where a good reward becomes a bad gradient.

Between the grader and the gradient sits a transform that both prior designs treat as plumbing. It
is not plumbing. It is scale-invariant, so it destroys dose control; it divides by a group standard
deviation, so an all-fail group amplifies whatever variance is left; and every one of its fields
changes what a downstream number means.

Seven instruments over six catalogue records:

    E1  `RecordedEstimator`, `estimator.spec`
        The transform that actually ran, read off the record with its undeclared fields named and
        with a replay check that says whether the recorded spec describes what the trainer did.
        Nothing kills it: it is the precondition for the rest.
    E2  `DegenerateGroups`, `estimator.degenerate_fraction`, and `AllFailGroups`,
        `estimator.allfail_fraction`
        How many groups had no reward spread, how many were all-fail, and which of four causes the
        first are attributable to.
    E3  `NoiseShare`, `estimator.noise_share`, and `NoiseAttribution`,
        `estimator.noise_attribution`
        veRL already ships the fraction. This reads it with its two censoring gates enforced, and
        then attributes it to the clip, to grader replication variance and to which K rollouts were
        drawn.
    E4  `AmplifierSafety`, `estimator.amplifier_safety`
        The variance ratio between all-fail and mixed groups, per reward component. The single most
        actionable instrument in the series for someone about to add a reward term.
    E5  `ClipAccounting`, `estimator.clip_fraction_effect`
        The clip fraction, and what it does to every quantity derived from the gradient norm. This
        is the instrument that tells veRL's `proxy1` it is lying.
    E6  `LogprobMismatch`, `policy.train_infer_logprob_mismatch`
        How far apart the sampling engine and the training engine are, in nats per token.

**The correction this series carries, and it changes one instrument's answer rather than a
footnote.** `verifiers`' `score_group` is mean-centred with no standard-deviation division
(`rubrics/rubric.py:406-409`). Amplifier safety is a statement about the z-score, so on that
framework the amplification mechanism E4 measures is **absent**, and E4 reports that rather than a
ratio nobody can act on: it returns the component magnitudes as a bound, because with no division
dose control survives and magnitude is the right diagnostic there.

**Every instrument here reads a record and nothing else.** No grader call, no policy, no gradients,
no torch. What that buys is that the whole series runs against a training run somebody else did, on
a laptop, after the fact.

`register_all` populates the estimator ladder for every quantity in the series. It is idempotent and
it is not called at import, because a registry whose contents depend on import order is not a
registry.
"""

from __future__ import annotations

from reward_lens.measure.estimator._base import EstimatorInstrument
from reward_lens.measure.estimator.amplifier import (
    AMPLIFIER_BASELINES,
    AMPLIFIER_BIAS,
    AMPLIFIER_ENVELOPE,
    DEGENERATE_FRACTION_MAX,
    DOMINANCE_AT,
    LIVE_AMPLIFIER_AT,
    PERSISTENCE_AT,
    SAFE_BELOW,
    AmplifierReading,
    AmplifierSafety,
    ComponentSafety,
    SafetyTrajectory,
    component_names,
    measure_amplifier_safety,
    pooled_within_variance,
)
from reward_lens.measure.estimator.clip import (
    CLIP_BASELINES,
    CLIP_BIAS,
    CLIP_ENVELOPE,
    RATIO_CLIP_KEYS,
    ClipAccounting,
    ClipEffect,
    measure_clip_effect,
)
from reward_lens.measure.estimator.groups import (
    CAUSES,
    DEGENERATE_BIAS,
    FRAMEWORK_ZERO_STD_KEY,
    GROUP_BASELINES,
    GROUP_CENSUS_ENVELOPE,
    PHASES,
    AllFailGroups,
    DegenerateGroups,
    FailureFloor,
    GroupCensus,
    census_groups,
    classify_groups,
    component_scores,
    group_phase,
    partition_by_floor,
    total_scores,
)
from reward_lens.measure.estimator.mismatch import (
    FLOAT32_EPS,
    MISMATCH_BASELINES,
    MISMATCH_BIAS,
    MISMATCH_ENVELOPE,
    LogprobMismatch,
    MismatchReading,
    StreamLengthCheck,
    check_stream_lengths,
    float32_floor_at,
    measure_mismatch,
    sequence_totals,
)
from reward_lens.measure.estimator.noise import (
    ATTRIBUTION_BIAS,
    DEFAULT_DRAWS,
    MECHANISMS,
    NOISE_BASELINES,
    NOISE_ENVELOPE,
    PROXY_BIAS,
    PROXY_KEYS,
    NoiseAttribution,
    NoiseReading,
    NoiseShare,
    ProxyStep,
    VarianceComponents,
    VarianceComponentsLike,
    advantage_variance,
    grader_injected_variance,
    measure_noise_share,
    read_proxies,
    sampling_injected_variance,
)
from reward_lens.measure.estimator.spec import (
    ALL_SUBSTRATES,
    DECISIVE_FIELDS,
    ESTIMATOR_SPEC_ENVELOPE,
    FRAMEWORK_DEFAULTS,
    RECORD_ACCESS,
    REPLAY_TOL,
    SPEC_BASELINES,
    EstimatorReading,
    RecordedEstimator,
    ReplayCheck,
    check_replay,
    collect_specs,
    iter_groups,
    iter_steps,
    read_estimator_spec,
)
from reward_lens.measure.estimator.units import EstimatorQuantity, difference, unit_of

#: The seven instruments, in the order the series is read: the spec first, because everything
#: downstream is conditioned on it.
ESTIMATOR: tuple[type, ...] = (
    RecordedEstimator,
    DegenerateGroups,
    AllFailGroups,
    NoiseShare,
    NoiseAttribution,
    AmplifierSafety,
    ClipAccounting,
    LogprobMismatch,
)

_REGISTERED = False


def register_all() -> None:
    """Register every estimator in series E on the quantity ladder. Idempotent."""
    global _REGISTERED
    if _REGISTERED:
        return

    from reward_lens.core.quantity import QUANTITIES as _Q
    from reward_lens.core.quantity import (
        BiasStatement,
        CostModel,
        EstimatorEntry,
        load_quantities,
        register_estimator,
    )

    if "estimator.spec" not in _Q:
        load_quantities()

    free = CostModel(note="a pass over the record; no grader call, no policy, no gradients")

    register_estimator(
        EstimatorEntry(
            quantity="estimator.spec",
            impl="estimator.spec.r0_read",
            requires=RECORD_ACCESS,
            envelope=ESTIMATOR_SPEC_ENVELOPE,
            rung=0,
            bias=BiasStatement(
                direction="unknown",
                why=(
                    "a transcription is right or wrong rather than biased. What it can be is "
                    "incomplete, and the reading names every field left at a placeholder instead "
                    "of reporting the placeholder as a value."
                ),
            ),
            cost=free,
            run=read_estimator_spec,
        )
    )
    for quantity, impl in (
        ("estimator.degenerate_fraction", "estimator.groups.r0_count"),
        ("estimator.allfail_fraction", "estimator.groups.r0_allfail"),
    ):
        register_estimator(
            EstimatorEntry(
                quantity=quantity,
                impl=impl,
                requires=RECORD_ACCESS,
                envelope=GROUP_CENSUS_ENVELOPE,
                rung=0,
                bias=DEGENERATE_BIAS,
                cost=free,
                run=census_groups,
            )
        )
    register_estimator(
        EstimatorEntry(
            quantity="estimator.degenerate_fraction",
            impl="estimator.groups.r1_attribute",
            requires=RECORD_ACCESS,
            envelope=GROUP_CENSUS_ENVELOPE,
            rung=1,
            bias=DEGENERATE_BIAS,
            cost=free,
            run=census_groups,
        )
    )
    register_estimator(
        EstimatorEntry(
            quantity="estimator.noise_share",
            impl="estimator.noise.r0_verl_proxy3",
            requires=NoiseShare.requires,
            envelope=NOISE_ENVELOPE,
            rung=0,
            bias=PROXY_BIAS,
            cost=free,
            run=measure_noise_share,
        )
    )
    register_estimator(
        EstimatorEntry(
            quantity="estimator.noise_attribution",
            impl="estimator.noise.r1_attribute",
            requires=NoiseAttribution.requires,
            envelope=NOISE_ENVELOPE,
            rung=1,
            bias=ATTRIBUTION_BIAS,
            cost=CostModel(
                note=(
                    "one replay per group per perturbation draw, plus one per rollout for the "
                    "delete-one term. Seconds on a record with hundreds of groups, and it needs "
                    "A2's variance components, which need GRADER:REPLICATE"
                )
            ),
            run=measure_noise_share,
        )
    )
    for rung in (0, 1, 2):
        register_estimator(
            EstimatorEntry(
                quantity="estimator.amplifier_safety",
                impl=f"estimator.amplifier.r{rung}",
                requires=RECORD_ACCESS,
                envelope=AMPLIFIER_ENVELOPE,
                rung=rung,
                bias=AMPLIFIER_BIAS,
                cost=free,
                run=measure_amplifier_safety,
            )
        )
    register_estimator(
        EstimatorEntry(
            quantity="estimator.clip_fraction_effect",
            impl="estimator.clip.r0_norm_ratio",
            requires=ClipAccounting.requires,
            envelope=CLIP_ENVELOPE,
            rung=0,
            bias=CLIP_BIAS,
            cost=free,
            run=measure_clip_effect,
        )
    )
    register_estimator(
        EstimatorEntry(
            quantity="policy.train_infer_logprob_mismatch",
            impl="estimator.mismatch.r0_mean_abs_gap",
            requires=LogprobMismatch.requires,
            envelope=MISMATCH_ENVELOPE,
            rung=0,
            bias=MISMATCH_BIAS,
            cost=free,
            run=measure_mismatch,
        )
    )
    _REGISTERED = True


__all__ = [
    "ALL_SUBSTRATES",
    "AMPLIFIER_BASELINES",
    "AMPLIFIER_BIAS",
    "AMPLIFIER_ENVELOPE",
    "ATTRIBUTION_BIAS",
    "CAUSES",
    "CLIP_BASELINES",
    "CLIP_BIAS",
    "CLIP_ENVELOPE",
    "DECISIVE_FIELDS",
    "DEFAULT_DRAWS",
    "DEGENERATE_BIAS",
    "DEGENERATE_FRACTION_MAX",
    "DOMINANCE_AT",
    "ESTIMATOR",
    "ESTIMATOR_SPEC_ENVELOPE",
    "FLOAT32_EPS",
    "FRAMEWORK_DEFAULTS",
    "FRAMEWORK_ZERO_STD_KEY",
    "GROUP_BASELINES",
    "GROUP_CENSUS_ENVELOPE",
    "LIVE_AMPLIFIER_AT",
    "MECHANISMS",
    "MISMATCH_BASELINES",
    "MISMATCH_BIAS",
    "MISMATCH_ENVELOPE",
    "NOISE_BASELINES",
    "NOISE_ENVELOPE",
    "PERSISTENCE_AT",
    "PHASES",
    "PROXY_BIAS",
    "PROXY_KEYS",
    "RATIO_CLIP_KEYS",
    "RECORD_ACCESS",
    "REPLAY_TOL",
    "SAFE_BELOW",
    "SPEC_BASELINES",
    "AllFailGroups",
    "AmplifierReading",
    "AmplifierSafety",
    "ClipAccounting",
    "ClipEffect",
    "ComponentSafety",
    "DegenerateGroups",
    "EstimatorInstrument",
    "EstimatorQuantity",
    "EstimatorReading",
    "FailureFloor",
    "GroupCensus",
    "LogprobMismatch",
    "MismatchReading",
    "NoiseAttribution",
    "NoiseReading",
    "NoiseShare",
    "ProxyStep",
    "RecordedEstimator",
    "ReplayCheck",
    "SafetyTrajectory",
    "StreamLengthCheck",
    "VarianceComponents",
    "VarianceComponentsLike",
    "advantage_variance",
    "census_groups",
    "check_replay",
    "check_stream_lengths",
    "classify_groups",
    "collect_specs",
    "component_names",
    "component_scores",
    "difference",
    "float32_floor_at",
    "grader_injected_variance",
    "group_phase",
    "iter_groups",
    "iter_steps",
    "measure_amplifier_safety",
    "measure_clip_effect",
    "measure_mismatch",
    "measure_noise_share",
    "partition_by_floor",
    "pooled_within_variance",
    "read_estimator_spec",
    "read_proxies",
    "register_all",
    "sampling_injected_variance",
    "sequence_totals",
    "total_scores",
    "unit_of",
]

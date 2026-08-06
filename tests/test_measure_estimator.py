"""Series E, the estimator, tested against arithmetic somebody can do on paper.

Four kinds of test, and the middle two are the ones that would have caught a real defect.

**Hand-computed values.** A group whose z-score is a fraction you can write down, and an all-fail
group where the amplification is analytic: the task component has zero within-group variance in an
all-fail group by construction, so its amplifier safety is exactly zero, and an auxiliary whose
variance does not depend on task success has a ratio you can compute from two numbers.

**The generated invariance test per instrument.** E4's ratio is a ratio of variances of the *same*
component, so it must be invariant under an affine rescaling of that component. That is checked
rather than asserted, because the way it could fail is subtle and real: if the partition into
all-fail and mixed moved with the component being rescaled, the ratio would move too, and that would
be a defect in the instrument rather than a property of the estimator.

**The one place the invariance genuinely fails, measured rather than tolerated.** E2's degenerate
flag is ``std <= eps``, an absolute threshold on a scale-carrying quantity, so the degenerate
fraction is affine-invariant only outside a band of width ``eps`` around zero spread. There are two
tests: one on a window with no group inside the band, which passes and is the merge gate, and one on
a window with a group placed inside it, which fails and records what the failure means. The
*quantity* as stated ("groups with zero reward spread") is invariant; the estimator
the trainer actually runs is not, and the difference is the epsilon.

**A refusal test per instrument, asserting the reason and the remedy string.** A refusal with no
remedy is a tool that looks broken instead of a tool that looks careful, and the remedy is the part
a user acts on.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from reward_lens.core.budget import LimitOfDetection
from reward_lens.core.envelope import RegimeCondition, RegimeReading
from reward_lens.core.invariance import (
    INVARIANT,
    InvariancePayload,
    check_invariance,
    check_unit_refusal,
)
from reward_lens.core.quantity import QUANTITIES, load_quantities
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.types import Access, Component
from reward_lens.measure.base import Context, lint_instrument
from reward_lens.measure.estimator import (
    CAUSES,
    ESTIMATOR,
    MECHANISMS,
    PROXY_KEYS,
    REPLAY_TOL,
    AllFailGroups,
    AmplifierSafety,
    ClipAccounting,
    DegenerateGroups,
    EstimatorQuantity,
    FailureFloor,
    LogprobMismatch,
    NoiseAttribution,
    NoiseShare,
    RecordedEstimator,
    VarianceComponents,
    census_groups,
    check_replay,
    difference,
    measure_amplifier_safety,
    measure_clip_effect,
    measure_mismatch,
    measure_noise_share,
    pooled_within_variance,
    read_estimator_spec,
    register_all,
    sequence_totals,
)
from reward_lens.record.schema import (
    EstimatorSpec,
    Group,
    GroupStats,
    InMemoryStepStream,
    OptimizerTelemetry,
    RegimeDeclaration,
    Run,
    Step,
    make_trajectory,
)
from reward_lens.record.scores import Leaf, WeightedSum
from reward_lens.record.turns import Turn

load_quantities()
register_all()

#: The estimator TRL 1.9.2 actually runs, with `clip_low`/`clip_high` left unset. The tap writes
#: TRL's `epsilon` into them and `replay_advantages` reads them as advantage bounds, which is the
#: defect this package found on the first real record; the fixtures here are built the way the
#: schema means them so the replay check has something that agrees to compare against.
GRPO = EstimatorSpec(
    family="grpo/dapo",
    group_centred=True,
    std_normalised=True,
    std_epsilon=1e-4,
    # TRL's own divisor: `nanstd` multiplies the variance by `count/(count - 1)` at
    # `trl/trainer/utils.py:877-879`, so the sample form. Declared here rather than left at None
    # because the two differ by 15.5% at K = 4 and `replay_advantages` refuses without it.
    std_ddof=1,
    degenerate_policy="zero",
    aggregation="token",
    loss_mask_policy="mask_environment",
)

MEAN_CENTRED = EstimatorSpec(
    family="verifiers/score_group",
    group_centred=True,
    std_normalised=False,
    degenerate_policy="keep",
    aggregation="sequence",
    loss_mask_policy="none",
)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def leafs(task: float, aux: float) -> WeightedSum:
    """A two-component composite: a binary task leaf and one auxiliary at weight 1."""
    return WeightedSum(
        name="reward",
        children=(
            Leaf(name="task", value=task, grader_call=None),
            Leaf(name="aux", value=aux, grader_call=None),
        ),
        weights=(1.0, 1.0),
    )


def group(
    ident: str,
    task: list[float],
    aux: list[float],
    *,
    spec: EstimatorSpec = GRPO,
    advantages: list[float] | None = None,
    turns: list[Turn] | None = None,
) -> Group:
    totals = [t + a for t, a in zip(task, aux)]
    trajectories = tuple(
        make_trajectory(
            id=f"{ident}-{i}",
            task_ref=ident,
            turns=turns or [Turn(index=0, role="assistant")],
            scores=leafs(t, a),
            advantage=None if advantages is None else advantages[i],
        )
        for i, (t, a) in enumerate(zip(task, aux))
    )
    return Group(
        id=ident,  # type: ignore[arg-type]
        task_ref=ident,  # type: ignore[arg-type]
        trajectories=trajectories,
        estimator=spec,
        group_stats=GroupStats.from_scores(totals, std_epsilon=spec.std_epsilon or 0.0),
    )


def step(index: int, groups: list[Group], **optimizer: object) -> Step:
    return Step(
        index=index,
        groups=tuple(groups),
        schedule={},
        optimizer=OptimizerTelemetry(**optimizer),  # type: ignore[arg-type]
    )


def run_of(steps: list[Step]) -> Run:
    return Run(
        id="run-under-test",  # type: ignore[arg-type]
        kind="train",
        components={},
        access={Component.RECORD: Access.RECORD, Component.ESTIMATOR: Access.RECORD},
        regime=RegimeDeclaration(),
        steps=InMemoryStepStream(steps),
    )


def z_advantages(totals: list[float], eps: float = 1e-4, ddof: int = 1) -> list[float]:
    """The group z-score, by hand: `(r - mean) / (std + eps)`.

    ``ddof`` defaults to 1, which is what both frameworks in scope do: veRL divides by
    ``torch.std``, whose default is ``correction=1`` (`core_algos.py:321`), and TRL's `nanstd`
    multiplies the variance by ``count/(count - 1)`` (`trl/trainer/utils.py:877-879`).

    ``record.scores.replay_advantages`` calls ``present.std()``, which is numpy's ``ddof=0``, so a
    test comparing it against a hand computation has to say which convention it is asserting. Every
    call site here that passes ``ddof=0`` is asserting the shipped behaviour rather than the
    framework's, and says so at the call.
    """
    arr = np.asarray(totals, dtype=float)
    return list((arr - arr.mean()) / (arr.std(ddof=ddof) + eps))


#: Two all-fail groups and two mixed ones, with the auxiliary's within-group variance chosen so the
#: ratio is a number you can do in your head.
def mixed_window(spec: EstimatorSpec = GRPO) -> list[Group]:
    return [
        # All-fail: the task leaf is identically 0, so its within-group variance is exactly 0. The
        # auxiliary spreads over {0, 2}, giving an unbiased within-group variance of 4/3.
        group("af-0", [0.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 2.0], spec=spec),
        group("af-1", [0.0, 0.0, 0.0, 0.0], [2.0, 0.0, 2.0, 0.0], spec=spec),
        # Mixed: the task leaf has spread, and the auxiliary spreads over {0, 1} for an unbiased
        # within-group variance of 1/3. The composite total varies inside every group here, so no
        # group is degenerate and E4's envelope holds.
        group("mx-0", [1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0], spec=spec),
        group("mx-1", [0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 1.0, 0.0], spec=spec),
    ]


FLOOR = FailureFloor(at=0.0, component="task")


def ctx() -> Context:
    return Context()


# ===========================================================================
# Hand-computed values
# ===========================================================================


def test_the_pooled_within_variance_is_the_dof_weighted_one_and_drops_singletons():
    """`sum (n-1) s^2 / sum (n-1)`, and a group of one contributes nothing rather than a zero."""
    v, used, dof = pooled_within_variance([[0.0, 2.0], [0.0, 1.0, 2.0], [5.0]])
    # (1 * 2.0 + 2 * 1.0) / 3
    assert v == pytest.approx(4.0 / 3.0)
    assert (used, dof) == (2, 3)

    # Averaging the per-group variances would give 1.5, which is the number this is not.
    assert v != pytest.approx(1.5)


def test_an_all_fail_group_has_zero_task_variance_so_the_task_component_is_analytically_safe():
    """The mechanism, on paper: a binary task leaf cannot vary inside a group where all rollouts
    failed, so its numerator is exactly zero and its amplifier safety is exactly zero."""
    reading = measure_amplifier_safety(mixed_window(), floor=FLOOR)
    assert not isinstance(reading, Refusal), reading
    assert reading.safety["task"] == 0.0
    assert reading.detail["task"]["var_allfail"] == 0.0
    assert reading.verdicts["task"] == "amplifier-safe"

    # The auxiliary: 4/3 inside the all-fail groups against 1/3 inside the mixed ones, so exactly 4.
    aux = reading.detail["aux"]
    assert aux["var_allfail"] == pytest.approx(4.0 / 3.0)
    assert aux["var_mixed"] == pytest.approx(1.0 / 3.0)
    assert aux["var_allfail"] == pytest.approx(np.var([0.0, 2.0, 0.0, 2.0], ddof=1))
    assert aux["var_mixed"] == pytest.approx(np.var([0.0, 1.0, 0.0, 1.0], ddof=1))
    assert reading.safety["aux"] == pytest.approx(4.0)
    assert reading.verdicts["aux"] == "live amplifier"

    # The mandatory baseline, rendered beside the ratio because the contrast is the argument.
    assert reading.baselines["magnitude/aux"] == pytest.approx(0.75)
    assert reading.baselines["magnitude/task"] == pytest.approx(0.25)
    assert "mean |r|" in reading.render()


def test_the_group_z_score_replay_agrees_with_the_hand_computation():
    """`check_replay` is the claim that the recorded spec describes the transform that ran.

    The divisor is now on the record, so the hand computation and the replay agree about which one
    they mean instead of sharing a default. `GRPO` declares ``std_ddof = 1``, which is TRL's, and
    `z_advantages` computes the same by default.
    """
    totals = [1.0, 2.0, 3.0, 4.0]
    hand = z_advantages(totals)
    assert hand[0] == pytest.approx((1.0 - 2.5) / (math.sqrt(5 / 3) + 1e-4))

    g = group("g", [1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0], advantages=hand)
    check = check_replay([g])
    assert check.checked and check.agrees
    assert check.max_abs_error < 1e-12
    assert "reproduces the recorded advantages" in check.render()

    # And a record whose advantages were written under the *other* convention does not agree, which
    # is the whole reason the divisor is a field. The two differ by the Bessel factor,
    # `sqrt(K/(K-1)) - 1`, which is 15.47% at K = 4 and 0.1797 in absolute terms here, against a
    # tolerance of 1e-4. Sharing `ddof=0` on both sides is what used to hide this.
    population = z_advantages(totals, ddof=0)
    off = group("off", totals, [0.0] * 4, advantages=population)
    missed = check_replay([off])
    assert missed.checked and not missed.agrees
    assert missed.max_abs_error == pytest.approx(0.1797, abs=5e-4)
    assert missed.max_abs_error > 1000 * REPLAY_TOL
    assert max(abs(p) for p in population) / max(abs(h) for h in hand) == pytest.approx(
        math.sqrt(4 / 3), rel=2e-4
    )


def test_the_replay_check_catches_a_spec_that_does_not_describe_the_transform():
    """A spec that declares the wrong variance divisor, which is the live version of this defect.

    This test used to build its disagreement out of TRL's ratio clip, which
    `record.scores.replay_advantages` applied as a bound on the advantage: with `epsilon = 0.2`
    every replayed advantage came back as exactly 0.2. That route is closed, because the ratio clip
    is a property of the loss and is no longer applied to the advantage (E50), so the
    disagreement here is built from the divisor instead. A record whose advantages TRL wrote under
    Bessel's correction and whose spec declares the population form misses by 0.1797 at K = 4,
    which is 1,797 times the tolerance.
    """
    totals = [1.0, 2.0, 3.0, 4.0]
    wrong_divisor = EstimatorSpec(
        family="grpo/dapo",
        group_centred=True,
        std_normalised=True,
        std_epsilon=1e-4,
        std_ddof=0,
        degenerate_policy="zero",
    )
    g = group("g", totals, [0.0] * 4, spec=wrong_divisor, advantages=z_advantages(totals, ddof=1))
    check = check_replay([g])
    assert check.checked and not check.agrees
    assert check.n_agree == 0
    assert check.max_abs_error == pytest.approx(0.1797, abs=5e-4)
    assert check.max_abs_error > 1000 * REPLAY_TOL
    assert "does not describe the transform that ran" in check.render()

    reading = read_estimator_spec([g])
    assert reading.replay["agrees"] == 0.0
    assert "does not describe the transform that ran" in reading.says
    # And the divisor is a decisive field, so E1 scores it against TRL's default and finds it off.
    assert "std_ddof" in reading.differs_from_default


def test_the_recorded_spec_reading_names_undeclared_and_ambiguous_fields_apart():
    bare = EstimatorSpec(group_centred=True, std_normalised=True)
    reading = read_estimator_spec([group("g", [0.0, 1.0], [0.0, 0.0], spec=bare)], replay=False)
    assert not isinstance(reading, Refusal)
    # `family`, `degenerate_policy`, `aggregation` and `loss_mask_policy` sit at "unknown"; a
    # z-scoring estimator with no epsilon is the fifth and one with no divisor is the sixth. Both
    # of those sit in the same denominator and neither is recoverable from the scores, so neither
    # is ambiguous: the trainer divided by something and the record does not say what.
    assert set(reading.undeclared) == {
        "family",
        "degenerate_policy",
        "aggregation",
        "loss_mask_policy",
        "std_epsilon",
        "std_ddof",
    }
    # `None` on an optional numeric means either absent or unrecorded, and it is reported as such.
    assert "clip_low" in reading.ambiguous and "clip_low" not in reading.undeclared
    assert "Undeclared:" in reading.says and "either absent or unrecorded" in reading.says


def test_the_census_partitions_the_degenerate_groups_into_four_named_causes():
    window = [
        # degenerate and all-fail: task 0 everywhere, auxiliary flat
        group("d-task", [0.0, 0.0], [1.0, 1.0]),
        # degenerate and saturated: task at the ceiling everywhere
        group("d-sat", [1.0, 1.0], [1.0, 1.0]),
        # degenerate at neither extreme: the grader gave four rollouts the same middling score
        group("d-res", [1.0, 0.0], [0.0, 1.0]),
        # live
        group("live", [1.0, 0.0], [1.0, 0.0]),
    ]
    census = census_groups(window, floor=FailureFloor(at=0.0, component="task", saturates_at=1.0))
    assert not isinstance(census, Refusal)
    assert census.n_groups == 4
    assert census.n_degenerate == 3
    assert census.causes["task_difficulty"] == 1
    assert census.causes["grader_saturation"] == 1
    assert census.causes["grader_resolution"] == 1
    assert sum(census.causes.values()) == census.n_degenerate
    assert set(census.causes) == set(CAUSES)
    assert census.baselines["baseline.nominal_group_size"] == 2.0


def test_the_noise_share_is_proxy3_over_proxy2_read_off_the_record():
    telemetry = {
        PROXY_KEYS["proxy1"]: 4.0,
        PROXY_KEYS["proxy2"]: 10.0,
        PROXY_KEYS["proxy3"]: 3.0,
    }
    r = run_of([step(0, mixed_window(), extra=telemetry)])
    reading = measure_noise_share(r)
    assert not isinstance(reading, Refusal), reading
    assert reading.noise_share == pytest.approx(0.3)
    assert reading.n_steps_with_proxies == 1
    assert reading.n_steps_proxy1_uncomputed == 0


def test_the_clip_effect_is_the_ratio_of_the_two_recorded_norms():
    r = run_of(
        [
            step(
                0,
                mixed_window(),
                grad_norm_clipped=1.0,
                grad_norm_unclipped=1.5,
                extra={"clip_ratio/region_mean": 0.18},
            ),
            step(
                1,
                mixed_window(),
                grad_norm_clipped=2.0,
                grad_norm_unclipped=2.5,
                extra={"clip_ratio/region_mean": 0.10},
            ),
        ]
    )
    reading = measure_clip_effect(r)
    assert not isinstance(reading, Refusal), reading
    assert reading.effect == pytest.approx((0.5 + 0.25) / 2)
    assert reading.clip_fraction == pytest.approx(0.14)
    assert "clip_ratio/region_mean" in reading.clip_fraction_source

    # The shrinkage the clip applied to the update: `clipped / unclipped`, per step then averaged.
    assert reading.shrinkage == pytest.approx((1.0 / 1.5 + 2.0 / 2.5) / 2)
    assert reading.shrinkage_per_step == [pytest.approx(2 / 3), pytest.approx(0.8)]
    assert reading.n_steps_shrunk == 2

    # `ratio_squared` is the mean of the per-step squares, and squaring the mean effect instead
    # would be low by exactly the variance of the per-step ratio. This assertion used to read
    # `proxy1_understatement == (1.375)**2 - 1.0`, which was both the square of a mean and a claim
    # that veRL's proxy1 is the post-clipping norm. It is the pre-clipping norm, so there is no
    # understatement to report; what is left is the Jensen gap, and it is measured here.
    assert reading.ratio_squared == pytest.approx((1.5**2 + 1.25**2) / 2)
    assert reading.ratio_squared == pytest.approx(1.90625)
    assert reading.ratio_squared_jensen_gap == pytest.approx(1.90625 - 1.375**2)
    assert reading.ratio_squared_jensen_gap == pytest.approx(0.015625)
    assert reading.ratio_squared_jensen_gap == pytest.approx(float(np.var([1.5, 1.25])))
    assert not hasattr(reading, "proxy1_understatement")


def test_the_mismatch_is_a_mean_absolute_per_token_gap_and_a_sequence_total():
    turn = Turn(
        index=0,
        role="assistant",
        token_ids=(1, 2, 3, 4),
        logprobs_sampling=(-1.0, -2.0, -3.0, -4.0),
        logprobs_train=(-1.5, -1.5, -3.5, -3.5),
    )
    reading = measure_mismatch(
        make_trajectory(id="t", task_ref="k", turns=[turn]),
    )
    assert not isinstance(reading, Refusal), reading
    # gaps: -0.5, +0.5, -0.5, +0.5
    assert reading.per_token == pytest.approx(0.5)
    assert reading.n_tokens == 4
    # and the sequence total cancels exactly, which is the point of carrying it
    assert reading.per_sequence == pytest.approx(0.0, abs=1e-12)
    assert sequence_totals([turn]) == [pytest.approx(0.0, abs=1e-12)]


def test_a_mismatch_below_the_numerics_floor_is_a_reading_rather_than_a_refusal():
    """E6's kill condition is good news, so it must not come back as `BELOW_LOD`."""
    turn = Turn(
        index=0,
        role="assistant",
        token_ids=(1, 2),
        logprobs_sampling=(-1.0, -2.0),
        logprobs_train=(-1.0 + 1e-9, -2.0 - 1e-9),
    )
    reading = measure_mismatch(
        make_trajectory(id="t", task_ref="k", turns=[turn]),
        lod=LimitOfDetection(sigma_blank=1e-4, sensitivity=1.0),
    )
    assert not isinstance(reading, Refusal)
    assert reading.below_floor is True
    assert "worth publishing" in reading.says


# ===========================================================================
# The attribution separates grader from sampling
# ===========================================================================


def test_the_attribution_separates_grader_from_sampling_and_names_its_residual():
    """E3's kill condition is that it never separates the two. On a window where the grader's
    error variance is a tenth of the score's spread, the two terms come back different."""
    telemetry = {
        PROXY_KEYS["proxy1"]: 6.0,
        PROXY_KEYS["proxy2"]: 10.0,
        PROXY_KEYS["proxy3"]: 4.0,
    }
    r = run_of([step(0, mixed_window(), extra=telemetry)])
    components = VarianceComponents(
        components={"item": 1.0, "rater": 0.1, "residual": 0.02},
        design="two-facet, hand-built for this test",
    )
    reading = measure_noise_share(r, components=components, draws=64, seed=7)
    assert not isinstance(reading, Refusal), reading
    assert set(reading.attribution) == set(MECHANISMS)
    # Named rather than only compared against MECHANISMS, because both moved together when the
    # clip term was withdrawn and a test that only compares the two would not have noticed.
    assert set(reading.attribution) == {"grader", "sampling", "unattributed"}
    assert sum(reading.attribution.values()) == pytest.approx(reading.noise_share, rel=1e-9)
    assert reading.attribution["grader"] > 0.0
    assert reading.attribution["sampling"] > 0.0
    assert reading.attribution["grader"] != reading.attribution["sampling"]
    assert reading.independent_error_assumed is True
    assert "grader replication variance" in reading.says


def test_a_larger_grader_error_moves_the_grader_share_and_leaves_sampling_alone():
    """The separation is a measurement rather than a split: only one term responds to A2."""
    telemetry = {
        PROXY_KEYS["proxy1"]: 6.0,
        PROXY_KEYS["proxy2"]: 10.0,
        PROXY_KEYS["proxy3"]: 4.0,
    }
    r = run_of([step(0, mixed_window(), extra=telemetry)])
    small = measure_noise_share(
        r, components=VarianceComponents(components={"rater": 0.01}), draws=64, seed=3
    )
    large = measure_noise_share(
        r, components=VarianceComponents(components={"rater": 0.25}), draws=64, seed=3
    )
    assert large.grader_variance > 5.0 * small.grader_variance
    assert large.sampling_variance == pytest.approx(small.sampling_variance)


# ===========================================================================
# Property tests
# ===========================================================================


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    a=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
    b=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
)
def test_amplifier_safety_is_invariant_under_rescaling_the_component(a: float, b: float):
    """Both variances are variances of the same component, so `a**2` cancels and `b` drops out."""
    base = measure_amplifier_safety(mixed_window(), floor=FLOOR)

    scaled = [
        group(
            g.id,
            [
                next(
                    leaf.value for leaf in g.trajectories[i].scores.children if leaf.name == "task"
                )
                for i in range(len(g.trajectories))
            ],
            [
                a
                * next(
                    leaf.value for leaf in g.trajectories[i].scores.children if leaf.name == "aux"
                )
                + b
                for i in range(len(g.trajectories))
            ],
        )
        for g in mixed_window()
    ]
    moved = measure_amplifier_safety(scaled, floor=FLOOR)
    assert moved.safety["aux"] == pytest.approx(base.safety["aux"], rel=1e-9, abs=1e-12)
    assert moved.safety["task"] == base.safety["task"] == 0.0


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    values=st.lists(
        st.lists(
            st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=6,
        ),
        min_size=1,
        max_size=8,
    )
)
def test_the_degenerate_fraction_is_a_fraction(values: list[list[float]]):
    window = [group(f"g{i}", [0.0] * len(vals), vals) for i, vals in enumerate(values)]
    census = census_groups(window)
    assert not isinstance(census, Refusal)
    assert 0.0 <= census.degenerate_fraction <= 1.0
    assert census.n_degenerate <= census.n_groups


# ===========================================================================
# The generated invariance test, per instrument
# ===========================================================================


def _payload_groups(payload: InvariancePayload) -> list[Group]:
    """Rebuild the four-group window from a transformed payload.

    `scores` carries the *auxiliary component* per rollout and `group_ids` the partition, so the
    affine action rescales the component and leaves the all-fail partition where it was. That
    separation is the whole point: if group membership moved with the component, E4's ratio would
    move too, and that would be a defect here rather than a property of the estimator.
    """
    scores = np.asarray(payload.scores, dtype=float)
    gids = np.asarray(payload.group_ids)
    task_by_group = payload.extra["task"]
    out = []
    for g in sorted(set(gids.tolist())):
        idx = np.flatnonzero(gids == g)
        out.append(group(f"g{g}", list(task_by_group[g]), [float(scores[i]) for i in idx]))
    return out


def _amplifier_payload() -> InvariancePayload:
    task = {
        0: [0.0, 0.0, 0.0, 0.0],
        1: [0.0, 0.0, 0.0, 0.0],
        2: [1.0, 0.0, 1.0, 0.0],
        3: [0.0, 1.0, 0.0, 1.0],
    }
    aux = [0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]
    return InvariancePayload(
        scores=np.asarray(aux, dtype=float),
        group_ids=np.repeat(np.arange(4), 4),
        extra={"task": task},
    )


def test_e4_passes_its_generated_invariance_test_under_reward_affine():
    report = check_invariance(
        AmplifierSafety(),
        "reward.affine",
        _amplifier_payload(),
        n=48,
        relation=INVARIANT,
        run=lambda inst, p: measure_amplifier_safety(_payload_groups(p), floor=FLOOR).safety["aux"],
    )
    assert report.passed, report.render()
    # Exact rather than approximate: `a**2` cancels in a ratio of two variances of one component.
    assert report.max_deviation < 1e-12, report.render()
    assert QUANTITIES.get("estimator.amplifier_safety").invariance == "reward.affine"


def test_e2_passes_reward_affine_away_from_the_epsilon_band():
    """The merge gate. With every group's spread far from the estimator's epsilon, the degenerate
    fraction does not move under an affine rescaling of the reward."""
    payload = InvariancePayload(
        scores=np.asarray([0.0, 1.0, 0.0, 2.0, 3.0, 5.0], dtype=float),
        group_ids=np.repeat(np.arange(3), 2),
    )

    def run(_inst, p):
        gids = np.asarray(p.group_ids)
        scores = np.asarray(p.scores, dtype=float)
        window = [
            group(
                f"g{g}",
                [0.0] * int((gids == g).sum()),
                [float(v) for v in scores[gids == g]],
            )
            for g in sorted(set(gids.tolist()))
        ]
        return census_groups(window).degenerate_fraction

    report = check_invariance(
        DegenerateGroups(), "reward.affine", payload, n=48, relation=INVARIANT, run=run
    )
    assert report.passed, report.render()


def test_e2_is_not_affine_invariant_inside_the_epsilon_band_and_that_is_a_finding():
    """A real property of the estimator rather than a test to loosen.

    `GroupStats.degenerate` is ``std <= std_epsilon``, an absolute threshold on a quantity that
    carries the reward's scale. A group whose spread sits inside a band of width `eps` around zero
    changes its verdict under a rescaling, so the fraction moves. The *quantity* as stated
    (groups with zero reward spread) is invariant; the estimator the trainer runs is not, and the
    gap is exactly the epsilon that makes `0 / (0 + eps)` finite.
    """
    eps = GRPO.std_epsilon or 1e-4
    payload = InvariancePayload(
        # One group whose standard deviation is 5e-5, half the 1e-4 epsilon, so it reads as
        # degenerate at a = 1 and stops reading as degenerate for any a above 2. The affine group
        # draws a ~ LogUniform(0.1, 10), so about a third of the draws cross it.
        scores=np.asarray([0.0, 1e-4, 0.0, 3.0], dtype=float),
        group_ids=np.repeat(np.arange(2), 2),
    )

    def run(_inst, p):
        gids = np.asarray(p.group_ids)
        scores = np.asarray(p.scores, dtype=float)
        window = [
            group(
                f"g{g}",
                [0.0] * int((gids == g).sum()),
                [float(v) for v in scores[gids == g]],
            )
            for g in sorted(set(gids.tolist()))
        ]
        return census_groups(window).degenerate_fraction

    report = check_invariance(
        DegenerateGroups(), "reward.affine", payload, n=48, relation=INVARIANT, run=run
    )
    assert not report.passed
    assert report.max_deviation == pytest.approx(0.5)
    assert "reading a level rather than a contrast" in report.interpretation
    worst = report.worst
    assert worst is not None and worst.params["a"] > 2.0
    # And the mechanism, stated as arithmetic: the group sits inside the band at a = 1 and outside
    # it once a lifts its spread past the estimator's own epsilon.
    assert float(np.std([0.0, 1e-4])) < eps < float(np.std([0.0, 1e-4])) * worst.params["a"]


def test_e1_declares_the_trivial_group_and_its_generated_test_passes_vacuously():
    """`none` in the registry is a declaration, not an omission (E11)."""
    assert QUANTITIES.get("estimator.spec").invariance == "trivial"
    report = check_invariance(
        RecordedEstimator(),
        "trivial",
        InvariancePayload(),
        relation=INVARIANT,
        run=lambda i, p: 1.0,
    )
    assert report.passed
    assert "no generators" in report.skipped


@pytest.mark.parametrize("instrument", [NoiseShare(), NoiseAttribution(), ClipAccounting()])
def test_the_units_group_routes_to_a_refusal_rather_than_a_value_relation(instrument):
    report = check_invariance(
        instrument, "units", InvariancePayload(), relation=INVARIANT, run=lambda i, p: 1.0
    )
    assert report.passed and "refusal" in report.skipped

    # The real assertion: a share of gradient power and a per-token logprob gap do not subtract.
    assert check_unit_refusal(
        difference,
        EstimatorQuantity("estimator.noise_share", 0.37),
        EstimatorQuantity("policy.train_infer_logprob_mismatch", 0.4),
    )
    out = difference(
        EstimatorQuantity("estimator.noise_share", 0.37),
        EstimatorQuantity("policy.train_infer_logprob_mismatch", 0.4),
    )
    assert isinstance(out, Refusal) and out.reason is RefusalReason.UNIT_MISMATCH
    assert "do not subtract" in out.remedy


def test_e6_is_invariant_under_a_faithful_retokenisation_of_its_sequence_total():
    """The default `tokenization` generator splits a token id and leaves the logprobs alone, which
    is not a re-tokenisation of anything. A real one divides the token's log-probability mass
    between the two pieces under *both* engines, and under that transform the sequence total is
    exactly invariant while the per-token mean is not.
    """
    from random import Random

    from reward_lens.core.invariance import GroupAction, InvarianceGroup

    base_s = (-1.0, -2.0, -3.0, -4.0)
    base_t = (-1.2, -1.7, -3.4, -3.6)

    def make(seed: int) -> GroupAction:
        def apply(p: InvariancePayload) -> InvariancePayload:
            rng = np.random.default_rng(seed)
            s = list(p.extra["sampling"])
            t = list(p.extra["train"])
            i = int(rng.integers(0, len(s)))
            f = float(rng.uniform(0.2, 0.8))
            s[i : i + 1] = [s[i] * f, s[i] * (1 - f)]
            t[i : i + 1] = [t[i] * f, t[i] * (1 - f)]
            return p.replace(extra={"sampling": tuple(s), "train": tuple(t)})

        return GroupAction(
            name=f"split one token's logprob mass, seed={seed}",
            apply=apply,
            params={"seed": float(seed)},
            sample=lambda rng: make(rng.randrange(2**31)),
        )

    faithful = InvarianceGroup(
        id="tokenization",
        generators=(make(0),),
        acts_on="tokens",
        admits=frozenset({"invariant"}),
        assertion="the sequence total is unchanged by an equivalent re-tokenisation",
    )

    def run(_inst, p):
        s, t = p.extra["sampling"], p.extra["train"]
        turn = Turn(
            index=0,
            role="assistant",
            token_ids=tuple(range(len(s))),
            logprobs_sampling=tuple(s),
            logprobs_train=tuple(t),
        )
        return measure_mismatch(make_trajectory(id="t", task_ref="k", turns=[turn])).per_sequence

    payload = InvariancePayload(extra={"sampling": base_s, "train": base_t})
    report = check_invariance(
        LogprobMismatch(), faithful, payload, n=48, relation=INVARIANT, run=run
    )
    assert report.passed, report.render()
    assert report.max_deviation < 1e-9

    # And the per-token mean is not, which is why the reading carries both.
    def per_token(_inst, p):
        s, t = p.extra["sampling"], p.extra["train"]
        turn = Turn(
            index=0,
            role="assistant",
            token_ids=tuple(range(len(s))),
            logprobs_sampling=tuple(s),
            logprobs_train=tuple(t),
        )
        return measure_mismatch(make_trajectory(id="t", task_ref="k", turns=[turn])).per_token

    moved = check_invariance(
        LogprobMismatch(), faithful, payload, n=16, relation=INVARIANT, run=per_token, seed=1
    )
    assert not moved.passed, "the per-token mean would be invariant, which it is not"
    assert Random  # the import is what makes the generator reproducible across runs


# ===========================================================================
# A refusal per instrument, with its reason and its remedy
# ===========================================================================


def test_every_instrument_refuses_with_a_remedy_when_it_is_handed_nothing():
    for cls in ESTIMATOR:
        out = cls().estimate(ctx())
        assert isinstance(out, Refusal), cls.__name__
        assert out.reason is RefusalReason.ACCESS_INSUFFICIENT, cls.__name__
        assert out.remedy.strip(), cls.__name__
        assert "pass `" in out.remedy or "run A2" in out.remedy, cls.__name__


def test_e2_refuses_the_all_fail_fraction_when_nobody_stated_what_failure_is():
    out = AllFailGroups(run_of([step(0, mixed_window())])).estimate(ctx())
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert "records the all-fail verdict without" in out.detail
    assert "FailureFloor(at=..., component=" in out.remedy
    assert out.is_bounded, "the degenerate count is still worth handing back"


def test_e3_refuses_when_verl_never_computed_the_proxies_and_names_the_config_flag():
    out = NoiseShare(run_of([step(0, mixed_window())])).estimate(ctx())
    assert isinstance(out, Refusal)
    assert "calculate_sum_pi_squared" in out.detail
    assert "actor.calculate_sum_pi_squared: true" in out.remedy


def test_e3_refuses_a_step_whose_proxy1_is_the_literal_zero_verl_emits_when_uncomputed():
    telemetry = {PROXY_KEYS["proxy1"]: 0.0, PROXY_KEYS["proxy2"]: 10.0, PROXY_KEYS["proxy3"]: 0.0}
    out = NoiseShare(run_of([step(0, mixed_window(), extra=telemetry)])).estimate(ctx())
    assert isinstance(out, Refusal)
    assert "exactly 0.0" in out.detail
    assert "actor/grad_norm" in out.remedy


def test_e3_refuses_the_attribution_when_the_replay_does_not_reproduce_the_advantages():
    """The attribution perturbs scores and replays them, so a spec that is not the transform gives
    shares of the wrong operator. The disagreement here is a declared divisor that is not the one
    the advantages were written under, which misses by 0.1797 at K = 4."""
    wrong_divisor = EstimatorSpec(
        family="grpo/dapo",
        group_centred=True,
        std_normalised=True,
        std_epsilon=1e-4,
        std_ddof=0,
        degenerate_policy="zero",
    )
    window = [
        group(
            "g0",
            [1.0, 2.0, 3.0, 4.0],
            [0.0] * 4,
            spec=wrong_divisor,
            advantages=z_advantages([1.0, 2.0, 3.0, 4.0], ddof=1),
        ),
        group(
            "g1",
            [0.0, 1.0, 2.0, 3.0],
            [0.0] * 4,
            spec=wrong_divisor,
            advantages=z_advantages([0.0, 1.0, 2.0, 3.0], ddof=1),
        ),
    ]
    telemetry = {PROXY_KEYS["proxy1"]: 6.0, PROXY_KEYS["proxy2"]: 10.0, PROXY_KEYS["proxy3"]: 4.0}
    out = NoiseAttribution(
        run_of([step(0, window, extra=telemetry)]),
        components=VarianceComponents(components={"rater": 0.1}),
        draws=8,
    ).estimate(ctx())
    assert isinstance(out, Refusal)
    assert "does not reproduce the recorded advantages" in out.detail
    assert "E1's `replay` field" in out.remedy
    assert out.is_bounded


def test_e3_refuses_the_attribution_with_no_variance_components_and_says_why_it_cannot_guess():
    telemetry = {PROXY_KEYS["proxy1"]: 6.0, PROXY_KEYS["proxy2"]: 10.0, PROXY_KEYS["proxy3"]: 4.0}
    out = NoiseAttribution(run_of([step(0, mixed_window(), extra=telemetry)])).estimate(ctx())
    assert isinstance(out, Refusal)
    assert "circular when grader errors are correlated" in out.detail
    assert "GRADER:REPLICATE" in out.remedy


def test_e4_reports_the_mechanism_as_absent_on_an_estimator_that_does_not_z_score():
    """`verifiers`' `score_group` is mean-centred with no standard-deviation division."""
    out = AmplifierSafety(mixed_window(spec=MEAN_CENTRED), floor=FLOOR).estimate(ctx())
    assert isinstance(out, Refusal)
    assert "does not divide by the group standard deviation" in out.detail
    assert "magnitude is the right diagnostic here" in out.remedy
    assert out.is_bounded
    bound = out.partial.value
    assert bound.mechanism_present is False
    assert bound.safety == {}
    assert bound.magnitude_ranking == ["aux", "task"]
    assert "mean-centres" in bound.render()


def test_e4_refuses_a_window_with_no_mixed_groups_rather_than_dividing_by_an_empty_set():
    only_allfail = [g for g in mixed_window() if str(g.id).startswith("af")]
    out = AmplifierSafety(only_allfail, floor=FLOOR).estimate(ctx())
    assert isinstance(out, Refusal)
    assert "the mixed side is empty" in out.detail
    assert "widen the window" in out.remedy
    assert out.statistics["n_mixed_groups"] == 0
    assert out.is_bounded and out.partial.value.magnitude_ranking


def test_e4_refuses_without_a_failure_floor_and_will_not_read_the_recorded_flag():
    out = AmplifierSafety(mixed_window()).estimate(ctx())
    assert isinstance(out, Refusal)
    assert "recorded flag cannot be read as authoritative" in out.detail
    assert "FailureFloor(at=0.0, component=" in out.remedy


def test_e5_refuses_the_effect_and_hands_back_the_clip_fraction_as_the_bound():
    r = run_of(
        [step(0, mixed_window(), grad_norm_clipped=1.0, extra={"clip_ratio/region_mean": 0.18})]
    )
    out = ClipAccounting(r).estimate(ctx())
    assert isinstance(out, Refusal)
    assert "none recorded both gradient norms" in out.detail
    assert "grad_norm_unclipped" in out.remedy
    assert out.is_bounded
    assert out.partial.value.clip_fraction == pytest.approx(0.18)


def test_e6_refuses_when_only_one_logprob_stream_was_recorded():
    turn = Turn(index=0, role="assistant", token_ids=(1, 2), logprobs_sampling=(-1.0, -2.0))
    out = LogprobMismatch(make_trajectory(id="t", task_ref="k", turns=[turn])).estimate(ctx())
    assert isinstance(out, Refusal)
    assert "1 of 1 turns have `logprobs_sampling` and 0 have `logprobs_train`" in out.detail
    assert "loss.py:23-24" in out.remedy


# ===========================================================================
# Declarations and the envelope
# ===========================================================================


def test_lint_instrument_is_empty_for_every_instrument_in_the_series():
    for cls in ESTIMATOR:
        assert lint_instrument(cls()) == [], cls.__name__


def test_the_series_needs_a_record_and_nothing_above_it():
    for cls in ESTIMATOR:
        assert Component.RECORD in cls.requires, cls.__name__
        levels = set(cls.requires.values())
        # E3's attribution is the one exception and it is a grader requirement, not a record one.
        assert levels <= {Access.RECORD, Access.REPLICATE}, cls.__name__
    assert NoiseAttribution.requires[Component.GRADER] is Access.REPLICATE


def test_e4_measures_its_envelope_rather_than_accepting_a_supplied_verdict():
    """The specification says "`GROUP_NONDEGENERATE` measured, not assumed"."""
    instrument = AmplifierSafety(mixed_window(), floor=FLOOR)
    measured = instrument.measure_nondegeneracy()
    assert measured is not None
    assert measured.condition is RegimeCondition.GROUP_NONDEGENERATE
    assert measured.holds is True
    assert measured.statistic == 0.0
    assert "measured over 4 groups" in measured.detail

    # A caller who asserts the opposite does not get to override the measurement, and the
    # disagreement is recorded on the reading.
    out = instrument.estimate(Context(regime_reading=RegimeReading.of(GROUP_NONDEGENERATE=False)))
    assert not isinstance(out, Refusal), out
    assert out.value.envelope_measured is True
    assert out.value.envelope_disagreed is True


def test_e4_refuses_when_its_own_measurement_says_the_groups_are_degenerate():
    flat = [group(f"g{i}", [0.0, 0.0], [1.0, 1.0]) for i in range(4)] + [
        group("live", [1.0, 0.0], [0.0, 1.0])
    ]
    out = AmplifierSafety(flat, floor=FLOOR).estimate(ctx())
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.ENVELOPE_VIOLATED
    assert "GROUP_NONDEGENERATE" in out.detail
    assert "restrict the window" in out.remedy


def test_e2_does_not_require_the_condition_it_measures():
    """The A4/B5 defect (E29), one series over: an envelope that makes an instrument
    refuse in exactly the regime it exists to report on."""
    for cls in (DegenerateGroups, AllFailGroups):
        assert cls.envelope.unconditional, cls.__name__
        assert not cls.envelope.requires, cls.__name__
        assert cls.envelope.justification


def test_every_reading_in_the_series_round_trips_through_the_store_codec():
    from reward_lens.core.evidence import ValueCodec

    codec = ValueCodec()
    telemetry = {PROXY_KEYS["proxy1"]: 6.0, PROXY_KEYS["proxy2"]: 10.0, PROXY_KEYS["proxy3"]: 4.0}
    r = run_of(
        [
            step(
                0,
                mixed_window(),
                grad_norm_clipped=1.0,
                grad_norm_unclipped=1.4,
                extra=telemetry,
            )
        ]
    )
    turn = Turn(
        index=0,
        role="assistant",
        token_ids=(1, 2),
        logprobs_sampling=(-1.0, -2.0),
        logprobs_train=(-1.1, -1.9),
    )
    readings = [
        read_estimator_spec(r),
        census_groups(r, floor=FLOOR),
        measure_noise_share(r),
        measure_amplifier_safety(r, floor=FLOOR),
        measure_clip_effect(r),
        measure_mismatch(make_trajectory(id="t", task_ref="k", turns=[turn])),
    ]
    for reading in readings:
        assert not isinstance(reading, Refusal), reading
        back = codec.decode(codec.encode(reading))
        assert type(back) is type(reading)
        assert back.render() == reading.render()
        # NaN is a legitimate baseline value (a comparator that could not be computed) and it does
        # not compare equal to itself, so the round trip is checked field by field with that in mind.
        for name, value in reading.baselines.items():
            other = back.baselines[name]
            assert (math.isnan(value) and math.isnan(other)) or value == other


def test_the_series_imports_no_torch():
    import subprocess
    import sys

    code = "import sys, reward_lens.measure.estimator; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False", out.stdout


def test_a2s_own_types_adapt_onto_the_interface_e3_declares():
    """Series A ships `ComponentSet` and `GaugeRR`; E3 asks for one number and adapts both.

    The two conversions must agree, because `gauge_rr`'s rule is "everything that is not the part is
    gauge" and that is exactly what `from_component_set` takes as error. Two routes to one number
    that disagreed would mean one of the two adapters was reading a different decomposition.
    """
    from reward_lens.measure.estimator import VarianceComponentsLike
    from reward_lens.stats.variance import gauge_rr, truncate_at_zero

    components = truncate_at_zero(
        {"p": 1.0, "r": 0.2, "pr": 0.05, "e": 0.03}, design="crossed part x rater"
    )
    from_set = VarianceComponents.from_component_set(components, part="p")
    assert isinstance(from_set, VarianceComponentsLike)
    assert from_set.error_variance == pytest.approx(0.28)
    assert from_set.universe_variance == pytest.approx(1.0)
    assert from_set.error_facets == ("r", "pr", "e")

    from_gauge = VarianceComponents.from_gauge_rr(gauge_rr(components, part="p", repeatability="e"))
    assert from_gauge.error_variance == pytest.approx(from_set.error_variance)
    assert from_gauge.universe_variance == pytest.approx(from_set.universe_variance)
    assert "gauge R&R over r, pr, e" == from_gauge.design

    # And the attribution runs on the adapted object rather than only on a hand-built one.
    telemetry = {PROXY_KEYS["proxy1"]: 6.0, PROXY_KEYS["proxy2"]: 10.0, PROXY_KEYS["proxy3"]: 4.0}
    r = run_of([step(0, mixed_window(), extra=telemetry)])
    reading = measure_noise_share(r, components=from_gauge, draws=32, seed=1)
    assert not isinstance(reading, Refusal), reading
    assert reading.attribution["grader"] > 0.0


def test_the_two_rankings_can_invert_which_is_the_whole_argument_for_the_ratio():
    """A large component that stops moving at mastery is safe; a small one that keeps moving is not.

    On the real GRPO record this package was built against the two rankings happen to agree, so the
    inversion is demonstrated here instead of being asserted about that record. `big` carries five
    times the magnitude of `small` and none of its variance survives into the all-fail groups;
    `small` carries all of its. Magnitude ranks them one way and amplifier safety the other, which is
    what the z-score does to a shaping term and why the reading renders both.
    """
    window = [
        # all-fail: `big` is flat at its own large value, `small` still spreads over {0, 0.2}
        group("af-0", [0.0] * 4, [10.0, 10.0, 10.0, 10.0], spec=GRPO),
        group("af-1", [0.0] * 4, [10.0, 10.0, 10.0, 10.0], spec=GRPO),
        group("mx-0", [1.0, 1.0, 0.0, 0.0], [0.0, 10.0, 0.0, 10.0], spec=GRPO),
        group("mx-1", [0.0, 0.0, 1.0, 1.0], [10.0, 0.0, 10.0, 0.0], spec=GRPO),
    ]
    # A third leaf is needed for the contrast, so it is added by hand rather than by `group`.
    from dataclasses import replace as dc_replace

    small_values = {
        "af-0": [0.0, 0.2, 0.0, 0.2],
        "af-1": [0.2, 0.0, 0.2, 0.0],
        "mx-0": [0.0, 0.2, 0.0, 0.2],
        "mx-1": [0.2, 0.0, 0.2, 0.0],
    }
    window = [
        dc_replace(
            g,
            trajectories=tuple(
                dc_replace(
                    t,
                    scores=dc_replace(
                        t.scores,
                        children=t.scores.children
                        + (Leaf(name="small", value=small_values[str(g.id)][i], grader_call=None),),
                        weights=t.scores.weights + (1.0,),
                    ),
                )
                for i, t in enumerate(g.trajectories)
            ),
        )
        for g in window
    ]
    reading = measure_amplifier_safety(window, floor=FLOOR)
    assert not isinstance(reading, Refusal), reading

    assert reading.baselines["magnitude/aux"] > 5.0 * reading.baselines["magnitude/small"]
    assert reading.safety["aux"] == 0.0
    assert reading.safety["small"] >= 1.0
    assert reading.magnitude_ranking[0] == "aux"
    assert reading.ranking[0] == "small"
    assert not reading.rankings_agree
    assert "They disagree, which is the point" in reading.render()

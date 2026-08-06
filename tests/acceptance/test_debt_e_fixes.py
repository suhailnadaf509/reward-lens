"""Regressions for the five series-E defects the statistical review found, with their numbers.

Each test here carries the wrong number and the right one, because a fix that lands without the
number it moved is indistinguishable from a fix that did nothing. Where the defect is in a module
this package does not own, the test measures the divergence against the framework's own arithmetic
and says what the one-line change is.

The five, in the order the review ranked them:

1. E3's clip correction. It omitted veRL's ``1/(N-1)``, so the clip term came back ``N-1`` times too
   large, `corrected` floored at zero, and the attribution reported grader and sampling variance as
   exactly 0.0 when both had been measured and differed by a factor of 1.36. That fires E3's own
   kill condition, which is "if the attribution never separates grader from sampling". The
   arithmetic error is real and the term it was in is gone, for the reason in item 2.
2. `actor/grad_norm` is the norm **before** clipping and three modules said it was after. The claim
   that rested on it, that the clip inflates veRL's reported noise share, does not hold, and E5's
   headline moved to what the clip really does: shrink the applied update.
3. E4's denominator pooled all-pass groups with mixed ones. On a five-group hand window that reads
   5.9925 against a correct 4.0.
4. `record.scores.replay_advantages` divides by ``std(ddof=0)`` where veRL and TRL both apply
   Bessel's correction. 15.47% at K = 4 against a `REPLAY_TOL` of 1e-4.
5. `Turn.logprob_gap` zips the two logprob streams, so a length difference truncates in silence.
   A length difference is the signature of the two engines tokenising differently, which is exactly
   what E6 exists to detect.
"""

from __future__ import annotations

import inspect
import math
import os
import pathlib
import sys
from dataclasses import replace

import numpy as np
import pytest

from reward_lens.core.quantity import load_quantities
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.measure.estimator import (
    FLOAT32_EPS,
    MECHANISMS,
    PROXY_KEYS,
    REPLAY_TOL,
    FailureFloor,
    check_stream_lengths,
    float32_floor_at,
    group_phase,
    measure_amplifier_safety,
    measure_clip_effect,
    measure_mismatch,
    measure_noise_share,
    partition_by_floor,
    pooled_within_variance,
    read_estimator_spec,
    register_all,
)
from reward_lens.record.schema import EstimatorSpec, make_trajectory
from reward_lens.record.scores import replay_advantages
from reward_lens.record.turns import Turn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from test_measure_estimator import (  # noqa: E402
    GRPO,
    VarianceComponents,
    group,
    mixed_window,
    run_of,
    step,
)

load_quantities()
register_all()

#: veRL's source tree, which is not vendored here. There is no default: point
#: ``REWARD_LENS_VERL_SOURCE`` at the inner ``verl/`` package directory of a veRL checkout,
#: or the two tests that read that source skip.
_VERL_ENV = os.environ.get("REWARD_LENS_VERL_SOURCE")
VERL_SOURCE = pathlib.Path(_VERL_ENV) if _VERL_ENV else None


# ===========================================================================
# 1. The clip correction, and the kill condition it fired
# ===========================================================================

#: A veRL-shaped batch. veRL's own `proxy3 = (1/(N-1)) * (proxy2 - proxy1)` at
#: `metric_utils.py:776`, so the telemetry below is internally consistent at this N.
N_BATCH = 512
PROXY1 = 6.0
PROXY2 = 10.0
PROXY3 = (PROXY2 - PROXY1) / (N_BATCH - 1)


def _verl_window():
    """One step carrying both gradient norms and a consistent set of variance proxies.

    Both norms are the condition the old defect needed: with only one of them `measure_clip_effect`
    refuses and the clip term was zero by accident rather than by derivation, which is why the
    existing unit tests never saw it.
    """
    return run_of(
        [
            step(
                0,
                mixed_window(),
                extra={
                    PROXY_KEYS["proxy1"]: PROXY1,
                    PROXY_KEYS["proxy2"]: PROXY2,
                    PROXY_KEYS["proxy3"]: PROXY3,
                },
                grad_norm_clipped=1.0,
                grad_norm_unclipped=1.5,
            )
        ]
    )


def test_verls_proxy3_identity_is_what_the_review_says_it_is():
    """The 1/(N-1) the correction omitted, read off veRL's source rather than off a summary."""
    if VERL_SOURCE is None:
        pytest.skip("no veRL source tree; set REWARD_LENS_VERL_SOURCE")
    src = VERL_SOURCE / "trainer" / "ppo" / "metric_utils.py"
    if not src.is_file():
        pytest.skip("the veRL source tree has no trainer/ppo/metric_utils.py")
    text = src.read_text()
    assert (
        "proxy3_pure_noise = (1.0 / (batch_size - 1)) * (proxy2_total_power - proxy1_signal_strength)"
        in text
    )
    assert "batch_size = advantages_scalar.shape[0]" in text
    # And the floor that makes proxy3 censored from below.
    assert "proxy3_pure_noise = max(" in text

    # The identity, at the batch size the fixture uses.
    assert PROXY3 == pytest.approx((PROXY2 - PROXY1) / (N_BATCH - 1))
    assert PROXY3 == pytest.approx(0.00782779, rel=1e-5)


def test_the_attribution_separates_grader_from_sampling_where_it_used_to_report_two_zeros():
    """E3's kill condition, on the window that fired it.

    Before: the clip term was ``proxy1 * u / proxy2`` with no ``1/(N-1)``, which at N = 512 and
    ``u = (1.5/1.0)**2 - 1 = 1.25`` is 0.75, against a noise share of 0.000783. That is 958 times
    the share it was a component of, and exactly ``N - 1 = 511`` times the correct value of
    0.00146771. `corrected = max(share - 0.75, 0)` floored to 0.0, so grader and sampling both came
    back as exactly 0.0 while both had been measured and differed by a third.

    The two variances behind them are 0.0605 and 0.0795 here. They read 0.0806 and 0.1098 when the
    replay divided by the population standard deviation; that divisor is now a recorded field and
    this fixture declares TRL's, which is Bessel's. The ratio between them, which is what the kill
    condition is about, is 1.32 either way.
    """
    components = VarianceComponents(
        components={"item": 1.0, "rater": 0.1, "residual": 0.02},
        design="two-facet, hand-built for this regression",
    )
    reading = measure_noise_share(_verl_window(), components=components, draws=64, seed=7)
    assert not isinstance(reading, Refusal), reading

    # The wrong numbers, kept so the regression is legible.
    u = (1.5 / 1.0) ** 2 - 1.0
    wrong_clip_points = PROXY1 * u / PROXY2
    right_clip_points = wrong_clip_points / (N_BATCH - 1)
    assert wrong_clip_points == pytest.approx(0.75)
    assert right_clip_points == pytest.approx(0.00146771, rel=1e-5)
    assert wrong_clip_points / right_clip_points == pytest.approx(N_BATCH - 1)
    assert wrong_clip_points / reading.noise_share == pytest.approx(958.125, rel=1e-4)

    # The right ones. Both terms are non-zero, they are different, and the two variances behind
    # them were measured and differ.
    assert reading.noise_share == pytest.approx(0.000782779, rel=1e-5)
    assert reading.attribution["grader"] > 0.0
    assert reading.attribution["sampling"] > 0.0
    assert reading.attribution["grader"] == pytest.approx(5.9169e-05, rel=1e-3)
    assert reading.attribution["sampling"] == pytest.approx(7.7819e-05, rel=1e-3)
    assert reading.grader_variance == pytest.approx(0.060458, rel=1e-4)
    assert reading.sampling_variance == pytest.approx(0.079515, rel=1e-4)
    assert reading.sampling_variance / reading.grader_variance == pytest.approx(1.3152, rel=1e-3)

    # The partition still sums to the share it apportions.
    assert sum(reading.attribution.values()) == pytest.approx(reading.noise_share, rel=1e-12)
    assert set(reading.attribution) == set(MECHANISMS) == {"grader", "sampling", "unattributed"}


def test_the_clip_is_reported_beside_the_shares_and_never_subtracted_from_them():
    """It is carried, because a reader will look for it, and it does not move the attribution."""
    components = VarianceComponents(components={"rater": 0.1})
    with_norms = measure_noise_share(_verl_window(), components=components, draws=32, seed=3)
    assert not isinstance(with_norms, Refusal), with_norms
    assert with_norms.clip_shrinkage == pytest.approx(1.0 / 1.5)
    assert "not subtracted from the noise share" in with_norms.clip_note
    assert "clip" not in with_norms.attribution

    # Strip the two norms and the shares do not move: the clip is not in the arithmetic.
    without = run_of(
        [
            step(
                0,
                mixed_window(),
                extra={
                    PROXY_KEYS["proxy1"]: PROXY1,
                    PROXY_KEYS["proxy2"]: PROXY2,
                    PROXY_KEYS["proxy3"]: PROXY3,
                },
            )
        ]
    )
    bare = measure_noise_share(without, components=components, draws=32, seed=3)
    assert not isinstance(bare, Refusal), bare
    assert math.isnan(bare.clip_shrinkage)
    assert "not computable" in bare.clip_note
    for name in MECHANISMS:
        assert bare.attribution[name] == pytest.approx(with_norms.attribution[name], rel=1e-12)


# ===========================================================================
# 2. `actor/grad_norm` is the pre-clipping norm
# ===========================================================================


def test_torch_clip_grad_norm_returns_the_norm_before_clipping():
    """The primary source, run rather than quoted.

    A single parameter whose gradient has norm 10 is clipped at 1. If the return were the
    post-clipping norm it would be 1.0; it is 10.0, and the gradient afterwards has norm 1.0.
    """
    torch = pytest.importorskip("torch")

    p = torch.nn.Parameter(torch.zeros(4))
    p.grad = torch.tensor([10.0, 0.0, 0.0, 0.0])
    returned = torch.nn.utils.clip_grad_norm_([p], max_norm=1.0)

    assert float(returned) == pytest.approx(10.0)
    assert float(p.grad.norm()) == pytest.approx(1.0)
    assert float(returned) != pytest.approx(float(p.grad.norm()))

    # And the source says so: the total norm is taken, then the gradients are scaled with it.
    src = inspect.getsource(torch.nn.utils.clip_grad_norm_)
    assert "total_norm = _get_total_norm(" in src
    assert src.index("total_norm = _get_total_norm(") < src.index("_clip_grads_with_norm_(")
    assert src.rstrip().endswith("return total_norm")


def test_transformers_and_verl_both_document_the_logged_norm_as_pre_clipping():
    """The other three sources. Read off installed or fetched source, not off documentation."""
    transformers = pytest.importorskip("transformers")
    trainer_src = pathlib.Path(inspect.getfile(transformers.Trainer)).read_text()
    assert "Returns the pre-clip gradient norm" in trainer_src
    assert 'logs["grad_norm"] =' in trainer_src

    root = VERL_SOURCE
    if root is None or not root.is_dir():
        pytest.skip("no veRL source tree; set REWARD_LENS_VERL_SOURCE")
    fsdp = (root / "workers/engine/fsdp/transformer_impl.py").read_text()
    assert "grad_norm (float): Norm of gradients before clipping." in fsdp
    megatron = (root / "workers/engine/megatron/transformer_impl.py").read_text()
    assert "The norm of the gradients before clipping or update." in megatron
    # And the value veRL returns is exactly what `clip_grad_norm_` handed it.
    assert "grad_norm = torch.nn.utils.clip_grad_norm_(" in fsdp
    assert "return grad_norm.item()" in fsdp


def test_no_module_in_this_package_still_says_the_logged_norm_is_post_clipping():
    """Three modules said it and two of them are here. The third is `tap/adapters/trl.py:612`."""
    here = pathlib.Path(__file__).resolve().parents[2] / "src/reward_lens/measure/estimator"
    offenders = []
    for path in sorted(here.glob("*.py")):
        text = path.read_text()
        for i, line in enumerate(text.splitlines(), 1):
            lowered = line.lower()
            if "post-clipping" in lowered or "post clipping" in lowered:
                # The two occurrences that survive are the ones recording that the claim was wrong.
                if "**post**-clipping" in line or "post-clipping norm" in line:
                    continue
                offenders.append(f"{path.name}:{i}: {line.strip()}")
    assert offenders == [], offenders


def test_e5_reports_the_shrinkage_on_the_update_rather_than_an_understatement_of_proxy1():
    """The headline claim, moved. `clipped / unclipped` is the multiplier on the applied step."""
    r = run_of(
        [
            step(0, mixed_window(), grad_norm_clipped=1.0, grad_norm_unclipped=1.5),
            step(1, mixed_window(), grad_norm_clipped=2.0, grad_norm_unclipped=2.5),
        ]
    )
    reading = measure_clip_effect(r)
    assert not isinstance(reading, Refusal), reading
    assert reading.shrinkage == pytest.approx((1.0 / 1.5 + 2.0 / 2.5) / 2)
    assert reading.n_steps_shrunk == 2
    assert "proportional to the gradient" in reading.says
    assert not hasattr(reading, "proxy1_understatement")
    assert not hasattr(reading, "proxy3_inflation_at_n")

    # And the Jensen gap, which squaring a mean effect would have swallowed.
    assert reading.ratio_squared == pytest.approx(1.90625)
    assert reading.ratio_squared_jensen_gap == pytest.approx(0.015625)
    assert reading.ratio_squared_jensen_gap == pytest.approx(float(np.var([1.5, 1.25])))


# ===========================================================================
# 3. The amplifier-safety denominator
# ===========================================================================

#: Two all-fail groups, two genuinely mixed, one all-pass whose auxiliary has run out of scale.
#: The auxiliary's within-group variance is 4/3 in the all-fail groups and 1/3 in the mixed ones,
#: so the correct ratio is exactly 4.
SATURATING_FLOOR = FailureFloor(at=0.0, component="task", saturates_at=1.0)


def _five_group_window():
    return [
        group("af-0", [0.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 2.0], spec=GRPO),
        group("af-1", [0.0, 0.0, 0.0, 0.0], [2.0, 0.0, 2.0, 0.0], spec=GRPO),
        group("mx-0", [1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0], spec=GRPO),
        group("mx-1", [0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 1.0, 0.0], spec=GRPO),
        group("ap-0", [1.0, 1.0, 1.0, 1.0], [0.5, 0.55, 0.5, 0.55], spec=GRPO),
    ]


def test_the_partition_is_three_ways_and_all_pass_is_not_the_complement_of_all_fail():
    window = _five_group_window()
    assert partition_by_floor(window, SATURATING_FLOOR) == [
        "all_fail",
        "all_fail",
        "mixed",
        "mixed",
        "all_pass",
    ]
    # And it does not need `saturates_at`: the failure floor alone settles which side a group is on.
    bare = FailureFloor(at=0.0, component="task")
    assert partition_by_floor(window, bare) == partition_by_floor(window, SATURATING_FLOOR)
    assert group_phase(window[4], bare) == "all_pass"


def test_the_denominator_is_the_mixed_groups_and_not_the_not_all_fail_ones():
    """Before: 5.9925, because the all-pass group's suppressed variance sat in the denominator.

    After: exactly 4.0, which is the analytic value on this window: 4/3 over 1/3.

    The inflation on this window is 1.4981x, not the 3x the review reported, and the difference is
    the mix rather than a disagreement about the mechanism. Pooling is degrees-of-freedom weighted,
    so the denominator moves from ``v_mixed`` to
    ``(dof_mixed * v_mixed + dof_allpass * v_allpass) / (dof_mixed + dof_allpass)``, and with the
    all-pass variance near zero the inflation is bounded above by
    ``1 + dof_allpass/dof_mixed``. Here that is 1 + 3/6 = 1.5 and the measurement is 1.4981.
    """
    window = _five_group_window()
    out = measure_amplifier_safety(window, floor=SATURATING_FLOOR)
    assert not isinstance(out, Refusal), out

    assert out.safety["aux"] == pytest.approx(4.0)
    assert out.detail["aux"]["var_allfail"] == pytest.approx(4.0 / 3.0)
    assert out.detail["aux"]["var_mixed"] == pytest.approx(1.0 / 3.0)
    assert out.detail["aux"]["n_mixed_groups"] == 2.0
    assert out.detail["aux"]["n_allpass_groups"] == 1.0
    assert (out.n_allfail_groups, out.n_mixed_groups, out.n_allpass_groups) == (2, 2, 1)
    assert out.n_allfail_groups + out.n_mixed_groups + out.n_allpass_groups == out.n_groups
    assert "all-pass held out of the denominator" in out.render()

    # The wrong denominator, recomputed here so the before number is in the file.
    pooled_wrong, _, _ = pooled_within_variance(
        [[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0], [0.5, 0.55, 0.5, 0.55]]
    )
    wrong_ratio = (4.0 / 3.0) / pooled_wrong
    assert pooled_wrong == pytest.approx(0.2225)
    assert wrong_ratio == pytest.approx(5.99251, rel=1e-5)
    assert wrong_ratio / out.safety["aux"] == pytest.approx(1.49813, rel=1e-5)
    # The bound the mechanism gives: dof 3 joining dof 6 can inflate by at most 1 + 3/6.
    assert wrong_ratio / out.safety["aux"] < 1.0 + 3.0 / 6.0


def test_the_eps_departure_from_scale_invariance_is_a_function_of_eps_over_sigma():
    """The docstring number, computed rather than quoted.

    `amplifier.py` used to say the z-score's departure from scale invariance is "about 1e-7 at
    eps = 1e-8", which pins an unstated group standard deviation. Under `r -> a*r + b` the advantage
    moves by ``eps * (a - 1) / (a * std + eps)``, so it is a function of ``eps/std`` and the quoted
    figure is the ``std = 0.05`` row at ``a = 2``.
    """

    def departure(a: float, eps: float, std: float) -> float:
        original = 1.0 / (std + eps)
        rescaled = a / (a * std + eps)
        return abs(rescaled - original) / original

    a, eps = 2.0, 1e-8
    table = {0.005: 1.0e-6, 0.05: 1.0e-7, 0.5: 1.0e-8, 5.0: 1.0e-9}
    for std, expected in table.items():
        assert departure(a, eps, std) == pytest.approx(expected, rel=2e-3), std
        assert departure(a, eps, std) == pytest.approx(eps * (a - 1) / (a * std + eps), rel=1e-12)

    # Three orders of magnitude across a range a real reward group covers, which is why the number
    # cannot be quoted without the standard deviation it was taken at.
    assert departure(a, eps, 0.005) / departure(a, eps, 5.0) == pytest.approx(1000.0, rel=1e-2)
    # And it is `eps/std` that governs it: hold the ratio and the departure does not move.
    assert departure(a, 1e-8, 0.05) == pytest.approx(departure(a, 1e-6, 5.0), rel=1e-6)


def test_a_window_of_only_all_fail_and_all_pass_refuses_instead_of_dividing_by_the_all_pass_ones():
    """The case the old denominator turned into a number: no mixed groups at all."""
    window = [
        group("af-0", [0.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 2.0], spec=GRPO),
        group("ap-0", [1.0, 1.0, 1.0, 1.0], [0.5, 0.55, 0.5, 0.55], spec=GRPO),
    ]
    out = measure_amplifier_safety(window, floor=SATURATING_FLOOR)
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert "the mixed side is empty" in out.detail
    assert "1 of the 2 groups are all-pass" in out.detail
    assert out.statistics["n_allpass_groups"] == 1
    assert out.is_bounded


# ===========================================================================
# 4. The replay divisor
# ===========================================================================

#: The divisor is now a recorded field, so the two conventions are two specs rather than one
#: undeclared default. `std_ddof=1` is what TRL and veRL both do.
Z_SPEC = EstimatorSpec(
    family="grpo",
    group_centred=True,
    std_normalised=True,
    std_epsilon=1e-6,
    std_ddof=1,
    degenerate_policy="keep",
    aggregation="sequence",
)
POPULATION_SPEC = replace(Z_SPEC, std_ddof=0)
UNDECLARED_SPEC = replace(Z_SPEC, std_ddof=None)
SCORES = [1.0, 2.0, 3.0, 6.0]


def _framework_advantages(scores, eps):
    """What veRL and TRL both compute, from their own arithmetic rather than from a description.

    veRL: ``(r - mean) / (torch.std(r) + eps)``, and `torch.std` defaults to ``correction=1``
    (`verl/trainer/ppo/core_algos.py:321`). TRL: the same with `nanstd`, which multiplies the
    variance by ``count/(count - 1)`` at `trl/trainer/utils.py:877-879`.
    """
    arr = np.asarray(scores, dtype=float)
    return list((arr - arr.mean()) / (arr.std(ddof=1) + eps))


def test_verl_and_trl_both_apply_bessels_correction_and_the_gap_is_a_known_size():
    torch = pytest.importorskip("torch")

    t = torch.tensor(SCORES)
    verl = ((t - t.mean()) / (torch.std(t) + 1e-6)).tolist()
    mean = float(np.mean(SCORES))
    var = float(np.mean((np.asarray(SCORES) - mean) ** 2)) * (len(SCORES) / (len(SCORES) - 1))
    trl = list((np.asarray(SCORES) - mean) / (math.sqrt(var) + 1e-6))

    # `rel=1e-6` because veRL's tensor is float32 and TRL's arithmetic here is float64. The two
    # agree to 6.5e-08 absolute, which is float32 rounding and not a difference of convention.
    assert verl == pytest.approx(trl, rel=1e-6)
    assert verl == pytest.approx(_framework_advantages(SCORES, 1e-6), rel=1e-6)
    assert max(abs(a - b) for a, b in zip(verl, trl)) < 1e-7

    # The size of the divergence, which is the Bessel factor and nothing else.
    for k, expected in (
        (2, 0.414214),
        (4, 0.154701),
        (8, 0.069045),
        (16, 0.032796),
        (64, 0.007905),
    ):
        assert math.sqrt(k / (k - 1)) - 1 == pytest.approx(expected, rel=1e-4)
    assert math.sqrt(4 / 3) - 1 == pytest.approx(0.15470054, rel=1e-6)


def test_the_replay_uses_the_divisor_both_frameworks_use():
    """The measurement, on the current code, and the number it moved from.

    `replay_advantages` called ``present.std()``, whose numpy default is ``ddof=0``, and the
    advantages came back 15.47% large at K = 4. In absolute terms that is 0.2148 against a
    `REPLAY_TOL` of 1e-4: three orders of magnitude past the tolerance, so a genuine veRL record
    would have refused for a reason internal to the replay rather than for anything about the
    record. It is now `EstimatorSpec.std_ddof`, declared per record, and it reproduces both
    frameworks exactly.
    """
    got = replay_advantages(SCORES, Z_SPEC, where="k4")
    assert not isinstance(got, Refusal), got
    framework = _framework_advantages(SCORES, 1e-6)
    assert list(got) == pytest.approx(framework, rel=1e-12)

    # The old behaviour, still reachable by declaring the population form, and the size of the gap.
    population = replay_advantages(SCORES, POPULATION_SPEC, where="k4")
    assert not isinstance(population, Refusal), population
    arr = np.asarray(SCORES)
    assert list(population) == pytest.approx(
        list((arr - arr.mean()) / (arr.std(ddof=0) + 1e-6)), rel=1e-12
    )
    gap = max(abs(a - b) for a, b in zip(population, framework))
    assert gap == pytest.approx(0.214837, rel=1e-4)
    assert gap > 1000 * REPLAY_TOL
    assert max(abs(v) for v in population) / max(abs(v) for v in framework) == pytest.approx(
        math.sqrt(4 / 3), rel=1e-4
    )


def test_the_declared_divisor_survives_a_round_trip_through_the_record():
    """Measured on the 24-step GRPO record: the tap writes 1 and the reader gives back None.

    In-process, `tap.finish()` hands over the live `EstimatorSpec` and everything works. The loss
    only appears once the record is written and read, which is the path every analysis of a
    finished run takes, so it is the path that matters.
    """
    spec = replace(Z_SPEC, std_ddof=1)
    assert spec.std_ddof == 1
    assert "std_ddof" in spec.__canonical__()
    assert EstimatorSpec.from_canonical(spec.__canonical__()).std_ddof == 1


def test_an_undeclared_divisor_refuses_rather_than_picking_one():
    """A near-certain default is still an assumption about a denominator, so it is refused."""
    out = replay_advantages(SCORES, UNDECLARED_SPEC, where="k4")
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.RECORD_INCOMPLETE
    assert out.statistics["std_ddof"] is None
    assert "15.5% at K=4" in out.detail
    assert "std_ddof" in out.remedy

    # And E1 names it as undeclared rather than as ambiguous, because the trainer divided by
    # something: `None` here cannot mean "this trainer does not do it".
    reading = read_estimator_spec(
        [group("g", [0.0, 1.0], [0.0, 0.0], spec=UNDECLARED_SPEC)], replay=False
    )
    assert not isinstance(reading, Refusal), reading
    assert "std_ddof" in reading.undeclared
    assert "std_ddof" not in reading.ambiguous


# ===========================================================================
# 5. Unequal logprob streams
# ===========================================================================

#: Two engines that tokenised the same text differently: sampling emitted six tokens, training
#: scored five. The tail the zip discards carries most of the disagreement, which is the usual
#: shape: the divergence starts somewhere and everything after it is misaligned.
SAMPLING = (-0.10, -0.20, -0.30, -0.40, -0.50, -0.60)
TRAIN = (-0.11, -0.22, -0.33, -0.44, -1.55)


def test_the_zip_truncates_and_the_number_it_produces_is_wrong_by_a_measurable_amount():
    """`Turn.logprob_gap` lives in `record/turns.py`, which this package does not own.

    The truncation is measured here rather than fixed here. The sequence total is the invariant
    object E6 reports, and over the zipped prefix it reads 1.15 nats against a true 0.55: 2.09x,
    and in the direction that makes the engines look worse. The direction is not signed, because
    the discarded tail can carry the gap either way.
    """
    turn = Turn(index=0, role="assistant", logprobs_sampling=SAMPLING, logprobs_train=TRAIN)
    gap = turn.logprob_gap()
    assert gap is not None
    assert len(gap) == 5 < len(SAMPLING)

    truncated_total = abs(sum(gap))
    true_total = abs(sum(TRAIN) - sum(SAMPLING))
    assert truncated_total == pytest.approx(1.15)
    assert true_total == pytest.approx(0.55)
    assert truncated_total / true_total == pytest.approx(2.0909, rel=1e-3)


def test_e6_refuses_on_unequal_streams_and_names_the_counts():
    """The signal the instrument exists to detect, reported as a refusal rather than averaged."""
    turn = Turn(index=0, role="assistant", logprobs_sampling=SAMPLING, logprobs_train=TRAIN)
    out = measure_mismatch(make_trajectory(id="t", task_ref="k", turns=[turn]))

    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.UNIT_MISMATCH
    assert "did not tokenise the same text the same way" in out.detail
    assert "worst 5 training against 6 sampling" in out.detail
    assert "tokenizer revisions" in out.remedy
    assert out.statistics["n_unequal"] == 1
    assert out.statistics["n_comparable"] == 1
    assert out.is_bounded

    check = check_stream_lengths([turn])
    assert (check.n_comparable, check.n_unequal) == (1, 1)
    assert (check.worst_train, check.worst_sampling) == (5, 6)
    assert not check.agrees


def test_the_bound_on_that_refusal_is_the_mismatch_over_the_turns_that_did_line_up():
    """A refusal carrying a measurement of the easy cases is more use than no number at all."""
    good = Turn(
        index=0,
        role="assistant",
        token_ids=(1, 2, 3, 4),
        logprobs_sampling=(-1.0, -2.0, -3.0, -4.0),
        logprobs_train=(-1.5, -1.5, -3.5, -3.5),
    )
    bad = Turn(index=1, role="assistant", logprobs_sampling=SAMPLING, logprobs_train=TRAIN)
    out = measure_mismatch(make_trajectory(id="t", task_ref="k", turns=[good, bad]))

    assert isinstance(out, Refusal)
    assert out.statistics["n_unequal"] == 1
    bound = out.partial.value
    # The good turn's gaps are -0.5, +0.5, -0.5, +0.5.
    assert bound.per_token == pytest.approx(0.5)
    assert bound.n_tokens == 4
    assert bound.n_turns_equal_length == 1
    assert bound.per_sequence == pytest.approx(0.0, abs=1e-12)
    assert "not comparable at all" in bound.says


def test_equal_streams_still_read_normally():
    """The guard fires on a length difference and on nothing else."""
    turn = Turn(
        index=0,
        role="assistant",
        token_ids=(1, 2, 3, 4),
        logprobs_sampling=(-1.0, -2.0, -3.0, -4.0),
        logprobs_train=(-1.5, -1.5, -3.5, -3.5),
    )
    out = measure_mismatch(make_trajectory(id="t", task_ref="k", turns=[turn]))
    assert not isinstance(out, Refusal), out
    assert out.per_token == pytest.approx(0.5)
    assert out.n_turns_equal_length == 1
    assert check_stream_lengths([turn]).agrees


def test_turn_post_init_does_not_catch_this_because_it_needs_token_ids():
    """Why the guard belongs in the instrument as well as in the schema.

    `Turn.__post_init__` compares each per-token array against `len(token_ids)` and returns early
    when `token_ids` is None. A tap that writes two logprob streams and no token ids, which is what
    reading logprobs off two engines produces, constructs cleanly with mismatched lengths.
    """
    ok = Turn(index=0, role="assistant", logprobs_sampling=SAMPLING, logprobs_train=TRAIN)
    assert len(ok.logprobs_sampling) != len(ok.logprobs_train)

    with pytest.raises(ValueError, match="against 6 token ids"):
        Turn(
            index=0,
            role="assistant",
            token_ids=(1, 2, 3, 4, 5, 6),
            logprobs_sampling=SAMPLING,
            logprobs_train=TRAIN,
        )


# ===========================================================================
# The numerics floor
# ===========================================================================


def test_the_float32_floor_is_taken_at_the_magnitude_the_logprobs_have():
    """`numpy.finfo(float32).eps` is the spacing at 1.0 and token logprobs are not near 1.0."""
    assert FLOAT32_EPS == pytest.approx(1.1920929e-07)
    assert float32_floor_at(1.0) == pytest.approx(FLOAT32_EPS)
    for magnitude, factor in ((2.0, 2), (4.0, 4), (8.0, 8), (10.0, 8), (16.0, 16), (20.0, 16)):
        assert float32_floor_at(magnitude) == pytest.approx(factor * FLOAT32_EPS), magnitude
    # Below 1.0 it is tighter, which is the same rule and not a special case.
    assert float32_floor_at(0.5) == pytest.approx(FLOAT32_EPS / 2)
    # Degenerate inputs fall back rather than returning the smallest subnormal.
    assert float32_floor_at(0.0) == FLOAT32_EPS
    assert float32_floor_at(math.nan) == FLOAT32_EPS


def test_the_reading_takes_its_floor_from_the_record_and_names_the_magnitude():
    """A stream at |logprob| ~ 16 gets a floor 16x the constant, and says which one it used."""
    turn = Turn(
        index=0,
        role="assistant",
        token_ids=(1, 2, 3, 4),
        logprobs_sampling=(-16.0, -16.0, -16.0, -16.0),
        logprobs_train=(-16.0 + 1e-6, -16.0 - 1e-6, -16.0 + 1e-6, -16.0 - 1e-6),
    )
    out = measure_mismatch(make_trajectory(id="t", task_ref="k", turns=[turn]))
    assert not isinstance(out, Refusal), out
    assert out.typical_magnitude == pytest.approx(16.0, abs=1e-5)
    assert out.floor == pytest.approx(16 * FLOAT32_EPS)
    assert out.floor > FLOAT32_EPS
    assert "median absolute logprob" in out.floor_source

    # The registered baseline is still the named constant, which is what that baseline is.
    assert out.baselines["baseline.float32_epsilon"] == pytest.approx(FLOAT32_EPS)

    # And the verdict this changes: a 1e-6 disagreement is above the 1.0 floor and below the
    # 16.0 one, so the old floor would have called it a real disagreement.
    assert out.per_token == pytest.approx(1e-6, rel=1e-3)
    assert out.per_token > FLOAT32_EPS
    assert out.below_floor is True

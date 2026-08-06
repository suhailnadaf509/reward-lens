"""A1's headline number stopped charging measurement noise twice, and this is the file that pins it.

`grader.effective_group_size` was `kish x reliability`. It is now `K x reliability`. The Kish count
is computed on observed scores that already contain grader noise and the reliability factor then
discounts for the same noise, so the product charged it twice.

Four things are asserted here.

1. **The two-point truth returns the right number now.** A two-point score distribution has a Kish
   shape factor of exactly 1.0 by construction, so noise is the only thing that can move the
   reading. At a reliability of 0.5 a group of sixteen carries eight independent observations. The
   old rule reported 5.59.
2. **The shape factor is reported beside the reading and is not multiplied into it.** It keeps its
   own bootstrap interval, because a statistic without one is not a measurement.
3. **Rung 0's declarations are unchanged and rung 0's number is not.** It still sets the
   reliability to 1.0, still names every error term as invisible, and is still biased upward. It
   returns K where it used to return the Kish count, which on Gaussian rewards is about `0.64K`.
4. **The before and the after are both recorded on the same real subjects.** The eleven open reward
   models series A ran on, fully crossed, 1,763 groups of four.

The kill condition, the intervals, the transfers and the rest of series A stay in
`test_w3_2a_metrology.py`; this file is about the correction itself.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest

from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.measure.metrology.gstudy import (
    EffectiveGroupSize,
    GroupScores,
    ReplicationDesign,
    effective_group_size,
    jackknife_reliability,
)
from reward_lens.record.convert.store import CampaignStore
from reward_lens.stats.variance import group_effective_size

#: The campaign evidence store, which is not in this repository. There is no default: point
#: ``REWARD_LENS_CAMPAIGN_STORE`` at the store directory or the tests that need it skip.
_CAMPAIGN_ENV = os.environ.get("REWARD_LENS_CAMPAIGN_STORE")
CAMPAIGN_STORE = Path(_CAMPAIGN_ENV) if _CAMPAIGN_ENV else None

NOT_A_GRADER = {"hackfore-flagged"}

#: K for the two-point construction. Sixteen because that is the number E41 stated the defect at,
#: and because eight of sixteen is an arithmetic a reader can check without running anything.
K = 16


# ---------------------------------------------------------------------------
# The two-point truth
# ---------------------------------------------------------------------------


def _two_point_groups(n_groups: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Half the group at +1 and half at -1, shuffled. Kish shape factor exactly 1.0, exactly."""
    base = np.array([1.0] * (k // 2) + [-1.0] * (k // 2), dtype=np.float64)
    return np.stack([rng.permutation(base) for _ in range(n_groups)])


def test_the_two_point_construction_really_does_have_a_shape_factor_of_one():
    """The premise, checked before it is used to settle anything.

    `group_effective_size` is `(E|dev|)^2 / E[dev^2]` on the centred scores, which is 1 exactly when
    every rollout sits the same distance from the group mean. A two-point group is the only
    distribution for which that holds, which is why it is the construction that isolates noise.
    """
    truth = _two_point_groups(4_000, K, np.random.default_rng(1))
    per_group = np.array([group_effective_size(g) for g in truth])
    assert per_group.mean() == pytest.approx(float(K), abs=1e-12)
    assert np.max(np.abs(per_group - K)) == 0.0


@pytest.fixture(scope="module")
def two_point_truth():
    """A reliability of 0.5 fitted from a real crossed design, and 1,000 observed groups of sixteen.

    The design is 800 objects with a two-point true score, each scored by four independent grader
    draws at `sigma_err = 1.0`. Then `sigma2(p) = 1` and `sigma2(pr,e) = 1`, so the single-score
    generalizability coefficient is `1 / (1 + 1) = 0.5` and the group carries `16 x 0.5 = 8`.
    Nothing here is asserted from the analytic value: the reliability is whatever the EMS inversion
    returns on the sampled design, and the assertions are against that.
    """
    rng = np.random.default_rng(11)
    n_p, n_r = 800, 4
    truth_flat = rng.permutation(np.array([1.0] * (n_p // 2) + [-1.0] * (n_p // 2)))
    design = ReplicationDesign(
        scores=truth_flat[:, None] + rng.normal(0.0, 1.0, (n_p, n_r)),
        raters=tuple(f"draw{i}" for i in range(n_r)),
        object_label="response",
    )
    rng2 = np.random.default_rng(13)
    observed = _two_point_groups(1_000, K, rng2) + rng2.normal(0.0, 1.0, (1_000, K))
    scored = GroupScores.of(observed, grader="two-point-truth")
    r0 = effective_group_size(scored, None, n_resamples=400, seed=0)
    r3 = effective_group_size(scored, design, n_resamples=400, seed=0)
    return scored, design, r0, r3


def test_the_two_point_truth_now_returns_the_right_number(two_point_truth):
    """*At reliability 0.5 the group carries eight independent observations of sixteen.*

    This is E41 item 2's own sentence, and the reproduction that preceded the fix reported 5.5938
    against a truth of 7.9905. The erratum was right.
    """
    _, _, _, r3 = two_point_truth
    assert r3.rung == 3
    assert r3.reliability == pytest.approx(0.4994, abs=5e-4)
    assert r3.k_nominal == float(K)

    truth = K * r3.reliability
    assert truth == pytest.approx(7.9905, abs=5e-4)
    assert r3.n_eff == pytest.approx(truth, rel=1e-12), "the reading is K x reliability"
    assert r3.n_eff == pytest.approx(7.99, abs=0.01)

    # What the old rule would have said on the identical reading, computed from the fields the
    # reading still carries. The gap is 30% of the answer.
    old = r3.kish * r3.reliability
    assert old == pytest.approx(5.5938, abs=5e-4)
    assert r3.n_eff / old == pytest.approx(1.428, abs=5e-3)


def test_the_double_charge_is_visible_across_the_whole_noise_range():
    """E41's table, reproduced. The shape factor falls with noise for the same reason reliability does.

    A two-point truth's shape factor is 1.0 before any noise is added, so every step down this
    column is measurement error being counted a second time. At `sigma_err = 1.0` the observed
    shape factor is 0.7011 where the truth is 1.0, and multiplying it into a reliability of 0.5 is
    where the missing 30% went.
    """
    expected = {
        # sigma_err: (reliability, observed shape factor, old n_eff / K, new n_eff / K)
        0.0: (1.0000, 1.0000, 1.0000, 1.0000),
        0.5: (0.8000, 0.8307, 0.6646, 0.8000),
        1.0: (0.5000, 0.7011, 0.3506, 0.5000),
        2.0: (0.2000, 0.6639, 0.1328, 0.2000),
    }
    for sigma_err, (want_rel, want_shape, want_old, want_new) in expected.items():
        rng = np.random.default_rng(7)
        truth = _two_point_groups(4_000, K, rng)
        observed = truth + rng.normal(0.0, sigma_err, truth.shape)
        reliability = 1.0 / (1.0 + sigma_err**2)
        shape = float(np.mean([group_effective_size(g) for g in observed])) / K
        assert reliability == pytest.approx(want_rel, abs=1e-9), sigma_err
        assert shape == pytest.approx(want_shape, abs=5e-4), sigma_err
        assert shape * reliability == pytest.approx(want_old, abs=5e-4), sigma_err
        assert reliability == pytest.approx(want_new, abs=5e-4), sigma_err
    # And the direction is not signable, which is why the shape factor is separated rather than
    # deconvolved: noise pulls it toward 2/pi from whichever side it starts. Here it starts above.
    assert expected[2.0][1] > 2.0 / math.pi


# ---------------------------------------------------------------------------
# The shape factor is beside the reading, not inside it
# ---------------------------------------------------------------------------


def test_the_shape_factor_is_reported_separately_and_is_not_multiplied_in(two_point_truth):
    _, _, r0, r3 = two_point_truth
    for reading in (r0, r3):
        assert reading.shape_factor == pytest.approx(reading.kish / reading.k_nominal)
        low, high = reading.shape_ci
        assert low < reading.shape_factor < high, "a statistic without an interval is not one"
        assert "shape factor" in reading.says()
    # The two readings share a shape factor, because it is computed from the same observed groups
    # and has nothing to do with the design that gave rung 3 its reliability.
    assert r0.shape_factor == pytest.approx(r3.shape_factor, rel=1e-12)
    assert r0.shape_ci == r3.shape_ci
    # And it is not a factor of either reading.
    assert r0.n_eff == pytest.approx(r0.k_nominal, rel=1e-12)
    assert r3.n_eff == pytest.approx(r3.k_nominal * r3.reliability, rel=1e-12)
    assert r3.n_eff != pytest.approx(r3.kish * r3.reliability)


def test_the_shape_factor_reaches_the_payload_with_its_interval(two_point_truth):
    """A statistic that lives only in a dataclass is invisible to anything reading the store."""
    scored, design, _, r3 = two_point_truth
    payload = EffectiveGroupSize(groups=scored, design=design).payload(r3)
    assert payload["shape_factor"] == pytest.approx(r3.shape_factor)
    assert payload["shape_factor_ci_low"] < payload["shape_factor"]
    assert payload["shape_factor"] < payload["shape_factor_ci_high"]
    assert payload["n_eff"] == pytest.approx(r3.k_nominal * r3.reliability)
    assert payload["determined"] is True


# ---------------------------------------------------------------------------
# Rung 0: the declarations are unchanged and the number is not
# ---------------------------------------------------------------------------


def test_rung_zero_keeps_every_declaration_it_had(two_point_truth):
    """*Rung 0 is unaffected: it sets reliability to 1.0 and already names every error term as
    invisible.* Both halves of that hold, and they are what this asserts."""
    _, _, r0, _ = two_point_truth
    assert r0.rung == 0
    assert r0.reliability == 1.0
    assert r0.invisible_terms == ("every error term",)
    assert r0.bias.direction == "upward"
    assert "cannot see correlated grader error at all" in r0.bias.why
    assert "assumed to be 1" in r0.universe
    assert r0.determined


def test_rung_zeros_number_moved_from_the_kish_count_to_k(two_point_truth):
    """The half of rung 0 that did change, asserted with both values so nothing moves it back quietly.

    Dropping the Kish factor out of the product takes rung 0 from `shape x K` to `K`, because rung
    0's reliability is 1. On these two-point groups at `sigma_err = 1.0` that is 11.20 to 16.00.
    Rung 0 is now exactly the number practitioners already quote, which is the honest thing for a
    rung that can see no grader error at all to say.
    """
    _, _, r0, _ = two_point_truth
    assert r0.n_eff == float(K)
    assert r0.kish == pytest.approx(11.2010, abs=5e-4), "what rung 0 used to report as n_eff"
    assert r0.n_eff / r0.kish == pytest.approx(1.4284, abs=5e-4)
    assert r0.wasted == 0.0, "a rung that can see no error must not report a wasted rollout"
    assert not r0.has_interval, "rung 0 assumes its reliability, so it has no interval to give"
    assert r0.ci_low == r0.ci_high == r0.n_eff
    assert "which is all of it" in r0.says()
    assert "score some objects twice" in r0.says()


def test_a_degenerate_decomposition_refuses_instead_of_reporting_zero(two_point_truth):
    """E42's triage, the item E41 fixed in `GaugeRR` and nothing had fixed here.

    All-zero components make `E rho^2` zero over zero, which returns 0.0, and `K x 0.0` is an
    effective group size of exactly 0.0. That is a measured-looking number for a design in which
    nothing varied.
    """
    scored, _, _, _ = two_point_truth
    flat = ReplicationDesign(scores=np.full((60, 4), -1.25))
    out = EffectiveGroupSize(groups=scored, design=flat).compute()
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.ENVELOPE_VIOLATED
    assert "no variance in it" in out.detail
    assert "measurement of nothing" in out.detail
    assert out.remedy.strip()
    assert out.statistics["components_total"] == 0.0

    raw = effective_group_size(scored, flat, n_resamples=100, seed=0)
    assert raw.n_eff == 0.0
    assert not raw.determined
    assert "undetermined" in raw.render()


# ---------------------------------------------------------------------------
# The before and after, on the eleven real reward models series A ran on
# ---------------------------------------------------------------------------

pytestmark_real = pytest.mark.skipif(
    CAMPAIGN_STORE is None or not (CAMPAIGN_STORE / "evidence.jsonl").exists(),
    reason=(
        "no campaign evidence store. The before-and-after has to be measured on the same eleven "
        "open reward models series A ran on, or it is not a comparison. Set "
        "REWARD_LENS_CAMPAIGN_STORE to the directory holding evidence.jsonl."
    ),
)


@pytest.fixture(scope="module")
def rb2_panel():
    """11 graders x 1,763 groups of K = 4, gauge-fixed, exactly as `test_w3_2a_metrology.py` loads it."""
    store = CampaignStore(CAMPAIGN_STORE)
    banks = {}
    for row in store.by_observable("campaign.scores"):
        if row.bank != "rb2-full" or row.roster_key in NOT_A_GRADER:
            continue
        value = store.value(row)
        if value["layout"] != "best-of-4":
            continue
        banks[row.roster_key] = (
            list(value["item_ids"]),
            np.asarray(value["scores"], dtype=np.float64),
        )
    graders = sorted(banks)
    reference = banks[graders[0]][0]
    assert all(banks[g][0] == reference for g in graders), "the banks are not the same items"
    raw = np.stack([banks[g][1] for g in graders], axis=0)
    n_r, n_groups, k = raw.shape
    flat = raw.reshape(n_r, n_groups * k)
    flat = (flat - flat.mean(axis=1, keepdims=True)) / flat.std(axis=1, ddof=1, keepdims=True)
    design = ReplicationDesign(
        scores=flat.T,
        raters=tuple(graders),
        object_label="response",
        facet_labels=("reward model", "occasion"),
    )
    _, se = jackknife_reliability(design)
    readings = {}
    for i, grader in enumerate(graders):
        scored = GroupScores.of(flat.reshape(n_r, n_groups, k)[i], grader=grader)
        readings[grader] = (
            effective_group_size(scored, None, n_resamples=600, seed=0),
            effective_group_size(scored, design, n_resamples=600, seed=0, reliability_se=se),
        )
    return graders, readings, se


@pytestmark_real
def test_the_before_and_after_are_recorded_on_the_eleven_real_reward_models(rb2_panel):
    """The published number and the corrected one, side by side, on the subjects that produced it.

    Both are computed from the same readings, so this is the correction's effect and not a
    re-measurement. `old` is the arithmetic this module performed until 2026-08-05.

        rung 0   old 2.9859 mean over eleven, spread 0.0863   new 4.0000 for all eleven
        rung 3   old 1.9097 mean over eleven, spread 0.0552   new 2.5582 for all eleven

    The rung-3 spread across graders was entirely the Kish shape factor. One crossed design gives
    one reliability, so the corrected reading is a property of the panel. That is the coefficient
    this design estimates, and a per-grader effective size would need repeated calls to each
    grader, which this store does not contain.
    """
    graders, readings, se = rb2_panel
    assert len(graders) == 11
    assert se == pytest.approx(0.1140, abs=5e-4)

    old_r0 = np.array([r0.kish * r0.reliability for r0, _ in readings.values()])
    new_r0 = np.array([r0.n_eff for r0, _ in readings.values()])
    old_r3 = np.array([r3.kish * r3.reliability for _, r3 in readings.values()])
    new_r3 = np.array([r3.n_eff for _, r3 in readings.values()])

    assert old_r0.mean() == pytest.approx(2.9859, abs=5e-4)
    assert old_r0.max() - old_r0.min() == pytest.approx(0.0863, abs=5e-4)
    assert np.all(new_r0 == 4.0)

    assert old_r3.mean() == pytest.approx(1.9097, abs=5e-4)
    assert old_r3.max() - old_r3.min() == pytest.approx(0.0552, abs=5e-4)
    assert new_r3 == pytest.approx(2.5582, abs=5e-4)
    assert new_r3.max() - new_r3.min() == pytest.approx(0.0, abs=1e-12)

    assert new_r3.mean() / old_r3.mean() == pytest.approx(1.3396, abs=5e-4)

    # Every grader in this panel shares one reliability, and it is what the reading is now made of.
    for _, r3 in readings.values():
        assert r3.reliability == pytest.approx(0.6396, abs=5e-4)
        assert r3.n_eff == pytest.approx(r3.k_nominal * r3.reliability, rel=1e-12)


@pytestmark_real
def test_the_shape_factor_is_where_the_per_grader_variation_actually_lives(rb2_panel):
    """It reads about 0.75 on all eleven, which is the uniform-spread anchor rather than a defect.

    A group of four best-of-four responses whose scores are roughly evenly spread has a shape
    factor near 0.75, and `group_effective_size`'s own docstring gives 0.75 as the uniform anchor.
    On a perfect grader the same number would come out, which is exactly why calling `0.75 x K` an
    effective group size read as "your grader costs you a quarter of your rollouts".
    """
    _, readings, _ = rb2_panel
    shapes = np.array([r0.shape_factor for r0, _ in readings.values()])
    assert shapes.min() == pytest.approx(0.7346, abs=5e-4)
    assert shapes.max() == pytest.approx(0.7562, abs=5e-4)
    assert shapes.mean() == pytest.approx(0.7465, abs=5e-4)
    assert shapes.max() - shapes.min() == pytest.approx(0.0216, abs=5e-4)
    for r0, _ in readings.values():
        low, high = r0.shape_ci
        assert low < r0.shape_factor < high


@pytestmark_real
def test_p13_the_kill_condition_still_does_not_fire_after_the_correction(rb2_panel):
    """**P13, pre-registered before this result existed:** correcting the quantity does not change
    whether A1's kill condition fires. It holds, and the margin got wider rather than narrower.

    The kill is *if r0 and r3 agree within their intervals on five graders, the ladder is
    decoration and only r0 ships*, and the overlap test is the one in `test_w3_2a_metrology.py`:
    `not (r0.ci_low > r3.ci_high or r3.ci_low > r0.ci_high)`. Both regimes, same eleven subjects,
    same single load:

        old rule   0 of 11 overlapping, smallest margin 0.3847 effective rollouts
        new rule   0 of 11 overlapping, smallest margin 0.5484 effective rollouts

    Rung 0 has no interval of its own now, so the question became whether the rung-3 interval
    reaches K, and it does not: 3.4516 against 4.0 on the widest version of itself, the one
    carrying the rater panel's leave-one-model-out uncertainty. The correction moved both rungs up
    and it did not move them past each other.
    """
    _, readings, _ = rb2_panel
    overlapping = [
        grader
        for grader, (r0, r3) in readings.items()
        if not (r0.ci_low > r3.ci_high or r3.ci_low > r0.ci_high)
    ]
    assert overlapping == [], overlapping
    assert len(readings) >= 5, "the kill condition needs five graders to be answerable at all"

    margins = [r0.ci_low - r3.ci_high for r0, r3 in readings.values()]
    assert min(margins) == pytest.approx(0.5484, abs=5e-4)
    assert min(margins) > 0.3847, (
        "the correction widened the gap between the rungs, not narrowed it"
    )

    for grader, (r0, r3) in readings.items():
        assert r3.n_eff < r0.n_eff, grader
        assert r3.ci_high < r0.n_eff, grader
        assert r3.n_eff / r0.n_eff == pytest.approx(0.6396, abs=5e-4), grader
    _, any_r3 = next(iter(readings.values()))
    assert any_r3.ci_low == pytest.approx(1.6648, abs=5e-4)
    assert any_r3.ci_high == pytest.approx(3.4516, abs=5e-4)

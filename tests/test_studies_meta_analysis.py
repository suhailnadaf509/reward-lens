"""Unit tests for the aggregation plan, its check, and the two family-level honesty items.

Hand-computed values throughout. The e-BH cases are chosen to include the one a plausible wrong
implementation gets wrong: a family where the largest e-value alone does not clear its threshold
but three of them together do, so a procedure that stops at the first failing rank rejects nothing
where the correct answer is three.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from reward_lens.core.types import TrustLevel
from reward_lens.studies.meta_analysis import (
    DEFAULT_REMEDY,
    AdjudicatedPrediction,
    CountClaim,
    ExploratoryReason,
    Stability,
    boundary_sensitivity,
    calibrate_p_to_e,
    correct_family,
    cover,
    e_bh,
    freeze_meta_plan,
    near_threshold,
    p_value_predictions,
    render_count,
    tally,
)


def _pred(**kw) -> AdjudicatedPrediction:
    fields = {
        "study": "S",
        "owner": "H1",
        "kind": "hypothesis",
        "metric": "m",
        "comparator": ">",
        "threshold": 0.5,
        "value": 0.7,
        "outcome": "confirmed",
    }
    fields.update(kw)
    return AdjudicatedPrediction(**fields)


def _plan(**kw):
    fields = {"id": "p", "studies": ("S",), "unit": "hypothesis", "labels": ("confirmed",)}
    fields.update(kw)
    return freeze_meta_plan("agg", "incl", "score", **fields)


# ---------------------------------------------------------------------------
# The plan and its hash
# ---------------------------------------------------------------------------


def test_the_hash_is_a_function_of_the_content_and_nothing_else() -> None:
    a = _plan(frozen_at="2026-01-01T00:00:00+00:00")
    b = _plan(frozen_at="2026-01-01T00:00:00+00:00")
    assert a.spec_hash == b.spec_hash
    assert a.hash_verified and b.hash_verified


def test_every_hashed_field_changes_the_hash() -> None:
    base = _plan(frozen_at="2026-01-01T00:00:00+00:00")
    variants = {
        "aggregation": freeze_meta_plan(
            "other", "incl", "score", id="p", studies=("S",), frozen_at=base.frozen_at
        ),
        "inclusion": freeze_meta_plan(
            "agg", "other", "score", id="p", studies=("S",), frozen_at=base.frozen_at
        ),
        "scoring": freeze_meta_plan(
            "agg", "incl", "other", id="p", studies=("S",), frozen_at=base.frozen_at
        ),
        "studies": _plan(frozen_at=base.frozen_at, studies=("S", "T")),
        "unit": _plan(frozen_at=base.frozen_at, unit="study"),
        "frozen_at": _plan(frozen_at="2026-01-02T00:00:00+00:00"),
    }
    for name, variant in variants.items():
        assert variant.spec_hash != base.spec_hash, name


def test_the_declared_study_order_does_not_change_the_hash() -> None:
    """Inclusion is a set. Two people writing the same set in different orders wrote one plan."""
    a = _plan(frozen_at="2026-01-01T00:00:00+00:00", studies=("S", "T"))
    b = _plan(frozen_at="2026-01-01T00:00:00+00:00", studies=("T", "S"))
    assert a.spec_hash == b.spec_hash


def test_an_edited_plan_no_longer_verifies() -> None:
    tampered = replace(_plan(), scoring="counted differently")
    assert not tampered.hash_verified


# ---------------------------------------------------------------------------
# The check, and the remedies it hands back
# ---------------------------------------------------------------------------


def test_every_reason_carries_a_remedy_written_as_an_instruction() -> None:
    """The refusal-test discipline: assert the reason and the remedy, not just that it failed."""
    assert set(DEFAULT_REMEDY) == set(ExploratoryReason)
    for reason, remedy in DEFAULT_REMEDY.items():
        assert len(remedy) > 60, reason
        assert remedy[0].isupper(), reason
        assert remedy.rstrip().endswith("."), reason


def test_an_uncovered_count_reports_every_reason_not_only_the_first() -> None:
    claim = CountClaim(label="mixed", value=3, unit="study", over=("A",))
    coverage = cover(claim, _plan())
    reasons = {r for r, _ in coverage.reasons}
    assert reasons == {
        ExploratoryReason.OUTSIDE_INCLUSION,
        ExploratoryReason.UNIT_UNDECLARED,
        ExploratoryReason.LABEL_UNDECLARED,
    }


def test_an_empty_evidence_timestamp_does_not_invent_an_ordering() -> None:
    """With nothing to compare against, the temporal clause abstains rather than guessing."""
    plan = _plan(frozen_at="2099-01-01T00:00:00+00:00")
    claim = CountClaim(label="confirmed", value=1, over=("S",), evidence_at="")
    reasons = {r for r, _ in cover(claim, plan).reasons}
    assert ExploratoryReason.PLAN_POSTDATES_EVIDENCE not in reasons


def test_the_trust_level_is_the_ladder_the_evidence_already_uses() -> None:
    claim = CountClaim(label="confirmed", value=1, over=("S",))
    assert cover(claim, None).trust is TrustLevel.EXPLORATORY
    assert cover(claim, _plan()).trust is TrustLevel.REGISTERED


def test_the_badge_is_short_enough_for_a_table_cell() -> None:
    claim = CountClaim(label="confirmed", value=1, over=("S",))
    assert cover(claim, None).badge == "[EXPLORATORY]"
    assert cover(claim, _plan()).badge.startswith("[REGISTERED under meta:p#")


def test_render_prints_the_number_in_both_states() -> None:
    claim = CountClaim(label="confirmed", value=41, over=("S",), statement="41 confirmed")
    assert "41 confirmed" in render_count(claim, None)
    assert "41 confirmed" in render_count(claim, _plan())


def test_tally_records_what_it_counted_over() -> None:
    rows = [
        _pred(study="A", outcome="confirmed"),
        _pred(study="A", outcome="refuted"),
        _pred(study="B", outcome="confirmed"),
        _pred(study="C", outcome="void", value=None),
    ]
    claim = tally(rows, "confirmed")
    assert claim.value == 2
    assert claim.over == ("A", "B", "C")


# ---------------------------------------------------------------------------
# e-BH
# ---------------------------------------------------------------------------


def test_e_bh_hand_computed() -> None:
    """n = 3, alpha = 0.05, so rank k needs an e-value of 3 / (0.05 k): 60, 30, 20."""
    assert e_bh([100.0, 40.0, 25.0], 0.05) == ((True, True, True), 3)
    assert e_bh([100.0, 40.0, 10.0], 0.05) == ((True, True, False), 2)
    assert e_bh([100.0, 10.0, 10.0], 0.05) == ((True, False, False), 1)
    assert e_bh([10.0, 10.0, 10.0], 0.05) == ((False, False, False), 0)


def test_e_bh_takes_the_largest_k_not_the_first() -> None:
    """The case a stop-at-first-failure implementation gets wrong.

    50 does not clear rank 1's threshold of 60, but the three together clear rank 3's 20. The
    correct answer is three rejections, and an implementation that breaks at rank 1 returns none.
    """
    mask, k = e_bh([50.0, 40.0, 25.0], 0.05)
    assert k == 3
    assert mask == (True, True, True)


def test_e_bh_rejects_the_largest_e_values() -> None:
    """n = 4, so rank k needs 4 / (0.05 k): 80, 40, 26.67, 20. 500 and 90 clear ranks 1 and 2."""
    mask, k = e_bh([10.0, 500.0, 1.0, 90.0], 0.05)
    assert k == 2
    assert mask == (False, True, False, True)


def test_e_bh_treats_an_undefined_e_value_as_no_evidence() -> None:
    """NaN is not evidence against a null, and dropping it would loosen everyone else's threshold."""
    assert e_bh([float("nan"), float("nan"), float("nan")], 0.05)[1] == 0
    mask, k = e_bh([100.0, float("nan"), 40.0, 25.0], 0.05)
    assert mask == (True, False, True, False)
    # n stays 4, so rank 3 needs 26.67 and the 25 misses it. Had the NaN been dropped, n would be
    # 3, rank 3 would need 20, and 25 would have been rejected on the strength of a test that
    # never ran.
    assert k == 2


def test_e_bh_on_an_empty_family() -> None:
    assert e_bh([], 0.05) == ((), 0)


@given(
    e=st.lists(st.floats(min_value=0.0, max_value=1e6, allow_nan=False), min_size=1, max_size=30),
    alpha=st.floats(min_value=0.001, max_value=0.5),
)
def test_e_bh_rejection_set_satisfies_its_own_defining_inequality(e, alpha) -> None:
    mask, k = e_bh(e, alpha)
    n = len(e)
    assert sum(mask) == k
    if k > 0:
        kth = sorted((x if math.isfinite(x) and x > 0 else 0.0 for x in e), reverse=True)[k - 1]
        assert kth >= n / (alpha * k)
    # k is maximal: no larger rank satisfies the inequality.
    ordered = sorted((x if math.isfinite(x) and x > 0 else 0.0 for x in e), reverse=True)
    for rank in range(k + 1, n + 1):
        assert ordered[rank - 1] < n / (alpha * rank)


@given(
    e=st.lists(st.floats(min_value=0.0, max_value=1e4, allow_nan=False), min_size=1, max_size=20),
    alpha=st.floats(min_value=0.001, max_value=0.5),
)
def test_e_bh_is_monotone_in_the_evidence(e, alpha) -> None:
    """Raising an e-value never reduces the number of rejections."""
    _, before = e_bh(e, alpha)
    _, after = e_bh([x * 2.0 for x in e], alpha)
    assert after >= before


@given(
    e=st.lists(
        st.floats(min_value=0.0, max_value=1e5, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=25,
    ),
    alpha=st.floats(min_value=0.005, max_value=0.5),
)
@settings(max_examples=60, deadline=None)
def test_e_bh_agrees_with_the_sequential_statistics_implementation(e, alpha) -> None:
    """The two e-BH implementations in the tree must agree before either can replace the other.

    `reward_lens.stats.sequential.ebh` is the same procedure written for the monitor's ledger of
    alarms. This module keeps its own pure-Python copy so that rendering a count does not pull
    `reward_lens.stats` and the 1.15 s of scipy and sklearn behind it. Two implementations of one
    procedure is one too many, and this test is what makes collapsing them a mechanical change
    rather than a judgement call: it skips while the other module is absent and fails the moment
    they disagree.
    """
    sequential = pytest.importorskip("reward_lens.stats.sequential")

    mask, k = e_bh(e, alpha)
    theirs = sequential.ebh(e, alpha)
    assert k == theirs.n_rejected
    assert list(mask) == [bool(x) for x in theirs.rejected]


# ---------------------------------------------------------------------------
# The p-to-e calibrator
# ---------------------------------------------------------------------------


def test_calibrator_hand_computed() -> None:
    """f(p) = kappa * p ** (kappa - 1). At kappa = 0.5 this is 0.5 / sqrt(p)."""
    assert calibrate_p_to_e(0.25, 0.5) == pytest.approx(1.0)
    assert calibrate_p_to_e(0.04, 0.5) == pytest.approx(2.5)
    assert calibrate_p_to_e(1.0, 0.5) == pytest.approx(0.5)


def test_calibrator_rejects_a_kappa_outside_its_domain() -> None:
    for bad in (0.0, 1.0, -0.5, 2.0):
        with pytest.raises(ValueError, match="kappa"):
            calibrate_p_to_e(0.1, bad)


def test_calibrator_rejects_an_impossible_p() -> None:
    with pytest.raises(ValueError, match="cannot exceed 1"):
        calibrate_p_to_e(1.5, 0.5)


@given(kappa=st.floats(min_value=0.05, max_value=0.95))
@settings(max_examples=25, deadline=None)
def test_the_calibrator_is_admissible(kappa) -> None:
    """The property that makes the output an e-value: it integrates to 1 over a uniform p.

    The antiderivative of ``kappa * p ** (kappa - 1)`` is ``p ** kappa``, so the integral over
    ``[a, 1]`` is exactly ``1 - a ** kappa``. Checked over ``[0.1, 1]`` rather than over the whole
    unit interval: the integrand has an integrable singularity at 0 and the midpoint rule converges
    too slowly there to distinguish a correct implementation from a slightly wrong one, which would
    make the test noise rather than evidence. Away from the singularity it pins both the
    coefficient and the exponent, so ``p ** (kappa + 1)`` or a missing ``kappa`` factor fails here.
    """
    n, lo, hi = 20000, 0.1, 1.0
    width = (hi - lo) / n
    total = sum(calibrate_p_to_e(lo + (i + 0.5) * width, kappa) for i in range(n)) * width
    assert total == pytest.approx(1.0 - lo**kappa, rel=1e-6)


@given(
    p=st.floats(min_value=1e-6, max_value=1.0),
    q=st.floats(min_value=1e-6, max_value=1.0),
    kappa=st.floats(min_value=0.05, max_value=0.95),
)
def test_the_calibrator_is_decreasing_in_p(p, q, kappa) -> None:
    assume(p < q)
    assert calibrate_p_to_e(p, kappa) >= calibrate_p_to_e(q, kappa)


# ---------------------------------------------------------------------------
# The family correction
# ---------------------------------------------------------------------------


def test_correct_family_hand_computed() -> None:
    """Four p-values, alpha 0.05. BH q for the smallest is 0.001 * 4 / 1 = 0.004."""
    entries = [("a", 0.001), ("b", 0.045), ("c", 0.4), ("d", 0.9)]
    out = correct_family(entries, alpha=0.05)
    assert out.n == 4
    assert out.bh_q[0] == pytest.approx(0.004)
    # b sits under 0.05 on its own and its BH q is 0.045 * 4 / 2 = 0.09, so it does not survive.
    assert out.bh_q[1] == pytest.approx(0.09)
    # BY is BH scaled by H_4 = 1 + 1/2 + 1/3 + 1/4.
    h4 = 1 + 0.5 + 1 / 3 + 0.25
    assert out.by_q[0] == pytest.approx(0.004 * h4)
    assert sum(out.uncorrected) == 2
    assert sum(out.bh_rejected) == 1
    assert out.lost_to_correction == ("b",)


def test_by_is_never_more_liberal_than_bh() -> None:
    out = correct_family([("a", 0.001), ("b", 0.02), ("c", 0.4)], alpha=0.05)
    assert sum(out.by_rejected) <= sum(out.bh_rejected)


def test_correct_family_on_an_empty_family() -> None:
    out = correct_family([], alpha=0.05)
    assert out.n == 0
    assert out.render().startswith("0 registered p-values")


def test_p_value_predictions_selects_on_name_comparator_and_threshold() -> None:
    rows = [
        _pred(owner="H1", metric="perm_p", comparator="<", threshold=0.05, value=0.01),
        _pred(owner="H2", metric="x_p_value", comparator="<=", threshold=0.05, value=0.2),
        # right name, wrong direction: a prediction that p exceeds a level is not a test.
        _pred(owner="H3", metric="other_p", comparator=">", threshold=0.05, value=0.01),
        # right shape, threshold outside (0, 1).
        _pred(owner="H4", metric="z_p", comparator="<", threshold=2.0, value=0.01),
        # not a p-value at all.
        _pred(owner="H5", metric="auroc", comparator="<", threshold=0.5, value=0.2),
        # never computed.
        _pred(owner="H6", metric="w_p", comparator="<", threshold=0.05, value=None),
    ]
    assert p_value_predictions(rows) == (("S/H1", 0.01), ("S/H2", 0.2))


# ---------------------------------------------------------------------------
# Null-boundary sensitivity
# ---------------------------------------------------------------------------


def test_flip_distance_hand_computed() -> None:
    rows = boundary_sensitivity([_pred(threshold=0.5, value=0.7, comparator=">")])
    assert rows[0].flip_distance == pytest.approx(0.2)
    assert rows[0].relative_flip == pytest.approx(0.2 / 0.7)


def test_flip_distance_uses_the_magnitude_for_an_abs_comparator() -> None:
    """`raw_cos abs< 0.02` at -0.015 is 0.005 from flipping, not 0.035."""
    rows = boundary_sensitivity([_pred(comparator="abs<", threshold=0.02, value=-0.015)])
    assert rows[0].flip_distance == pytest.approx(0.005)


def test_a_recorded_uncertainty_turns_the_ranking_into_a_verdict() -> None:
    close = boundary_sensitivity([_pred(threshold=0.5, value=0.55, uncertainty=0.1)])[0]
    assert close.z == pytest.approx(0.5)
    assert close.stability is Stability.SENSITIVE
    assert "could have produced the other verdict" in close.note

    far = boundary_sensitivity([_pred(threshold=0.5, value=0.9, uncertainty=0.1)])[0]
    assert far.z == pytest.approx(4.0)
    assert far.stability is Stability.STABLE


def test_a_missing_uncertainty_is_unknown_rather_than_assumed() -> None:
    row = boundary_sensitivity([_pred(uncertainty=None)])[0]
    assert row.stability is Stability.UNKNOWN
    assert row.z is None
    assert "Record ci_low and ci_high" in row.note


def test_an_equality_comparator_is_discrete_rather_than_maximally_sensitive() -> None:
    """A flip distance of zero on an indicator means an exact match, not a verdict about to fall."""
    row = boundary_sensitivity([_pred(comparator="==", threshold=1.0, value=1.0)])[0]
    assert row.stability is Stability.DISCRETE
    assert row.flip_distance == 0.0
    assert near_threshold([row]) == ()


def test_predictions_with_no_value_have_no_verdict_to_be_sensitive_about() -> None:
    assert boundary_sensitivity([_pred(value=None, outcome="void")]) == ()


def test_rows_come_back_closest_first() -> None:
    rows = boundary_sensitivity(
        [
            _pred(owner="far", threshold=0.5, value=1.0),
            _pred(owner="near", threshold=0.5, value=0.52),
            _pred(owner="mid", threshold=0.5, value=0.7),
        ]
    )
    assert [r.owner for r in rows] == ["near", "mid", "far"]


@given(
    threshold=st.floats(min_value=-100, max_value=100, allow_nan=False),
    value=st.floats(min_value=-100, max_value=100, allow_nan=False),
)
def test_flip_distance_is_symmetric_and_non_negative(threshold, value) -> None:
    row = boundary_sensitivity([_pred(threshold=threshold, value=value, comparator=">")])[0]
    assert row.flip_distance >= 0
    assert row.flip_distance == pytest.approx(abs(value - threshold))
    if row.relative_flip is not None:
        assert row.relative_flip >= 0

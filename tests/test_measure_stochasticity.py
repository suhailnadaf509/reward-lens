"""Unit tests for A6's two declaring instruments and the design diagnostic the split forced.

The acceptance file argues the split on real data. This one covers the edges: the refusals, the
determinism case that is a reading rather than a failure, the ragged and partly-replicated designs
that a real replication budget produces, and the closed form for how many pairs a design left with
nothing to say.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from reward_lens.core.evidence import ValueCodec
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.measure.metrology.distribution import RepeatedScores
from reward_lens.measure.metrology.stochasticity import (
    GraderFlipRate,
    GraderScoreSigma,
    ScoreSpread,
    VerdictStability,
    facet_clause,
    stochasticity_profile,
    unanimous_pair_fraction,
)


def _panel(seed: int = 5, n_items: int = 18, n_repeats: int = 4) -> RepeatedScores:
    rng = np.random.default_rng(seed)
    scores = rng.normal(size=(n_items, n_repeats)) + rng.normal(size=(n_items, 1)) * 2.5
    return RepeatedScores(scores=scores, grader="synthetic")


# ---------------------------------------------------------------------------
# The refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [GraderScoreSigma, GraderFlipRate])
def test_no_data_refuses_with_a_remedy_that_names_the_cheapest_facet(cls: type) -> None:
    out = cls().compute()
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert "Presentation order is the cheapest facet" in out.remedy


@pytest.mark.parametrize("cls", [GraderScoreSigma, GraderFlipRate])
def test_one_draw_per_item_refuses_rather_than_returning_the_zero_it_could(cls: type) -> None:
    """The refusal that matters more for the flip rate than for the spread.

    A variance over one observation is visibly undefined: it comes back NaN and nothing downstream
    mistakes it for a measurement. A flip rate over one draw per item comes back exactly 0.0,
    because a single comparison always agrees with itself, and 0.0 is in range and looks like a
    deterministic grader. Both refuse, and the detail says which of the two zeros this would be.
    """
    data = RepeatedScores(scores=np.arange(9.0)[:, None])
    out = cls(data).compute()
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert "GRADER:REPLICATE" in out.detail
    assert out.statistics["items_with_two_or_more"] == 0
    assert out.statistics["max_draws_per_item"] == 1


def test_the_zero_the_refusal_prevents_is_real() -> None:
    """The estimator does return 0.0 there, which is why the guard is a refusal and not a comment."""
    from reward_lens.measure.metrology.distribution import flip_rates

    assert flip_rates(RepeatedScores(scores=np.arange(9.0)[:, None])).flip_rate == 0.0


# ---------------------------------------------------------------------------
# Determinism is a reading
# ---------------------------------------------------------------------------


def test_a_deterministic_grader_reads_zero_on_both_and_says_so() -> None:
    """A6's kill condition, which is the honest scope limit rather than a failure.

    Sigma is exactly zero and the flip rate is exactly zero, both over a design that had every
    chance to show otherwise, and both readings say that replication buys nothing on this grader
    class. That is the sentence a scalar-head user needs and it is the reason not to pay for repeats.
    """
    scores = np.repeat(np.arange(12.0)[:, None], 5, axis=1)
    spread, verdict = stochasticity_profile(RepeatedScores(scores=scores, grader="scalar head"))
    assert isinstance(spread, ScoreSpread) and isinstance(verdict, VerdictStability)
    assert spread.sigma == 0.0
    assert spread.deterministic is True
    assert "deterministic" not in spread.says  # it says what happened, not the flag name
    assert "identical score" in spread.says
    assert verdict.flip_rate == 0.0
    assert verdict.never_flipped is True
    assert "never changed its pairwise verdict" in verdict.says


def test_never_flipped_is_not_set_when_the_design_gave_no_pair_a_choice() -> None:
    """The two zeros stay distinguishable, which is the whole argument for the refusal above.

    Constructed so the flip rate is computable (two items carry repeats) while most pairs carry one
    combination. `never_flipped` requires more than one combination behind the median pair, so it
    reports False here and `unanimous_by_construction` says why.
    """
    scores = np.full((8, 3), np.nan)
    scores[:, 0] = np.arange(8.0)
    scores[0, 1] = 0.0
    scores[1, 1] = 1.0
    verdict = GraderFlipRate(RepeatedScores(scores=scores)).compute()
    assert isinstance(verdict, VerdictStability)
    assert verdict.flip_rate == 0.0
    assert verdict.combinations_per_pair == 1
    assert verdict.never_flipped is False
    assert verdict.singleton_items == 6
    assert verdict.unanimous_by_construction > 0.0
    assert "unanimous whatever the grader does" in verdict.says


# ---------------------------------------------------------------------------
# The design diagnostic
# ---------------------------------------------------------------------------


def _brute_unanimous(data: RepeatedScores) -> float:
    counts = data.counts()
    idx = [i for i in range(len(counts)) if counts[i] >= 1]
    total = one = 0
    for i, j in itertools.combinations(idx, 2):
        combos = min(counts[i], counts[j]) if data.paired_occasions else counts[i] * counts[j]
        total += 1
        one += combos == 1
    return one / total if total else float("nan")


@settings(max_examples=60, deadline=None)
@given(
    counts=st.lists(st.integers(min_value=0, max_value=4), min_size=2, max_size=9),
    paired=st.booleans(),
)
def test_the_closed_form_for_unanimous_pairs_matches_counting_them(
    counts: list[int], paired: bool
) -> None:
    """It is derived rather than enumerated, so it is checked against the enumeration.

    `n choose 2` over a thousand items is half a million pairs, and the fraction is a property of
    the draw counts alone, so counting it would be work the design already answers.
    """
    width = max(max(counts), 2)
    scores = np.full((len(counts), width), np.nan)
    for i, c in enumerate(counts):
        scores[i, :c] = np.arange(float(c))
    data = RepeatedScores(scores=scores, paired_occasions=paired)
    if int(np.sum(data.counts() >= 1)) < 2:
        return
    assert unanimous_pair_fraction(data) == pytest.approx(_brute_unanimous(data))


def test_a_fully_replicated_design_leaves_no_pair_unanimous_by_construction() -> None:
    assert unanimous_pair_fraction(_panel()) == 0.0


def test_a_single_usable_item_has_no_pairs_and_reports_nan() -> None:
    scores = np.full((3, 2), np.nan)
    scores[0] = [1.0, 2.0]
    assert np.isnan(unanimous_pair_fraction(RepeatedScores(scores=scores)))


# ---------------------------------------------------------------------------
# Ragged designs, which is what a real replication budget produces
# ---------------------------------------------------------------------------


def test_a_ragged_design_pools_by_degrees_of_freedom_and_reports_what_it_used() -> None:
    """Items with more repeats carry more information about the spread, and pooling keeps it.

    The reading carries `n_items_used` beside `n_items` so a design where two thirds of the items
    contributed nothing to the variance says so rather than looking like a 30-item measurement.
    """
    scores = np.full((30, 4), np.nan)
    rng = np.random.default_rng(2)
    scores[:10] = rng.normal(size=(10, 4))
    scores[10:, 0] = rng.normal(size=20)
    spread = GraderScoreSigma(RepeatedScores(scores=scores)).compute()
    assert isinstance(spread, ScoreSpread)
    assert spread.n_items == 30
    assert spread.n_items_used == 10
    assert spread.df == 30
    assert spread.n_scores == 60
    assert spread.sigma > 0.0


def test_the_two_halves_refuse_independently() -> None:
    """A design can carry a usable spread and a flip rate that is mostly zeros by construction.

    Both are computed here and both are readings, which is the honest outcome: the spread rests on
    the ten replicated items and the flip rate says on its own face that 89% of its pairs never had
    a choice. Refusing the flip rate outright would throw away the ten items that do speak.
    """
    scores = np.full((30, 4), np.nan)
    rng = np.random.default_rng(4)
    scores[:10] = rng.normal(size=(10, 4))
    scores[10:, 0] = rng.normal(size=20)
    spread, verdict = stochasticity_profile(RepeatedScores(scores=scores))
    assert isinstance(spread, ScoreSpread)
    assert isinstance(verdict, VerdictStability)
    assert verdict.singleton_items == 20
    assert verdict.unanimous_by_construction == pytest.approx(190 / 435)


# ---------------------------------------------------------------------------
# Rung 1 stays with the spread
# ---------------------------------------------------------------------------


def test_the_facet_attribution_belongs_to_the_spread_and_not_to_the_verdict() -> None:
    """The catalogue's rung 1 attributes the within-item variance, so it lands on sigma.

    That the ladder splits cleanly at the same seam as the invariance is some evidence the seam is
    real: rungs 0 and 1 are about the spread and rung 2 is about the verdict, and the two halves
    transform differently.
    """
    rng = np.random.default_rng(9)
    scores = rng.normal(size=(20, 4)) * 0.2
    scores[:, 2:] += 1.5  # a systematic shift on the second level of the facet
    data = RepeatedScores(scores=scores, facets={"order": np.array([0, 0, 1, 1])})
    spread = GraderScoreSigma(data).compute()
    assert isinstance(spread, ScoreSpread)
    assert "order" in spread.facet_shares
    assert spread.facet_effects["order"].significant is True
    assert "systematic shift" in spread.says
    assert GraderScoreSigma.rung == 1
    assert GraderFlipRate.rung == 2
    assert not hasattr(GraderFlipRate(data).compute(), "facet_shares")


def test_the_facet_clause_never_quotes_a_share_without_its_own_null() -> None:
    """Eta-squared has a large positive expectation under the null, so the null travels with it.

    A two-level facet across six repeats explains 20% of the within-item variance by chance alone,
    and 0.18 reported on its own reads as a finding. Either the clause quotes both numbers, or it
    says the facet did not clear its own null, or it reports a main effect instead, which is a
    different statistic and does not have the problem.
    """
    from reward_lens.measure.metrology.distribution import facet_effects, facet_shares

    rng = np.random.default_rng(12)
    data = RepeatedScores(
        scores=rng.normal(size=(40, 6)), facets={"noise": np.array([0, 0, 0, 1, 1, 1])}
    )
    shares, null = facet_shares(data)
    assert null["noise"] == pytest.approx(1.0 / 5.0)
    clause = facet_clause(shares, null, facet_effects(data))
    quotes_a_share = "of the within-item variance" in clause
    assert (not quotes_a_share) or "null share" in clause or "systematic shift" in clause
    assert "null share" in clause, clause


def test_the_facet_clause_is_empty_when_nothing_was_varied() -> None:
    assert facet_clause({}, {}, {}) == ""


# ---------------------------------------------------------------------------
# The payloads
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [GraderScoreSigma, GraderFlipRate])
def test_the_payload_carries_the_baselines_key_and_the_says_sentence(cls: type) -> None:
    inst = cls(_panel())
    computed = inst.compute()
    payload = inst.payload(computed)
    assert payload["baselines"] == {"baseline.assume_determinism": 0.0}
    assert payload["says"] == computed.says
    assert payload["grader"] == "synthetic"


@pytest.mark.parametrize("cls", [GraderScoreSigma, GraderFlipRate])
def test_the_reading_round_trips_through_the_value_codec(cls: type) -> None:
    """Registered payloads decode back to their own type rather than to a bare dict.

    Two ids in two stores is the point of the split, and a payload that decoded to a dict would
    lose the type that says which id it is.
    """
    computed = cls(_panel()).compute()
    codec = ValueCodec()
    back = codec.decode(codec.encode(computed))
    assert type(back) is type(computed)
    assert back.says == computed.says


def test_the_spread_quotes_the_typical_item_against_the_worst_decile() -> None:
    """A uniformly noisy grader and a grader with a bad decile are different objects.

    Both numbers are always quoted, and on a uniformly noisy grader they coincide, which is the
    reading rather than an omission. The pooled sigma is a root-mean-square over items, so it sits
    above the median whenever the per-item spreads are unequal and a clause gated on "p90 above the
    pool" would almost never fire even on a grader with a genuinely bad decile.
    """
    uniform = np.tile(np.array([0.0, 1.0]), (20, 1)) + np.arange(20.0)[:, None]
    spread = GraderScoreSigma(RepeatedScores(scores=uniform)).compute()
    assert isinstance(spread, ScoreSpread)
    assert spread.sigma_quantiles["p90"] == pytest.approx(spread.sigma_quantiles["p50"])
    assert spread.sigma_quantiles["p90"] == pytest.approx(spread.sigma)
    assert "A typical item costs 0.707 and the worst decile costs 0.707" in spread.says

    heavy = uniform.copy()
    heavy[:2, 1] += 40.0
    with_tail = GraderScoreSigma(RepeatedScores(scores=heavy)).compute()
    assert with_tail.sigma_quantiles["p90"] > 4.0 * with_tail.sigma_quantiles["p50"]
    assert "the worst decile costs" in with_tail.says

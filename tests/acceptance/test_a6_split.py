"""Acceptance for A6's quantity split: one id over two things that transform differently.

The clause this file discharges: *both quantities are declared by an instrument; `lint_instrument`
passes on each; the generated invariance test passes for each under its own group and relation, with
sigma's covariance under `reward.affine` asserted at the correct power and the flip rate's invariance
asserted as equality; and both produce a reading on a real grader.*

`grader.score_distribution` carried the spread and the flip rate under one id. It could not. Rescale
a grader's output by `a` and the standard deviation scales by `|a|` while the flip rate does not move
at all, so one `invariance_group` field would have to be wrong about one of them, and one
`Instrument.quantity` field would have to be wrong the same way one level up. This file is the
executable form of that argument: the same estimator, the same data, two declarations, and the two
declarations disagree on exactly the thing the split is about.

**The subject is the one series A already used**, so the split is comparable with what shipped
rather than with a new measurement. `skywork-critic` on 1,000 preference pairs from the campaign store,
scored in the presented order and again in the swapped order. The verdict readout is a margin, so
swapping the two responses negates it for a judge with no position preference and ``-swapped`` is the
same measurement made a second time. Series A read sigma 1.3341 and a flip rate of 9.0% off that
design through the superseded single instrument; the two new instruments reproduce both exactly, because
they call the same estimators on the same array.

**The affine element used here is real rather than drawn.** The same store holds eleven open reward
models scored over one shared bank of 7,052 responses, and their raw standard deviations span more
than two orders of magnitude. The ratio between the widest and the narrowest is an element of
`reward.affine` that somebody's choice of output scale actually produced, and applying it to the
judge's margins is what a cross-grader comparison does implicitly. Sigma moves by exactly that
factor. The flip rate does not move at all.
"""

from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest

from reward_lens.core.invariance import (
    COVARIANT_LINEAR,
    INVARIANT,
    InvariancePayload,
    Relation,
    check_invariance,
    parse_group_field,
    resolve_relation,
)
from reward_lens.core.quantity import QUANTITIES
from reward_lens.core.reading import Refusal
from reward_lens.measure.base import Context, lint_instrument
from reward_lens.measure.metrology.distribution import (
    GraderStochasticity,
    RepeatedScores,
    flip_rate_from_scores,
    sigma_from_scores,
)
from reward_lens.measure.metrology.stochasticity import (
    AFFINE_RELATIONS,
    GraderFlipRate,
    GraderScoreSigma,
    ScoreSpread,
    VerdictStability,
    stochasticity_profile,
)

#: The campaign evidence store, which is not in this repository. There is no default: point
#: ``REWARD_LENS_CAMPAIGN_STORE`` at the store directory or the tests that need it skip.
_CAMPAIGN_ENV = os.environ.get("REWARD_LENS_CAMPAIGN_STORE")
CAMPAIGN = pathlib.Path(_CAMPAIGN_ENV) if _CAMPAIGN_ENV else None
SIDECARS = (CAMPAIGN.parent.parent / "store",) if CAMPAIGN is not None else ()

#: `hackfore-flagged` is not a reward model: one of its two rb2-full rows is byte-identical to
#: `grm-gemma2-2b`, so counting it would put a duplicated column into the panel. Same exclusion as
#: series A, for the same reason.
NOT_A_GRADER = {"hackfore-flagged"}


# ---------------------------------------------------------------------------
# The real store
# ---------------------------------------------------------------------------


def _store():
    if CAMPAIGN is None or not (CAMPAIGN / "evidence.jsonl").exists():
        pytest.skip("no campaign evidence store; set REWARD_LENS_CAMPAIGN_STORE")
    from reward_lens.record.convert import CampaignStore

    return CampaignStore(CAMPAIGN, sidecar_dirs=[p for p in SIDECARS if p.exists()])


def _flat_bank(store, base: str, roster: str) -> np.ndarray:
    """Reassemble a partitioned flat score bank in part order."""
    parts = []
    for row in store.by_observable("campaign.scores"):
        name = row.slice_name or ""
        if name.split("::part")[0] == base and row.roster_key == roster:
            value = store.value(row)
            parts.append(
                (int(value["meta"]["part"]), np.asarray(value["scores"], dtype=np.float64))
            )
    if not parts:
        pytest.skip(f"slice {base!r} for {roster!r} is not in this store")
    parts.sort(key=lambda t: t[0])
    return np.concatenate([p for _, p in parts])


@pytest.fixture(scope="module")
def critic_order_facet() -> RepeatedScores:
    """`skywork-critic` on 1,000 preference pairs, scored in both presentation orders.

    Identical construction to series A's fixture, deliberately: the point of this file is that the two
    new instruments read the same design and return the same numbers as the one they replace.
    """
    store = _store()
    original = _flat_bank(store, "judge-pairs-1000", "skywork-critic")
    swapped = _flat_bank(store, "judge-pairs-1000::swapped", "skywork-critic")
    assert original.shape == swapped.shape == (1000,)
    return RepeatedScores(
        scores=np.stack([original, -swapped], axis=1),
        facets={"order": np.array([0, 1])},
        grader="skywork-critic",
        paired_occasions=True,
    )


@pytest.fixture(scope="module")
def real_scale_ratio() -> float:
    """The widest over the narrowest raw output scale among the eleven open reward models.

    An affine element nobody drew: it is the ratio between two real graders' choices of output
    scale on one shared bank of 7,052 responses. Comparing their sigmas without accounting for it is
    the coordinate artifact the covariant declaration exists to make visible.
    """
    store = _store()
    sds: dict[str, float] = {}
    sizes: set[int] = set()
    for row in store.by_observable("campaign.scores"):
        if row.bank != "rb2-full" or row.roster_key in NOT_A_GRADER:
            continue
        value = store.value(row)
        if value["layout"] != "best-of-4":
            continue
        scores = np.asarray(value["scores"], dtype=np.float64)
        sizes.add(int(scores.size))
        sds[row.roster_key] = float(scores.std(ddof=1))
    if len(sds) != 11:
        pytest.skip(f"expected the eleven-model rb2-full panel; found {len(sds)}")
    # Fully crossed: one score per (model, response), same responses for every model.
    assert sizes == {7052}, sizes
    ratio = max(sds.values()) / min(sds.values())
    print(
        f"\n[scale] eleven reward models on rb2-full, {sizes.pop()} responses each: raw sigma "
        f"{min(sds.values()):.4g} to {max(sds.values()):.4g}, ratio {ratio:.4f}x"
    )
    return ratio


# ---------------------------------------------------------------------------
# Clause 1: both quantities are declared by an instrument
# ---------------------------------------------------------------------------


def test_both_quantities_are_registered_and_each_is_declared_by_one_instrument() -> None:
    """The split, at the level the registry and the protocol can see it.

    Two ids, two instruments, one quantity each. `Instrument.quantity` is singular by design, so the
    only way to honour a split into two quantities is a second instrument. Attaching the second id
    to the first instrument as a spare attribute was the alternative and it is worse than doing
    nothing: `hodge.py` carries `secondary_quantity` that way and nothing in the kernel reads it, so
    a quantity declared there gets no lint and no generated invariance test, which are the two things
    a declaration is for.
    """
    assert "grader.score_sigma" in QUANTITIES
    assert "grader.flip_rate" in QUANTITIES
    declared = {inst.quantity: inst for inst in (GraderScoreSigma(), GraderFlipRate())}
    assert declared.keys() == {"grader.score_sigma", "grader.flip_rate"}
    assert declared["grader.score_sigma"].name == "GraderScoreSigma"
    assert declared["grader.flip_rate"].name == "GraderFlipRate"
    # The registry puts both rows under the group that separates them, and the instruments carry the
    # relations that differ. That division is E13: the group is a property of the quantity, the
    # relation is a property of the instrument.
    assert QUANTITIES.get("grader.score_sigma").invariance == "reward.affine"
    assert QUANTITIES.get("grader.flip_rate").invariance == "reward.affine"


@pytest.mark.parametrize("instrument", [GraderScoreSigma(), GraderFlipRate()], ids=lambda i: i.name)
def test_lint_instrument_passes_on_each(instrument: object) -> None:
    """Standing rule 3. A registered quantity, a non-empty baselines list, an envelope, a group."""
    assert lint_instrument(instrument) == []


@pytest.mark.parametrize("instrument", [GraderScoreSigma(), GraderFlipRate()], ids=lambda i: i.name)
def test_each_declares_exactly_one_baseline_and_it_is_the_one_assay_prints(
    instrument: object,
) -> None:
    """E26's check, on this package's own field.

    The design says "Base assume determinism, and show the error that induces". That is one
    baseline whose second clause is an instruction about reporting, and splitting it on the comma
    would put a phantom member into the exact field `lint_instrument` reads, where it would read as
    more rigour than the instrument has.
    """
    assert instrument.baselines == ("baseline.assume_determinism",)  # type: ignore[attr-defined]


@pytest.mark.parametrize("instrument", [GraderScoreSigma(), GraderFlipRate()], ids=lambda i: i.name)
def test_an_instrument_with_no_data_refuses_rather_than_raising(instrument: object) -> None:
    """Standing rule 1, at the one place every instrument in this package shares."""
    out = instrument.compute()  # type: ignore[attr-defined]
    assert isinstance(out, Refusal)
    assert out.remedy.strip()


# ---------------------------------------------------------------------------
# Clause 2: the generated invariance test, under each instrument's own relation
# ---------------------------------------------------------------------------


def _payload(scores: np.ndarray) -> InvariancePayload:
    return InvariancePayload(
        scores=scores.ravel(),
        group_ids=np.repeat(np.arange(scores.shape[0]), scores.shape[1]),
    )


#: A panel with a real item effect on top of the within-item spread, so permuting repeats within an
#: item and permuting them across items are visibly different operations.
def _panel(seed: int = 11, n_items: int = 24, n_repeats: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_items, n_repeats)) + rng.normal(size=(n_items, 1)) * 3.0


PROBES = {
    "GraderScoreSigma": sigma_from_scores,
    "GraderFlipRate": flip_rate_from_scores,
}


@pytest.mark.parametrize("instrument", [GraderScoreSigma(), GraderFlipRate()], ids=lambda i: i.name)
def test_the_generated_invariance_test_passes_under_every_group_each_declares(
    instrument: object,
) -> None:
    """Standing rule 4, run from the declaration rather than from a hand-picked relation.

    The relation comes out of `resolve_relation`, which reads the instrument's own mapping, so this
    test cannot pass by being handed the answer. Both instruments declare both groups and the
    mapping differs on exactly one entry, which is the split.
    """
    scores = _panel()
    payload = _payload(scores)
    probe = PROBES[instrument.name]  # type: ignore[attr-defined]
    groups = parse_group_field(instrument.invariance)  # type: ignore[attr-defined]
    assert groups == ["reward.affine", "group.permutation"]

    for gid in groups:
        relation = resolve_relation(instrument, gid)
        report = check_invariance(
            instrument,
            gid,
            payload,
            n=64,
            relation=relation,
            run=lambda _i, p: probe(np.asarray(p.scores), scores.shape[1]),
        )
        assert report.passed, report.render()
        assert report.relation == relation
        print(f"\n[{instrument.name}] {report.render()}")  # type: ignore[attr-defined]


def test_sigma_declares_covariance_at_weight_one_and_the_two_wrong_answers_fail() -> None:
    """The power, asserted by ruling out its neighbours. This is what E13 bought.

    A status on the group and not on the instrument leaves nowhere to put the power, so the only
    assertion a generated test can make is equality, and every covariant instrument then declares
    itself invariant and passes. Both wrong declarations are run here so the passing one is a
    measurement rather than a default: invariant misses by the size of the reading itself, and
    weight 2 misses by more, because `Var` scales by `a**2` and `sigma` does not.
    """
    scores = _panel()
    payload = _payload(scores)
    probe = lambda p: sigma_from_scores(np.asarray(p.scores), scores.shape[1])  # noqa: E731

    right = check_invariance(
        GraderScoreSigma(),
        "reward.affine",
        payload,
        n=32,
        relation=COVARIANT_LINEAR,
        run=lambda _i, p: probe(p),
    )
    assert right.passed, right.render()
    assert right.relation.status == "covariant"
    assert right.relation.weight == 1.0

    for wrong in (INVARIANT, Relation("covariant", weight=2.0), Relation("covariant", weight=0.5)):
        report = check_invariance(
            GraderScoreSigma(),
            "reward.affine",
            payload,
            n=16,
            relation=wrong,
            run=lambda _i, p: probe(p),
        )
        assert not report.passed, (
            f"a sigma declared {wrong.status} at weight {wrong.weight} passed its own test, so the "
            f"generated test is not asserting the power"
        )
        print(
            f"\n[weight] {wrong.status} weight {wrong.weight}: max deviation "
            f"{report.max_deviation:.4g} against tol {report.tol:.4g}"
        )
    assert right.max_deviation < 1e-12


def test_the_flip_rate_invariance_is_asserted_as_equality_and_is_exact() -> None:
    """Equality, not a tolerance that would hide a small covariance.

    The verdict is the sign of a difference and `(a·r_i + b) - (a·r_j + b) = a·(r_i - r_j)`, so a
    positive `a` preserves every sign. The reading is a count statistic over those signs, so it does
    not merely stay close under the group, it is bit-identical, and asserting exact equality is
    therefore a stronger and more honest check than `report.passed`.
    """
    scores = _panel()
    payload = _payload(scores)
    report = check_invariance(
        GraderFlipRate(),
        "reward.affine",
        payload,
        n=64,
        relation=INVARIANT,
        run=lambda _i, p: flip_rate_from_scores(np.asarray(p.scores), scores.shape[1]),
    )
    assert report.passed, report.render()
    assert report.relation.status == "invariant"
    assert report.max_deviation == 0.0, (
        "the flip rate moved under an affine rescaling. It is a function of the signs of pairwise "
        "differences, so any movement is cancellation crossing zero on a near-tied pair, and the "
        "tie rate is where that would show."
    )


def test_the_two_declarations_differ_on_exactly_one_entry() -> None:
    """The whole split, in two lines. If these ever agree, one id would have been enough."""
    assert AFFINE_RELATIONS["grader.score_sigma"] == COVARIANT_LINEAR
    assert AFFINE_RELATIONS["grader.flip_rate"] == INVARIANT
    assert resolve_relation(GraderScoreSigma(), "reward.affine") == COVARIANT_LINEAR
    assert resolve_relation(GraderFlipRate(), "reward.affine") == INVARIANT
    assert resolve_relation(GraderScoreSigma(), "group.permutation") == INVARIANT
    assert resolve_relation(GraderFlipRate(), "group.permutation") == INVARIANT


# ---------------------------------------------------------------------------
# Clause 3: both produce a reading on a real grader
# ---------------------------------------------------------------------------


def test_both_readings_on_a_real_judge(critic_order_facet: RepeatedScores) -> None:
    """`skywork-critic`, 1,000 real preference pairs, both halves of A6.

    Rung 1 is the part worth reading twice. Eta-squared cannot separate the order facet from
    occasion noise on a design with one observation per cell and reports 1.00 against a null of 1.00
    for exactly that reason. The main effect is still identifiable because it is estimated across
    items, and it is what decides the remedy: a systematic shift is removed by presenting both orders
    and averaging, and noise is not.
    """
    spread, verdict = stochasticity_profile(critic_order_facet, max_pairs=20_000, seed=0)
    assert isinstance(spread, ScoreSpread), spread
    assert isinstance(verdict, VerdictStability), verdict

    print(f"\n[grader.score_sigma] {spread.says}")
    print(
        f"[grader.score_sigma] sigma {spread.sigma:.4f} on {spread.n_scores} draws over "
        f"{spread.n_items_used} items; per-item quantiles p50 {spread.sigma_quantiles['p50']:.4f} "
        f"p90 {spread.sigma_quantiles['p90']:.4f} max {spread.sigma_quantiles['max']:.4f}"
    )
    print(f"[grader.flip_rate]  {verdict.says}")
    print(
        f"[grader.flip_rate]  flip {verdict.flip_rate:.4f} over {verdict.n_pairs} pairs at "
        f"{verdict.combinations_per_pair} combinations each; ties {verdict.tie_rate:.4f}; "
        f"undetermined {verdict.undetermined_pairs:.4f}"
    )

    assert spread.n_items == 1000
    assert spread.n_repeats == 2
    assert spread.deterministic is False
    assert spread.sigma > 0.0
    assert 0.0 < verdict.flip_rate <= 0.5
    assert verdict.n_pairs == 20_000
    assert verdict.combinations_per_pair == 2
    assert verdict.never_flipped is False
    # Fully replicated design: no pair is unanimous because the design gave it no choice.
    assert verdict.unanimous_by_construction == 0.0
    assert verdict.singleton_items == 0

    # The direct, assumption-free version of the flip rate: how often the sign of the verdict
    # survives the swap. Over two occasions a flipped pair is exactly a pair whose two verdicts
    # disagree, so the two agree to within the tie rate.
    original = critic_order_facet.scores[:, 0]
    swapped_back = critic_order_facet.scores[:, 1]
    sign_disagreement = float(np.mean(np.sign(original) != np.sign(swapped_back)))
    assert verdict.flip_rate == pytest.approx(sign_disagreement / 2.0, abs=0.02)

    order = spread.facet_effects["order"]
    assert order.significant is True
    assert spread.facet_shares["order"] == pytest.approx(spread.facet_null["order"], abs=1e-6)


def test_the_two_readings_reproduce_the_superseded_single_instrument_exactly(
    critic_order_facet: RepeatedScores,
) -> None:
    """The split changes what is declared, not what is measured.

    Bit-for-bit on every number both carry, because both call the same estimators on the same array.
    What differs is the `says` sentence, which each instrument now writes for its own reading alone,
    and the declarations, which is the point.
    """
    old = GraderStochasticity(critic_order_facet, max_pairs=20_000, seed=0).compute()
    assert not isinstance(old, Refusal), old
    spread, verdict = stochasticity_profile(critic_order_facet, max_pairs=20_000, seed=0)

    assert spread.sigma == old.sigma
    assert spread.variance == old.variance
    assert spread.df == old.df
    assert spread.deterministic == old.deterministic
    assert spread.n_scores == old.n_scores
    assert dict(spread.sigma_quantiles) == dict(old.sigma_quantiles)
    assert dict(spread.facet_shares) == dict(old.facet_shares)
    assert dict(spread.facet_null) == dict(old.facet_null)

    assert verdict.flip_rate == old.flip_rate
    assert verdict.tie_rate == old.tie_rate
    assert verdict.n_pairs == old.n_pairs
    assert verdict.repeats_needed == old.repeats_needed
    assert verdict.achieved_agreement == old.achieved_agreement
    assert verdict.combinations_per_pair == old.combinations_per_pair
    assert verdict.undetermined_pairs == old.undetermined_pairs

    print(
        f"\n[reproduce] sigma {spread.sigma:.6f} and flip {verdict.flip_rate:.6f} match "
        f"GraderStochasticity exactly on {old.n_scores} real scores"
    )


def test_on_the_real_judge_a_real_grader_scale_moves_sigma_and_leaves_the_flip_rate_alone(
    critic_order_facet: RepeatedScores, real_scale_ratio: float
) -> None:
    """The split, demonstrated with an affine element that a real roster produced.

    Not a draw from LogUniform(0.1, 10): the eleven open reward models in this store chose output
    scales spanning this ratio, so a reader comparing two of their sigmas is applying it whether or
    not they meant to. Sigma scales by exactly the factor. The flip rate is bit-identical.
    """
    rescaled = RepeatedScores(
        scores=real_scale_ratio * critic_order_facet.scores + 3.7,
        facets=dict(critic_order_facet.facets),
        grader=critic_order_facet.grader,
        paired_occasions=critic_order_facet.paired_occasions,
    )
    base_spread, base_verdict = stochasticity_profile(critic_order_facet, seed=0)
    new_spread, new_verdict = stochasticity_profile(rescaled, seed=0)

    assert new_spread.sigma == pytest.approx(real_scale_ratio * base_spread.sigma, rel=1e-12)
    assert new_verdict.flip_rate == base_verdict.flip_rate
    print(
        f"\n[real affine] a = {real_scale_ratio:.4f}: sigma {base_spread.sigma:.4f} -> "
        f"{new_spread.sigma:.4f} (factor {new_spread.sigma / base_spread.sigma:.4f}); "
        f"flip rate {base_verdict.flip_rate:.6f} -> {new_verdict.flip_rate:.6f}"
    )


def test_both_emit_gated_evidence_carrying_their_own_quantity(
    critic_order_facet: RepeatedScores,
) -> None:
    """The reading reaches a store attributed to the right id, which is what the split is for.

    E35: `Context.emit` forwards the instrument's quantity, so a per-grader sigma and a
    dimensionless flip rate can no longer be ranked against each other by anything that reads a
    store. Before the split they carried one id and nothing could tell them apart.
    """
    ctx = Context()
    sigma_reading = GraderScoreSigma(critic_order_facet).estimate(ctx)
    flip_reading = GraderFlipRate(critic_order_facet).estimate(ctx)
    assert not isinstance(sigma_reading, Refusal), sigma_reading
    assert not isinstance(flip_reading, Refusal), flip_reading
    assert sigma_reading.quantity == "grader.score_sigma"
    assert flip_reading.quantity == "grader.flip_rate"
    assert sigma_reading.observable == "GraderScoreSigma"
    assert flip_reading.observable == "GraderFlipRate"
    assert sigma_reading.value["baselines"] == {"baseline.assume_determinism": 0.0}
    assert flip_reading.value["baselines"] == {"baseline.assume_determinism": 0.0}
    # Gate 2: the spread is covariant, so a cross-grader comparison of it needs a frame and the
    # flip rate's does not. That distinction did not exist while both were one instrument.
    assert GraderScoreSigma().gauge_status.value == "covariant"
    assert GraderFlipRate().gauge_status.value == "invariant"

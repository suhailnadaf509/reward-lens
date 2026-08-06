"""Acceptance: the reference certificate's trust cap, and VERIF-PRM re-adjudicated.

The clause this file discharges, in two halves:

    *An organism family reports all three uncertainty terms or `None` with the trust cap applied;
    and `VERIF-PRM`'s stored artifacts are re-scored against a position-stratified null and the
    number is published either way.*

**Half one.** Every prior version of this project calibrated against planted organisms and reported
the result as a scorecard. A planted organism is a reference material, and no reference material in
this field ships with an uncertainty on its assigned value. The rule that fixes it is one line and
it has to be enforced rather than documented: a reference whose homogeneity was never measured caps
the trust ladder at `CALIBRATED`, so a preregistered result calibrated against an uncharacterised
organism does not reach `REGISTERED`. Freezing a prediction against a ruler of unknown length does
not make the reading better; it makes the prediction precise about something unmeasured.

The campaign's own cal-transfer organisms are the test case, and they fail it. Three doses, one
seed, no stability re-measurement: `u_homogeneity is None`, `u_stability is None`, and the trust cap
fires on a reading taken under a frozen study.

**Half two.** `VERIF-PRM` reported a dense localisation AUC of 0.2821, below chance, and adjudicated
the hypothesis refuted. Below chance is information. A localisation AUC pooled over every candidate
step scores the localiser against a uniform prior over positions, and an earliest-error step is not
uniformly distributed over a solution, so the pooled number mixes the localiser's discrimination
with an agreement between two position distributions. This half re-scores the stored artifacts
against a position-stratified null and asserts the number is produced, whichever way it falls.

The store is the only copy of a $17.73 experiment and is outside this repository, so the half that
reads it skips when it is absent and the synthetic half runs unconditionally. The measured values
are pinned below: a change in them is a test failure rather than a silent drift.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from reward_lens.core.quantity import load_quantities
from reward_lens.core.reference import MatrixDescription
from reward_lens.core.types import TrustLevel
from reward_lens.measure.base import Context
from reward_lens.measure.labels import (
    Plant,
    PositionStratifiedNull,
    ReferenceCertificate,
    StabilityPoint,
    campaign_plants,
    certify,
    load_step_scores,
    register_all,
    rescore_against_position,
)
from reward_lens.measure.labels.position import CHANCE

load_quantities()
register_all()

#: The campaign evidence store, which is not in this repository. There is no default: point
#: ``REWARD_LENS_CAMPAIGN_STORE`` at the store directory or the tests that need it skip.
_CAMPAIGN_ENV = os.environ.get("REWARD_LENS_CAMPAIGN_STORE")
STORE = Path(_CAMPAIGN_ENV) if _CAMPAIGN_ENV else None

has_store = pytest.mark.skipif(
    STORE is None or not (STORE / "evidence.jsonl").exists(),
    reason="no campaign evidence store; set REWARD_LENS_CAMPAIGN_STORE",
)

# -- What VERIF-PRM stored, verbatim from `campaign.result.VERIF-PRM`. -------------------------
STORED_DENSE_LOCALIZATION_AUC = 0.2821441611813727
STORED_POSITION_BASELINE_AUC = 0.3445858324230701
STORED_N_ITEMS_WITH_ERROR = 2221
STORED_N_POOLED_STEPS = 19783

# -- What the re-scoring produced, on 2026-08-01, at n_boot=1000 and seed=0. -------------------
MEASURED_STRATIFIED_AUC = 0.21901567736283137
MEASURED_EXACT_POSITION_AUC = 0.21465277110748246
MEASURED_CI = (0.20688864767754941, 0.2314496458393615)


# ---------------------------------------------------------------------------
# Half one: an organism family reports all three terms or None, with the cap
# ---------------------------------------------------------------------------


def _plants(seeds: tuple[int, ...], *, pipeline: str = "acceptance") -> list[Plant]:
    """A family planted at three doses, with realised doses that scatter about the nominal ones.

    Scatter rather than a perfect line, because a perfect line has zero residual and therefore zero
    characterisation uncertainty, which is arithmetic rather than a characterised reference. The
    fit says so in its note and a test that leaned on the noiseless case would be testing nothing.
    """
    rng = np.random.default_rng(0)
    return [
        Plant(dose, dose + 0.01 * rng.normal(), seed=seed, pipeline=pipeline)
        for dose in (0.6, 0.8, 1.0)
        for seed in seeds
    ]


def test_a_single_seed_family_reports_none_and_the_cap_fires():
    """The mandatory baseline: one seed, nominal dose. Two terms are None and trust is capped."""
    inst = ReferenceCertificate(
        _plants((0,), pipeline="baseline"),
        reference_id="single-seed-nominal-dose",
        assigned_dose=0.8,
        matrix=MatrixDescription(system="acceptance"),
    )
    certificate = inst.certificate()

    # All three terms are reported. Two of them are reported as None, which is a state and not a gap.
    assert certificate.u_characterisation > 0.0
    assert certificate.u_homogeneity is None
    assert certificate.u_stability is None
    assert certificate.material().uncharacterised == ("u_homogeneity", "u_stability")
    assert "reference uncertainty not characterised" in certificate.material().status_line()

    # And the cap fires, under a frozen study, which is where it has to fire to mean anything.
    reading = inst.estimate(Context(study="study:acceptance@v1#0"))
    assert reading.trust is TrustLevel.CALIBRATED
    assert reading.reference == "single-seed-nominal-dose"


def test_a_fully_characterised_family_reports_three_numbers_and_lifts_the_cap():
    inst = ReferenceCertificate(
        _plants((0, 1, 2)),
        reference_id="three-seed-with-stability",
        assigned_dose=0.8,
        matrix=MatrixDescription(system="acceptance"),
        stability=[
            StabilityPoint(steps=t, response=0.8 + 1e-4 * i) for i, t in enumerate((0, 100, 200))
        ],
        horizon_steps=200,
    )
    certificate = inst.certificate()
    assert certificate.u_characterisation > 0.0
    assert certificate.u_homogeneity is not None
    assert certificate.u_stability is not None
    assert certificate.material().is_certified
    assert certificate.material().u_crm is not None
    assert certificate.rung == 2

    reading = inst.estimate(Context(study="study:acceptance@v1#0"))
    assert reading.trust is TrustLevel.REGISTERED


def test_calibrating_against_an_uncertified_reference_refuses_with_a_bound():
    """The other half of the rule: describing an uncertified reference is fine, calibrating is not."""
    inst = ReferenceCertificate(
        _plants((0,)),
        reference_id="single-seed-nominal-dose",
        assigned_dose=0.8,
        matrix=MatrixDescription(system="acceptance"),
        require_certified=True,
    )
    refusal = inst.estimate(Context())
    assert refusal.reason.name == "REFERENCE_UNCERTIFIED"
    assert "at least three seeds" in refusal.remedy
    assert "uncalibrated ruler" in refusal.remedy
    # A refusal that still answers: u_CRM is not computable and it is at least this bad.
    assert refusal.is_bounded
    assert refusal.partial.value > 0.0
    assert refusal.partial.reference == "single-seed-nominal-dose"
    # The cap travels onto the bound too. It is not binding here, because a bound taken outside a
    # frozen study is EXPLORATORY on its own account, and the assertion is that the cap holds
    # rather than that it is what set the level.
    assert refusal.partial.trust <= TrustLevel.CALIBRATED


@has_store
def test_the_campaigns_own_organisms_fail_the_clause_and_are_capped():
    """The case this instrument was built for: the project's own CAL-TRANSFER reference."""
    by_arm = campaign_plants(str(STORE))
    assert set(by_arm) == {"cpu", "real"}

    for arm, plants in by_arm.items():
        assert len({p.seed for p in plants}) == 1, "the campaign planted one seed per dose"
        certificate = certify(
            plants,
            reference_id=f"caltransfer-organisms::{arm}",
            assigned_dose=1.0,
            matrix=MatrixDescription(system="skywork-v2-qwen3-0.6b", scale=arm),
        )
        assert certificate.u_characterisation > 0.0
        assert certificate.u_homogeneity is None
        assert certificate.u_stability is None
        assert certificate.material().u_crm is None
        assert certificate.material().trust_cap() is TrustLevel.CALIBRATED

        # Rung 0, so nothing else carries the individual-plant term and `u_characterisation` is
        # the full inverse-prediction uncertainty. It used to be the bare ratio `s_resid/|slope|`,
        # which on this sweep is 35.40% smaller: the assigned dose is 1.0 and the sweep is centred
        # at 0.8, so both the `1` and the leverage term are omitted at once.
        ratio = certificate.fit.sensitivity_scaled_residual
        assert certificate.u_characterisation == pytest.approx(CAMPAIGN_U_CHAR_AFTER[arm], rel=1e-9)
        assert ratio == pytest.approx(CAMPAIGN_U_CHAR_BEFORE[arm], rel=1e-9)
        assert certificate.u_characterisation / ratio == pytest.approx(1.35399, rel=1e-4)

        # And the degrees of freedom now travel with the term, which is what makes the coverage
        # factor `t_0.975(1) = 12.706` rather than the conventional 2 on the reference path.
        assert certificate.material().dof_characterisation == pytest.approx(1.0)

        inst = ReferenceCertificate(
            plants,
            reference_id=f"caltransfer-organisms::{arm}",
            assigned_dose=1.0,
            matrix=MatrixDescription(system="skywork-v2-qwen3-0.6b", scale=arm),
        )
        reading = inst.estimate(Context(study="study:campaign-cal-transfer@v1#0"))
        assert reading.trust is TrustLevel.CALIBRATED


@has_store
def test_the_reference_is_worse_characterised_in_the_matrix_it_is_applied_in():
    """A matrix effect, measured: the same dose sweep is pinned down four times less well on the target.

    Not a threshold anyone chose. The two numbers come out of the same fit on the same three doses,
    and the assertion is only that the clean matrix beats the real one, which is the direction
    the matrix-effect diagnosis predicts.
    """
    by_arm = campaign_plants(str(STORE))
    u_char = {}
    for arm, plants in by_arm.items():
        certificate = certify(
            plants,
            reference_id=arm,
            assigned_dose=1.0,
            matrix=MatrixDescription(system="skywork-v2-qwen3-0.6b", scale=arm),
        )
        u_char[arm] = certificate.u_characterisation
    assert u_char["cpu"] < u_char["real"]


# ---------------------------------------------------------------------------
# Half two: VERIF-PRM re-scored against a position-stratified null
# ---------------------------------------------------------------------------


VERDICTS = {
    "indistinguishable from chance",
    "localises",
    "localises, and position inflated the pooled reading",
    "position confound",
    "below chance, and not from position",
}

# -- What the confound interval reads on the same run. -----------------------------------------
# -- The cal-transfer certificate, before and after the inverse-prediction correction. ---------
CAMPAIGN_U_CHAR_BEFORE = {"cpu": 0.017921695486774216, "real": 0.07330814893467337}
CAMPAIGN_U_CHAR_AFTER = {"cpu": 0.024266090401790785, "real": 0.09925970288634321}

MEASURED_CONFOUND_SIZE = -0.06312848381854133
MEASURED_CONFOUND_CI = (-0.06729916726487668, -0.05862347654637756)
STORED_DENSE_LOCALIZATION_AUC_RAW = 0.2815512146302947


@has_store
def test_verif_prm_is_rescored_and_the_number_is_published_either_way():
    series = load_step_scores(
        str(STORE),
        observable="campaign.prm.steps",
        slice_name="processbench-full",
        roster_key="qwen-prm",
        higher_is_positive=False,
    )
    prior = rescore_against_position(
        series, n_boot=1000, seed=0, source="VERIF-PRM qwen-prm processbench-full"
    )

    # The re-scoring reproduces what VERIF-PRM stored, exactly. Without this the rest is a
    # measurement of a different thing that happens to be nearby.
    assert prior.uniform_auc == pytest.approx(STORED_DENSE_LOCALIZATION_AUC, abs=1e-15)
    assert prior.position_only_auc == pytest.approx(STORED_POSITION_BASELINE_AUC, abs=1e-15)
    assert prior.n_items == STORED_N_ITEMS_WITH_ERROR
    assert prior.n_candidates == STORED_N_POOLED_STEPS

    # The control that says the stratification controlled what it claims to: within an
    # exact-position stratum every position comparison is a tie, so the position-only AUC is 0.5.
    assert prior.exact_position_check == pytest.approx(CHANCE, abs=1e-12)

    # The number is produced, with an interval, and it is published whichever way it falls.
    assert np.isfinite(prior.stratified_auc)
    assert np.isfinite(prior.ci_low) and np.isfinite(prior.ci_high)
    assert prior.ci_low <= prior.stratified_auc <= prior.ci_high
    assert prior.verdict in VERDICTS
    assert prior.interpretation
    assert prior.stratified_auc == pytest.approx(MEASURED_STRATIFIED_AUC, abs=1e-12)
    assert prior.exact_position_auc == pytest.approx(MEASURED_EXACT_POSITION_AUC, abs=1e-12)
    assert (prior.ci_low, prior.ci_high) == pytest.approx(MEASURED_CI, abs=1e-12)

    # What was actually found, pinned so a change in it is a failure rather than a drift: the
    # conditioned reading moved further from chance, not toward it, so the pooled number was not a
    # position artifact and the orientation of the readout is the remaining explanation.
    assert prior.stratified_auc < prior.uniform_auc < CHANCE
    assert prior.verdict == "below chance, and not from position"
    assert prior.inverted_stratified_auc == pytest.approx(1.0 - prior.stratified_auc)

    # X9's answer does not move under the `_verdict` reordering that made the above-chance
    # confound reachable. The reordering only changes readings whose conditioned statistic is
    # above chance, and this one is 0.2190. Pinned explicitly so a future change to the rule has
    # to confront the published answer rather than drift past it.
    assert abs(prior.stratified_auc - CHANCE) > abs(prior.uniform_auc - CHANCE)
    assert prior.confound_size == pytest.approx(MEASURED_CONFOUND_SIZE, abs=1e-12)
    assert prior.confound_ci == pytest.approx(MEASURED_CONFOUND_CI, abs=1e-12)
    assert prior.confound_ci[1] < 0.0, "conditioning moved the reading away from chance, not toward"

    # And which scores the pooled statistic was computed on, now that both are reported. The
    # standardised one is what VERIF-PRM stored; the raw one is what a study that pools raw step
    # scores would publish, and on this artifact they differ by 5.9e-4.
    assert prior.uniform_auc_raw == pytest.approx(STORED_DENSE_LOCALIZATION_AUC_RAW, abs=1e-15)
    assert abs(prior.uniform_auc - prior.uniform_auc_raw) == pytest.approx(5.93e-4, abs=1e-5)


@has_store
def test_the_rescored_number_reaches_the_evidence_with_its_comparators():
    """Published means it lands on the reading with the baselines a card prints beside it."""
    series = load_step_scores(
        str(STORE),
        observable="campaign.prm.steps",
        slice_name="processbench-full",
        roster_key="qwen-prm",
        higher_is_positive=False,
    )
    reading = PositionStratifiedNull(series, n_boot=100).estimate(Context())
    assert reading.quantity == "labels.position_prior"
    assert reading.baselines["baseline.uniform_prior"] == pytest.approx(
        STORED_DENSE_LOCALIZATION_AUC, abs=1e-15
    )
    assert reading.baselines["baseline.position_only"] == pytest.approx(
        STORED_POSITION_BASELINE_AUC, abs=1e-15
    )
    assert reading.uncertainty.ci_low is not None
    assert reading.value.verdict in VERDICTS
    assert "0.2821" in reading.value.render()


def test_the_three_explanations_for_a_below_chance_reading_are_distinguished():
    """The rule that makes half two mean something, on cases where the answer is known.

    Three synthetic localisers, all scoring below chance under a uniform prior over positions, and
    the three explanations the instrument has to tell apart.
    """
    rng = np.random.default_rng(0)

    def series(kind: str, n_items: int = 400):
        values, offsets, labels = [], [0], []
        for _ in range(n_items):
            k = int(rng.integers(6, 10))
            position = np.arange(k) / (k - 1)
            if kind == "position_only":
                # The label sits early and the statistic falls with position, so the detector
                # statistic (the negated score) rises with position and lands below chance from
                # position alone, with nothing else in it.
                step = -position + 0.05 * rng.normal(size=k)
                err = int(rng.integers(0, max(2, k // 3)))
            elif kind == "noise":
                step = rng.normal(size=k)
                err = int(rng.integers(0, k))
            else:  # a real but inverted localiser
                step = rng.normal(0.0, 0.3, k)
                err = int(rng.integers(0, k))
                step[err] += 3.0
            values.extend(step.tolist())
            offsets.append(len(values))
            labels.append(err)
        from reward_lens.measure.labels import LocalisationSeries

        return LocalisationSeries(
            values=np.asarray(values),
            offsets=np.asarray(offsets),
            labels=np.asarray(labels),
            higher_is_positive=False,
        )

    confound = rescore_against_position(series("position_only"), n_boot=200)
    assert confound.uniform_auc < CHANCE
    assert confound.verdict in {"position confound", "indistinguishable from chance"}
    assert abs(confound.stratified_auc - CHANCE) < abs(confound.uniform_auc - CHANCE)

    noise = rescore_against_position(series("noise"), n_boot=200)
    assert noise.verdict == "indistinguishable from chance"

    inverted = rescore_against_position(series("inverted"), n_boot=200)
    assert inverted.uniform_auc < CHANCE
    assert inverted.verdict == "below chance, and not from position"
    assert inverted.inverted_stratified_auc > 0.9

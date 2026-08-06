"""Acceptance: the meta-instruments M1, M2, M7, M8, M9 and M11 on real readings.

The clause: *on readings this repository can actually produce, M9 reports an incremental validity
with an interval for one instrument against a set already run, and M11 reports a real rung
disagreement from a real ladder. M1 and M7 report a noise floor and a budget on a real substrate. M2
and M8 either run on something real or refuse with a remedy naming what they would need.*

**Every number below is measured here.** The subject is the campaign store's eleven open reward
models on the RewardBench-2 bank: 1,763 groups of four responses, every model scoring every
response, one score per cell. That panel is what makes each of these a measurement rather than a
demonstration, and it is used differently by each instrument:

    M11  A1's `grader.effective_group_size` ladder, rung 0 against rung 3, on one grader.
    M9   the eleven models as eleven detectors of which response is best, one against the other ten.
    M8   the eleven models as eleven laboratories reporting one measurand with their own error bars.
    M1   the noise floor of two different readings on that panel, one of them M11's own subject.
    M7   the budget for A1's reading, composed from M1's floor and M11's transfer term.
    M2   the tap's own overhead, measured on this machine on a CPU-only loop.

M2 is the one whose subject is not the campaign. Its measurand is what instrumenting a grader costs,
which is a property of this library rather than of the store, so it is measured by running the
library's own tap and timing it. That is real in the sense the clause asks for and its numbers are
machine-dependent, so what is asserted here is structure and sign rather than a duration.

**Two things that are found rather than asserted, and both are results.** The campaign store carries
no replicate of any scoring pass, which is checked directly below and is why M1's engine-to-engine
rung is registered and not run. And the eleven laboratories disagree by roughly ten times what their
own error bars allow, against a control panel that comes back at about one, which is what an
interlaboratory comparison is for.
"""

from __future__ import annotations

import math
import os
import re
import time
from pathlib import Path

import numpy as np
import pytest

from reward_lens.core.budget import BudgetTerm, CalibrationCurve
from reward_lens.core.evidence import ValueCodec
from reward_lens.core.quantity import QUANTITIES, load_quantities
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.reference import Transfer
from reward_lens.measure.base import Context, lint_instrument
from reward_lens.measure.meta import (
    BlankReplicates,
    ControlPanel,
    Detector,
    DoseSweep,
    IncrementalValidityReading,
    InstrumentEffectReading,
    InterlaboratoryComparison,
    Lab,
    RungDisagreement,
    StepBasis,
    SubstrateNoiseFloor,
    UncertaintyBudgetReading,
    bootstrap_control,
    compose,
    instances,
    rung_from_effective_size,
    two_arm_blanks,
    verdict_for,
)
from reward_lens.measure.meta.floor import HillFit, limits_from
from reward_lens.measure.metrology.gstudy import (
    GroupScores,
    ReplicationDesign,
    effective_group_size,
    jackknife_reliability,
)
from reward_lens.record.convert.store import CampaignStore
from reward_lens.stats.variance import group_effective_size
from reward_lens.tap.grader_wrap import instrument_grader

#: The campaign evidence store, which is not in this repository. There is no default: point
#: ``REWARD_LENS_CAMPAIGN_STORE`` at the store directory or the tests that need it skip.
_CAMPAIGN_ENV = os.environ.get("REWARD_LENS_CAMPAIGN_STORE")
CAMPAIGN_STORE = Path(_CAMPAIGN_ENV) if _CAMPAIGN_ENV else None

pytestmark = pytest.mark.skipif(
    CAMPAIGN_STORE is None or not (CAMPAIGN_STORE / "evidence.jsonl").exists(),
    reason=(
        "no campaign evidence store. The eleven reward models it holds are what make these "
        "readings measurements rather than demonstrations. Set REWARD_LENS_CAMPAIGN_STORE."
    ),
)

#: Not a reward model: one of its two rb2-full rows is byte-identical to `grm-gemma2-2b`, so it is a
#: derived marker and counting it would put a duplicated column into every panel here.
NOT_A_GRADER = {"hackfore-flagged"}

#: The grader M11, M1 and M7 are read on. Chosen because it is mid-panel on accuracy and because its
#: rung-0 and rung-3 effective group sizes are far apart, which is the disagreement being published.
SUBJECT_GRADER = "tulu-rm"

#: The instrument M9 asks about, and the one it asks about second. The first is strong and partly
#: decorrelated from the panel; the second is the weakest model in the panel and is the case the
#: reading has to be able to report honestly.
STRONG_DETECTOR = "skywork-v2-qwen3-8b"
WEAK_DETECTOR = "tulu-dpo"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _registry() -> None:
    """The catalogue, loaded once. Every instrument here declares a registered quantity."""
    if "substrate.noise_floor" not in QUANTITIES:
        load_quantities()


@pytest.fixture(scope="module")
def store() -> CampaignStore:
    return CampaignStore(CAMPAIGN_STORE)


@pytest.fixture(scope="module")
def panel(store: CampaignStore):
    """11 graders x 1,763 groups of K = 4, gauge-fixed. Returns (graders, z, flat).

    Gauge fixing is a per-rater affine map and therefore an element of `reward.affine`. Without it
    the models' unit choices, which differ by more than two orders of magnitude on this bank, would
    be attributed to the rater facet in every reading below.
    """
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
    return graders, flat.reshape(n_r, n_groups, k), flat


@pytest.fixture(scope="module")
def margins(panel):
    """Per-grader, per-group margin: the chosen response's score minus the best rival's."""
    _, z, _ = panel
    return z[:, :, 0] - z[:, :, 1:].max(axis=2)


@pytest.fixture(scope="module")
def a1_ladder(panel):
    """A1's rung 0 and rung 3 for the subject grader, on the same 1,763 groups."""
    graders, z, flat = panel
    design = ReplicationDesign(
        scores=flat.T,
        raters=tuple(graders),
        object_label="response",
        facet_labels=("reward model", "occasion"),
    )
    _, se = jackknife_reliability(design)
    i = graders.index(SUBJECT_GRADER)
    scored = GroupScores.of(z[i], grader=SUBJECT_GRADER)
    r0 = effective_group_size(scored, None, n_resamples=600, seed=0)
    r3 = effective_group_size(scored, design, n_resamples=600, seed=0, reliability_se=se)
    return scored, r0, r3, se


@pytest.fixture(scope="module")
def disagreement(a1_ladder):
    """M11's reading on A1's ladder, which is the clause's own example."""
    _, r0, r3, _ = a1_ladder
    instrument = RungDisagreement(
        rung_from_effective_size(r0),
        rung_from_effective_size(r3),
        quantity_id="grader.effective_group_size",
        subject=f"{SUBJECT_GRADER} on rb2-full",
    )
    return instrument, instrument.compute()


@pytest.fixture(scope="module")
def neff_floor(a1_ladder):
    """M1's floor for A1's own reading, in n_eff units, so it composes into M7's budget."""
    scored, r0, _, _ = a1_ladder
    per_group = np.array([group_effective_size(g) for g in scored.groups], dtype=np.float64)
    blanks = two_arm_blanks(
        per_group,
        n_replicates=400,
        seed=0,
        method=(
            "two disjoint halves of the 1,763 groups, the rung-0 Kish count on each, 400 random "
            "splits"
        ),
        scale=float(r0.k_nominal),
    )
    instrument = SubstrateNoiseFloor(blanks)
    return instrument, instrument.compute(), blanks


# ---------------------------------------------------------------------------
# What the store does and does not contain
# ---------------------------------------------------------------------------


def test_the_store_carries_no_replicate_of_any_scoring_pass(store):
    """Why M1's engine-to-engine rung is registered and not run, checked rather than asserted.

    Nine of the models were written twice on this bank, once as a flat per-response bank and once
    grouped into fours. If those were two scoring passes the difference between them would be a
    two-engine noise floor for free. They are bit-identical, so they are one pass written twice, and
    the highest rung of `substrate.noise_floor` has no instance in this store.
    """
    flat: dict[str, dict[int, float]] = {}
    grouped: dict[str, np.ndarray] = {}
    for row in store.by_observable("campaign.scores"):
        if row.bank != "rb2-full" or row.roster_key in NOT_A_GRADER:
            continue
        value = store.value(row)
        scores = np.asarray(value["scores"], dtype=np.float64)
        if value["layout"] == "flat":
            bucket = flat.setdefault(row.roster_key, {})
            for item_id, score in zip(value["item_ids"], scores.ravel()):
                bucket[int(re.search(r"row(\d+)", item_id).group(1))] = float(score)
        elif value["layout"] == "best-of-4":
            grouped[row.roster_key] = scores

    shared = sorted(set(flat) & set(grouped))
    assert len(shared) >= 9, shared
    for model in shared:
        rows = flat[model]
        ordered = np.array([rows[i] for i in range(len(rows))], dtype=np.float64)
        assert np.array_equal(ordered.reshape(-1, 4), grouped[model]), (
            f"{model} differs between its two writings, which would make them two passes"
        )


# ---------------------------------------------------------------------------
# Clause: M11 reports a real rung disagreement from a real ladder
# ---------------------------------------------------------------------------


def test_m11_reproduces_a1s_rung_disagreement_on_a_real_ladder(disagreement):
    instrument, reading = disagreement
    assert not isinstance(reading, Refusal), reading
    assert reading.quantity == "grader.effective_group_size"
    assert reading.cheap_rung == 0 and reading.expensive_rung == 3
    # Re-pinned when A1 stopped multiplying the Kish shape factor into the effective size. Rung 0 is
    # now exactly the nominal group size, because it sets reliability to 1.0 and the shape factor no
    # longer enters; rung 3 is `K * reliability`. The old pins were 2.9-3.1 and 1.5-2.3. The
    # disagreement M11 exists to report is unchanged in kind and wider in margin: the smallest gap
    # between the two intervals went from 0.3847 to 0.5484 effective rollouts.
    assert reading.cheap_value == pytest.approx(4.0), reading.cheap_value
    assert 2.5 < reading.expensive_value < 2.6, reading.expensive_value
    assert reading.delta > 1.0
    assert reading.n == 1763
    assert reading.relative_delta > 0.3
    assert not reading.intervals_overlap, "the two rungs would not be disagreeing"
    assert "GRADER: RECORD" in reading.cheap_access
    assert "GRADER: REPLICATE" in reading.expensive_access
    unc = instrument.uncertainty(reading)
    assert unc is not None and unc.ci_low < reading.delta < unc.ci_high
    assert "upper bound" in unc.method


def test_the_disagreement_is_a_transfer_term_that_composes(disagreement):
    _, reading = disagreement
    transfer = reading.transfer()
    assert isinstance(transfer, Transfer)
    assert transfer.name == "t21"
    assert transfer.value == pytest.approx(reading.delta)
    assert transfer.n == 1763
    term = transfer.as_term()
    assert term.dof == 1762
    assert term.value == pytest.approx(reading.delta)


def test_m11_publishes_both_numbers_and_resolves_nothing(disagreement):
    """The design constraint, as a property of the payload rather than as a promise in a docstring."""
    instrument, reading = disagreement
    payload = instrument.payload(reading)
    assert payload["cheap_value"] == pytest.approx(reading.cheap_value)
    assert payload["expensive_value"] == pytest.approx(reading.expensive_value)
    forbidden = ("preferred", "resolved", "winner", "chosen_rung", "best_value", "corrected")
    assert not [k for k in payload if any(f in k for f in forbidden)], payload.keys()
    assert not [a for a in dir(reading) if not a.startswith("_") and "prefer" in a]
    text = reading.says()
    assert "both numbers stand" in text
    assert f"{reading.cheap_value:.4g}" in text and f"{reading.expensive_value:.4g}" in text


def test_m11_emits_evidence_carrying_the_quantity_and_the_interval(disagreement):
    instrument, _ = disagreement
    evidence = instrument.estimate(Context())
    assert not isinstance(evidence, Refusal), evidence
    assert evidence.quantity == "ladder.rung_disagreement"
    assert evidence.observable == "RungDisagreement"
    assert evidence.uncertainty.ci_high > evidence.uncertainty.ci_low
    assert set(evidence.baselines) == {
        "baseline.rungs_agree",
        "baseline.cheap_rung_interval_width",
    }


def test_m11_refuses_one_rung_and_names_the_rung_that_would_help(a1_ladder):
    _, r0, _, _ = a1_ladder
    out = RungDisagreement(
        rung_from_effective_size(r0), quantity_id="grader.effective_group_size"
    ).compute()
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert out.remedy.strip()


# ---------------------------------------------------------------------------
# Clause: M9 reports an incremental validity with an interval
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def detectors(panel, margins):
    graders, _, _ = panel
    return graders, [Detector(id=g, margin=margins[i]) for i, g in enumerate(graders)]


def _increment(detectors, own_id: str):
    graders, dets = detectors
    i = graders.index(own_id)
    instrument = IncrementalValidityReading(
        dets[i], [d for j, d in enumerate(dets) if j != i], n_resamples=2000, seed=0
    )
    return instrument, instrument.compute()


def test_m9_reports_an_increment_with_an_interval_against_the_ten_already_run(detectors):
    instrument, reading = _increment(detectors, STRONG_DETECTOR)
    assert not isinstance(reading, Refusal), reading
    assert reading.own_id == STRONG_DETECTOR
    assert len(reading.baseline_ids) == 10
    assert reading.n == 1763
    assert reading.ci_low < reading.increment < reading.ci_high
    assert reading.ci_low > 0.0, "this one is complementary and the interval should say so"
    assert not reading.adds_nothing and not reading.subtracts
    # Measured, not asserted: the errors are correlated with the best baseline's, and the increment
    # survives anyway, which is the case the ensemble-gain-only literature cannot distinguish.
    assert 0.3 < reading.error_correlation < 0.8
    assert reading.own_score < reading.best_baseline_score, (
        "the instrument is worse than the best baseline and still adds something, which is the "
        "whole argument for decorrelation over superiority"
    )
    unc = instrument.uncertainty(reading)
    assert unc.ci_low == pytest.approx(reading.ci_low)
    assert "paired percentile bootstrap" in unc.method


def test_m9_reports_a_negative_increment_as_a_negative_increment(detectors):
    """The weakest model in the panel makes the ensemble worse, and the reading has to say that."""
    _, reading = _increment(detectors, WEAK_DETECTOR)
    assert not isinstance(reading, Refusal), reading
    assert reading.increment < 0.0
    assert reading.ci_high < 0.0
    assert reading.subtracts and not reading.adds_nothing
    assert "makes the set worse" in reading.says()


def test_m9_emits_the_kernel_incremental_validity_record(detectors):
    instrument, reading = _increment(detectors, STRONG_DETECTOR)
    record = reading.record
    assert record.baseline_id in reading.baseline_ids
    assert record.ensemble_score == pytest.approx(reading.ensemble_with_own)
    evidence = instrument.estimate(Context())
    assert not isinstance(evidence, Refusal), evidence
    assert evidence.quantity == "instrument.incremental_validity"
    assert evidence.value["incremental_validity"]["error_correlation"] == pytest.approx(
        reading.error_correlation
    )
    assert evidence.uncertainty.ci_low == pytest.approx(reading.ci_low)


def test_m9_refuses_with_no_set_already_run(detectors):
    graders, dets = detectors
    out = IncrementalValidityReading(dets[0], []).compute()
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert "M3" in out.remedy


def test_m9_refuses_when_the_instruments_did_not_see_the_same_items(detectors):
    graders, dets = detectors
    short = Detector(id="short", margin=dets[1].margin[:100])
    out = IncrementalValidityReading(dets[0], [short]).compute()
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.RECORD_INCOMPLETE
    assert "identical items" in out.remedy


# ---------------------------------------------------------------------------
# Clause: M1 reports a noise floor on a real substrate
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def accuracy_floor(panel):
    """M1 with a real blank and a real dose sweep on the subject grader's own scores.

    The dose is an additive shift on the chosen response's standardised score, so a dose of 1.0 is
    one standard deviation of that grader's own output. The reading is the accuracy of a dosed arm
    minus the accuracy of an undosed one on the disjoint other half, which is zero in expectation at
    dose zero and is therefore a blank there.
    """
    graders, z, _ = panel
    i = graders.index(SUBJECT_GRADER)
    chosen = z[i][:, 0]
    rival = z[i][:, 1:].max(axis=1)
    n = chosen.size
    blanks = two_arm_blanks((chosen > rival).astype(np.float64), n_replicates=200, seed=1)

    rng = np.random.default_rng(7)
    doses = np.array([0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])
    responses = []
    for dose in doses:
        arm = []
        for _ in range(40):
            perm = rng.permutation(n)
            a, b = perm[: n // 2], perm[n // 2 :]
            arm.append(float(np.mean(chosen[a] + dose > rival[a]) - np.mean(chosen[b] > rival[b])))
        responses.append(float(np.mean(arm)))
    sweep = DoseSweep(doses, np.array(responses), dose_unit="standardised score units")
    instrument = SubstrateNoiseFloor(blanks, sweep)
    return instrument, instrument.compute()


def test_m1_reports_a_noise_floor_a_lod_and_a_loq_on_a_real_substrate(accuracy_floor):
    instrument, reading = accuracy_floor
    assert not isinstance(reading, Refusal), reading
    assert reading.blank_n == 200
    assert 0.005 < reading.sigma_blank < 0.05, reading.sigma_blank
    # A blank whose mean is far from zero is not a blank. Two standard errors is the check.
    assert abs(reading.blank_mean) < 2.0 * reading.sigma_blank / math.sqrt(reading.blank_n) + 0.01
    assert reading.is_determinate
    assert reading.sensitivity > 0
    assert reading.lod > 0 and math.isfinite(reading.lod)
    assert reading.loq == pytest.approx(reading.lod * 10.0 / 3.3)
    assert reading.ec50 > 0 and reading.hill_n > 0
    assert 0 < reading.relative_floor < 0.2
    assert "limit of detection" in reading.says()


def test_m1_composes_into_the_kernels_limit_of_detection_and_its_three_outcomes(accuracy_floor):
    _, reading = accuracy_floor
    limits = limits_from(
        BlankReplicates(
            values=np.array([0.0, reading.sigma_blank, -reading.sigma_blank]),
            method="reconstructed from the reading, to reach the kernel's own three-outcome rule",
        ),
        HillFit(
            curve=CalibrationCurve(
                e_max=reading.e_max,
                ec50=reading.ec50,
                hill_n=reading.hill_n,
                at_dose=reading.at_dose,
            )
        ),
    )
    assert limits.is_determinate
    assert verdict_for(reading.lod * 0.5, limits) == "below_lod"
    assert verdict_for(reading.lod * 1.5, limits) == "above_lod_below_loq"
    assert verdict_for(reading.loq * 2.0, limits) == "quantifiable"


def test_m1_measures_a_floor_with_no_sweep_and_declines_to_quote_a_limit(neff_floor):
    """The honest partial answer: the floor exists, the limits do not, and the reading says which."""
    instrument, reading, blanks = neff_floor
    assert not isinstance(reading, Refusal), reading
    assert reading.blank_n == 400
    assert reading.sigma_blank > 0
    assert not reading.is_determinate
    assert reading.lod is None and reading.loq is None and reading.ec50 is None
    assert "no dose sweep was supplied" in reading.fit_reason
    unc = instrument.uncertainty(reading)
    assert unc.ci_low < reading.sigma_blank < unc.ci_high
    assert "chi-squared" in unc.method


def test_m1_refuses_a_floor_from_too_few_replicates(neff_floor):
    _, _, blanks = neff_floor
    thin = BlankReplicates(values=blanks.values[:2], method=blanks.method)
    out = SubstrateNoiseFloor(thin).compute()
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert "n_replicates" in out.remedy


# ---------------------------------------------------------------------------
# Clause: M7 reports a budget on a real substrate
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def budget_reading(a1_ladder, disagreement, neff_floor, panel):
    """A1's own uncertainty budget, composed from M1's floor and M11's transfer.

    Every term is in the units of `grader.effective_group_size`, which is the discipline a budget
    imposes and the reason M1 is run a second time on this reading rather than the accuracy floor
    being borrowed. A budget mixing a floor in accuracy units with a transfer in effective-rollout
    units would compose cleanly and mean nothing.
    """
    graders, _, _ = panel
    _, r0, r3, se = a1_ladder
    _, disagree = disagreement
    _, _, blanks = neff_floor
    # The Kish bootstrap term is gone, and dropping it is the budget half of A1's correction rather
    # than bookkeeping. `n_eff = K * reliability`: `K` is a design constant nobody sampled, so the
    # only sampling uncertainty in the measurand is the reliability's, which is the rater jackknife.
    # The Kish bootstrap measured the *shape factor's* sampling uncertainty, and the shape factor is
    # no longer part of this quantity. Carrying it here would put it back into `n_eff` through the
    # uncertainty budget immediately after it was taken out of the point estimate, which is the same
    # conflation wearing different clothes. It travels on `run.group_shape_factor` instead.
    budget = compose(
        transfers=[disagree.transfer()],
        limits=limits_from(blanks, HillFit(curve=None, reason="no sweep on this reading")),
        extra=[
            BudgetTerm(
                name="reliability_jackknife",
                value=r3.k_nominal * se,
                kind="A",
                dof=len(graders) - 1,
                note=(
                    "rater-panel jackknife on the generalizability coefficient, times the nominal "
                    "group size, which is what carries it into effective-rollout units"
                ),
            )
        ],
    )
    instrument = UncertaintyBudgetReading(
        budget,
        quantity_id="grader.effective_group_size",
        value=r3.n_eff,
        sampling_terms=("reliability_jackknife",),
    )
    return instrument, instrument.compute()


def test_m7_composes_a_budget_whose_largest_term_is_not_sampling_noise(budget_reading):
    instrument, reading = budget_reading
    assert not isinstance(reading, Refusal), reading
    assert reading.quantity == "grader.effective_group_size"
    # Three terms, not four: the Kish bootstrap left with the shape factor it was measuring. See the
    # fixture for why re-routing it here would have reimported the conflation A1 removed.
    assert reading.n_terms == 3
    assert set(reading.shares) == {
        "t21",
        "substrate_noise",
        "reliability_jackknife",
    }
    assert reading.dominant_term == "t21"
    assert reading.dominant_share > 0.5
    assert reading.combined > 1.0
    assert reading.expanded == pytest.approx(2.0 * reading.combined)
    assert reading.effective_dof is not None
    assert reading.type_b_terms == 1
    assert sum(reading.shares.values()) == pytest.approx(1.0)
    # The reading is 2.56 effective rollouts and its combined uncertainty is more than half of that.
    assert reading.relative > 0.3


def test_m7_composition_is_arithmetic_and_checkable(budget_reading):
    instrument, reading = budget_reading
    terms = instrument.budget.terms
    expected = math.sqrt(sum(t.contribution**2 for t in terms))
    assert reading.combined == pytest.approx(expected)
    derived = instrument.as_uncertainty(reading)
    assert derived.ci_high == pytest.approx(reading.expanded)
    assert derived.ci_low == pytest.approx(-reading.expanded)


def test_m7_names_what_the_budget_is_still_missing(budget_reading):
    _, reading = budget_reading
    assert reading.findings, "a budget with no declared correlations has an assumption in it"
    assert any("correlations" in f for f in reading.findings)


def test_m7_emits_evidence_under_the_subjects_quantity(budget_reading):
    instrument, _ = budget_reading
    evidence = instrument.estimate(Context())
    assert not isinstance(evidence, Refusal), evidence
    assert evidence.quantity == "grader.effective_group_size"
    assert evidence.value["dominant_term"] == "t21"
    assert "t21" in evidence.value["table"]


def test_m7_refuses_a_budget_with_no_subject_and_a_budget_with_no_terms(budget_reading):
    instrument, _ = budget_reading
    orphan = UncertaintyBudgetReading(instrument.budget).compute()
    assert isinstance(orphan, Refusal)
    assert orphan.reason is RefusalReason.RECORD_INCOMPLETE
    empty = UncertaintyBudgetReading(None, quantity_id="grader.effective_group_size").compute()
    assert isinstance(empty, Refusal)
    assert empty.reason is RefusalReason.RECORD_INCOMPLETE
    assert "zero is the one value an uncertainty can never honestly take" in empty.remedy


# ---------------------------------------------------------------------------
# Clause: M8 runs on something real, or refuses with a remedy
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def interlab(panel):
    """The eleven models as eleven laboratories, with a matched control panel of eleven."""
    graders, z, _ = panel
    correct = z[:, :, 0] > z[:, :, 1:].max(axis=2)
    n = correct.shape[1]
    labs = [
        Lab(
            id=g,
            value=float(correct[i].mean()),
            u=float(np.sqrt(correct[i].mean() * (1 - correct[i].mean()) / n)),
            n=n,
            method="best-of-4 accuracy over 1,763 groups, standard error over groups",
        )
        for i, g in enumerate(graders)
    ]
    anchor = correct[graders.index(SUBJECT_GRADER)].astype(np.float64)
    control = bootstrap_control(anchor, k=len(labs), seed=3, measurand="best-of-4 accuracy")
    instrument = InterlaboratoryComparison(
        labs, control, measurand="best-of-4 accuracy on rb2-full"
    )
    return instrument, instrument.compute(), labs, control


def test_m8_runs_on_the_real_panel_and_reports_s_l_and_the_birge_ratio(interlab):
    instrument, reading, labs, _ = interlab
    assert not isinstance(reading, Refusal), reading
    assert reading.k == 11
    assert 0.6 < reading.consensus < 0.85
    assert 0.05 < reading.s_l < 0.2, reading.s_l
    assert reading.birge > 5.0, "the panel disagrees far beyond its own error bars"
    assert reading.q_df == 10
    assert reading.i2 > 90.0
    assert not reading.labs_understand_their_errors
    assert reading.typical_within_u < reading.s_l / 5
    assert reading.outliers, "at least one lab is beyond its own expanded uncertainty"
    unc = instrument.uncertainty(reading)
    assert unc.ci_low < reading.s_l < unc.ci_high
    assert "Q-profile" in unc.method


def test_the_matched_control_is_what_makes_the_birge_ratio_readable(interlab):
    _, reading, _, _ = interlab
    assert reading.control_k == 11
    assert 0.5 < reading.control_birge < 2.0, (
        "a panel that shares a measurand by construction should come back at about one"
    )
    assert reading.control_s_l < reading.s_l / 10
    assert reading.excess_dispersion > 5.0
    assert "bootstrap resamples" in reading.control_how


def test_m8_refuses_without_a_matched_control_and_names_what_would_fix_it(interlab):
    _, _, labs, _ = interlab
    out = InterlaboratoryComparison(labs).compute()
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.NO_MATCHED_CONTROL
    assert "bootstrap_control" in out.remedy
    assert out.statistics["k"] == 11


def test_m8_refuses_a_control_that_is_not_identically_powered(interlab):
    _, _, labs, control = interlab
    smaller = ControlPanel(labs=control.labs[:5], how=control.how, measurand=control.measurand)
    out = InterlaboratoryComparison(labs, smaller).compute()
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.NO_MATCHED_CONTROL
    assert "identically powered" in out.detail


def test_m8_refuses_a_lab_that_states_no_uncertainty_of_its_own(interlab):
    _, _, labs, control = interlab
    silent = [Lab(id=labs[0].id, value=labs[0].value, u=0.0, n=labs[0].n), *labs[1:]]
    out = InterlaboratoryComparison(silent, control).compute()
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.RECORD_INCOMPLETE


def test_m8_emits_evidence_under_study_tau2(interlab):
    instrument, _, _, _ = interlab
    evidence = instrument.estimate(Context())
    assert not isinstance(evidence, Refusal), evidence
    assert evidence.quantity == "study.tau2"
    assert evidence.value["s_L"] == pytest.approx(math.sqrt(evidence.value["tau2"]))


# ---------------------------------------------------------------------------
# Clause: M2 runs on something real, or refuses with a remedy
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tap_overhead(panel):
    """The library's own tap, timed on this machine over a CPU-only loop of real scores.

    The measurand is what instrumenting a grader costs, which is a property of this library rather
    than of the campaign, so the honest way to measure it is to run it. Nothing here asserts a
    duration: wall-clock numbers are machine-dependent and a test that pinned one would fail on a
    loaded machine for a reason that has nothing to do with the instrument.
    """
    _, z, _ = panel

    def grader(prompt: str, responses: np.ndarray) -> float:
        arr = np.asarray(responses, dtype=np.float64)
        return float(arr.mean() + 0.01 * arr.std())

    wrapped = instrument_grader(grader, name="rb2-margin-grader")
    steps, per_step_calls = 25, 64
    before = wrapped.effect()
    started = time.perf_counter()
    for step in range(steps):
        for j in range(per_step_calls):
            wrapped("prompt", z[0][(step * per_step_calls + j) % z.shape[1]])
    wall = time.perf_counter() - started
    after = wrapped.effect()
    basis = StepBasis(steps=steps, calls=steps * per_step_calls, wall_seconds=wall)
    instrument = InstrumentEffectReading(before=before, after=after, basis=basis)
    return instrument, instrument.compute(), after, basis


def test_m2_measures_the_taps_own_overhead_per_step(tap_overhead):
    instrument, reading, effect, _ = tap_overhead
    assert not isinstance(reading, Refusal), reading
    assert reading.tap == "rb2-margin-grader"
    assert reading.steps == 25
    assert reading.calls == 1600
    assert reading.calls_per_step == pytest.approx(64.0)
    assert reading.added_ms_per_step > 0.0 and math.isfinite(reading.added_ms_per_step)
    assert reading.added_ms_per_call_p99 >= reading.added_ms_per_call_p50 > 0.0
    assert 0.0 < reading.fraction_of_grader_time_mean < 10.0
    assert reading.fraction_of_step_time is not None and reading.fraction_of_step_time > 0.0
    assert reading.resident_bytes > 0
    assert reading.enabled and reading.recorder_exceptions == 0
    # tracemalloc is not tracing in a plain test process, so the allocation half is unmeasured.
    # Unmeasured and not zero, which is the whole content of `unchecked`.
    assert reading.added_alloc_bytes_per_step is None
    assert any("allocation" in u for u in reading.unchecked)


def test_the_overhead_composes_as_a_type_b_budget_term(tap_overhead):
    _, reading, _, _ = tap_overhead
    term = reading.as_term()
    assert term.kind == "B"
    assert term.distribution == "rectangular"
    assert term.value == pytest.approx(reading.fraction_of_grader_time_p99 / math.sqrt(3.0))
    composed = compose(sampling=0.01, overhead=term)
    assert composed.dominant is not None


def test_m2_refuses_to_convert_per_call_into_per_step_without_a_basis(tap_overhead):
    _, _, effect, _ = tap_overhead
    out = InstrumentEffectReading(effect).compute()
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.UNIT_MISMATCH
    assert "StepBasis" in out.remedy
    assert "property of the run rather than of the unit" in out.detail


def test_m2_refuses_a_tap_that_was_installed_and_never_called():
    wrapped = instrument_grader(lambda x: x, name="never-called")
    out = InstrumentEffectReading(wrapped.effect(), StepBasis(steps=1, calls=0)).compute()
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.RECORD_INCOMPLETE
    assert "never invoked" in out.remedy


# ---------------------------------------------------------------------------
# The declarations
# ---------------------------------------------------------------------------


def test_lint_instrument_is_empty_for_all_six(_registry):
    for inst in instances():
        assert lint_instrument(inst) == [], inst.name


def test_every_meta_instrument_declares_an_envelope_and_a_group(_registry):
    for inst in instances():
        assert inst.envelope is not None, inst.name
        assert inst.invariance, inst.name
        assert inst.baselines, inst.name
        if inst.envelope.requires:
            assert set(inst.envelope.measured_by) >= inst.envelope.requires, inst.name
        else:
            assert inst.envelope.unconditional and inst.envelope.justification, inst.name


def test_every_reading_encodes_through_the_store_codec(
    disagreement, accuracy_floor, budget_reading, interlab, tap_overhead, detectors
):
    """A reading that cannot round-trip is a reading nothing downstream can cite."""
    codec = ValueCodec()
    m9_instrument, m9_reading = _increment(detectors, STRONG_DETECTOR)
    pairs = [
        disagreement,
        accuracy_floor,
        budget_reading,
        (interlab[0], interlab[1]),
        (tap_overhead[0], tap_overhead[1]),
        (m9_instrument, m9_reading),
    ]
    for instrument, reading in pairs:
        payload = instrument.payload(reading)
        encoded = codec.encode(payload, None)
        assert codec.decode(encoded, None)["says"] == payload["says"], instrument.name

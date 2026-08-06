"""Acceptance: the information barrier, and the campaign re-scored through the new ledger.

The clause has two halves and both are exercised on real data rather than on a fixture.

**A forecast constructed with a future input raises at construction naming both timestamps.** The
subject is the campaign's LADDER card, which is the one campaign forecast with recorded inputs and
whose timing is genuinely tight: the three small rungs were measured at 13:29, 13:31 and 13:38 on
2026-07-19, the 80 percent prediction intervals were frozen at 14:38, and the held-out 8B rung was
measured at 15:34, 56 minutes after the freeze. Passing the 8B row is therefore a real leak on real
data, and the barrier finds it through the parent DAG rather than by inspecting the ids.

**The campaign's existing directional calls are re-scored through the new ledger and reproduce the
published Brier of 0.26.** Every verdict is recomputed from the frozen comparator and threshold in
the card's own spec file applied to the metric the adjudication row recorded, so this reproduces the
number rather than reprinting it.

The rest of the file points the package at the real GRPO run in ``tests/fixtures/grpo_run/long``:
200 real optimiser steps, real sampling, a grader that abstains on every seventh completion. That
run is a real optimisation trace and it is not a reward-hacking transition, so what is asserted on
it is the mechanics and the distribution-free null, never a transition claim it cannot support.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from reward_lens.core.evidence import Evidence, make_evidence
from reward_lens.core.provenance import Provenance
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.store import EvidenceStore
from reward_lens.core.types import SubjectRef
from reward_lens.forecast import (
    BaselineKind,
    BinaryProbability,
    CalibrationLedger,
    Comparator,
    DecisionSpec,
    ForecastCalibration,
    ForecastError,
    ForecastLeakageError,
    HorizonSpec,
    InformationTime,
    ReferenceClass,
    ResolutionRule,
    Resolved,
    climatology,
    contrastive_belief_flip,
    corpus_from_runs,
    decision_value,
    dumb_statistic,
    dumb_statistic_from_gradnorm,
    forecast_lead_time,
    issue,
    persistence,
    records_test,
    resolve,
)
from reward_lens.forecast.campaign import (
    campaign_reference_class,
    held_out_rung_id,
    ladder_interval_forecast,
    load_freeze,
    rescore_campaign,
)

#: The campaign archive, which is not in this repository. Neither path has a default: point
#: ``REWARD_LENS_CAMPAIGN_STORE`` at the evidence store and ``REWARD_LENS_CAMPAIGN_SPECS`` at
#: the frozen specs, or the tests that need them skip.
_CAMPAIGN_ENV = os.environ.get("REWARD_LENS_CAMPAIGN_STORE")
_SPECS_ENV = os.environ.get("REWARD_LENS_CAMPAIGN_SPECS")
CAMPAIGN_STORE = Path(_CAMPAIGN_ENV) if _CAMPAIGN_ENV else None
CAMPAIGN_SPECS = Path(_SPECS_ENV) if _SPECS_ENV else None
GRPO_LONG = Path(__file__).resolve().parents[1] / "fixtures" / "grpo_run" / "long"
GRPO_SHORT = Path(__file__).resolve().parents[1] / "fixtures" / "grpo_run" / "short"

campaign_only = pytest.mark.skipif(
    CAMPAIGN_STORE is None
    or CAMPAIGN_SPECS is None
    or not (CAMPAIGN_STORE / "evidence.jsonl").exists()
    or not (CAMPAIGN_SPECS / "manifest.json").exists(),
    reason=(
        "no campaign store or no frozen specs. Both are outside the repository; set "
        "REWARD_LENS_CAMPAIGN_STORE and REWARD_LENS_CAMPAIGN_SPECS to run these tests."
    ),
)
grpo_only = pytest.mark.skipif(
    not (GRPO_LONG / "runs").exists() and not (GRPO_SHORT / "runs").exists(),
    reason=f"no GRPO fixture at {GRPO_LONG} or {GRPO_SHORT}",
)

#: The records-theory reading on each real run, measured once and pinned here. The expected count is
#: theory (`H_n`) and is asserted against `harmonic` rather than against these; the observed count,
#: the z and the gradient-norm peak are facts about the fixture and change only when it is
#: regenerated, which is what pinning them is for.
GRPO_RECORDS: dict[int, dict[str, float]] = {
    200: {"n_records": 9, "z": 1.52, "p_low": 0.06, "p_high": 0.07, "gradnorm_peak": 109},
    12: {"n_records": 4, "z": 0.72, "p_low": 0.23, "p_high": 0.24, "gradnorm_peak": 4},
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def campaign_store() -> EvidenceStore:
    return EvidenceStore(CAMPAIGN_STORE, readonly=True)


@pytest.fixture(scope="module")
def freeze():
    return load_freeze(CAMPAIGN_SPECS)


@pytest.fixture(scope="module")
def rescored():
    ledger, freeze_doc, findings = rescore_campaign(CAMPAIGN_STORE, CAMPAIGN_SPECS)
    return ledger, freeze_doc, findings


def _open(root: Path):
    from reward_lens.record.reader import open_run

    run_id = next(p.name for p in (root / "runs").iterdir())
    return open_run(root, run_id.replace("run_", "run:"))


@pytest.fixture(scope="module")
def long_run():
    """The 200-step run, falling back to the 12-step one while `long/` is being regenerated.

    Both are real GRPO records from the same configuration, so every mechanical assertion holds on
    either. The numbers that are facts about the fixture rather than about the code are looked up in
    `GRPO_RECORDS` by step count, so a fallback run is still checked against measured values rather
    than against a loosened bound.
    """
    root = GRPO_LONG if (GRPO_LONG / "runs").exists() else GRPO_SHORT
    return _open(root)


@pytest.fixture(scope="module")
def short_run():
    return _open(GRPO_SHORT)


@pytest.fixture(scope="module")
def long_series(long_run):
    """Per-step mean grader value and per-step clipped gradient norm, from the real record."""
    means: list[float] = []
    grads: list[float] = []
    for step in long_run.steps:
        values = [
            t.scores.value
            for g in step.groups
            for t in g.trajectories
            if t.scores is not None and t.scores.value is not None
        ]
        means.append(float(np.mean(values)))
        grads.append(float(step.optimizer.grad_norm_clipped))
    return means, grads


# ---------------------------------------------------------------------------
# The clause, first half: the barrier raises naming both timestamps
# ---------------------------------------------------------------------------


@campaign_only
def test_ladder_forecast_issues_on_inputs_that_predate_it(campaign_store, freeze):
    """The three small rungs all predate the interval freeze, so the forecast is issuable."""
    forecast = ladder_interval_forecast(campaign_store, freeze)
    assert len(forecast.inputs) == 3
    assert forecast.issued_at.instant == "2026-07-19T14:38:14.080678+00:00"
    # The barrier recorded the inputs as parents, so the forecast could be appended to the store
    # it was checked against.
    assert set(forecast.provenance.parents) == set(forecast.inputs)


@campaign_only
def test_future_input_raises_at_construction_naming_both_timestamps(campaign_store, freeze):
    """The clause, first half, on real data with real timestamps."""
    held_out = held_out_rung_id(CAMPAIGN_STORE)

    with pytest.raises(ForecastLeakageError) as excinfo:
        ladder_interval_forecast(campaign_store, freeze, extra_inputs=(held_out,))

    error = excinfo.value
    message = str(error)

    # It names the offending id.
    assert error.evidence_id == held_out
    assert held_out in message

    # It names BOTH timestamps, in the message and on the exception.
    assert error.information_time.instant == "2026-07-19T15:34:03.449190+00:00"
    assert error.issued_at.instant == "2026-07-19T14:38:14.080678+00:00"
    assert "2026-07-19T15:34:03.449190+00:00" in message
    assert "2026-07-19T14:38:14.080678+00:00" in message

    # And it says how far out, and how to fix it.
    assert "+3349.369 s" in message
    assert "There is no override" in message


@campaign_only
def test_the_barrier_walks_the_parent_dag_not_just_the_inputs(campaign_store, freeze):
    """A leak two hops behind an input is still a leak.

    The 8B index table's own grandparent is a `campaign.capture` row written at 15:15:35, which is
    also after the freeze. Dropping the direct input but keeping the grandparent must still raise.
    """
    from reward_lens.forecast.barrier import ancestry

    held_out = held_out_rung_id(CAMPAIGN_STORE)
    reached = ancestry(campaign_store, [held_out])
    deep = [eid for eid, path in reached.items() if len(path) == 3]
    assert deep, "the 8B row should have at least one grandparent in this store"

    with pytest.raises(ForecastLeakageError) as excinfo:
        ladder_interval_forecast(campaign_store, freeze, extra_inputs=(deep[0],))
    assert excinfo.value.evidence_id == deep[0]


@campaign_only
def test_every_campaign_measurement_postdates_the_freeze(campaign_store, freeze):
    """The pre-registration, checked mechanically instead of asserted in a README.

    Zero of the 1,363 rows in the campaign store predate the 2026-07-18T23:46:57.951556+00:00
    freeze. That is what a pre-registration is supposed to look like, and it means a forecast issued
    at the freeze can consume no campaign measurement at all, which is why 22 of the 23 directional
    calls carry no inputs.
    """
    from reward_lens.forecast.barrier import _envelopes

    envelopes = _envelopes(campaign_store)
    freeze_instant = freeze.frozen_at
    before = [e for e in envelopes.values() if e["created_at"] < freeze_instant]
    assert before == []
    assert len(envelopes) == 1363


def test_there_is_no_override_anywhere_in_the_barrier():
    """No flag, no keyword, no environment variable turns the barrier off.

    Checked against the parsed module rather than against its text, because the word "enforced"
    contains "force" and a substring scan over a docstring is a test that fails on prose.
    """
    import ast
    import inspect

    from reward_lens.forecast import barrier

    escapes = {
        "force",
        "override",
        "allow_leak",
        "skip",
        "skip_barrier",
        "unsafe",
        "strict",
        "check",
        "disable",
        "bypass",
    }
    tree = ast.parse(inspect.getsource(barrier))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names = {a.arg for a in node.args.args} | {a.arg for a in node.args.kwonlyargs}
            assert not (names & escapes), f"{node.name} takes {names & escapes}"
        if isinstance(node, ast.Name):
            assert node.id not in {"os", "environ", "getenv"}
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = getattr(node, "module", "") or ""
            assert not module.startswith("os")
            assert all(not a.name.startswith("os") for a in node.names)

    signature = inspect.signature(barrier.issue)
    assert "store" in signature.parameters
    assert signature.parameters["store"].default is inspect.Parameter.empty


# ---------------------------------------------------------------------------
# The clause, second half: the campaign reproduces 0.26
# ---------------------------------------------------------------------------


@campaign_only
def test_campaign_rescored_reproduces_the_published_brier(rescored):
    """The clause, second half. 0.26 over 16 calls, against a coin at 0.25."""
    ledger, _, findings = rescored
    score = ledger.score()

    assert findings == (), f"recomputed verdicts disagree with the store: {findings}"
    assert score.n_directional == 16
    assert round(score.directional_brier, 2) == 0.26
    assert score.coin_brier == 0.25
    assert not score.beats_coin


@campaign_only
def test_campaign_rescored_reproduces_the_published_coverage(rescored):
    """Interval coverage 0.75 over four intervals against a registered nominal of 0.80."""
    ledger, _, _ = rescored
    score = ledger.score()
    assert score.n_intervals == 4
    assert score.coverage is not None
    assert score.coverage.coverage == 0.75
    assert score.coverage.nominal == 0.80
    # Four intervals is a companion diagnostic and not a kill metric, and the Wilson interval is
    # how that is said in numbers rather than in a caveat.
    assert score.coverage.covers_nominal


@campaign_only
def test_the_seven_inconclusive_calls_are_void_and_not_misses(rescored):
    """An expired or undecided forecast is void, never a miss, and the reason is on the row.

    Seven of the 23 registered directional calls went void, and one of them carried the campaign's
    most confident call at 0.9. Scoring those as misses would move the published Brier a long way
    and every step of the move would be a lie about what was measured.
    """
    ledger, _, _ = rescored
    voids = [e for e in ledger.voids if e.probability is not None]
    assert len(voids) == 7
    assert {e.voided for e in voids} == {"inconclusive"}
    assert all(e.void_detail for e in voids)

    void_cards = {e.target.split(".")[1] for e in voids}
    assert void_cards == {
        "STYLE-RMB",
        "PPE-BON",
        "HUMP",
        "GAUGE-E19",
        "GAUGE-XFAM",
        "HACK-FORE",
        "VALUES-CONTEST",
    }
    # If they were scored as misses instead, the Brier would move by this much.
    with_misses = ledger.directional_brier_if_voids_were_misses()
    assert round(with_misses, 4) != round(ledger.score().directional_brier, 4)


@campaign_only
def test_the_murphy_decomposition_closes_exactly(rescored):
    """`BS = REL - RES + UNC` to machine precision, because the campaign issued five distinct values."""
    ledger, _, _ = rescored
    murphy = ledger.score().murphy
    assert murphy is not None
    assert murphy.n == 16
    assert murphy.n_bins == 5
    assert murphy.binning.startswith("distinct forecast value")
    assert abs(murphy.residual) < 1e-12
    assert round(murphy.reliability, 4) == 0.0591
    assert round(murphy.resolution, 4) == 0.0452
    assert round(murphy.uncertainty, 4) == 0.2461


@campaign_only
def test_the_ledger_header_prints_the_honest_starting_position(rescored):
    """The published artifact says its own worst result first, and it recomputes it."""
    ledger, _, _ = rescored
    header = ledger.header()
    assert "Directional Brier 0.26 over 16 calls" in header
    assert "did not beat the always-guess-half coin at 0.25" in header
    assert "Interval coverage 0.75 over 4 intervals against a registered nominal 0.80" in header
    assert "The meta kill criterion fired" in header
    assert "8 registered call(s) went void" in header
    assert "we are the only ones publishing the score" in header


@campaign_only
def test_the_campaign_shipped_its_calls_against_no_comparator(rescored):
    """Every mandatory baseline refuses on every campaign row, and every refusal is on the record.

    This is what the ledger adds that the published scoreboard could not: the campaign's 23
    directional calls ran against zero comparators, and until there was somewhere for a refused
    baseline to be written down, a call that never ran the black-box comparator and one that ran it
    and won looked identical.
    """
    ledger, _, _ = rescored
    for entry in ledger:
        assert set(entry.baselines) == {k.value for k in BaselineKind}
        assert all(v is None for v in entry.baselines.values())
    # So the only skill score computable is against the coin.
    skills = ledger.score().skill
    assert len(skills) == 1
    assert skills[0].baseline_id.startswith("coin")
    assert skills[0].skill < 0
    assert skills[0].covers_zero


@campaign_only
def test_the_campaigns_decision_value_is_exactly_zero(rescored):
    """The number nobody in this field computes, on the real artifact, and it is zero.

    Under any decision whose cost-loss threshold is below 0.5, every one of the campaign's sixteen
    directional calls is above the threshold, so the forecaster acts every time and so does
    climatology. Seven hits, nine false alarms, no misses and no correct rejections: the
    probabilities never crossed a threshold in a way that would have changed an action, so the
    expected loss saved is 0.000 per event. That is a sharper statement than the Brier score, which
    only says the calls were badly calibrated, and it is invisible without a `DecisionSpec`.
    """
    ledger, _, _ = rescored
    spec = DecisionSpec(
        action="run the confirmatory arc for this card", cost=0.30, loss=1.50, unit="gpu_hours"
    )
    value = ledger.score(decision=spec).value
    assert value is not None
    assert value.threshold == pytest.approx(0.2)
    assert value.hits == 7
    assert value.false_alarms == 9
    assert value.misses == 0
    assert value.correct_rejections == 0
    assert value.value == 0.0
    assert value.loss_saved == 0.0
    assert value.climatology_action == "always act"


@campaign_only
def test_the_campaign_reference_class_has_no_base_rate(rescored):
    """Climatology refuses, and the refusal is the finding rather than a gap in the code."""
    reference = campaign_reference_class()
    assert not reference.has_climatology
    baseline = climatology(reference)
    assert not baseline.is_scored
    assert "no counted base rate" in baseline.refused


# ---------------------------------------------------------------------------
# The instrument
# ---------------------------------------------------------------------------


@campaign_only
def test_the_instrument_emits_a_gated_reading_over_the_campaign_ledger(rescored):
    ledger, _, _ = rescored
    instrument = ForecastCalibration(ledger)
    reading = instrument.estimate()
    assert isinstance(reading, Evidence)
    assert reading.observable == "ForecastCalibration"
    assert reading.quantity == "forecast.brier_score"
    assert round(reading.value.directional_brier, 2) == 0.26
    assert reading.baselines["coin"] == 0.25
    assert reading.uncertainty.method.startswith("paired percentile bootstrap")
    assert reading.subject.extra["n_directional"] == 16
    assert reading.subject.extra["n_void"] == 8


def test_the_instrument_refuses_an_empty_ledger():
    reading = ForecastCalibration(CalibrationLedger()).estimate()
    assert isinstance(reading, Refusal)
    assert reading.reason is RefusalReason.RECORD_INCOMPLETE
    assert "An empty ledger has no Brier score" in reading.remedy


def test_the_instrument_refuses_a_ledger_of_nothing_but_voids():
    """The refusal that matters: twenty voids and no resolutions has no denominator."""
    from reward_lens.forecast.ledger import LedgerEntry

    ledger = CalibrationLedger()
    for i in range(3):
        ledger.append(
            LedgerEntry(
                forecast_id=f"fc:{i}",
                target=f"t{i}",
                reference_class="rc",
                issued_at="2026-01-01T00:00:00+00:00",
                method="prior",
                probability=0.6,
                baselines={},
                voided="metric_absent",
                void_detail="the analysis produced no such metric",
            )
        )
    reading = ForecastCalibration(ledger).estimate()
    assert isinstance(reading, Refusal)
    assert reading.reason is RefusalReason.RECORD_INCOMPLETE
    assert "metric_absent" in reading.remedy
    assert "No amount of extra access fixes this" in reading.remedy
    assert reading.statistics["n_void"] == 3


# ---------------------------------------------------------------------------
# The real GRPO run
# ---------------------------------------------------------------------------


@grpo_only
def test_records_theory_on_the_real_run(long_series):
    """The distribution-free null on a real optimiser trace.

    On the 200-step run: nine running records against 5.88 expected under exchangeability,
    z = +1.52, one-sided p = 0.065. So the mean grader value drifts upward and the honest null does
    not call it significant, which is the correct reading of a 0.6M-parameter model optimising
    against a length grader. On the 12-step run: four records against 3.10 expected, z = +0.72,
    p = 0.235.

    The expected count is checked against `H_n` directly rather than against a stored number,
    because that half is theory and does not depend on the fixture at all.
    """
    from reward_lens.forecast import harmonic

    means, _ = long_series
    test = records_test(means)
    expected = GRPO_RECORDS[test.n]
    assert test.n == len(means)
    assert test.expected == pytest.approx(harmonic(test.n))
    assert test.variance == pytest.approx(harmonic(test.n) - harmonic(test.n, 2))
    assert test.n_records == expected["n_records"]
    assert round(test.z, 2) == expected["z"]
    assert expected["p_low"] < test.p_value < expected["p_high"]
    assert test.record_steps[0] == 0
    # Not significant at any conventional level on either run, which is the honest reading.
    assert test.p_value > 0.05


@grpo_only
def test_the_gradnorm_baseline_reads_the_real_logged_series(long_series):
    """The free dumb baseline, from the number the trainer already logs.

    Nothing here differentiates anything: `stats.baselines.series.gradnorm_peak` reads a logged
    scalar, which is why this baseline is available at RECORD access on somebody else's run.
    """
    means, grads = long_series
    expected = GRPO_RECORDS[len(means)]
    issue_step = len(means) // 2
    baseline = dumb_statistic_from_gradnorm(grads, issued_step=issue_step, horizon_steps=len(means))
    assert baseline.kind is BaselineKind.DUMB_STATISTIC
    assert baseline.is_scored
    assert f"peak at step {int(expected['gradnorm_peak'])}" in baseline.detail
    fired = expected["gradnorm_peak"] <= issue_step
    assert (baseline.distribution.p > 0.5) is fired


@grpo_only
def test_a_forecast_on_the_real_run_issues_and_resolves(tmp_path, long_series):
    """A real forecast, issued mid-run from the record's own rows, with the barrier live.

    The record carries no absolute per-step timestamps, so the information time of a step's reading
    has to come from whoever is reading it. Here it is a declared replay epoch plus the cumulative
    ``step_time`` the trainer actually recorded, and the basis string says exactly that. Inventing a
    clock and not saying so is the failure this whole package is about.
    """
    means, grads = long_series
    assume_enough = len(means) >= 12
    assert assume_enough, "both GRPO fixtures are at least 12 steps"
    store = EvidenceStore(tmp_path / "store")
    epoch = np.datetime64("2026-08-01T00:00:00")

    ids: list[str] = []
    parent: str | None = None
    for step, value in enumerate(means):
        when = str(epoch + np.timedelta64(int(step * 1000), "ms")) + "+00:00"
        evidence = make_evidence(
            observable="grpo.step_mean_reward",
            observable_version="1.0",
            subject=SubjectRef(extra={"step": step}),
            value=float(value),
            provenance=Provenance(parents=(parent,) if parent else ()),
            created_at=when,
            information_time=when,
        )
        store.append(evidence)
        ids.append(evidence.id)
        parent = evidence.id

    issue_step = len(means) // 2
    issued_at = InformationTime.parse(
        str(epoch + np.timedelta64(issue_step * 1000, "ms")) + "+00:00",
        basis="replay clock: a declared replay epoch plus the recorded per-step cadence",
    )
    reference = ReferenceClass(
        id="grpo.tiny-qwen3.length-grader",
        definition="a GRPO run of tiny-Qwen3 against a length grader, seed 1234, K=4, CPU",
        n=2,
        base_rate=0.5,
    )
    tail = max(2, len(means) // 4)
    rule = ResolutionRule(
        metric="mean_reward_tail",
        comparator=Comparator.GT,
        threshold=float(np.mean(means[:issue_step])),
        definition=(
            f"the mean grader value over the last {tail} steps exceeds the mean over the first "
            f"{issue_step}"
        ),
    )
    forecast = issue(
        target="grpo.mean_reward_improves",
        subject=SubjectRef(extra={"run": "long"}),
        resolution=rule,
        distribution=BinaryProbability(0.6),
        inputs=tuple(ids[: issue_step - 1]),
        at=issued_at,
        store=store,
        reference_class=reference,
        horizon=HorizonSpec(kind="steps", value=100.0),
        method="records-theory drift call on the first half of the run",
        baselines=(
            climatology(reference),
            persistence(True, confidence=0.7, detail="the last few steps were above the mean"),
            dumb_statistic_from_gradnorm(
                grads[:issue_step], issued_step=issue_step, horizon_steps=len(means) - issue_step
            ),
            contrastive_belief_flip((), (), judge=None),
        ),
        issued_step=issue_step,
    )
    assert len(forecast.inputs) == issue_step - 1

    # A step from after the issue instant is a leak, found through the chain of parents.
    leaked = ids[-1]
    with pytest.raises(ForecastLeakageError) as excinfo:
        issue(
            target="grpo.mean_reward_improves",
            subject=SubjectRef(extra={"run": "long"}),
            resolution=rule,
            distribution=BinaryProbability(0.6),
            inputs=(leaked,),
            at=issued_at,
            store=store,
            reference_class=reference,
            horizon=HorizonSpec(kind="steps", value=float(len(means) - issue_step)),
            method="records-theory drift call on the first half of the run",
            baselines=forecast.baselines,
        )
    assert excinfo.value.evidence_id == leaked
    assert excinfo.value.issued_at.instant == issued_at.instant

    outcome = resolve(
        forecast,
        {"mean_reward_tail": float(np.mean(means[-tail:]))},
        at=InformationTime.parse(
            str(epoch + np.timedelta64(len(means) * 1000, "ms")) + "+00:00",
            basis="the run finished and the record was closed",
        ),
    )
    assert isinstance(outcome, Resolved)
    assert outcome.metric == "mean_reward_tail"


@grpo_only
def test_lead_time_refuses_on_a_run_with_no_transition(long_series):
    """The GRPO fixture is a real optimisation trace and it is not a transition.

    So there is no fitted width to divide by, and asking for a lead time in width units returns the
    refusal that says so rather than a fraction of a width nobody fitted. This package does not fit
    transition widths: `forecast_lead_time` calls instrument H4 and passes its answer through, so
    the refusal below is H4's and its remedy is H4's.
    """
    means, _ = long_series
    reading = forecast_lead_time(len(means) // 2, means)
    assert isinstance(reading, Refusal)
    assert reading.remedy


def test_lead_time_on_a_planted_transition_is_in_width_units():
    """With a transition present, the lead comes back in width units.

    A logistic planted at midpoint 120 with slope 0.2 has a 10-to-90 width of 21.9722 steps. An
    alarm at step 100 is therefore 20 steps early, which is 0.910 of a width from the midpoint and
    0.410 from the start of the rise, and the rendered string says the step count is not comparable
    across runs.
    """
    series = 1.0 / (1.0 + np.exp(-0.2 * (np.arange(200.0) - 120.0)))
    lead = forecast_lead_time(100.0, series)
    assert not isinstance(lead, Refusal)
    assert lead.fit.width == pytest.approx(2 * np.log(9) / 0.2, rel=1e-6)
    assert lead.steps == pytest.approx(20.0, abs=1e-6)
    assert lead.widths == pytest.approx(0.910, abs=1e-3)
    assert lead.widths_from_onset == pytest.approx(0.410, abs=1e-3)
    assert lead.resolved
    assert "not comparable across runs" in lead.render()


@pytest.mark.skipif(
    not (GRPO_LONG / "runs").exists() or not (GRPO_SHORT / "runs").exists(),
    reason="the corpus clause needs both GRPO fixtures",
)
def test_the_corpus_splits_over_runs_and_time_on_the_two_real_runs(long_run, short_run):
    """212 real steps over two real runs, split three ways, with the embargo enforced."""
    walk = corpus_from_runs(
        [short_run, long_run],
        reference_class="grpo.tiny-qwen3.length-grader",
        split="walk_forward",
        embargo_steps=5,
    )
    assert walk.n_runs == 2
    assert walk.n_points == 212
    folds = walk.folds(n_folds=4, min_train=20)
    assert len(folds) == 4
    for fold in folds:
        train_max = max(s for _, s in fold.train)
        test_min = min(s for _, s in fold.test)
        assert test_min - train_max > walk.embargo_steps

    purged = corpus_from_runs(
        [short_run, long_run],
        reference_class="grpo.tiny-qwen3.length-grader",
        split="purged_kfold",
        embargo_steps=5,
    )
    for fold in purged.folds(n_folds=4):
        assert fold.purged > 0
        lo = min(s for _, s in fold.test)
        hi = max(s for _, s in fold.test)
        assert all(not (lo - 5 <= s <= hi + 5) for _, s in fold.train)

    loro = corpus_from_runs(
        [short_run, long_run],
        reference_class="grpo.tiny-qwen3.length-grader",
        split="leave_one_run_out",
    )
    for fold in loro.folds():
        assert fold.is_run_disjoint


def test_a_temporal_split_with_no_embargo_is_refused():
    """The embargo is not optional and zero is not a default anybody should get by accident."""
    from reward_lens.forecast import RunCorpus

    with pytest.raises(ForecastError) as excinfo:
        RunCorpus(
            runs=("a", "b"),
            reference_class="rc",
            split="walk_forward",
            embargo_steps=0,
            steps={"a": (0, 1, 2), "b": (0, 1, 2)},
        )
    assert "leaks across the boundary" in str(excinfo.value)
    # Asserting the features and labels are instantaneous is allowed and is a statement on the record.
    RunCorpus(
        runs=("a", "b"),
        reference_class="rc",
        split="walk_forward",
        embargo_steps=0,
        steps={"a": (0, 1, 2), "b": (0, 1, 2)},
        instantaneous=True,
    )


# ---------------------------------------------------------------------------
# Decision value
# ---------------------------------------------------------------------------


def test_decision_value_on_a_hand_computed_case():
    """The number that answers "so what", checked against arithmetic done by hand.

    Cost 90 GPU-hours to kill and restart, loss 480 to let a hacked run finish, so the cost-loss
    threshold is 0.1875. Four events: two hacked and forecast at 0.9, two clean and forecast at 0.05.
    The forecaster acts on exactly the two hacked ones, so it pays 2 x 90 over 4 events, which is 45.
    A perfect forecast pays the same 45. Climatology has a base rate of 0.5, so never acting costs
    0.5 x 480 = 240 and always acting costs 90; the cheaper is 90. Value is (90 - 45)/(90 - 45) = 1.
    """
    spec = DecisionSpec(action="kill and restart with a fixed grader", cost=90.0, loss=480.0)
    assert spec.threshold == 0.1875

    value = decision_value([0.9, 0.9, 0.05, 0.05], [True, True, False, False], spec)
    assert value.hits == 2
    assert value.false_alarms == 0
    assert value.misses == 0
    assert value.correct_rejections == 2
    assert value.expected_loss == 45.0
    assert value.perfect_loss == 45.0
    assert value.climatology_loss == 90.0
    assert value.value == 1.0
    assert value.loss_saved == 45.0
    assert value.loss_saved_total == 180.0

    # A forecaster that always says 0.5 acts every time, pays the cost every time, and adds nothing
    # over climatology, which would also always act.
    flat = decision_value([0.5] * 4, [True, True, False, False], spec)
    assert flat.hits == 2 and flat.false_alarms == 2
    assert flat.expected_loss == 90.0
    assert flat.value == 0.0


def test_a_forecast_without_four_baselines_cannot_be_constructed():
    """The stated minimum, enforced at construction and not by a linter."""
    from reward_lens.forecast import Forecast, forecast_id

    reference = ReferenceClass(id="rc", definition="a population", n=10, base_rate=0.2)
    rule = ResolutionRule(metric="m", comparator=Comparator.GT, threshold=0.0)
    subject = SubjectRef()
    issued = InformationTime.parse("2026-01-01T00:00:00+00:00", basis="a test")
    kwargs = dict(
        target="q",
        subject=subject,
        resolution=rule,
        issued_at=issued,
        horizon=HorizonSpec(),
        reference_class=reference,
        distribution=BinaryProbability(0.5),
        method="a guess",
        inputs=(),
    )
    with pytest.raises(ForecastError) as excinfo:
        Forecast(
            id=forecast_id(
                target="q",
                subject=subject,
                resolution=rule,
                issued_at=issued,
                distribution=BinaryProbability(0.5),
                inputs=(),
                method="a guess",
            ),
            baselines=(climatology(reference),),
            **kwargs,
        )
    message = str(excinfo.value)
    assert "persistence" in message and "dumb_statistic" in message
    assert "scaffolded_black_box" in message
    assert "look identical from the outside" in message

    # A refused baseline satisfies the requirement, which is the whole point of recording refusals.
    Forecast(
        id=forecast_id(
            target="q",
            subject=subject,
            resolution=rule,
            issued_at=issued,
            distribution=BinaryProbability(0.5),
            inputs=(),
            method="a guess",
        ),
        baselines=(
            climatology(reference),
            persistence(None),
            dumb_statistic(None, name="none"),
            contrastive_belief_flip((), (), judge=None),
        ),
        **kwargs,
    )


def test_information_time_cannot_be_built_from_a_run_position():
    """The third clock is in the type, so the commonest mistake fails where it is made."""
    with pytest.raises(ForecastError) as excinfo:
        InformationTime.from_run_position(200)
    assert "is a run position, not an information time" in str(excinfo.value)

    with pytest.raises(ForecastError):
        InformationTime.parse("2026-01-01T00:00:00", basis="naive, no offset")

    with pytest.raises(ForecastError):
        InformationTime.parse("2026-01-01T00:00:00+00:00", basis="   ")

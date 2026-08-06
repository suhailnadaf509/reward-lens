"""Acceptance: F1, the selection term and residual, and F2, `Λ` and `η_eff`.

**The clause.** *On the AISI labelled series, `Λ` is computed per step and plotted against the
labelled hack rate. Whatever the relationship, it is the first such measurement.*

It is discharged in two halves and both halves are here.

The **mechanics** are asserted on the real GRPO record under `tests/fixtures/grpo_run/`: a real
`GRPOTrainer`, real weights that really changed, real sampling, real advantages. Every instrument
runs on it end to end, `Λ` is computed per step, and the envelope loop that makes `Λ` a validity
certificate for F1 is exercised on it. That run is a real optimisation trace and it is **not** a
reward-hacking transition, so it can discharge "`Λ` is computed per step" and it cannot discharge
"plotted against the labelled hack rate".

The 200-step record is used when it is present and the 12-step one when it is not; every count
asserted below is derived from the record rather than written down, so the file tests the ledger
rather than the fixture. **On the 200-step record**: 199 step pairs, 189 sliding `Λ` windows,
`Λ` = 0.0012 [0.0000, 0.0195] and `η_eff` = -0.032 +/- 0.052, which is indistinguishable from zero.
Two of the five surface features are constant there (the model emits twelve distinct tokens every
time and every trajectory has two turns) and are dropped by name. Every consecutive pair draws a
disjoint prompt set, so `Δz` is a difference between two task samples as well as between two
policies, and the residual dominates the selection term on every varying feature. That is the
honest reading of a 0.6M-parameter model on a toy length grader and it is what F1's own kill
condition looks like when it fires.

The **relationship** is measured on the AISI labelled series, which the clause names:
`ai-safety-institute/reward-hacking-olmo3.1-32b-kl0.0-seed2-rollouts`, 25,664 rollouts over 401
training steps with a per-rollout `reward_hacked` label. That is a 189 MB parquet and it is not
vendored into this repository, so those tests skip unless it is on disk. To run them:

    curl -L -o /path/to/kl0.0.parquet \\
      https://huggingface.co/datasets/ai-safety-institute/reward-hacking-olmo3.1-32b-kl0.0-seed2-rollouts/resolve/main/data/train-00000-of-00001.parquet
    REWARD_LENS_AISI_ROLLOUTS=/path/to/kl0.0.parquet pytest tests/acceptance/test_w4_3_ledger.py

**What was measured there**, recorded so that a skipped run still carries the result. The labelled
hack rate goes from 0.002 to 0.989 with a fitted midpoint at step 106.0 and a 10-to-90 width of
23.9 steps. `Λ` over the whole 400 step pairs is 0.0371 [0.0128, 0.0789], with `η_eff` = 0.172 +/-
0.028. Per step over a sliding eleven-pair window, `Λ` sits in the 0.00 to 0.02 band until roughly
step 120 and rises to 0.3 to 0.7 afterwards; its own fitted midpoint is step 128.3. So on this
series **`Λ` lags the labelled hack rate by 0.93 transition widths**, and the assertions below fix
the direction rather than the magnitude, because the magnitude moves with how degenerate groups are
handled while the sign does not.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest

from reward_lens.core.envelope import RegimeCondition
from reward_lens.core.evidence import Evidence
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.types import Access, Component, Phase, Substrate, TrustLevel
from reward_lens.measure.base import Context, Instrument, lint_instrument
from reward_lens.measure.ledger import (
    EffectiveStepSize,
    SelectionExplainedFraction,
    SelectionResidual,
    SelectionTerm,
    SurfaceFeatures,
    feature_scales,
    fit_lambda,
    lambda_by_step,
    ledger_series,
    steps_from_run,
)
from reward_lens.measure.ledger.nulls import permuted_step_null, summarise
from reward_lens.measure.ledger.prediction import (
    LAMBDA_LEAD_TIME_SPEC,
    METRIC_MIDPOINT_LEAD,
    freeze_prediction,
    score_lead_time,
    transition_window,
)
from reward_lens.measure.rate.regime import RegimeInputs, measure_regime
from reward_lens.record.reader import open_run

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "grpo_run"
ACCESS = {Component.RECORD: Access.RECORD}

#: Where the AISI rollout parquet is expected. Overridable, because a 189 MB artifact does not
#: belong in a git repository and every reader will keep it somewhere different.
AISI_PATH = os.environ.get("REWARD_LENS_AISI_ROLLOUTS", "")


def _open(which: str):
    root = FIXTURES / which
    if not root.exists() or not (root / "runs").exists():
        return None
    try:
        run_id = next(p.name for p in (root / "runs").iterdir())
    except StopIteration:
        return None
    return open_run(root, run_id.replace("run_", "run:"))


def _ctx(**kwargs) -> Context:
    return Context(access=dict(ACCESS), substrate=Substrate.PROGRAM, phase=Phase.POST_RUN, **kwargs)


@pytest.fixture(scope="module")
def long_run():
    """The longest real GRPO record on disk, which is the 200-step one when it is there.

    Falls back to the 12-step record rather than skipping, because everything asserted below is a
    property of the ledger rather than of the run length and the counts are derived from the record
    rather than written down. A `Λ` fit needs a series; twelve steps is a short one and it is still
    a series, and a package that only tests when the long fixture happens to be present is a
    package with no test.
    """
    run = _open("long") or _open("short")
    if run is None:
        pytest.skip(f"no GRPO record is on disk under {FIXTURES}")
    return run


@pytest.fixture(scope="module")
def aisi_table():
    if not AISI_PATH or not Path(AISI_PATH).exists():
        pytest.skip(
            "the AISI rollout parquet is not on disk. Set REWARD_LENS_AISI_ROLLOUTS to it; the "
            "download command is in this module's docstring."
        )
    from reward_lens.measure.ledger.labelled import read_parquet

    return read_parquet(AISI_PATH)


# ---------------------------------------------------------------------------
# 1. The four instruments exist, declare what the contract requires, and run
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls", [SelectionTerm, SelectionResidual, SelectionExplainedFraction, EffectiveStepSize]
)
def test_every_instrument_satisfies_the_protocol_and_passes_lint(cls, long_run):
    instrument = cls(long_run, SurfaceFeatures())
    assert isinstance(instrument, Instrument)
    assert lint_instrument(instrument) == []
    assert instrument.requires == {Component.RECORD: Access.RECORD}
    assert instrument.phases == frozenset({Phase.IN_RUN, Phase.POST_RUN})
    assert instrument.baselines


def test_f1_declares_the_three_envelope_conditions_the_catalogue_gives_it(long_run):
    envelope = SelectionTerm(long_run, SurfaceFeatures()).envelope
    assert envelope.requires == frozenset(
        {
            RegimeCondition.LINEAR_RESPONSE,
            RegimeCondition.GROUP_NONDEGENERATE,
            RegimeCondition.NEAR_POLICY,
        }
    )
    # Every condition in `requires` resolves to a registered measuring quantity, which
    # `EnvelopeSpec.__post_init__` enforces and this restates as a readable assertion.
    assert envelope.measured_by[RegimeCondition.LINEAR_RESPONSE] == "selection.explained_fraction"


def test_f2_does_not_require_the_condition_it_measures(long_run):
    """`Λ` is what `LINEAR_RESPONSE` is measured by, so requiring it would be circular."""
    envelope = SelectionExplainedFraction(long_run, SurfaceFeatures()).envelope
    assert RegimeCondition.LINEAR_RESPONSE not in envelope.requires
    assert envelope.requires == frozenset(
        {RegimeCondition.GROUP_NONDEGENERATE, RegimeCondition.NEAR_POLICY}
    )


# ---------------------------------------------------------------------------
# 2. The mechanics, on the real 200-step GRPO record
# ---------------------------------------------------------------------------


def test_lambda_is_computed_per_step_on_the_real_grpo_record(long_run):
    """The clause's first half: `Λ` computed on a real optimisation trace, per step and pooled."""
    n_steps = len(long_run.steps.indices)
    assert n_steps >= 12

    featuriser = SurfaceFeatures()
    samples = steps_from_run(long_run, featuriser)
    assert len(samples) == n_steps
    assert all(s.n == 8 for s in samples)
    assert all(s.advantage_source == "recorded" for s in samples)

    ledgers = ledger_series(samples, eta=1.0)
    assert len(ledgers) == n_steps - 1

    scales = feature_scales(samples)
    fit = fit_lambda(ledgers, scales, n_bootstrap=500)
    assert fit is not None
    assert fit.n_steps == n_steps - 1
    # Two of the five surface features are constant on this run: the model emits twelve distinct
    # tokens every time, so the type-token ratio is exactly 1, and every trajectory has two turns.
    # A constant feature has no scale to be expressed in and is dropped by name rather than
    # divided by zero.
    assert set(fit.dropped) == {"type_token_ratio", "n_turns"}
    assert fit.n_features == 3
    assert 0.0 <= fit.lambda_ <= 1.0
    assert fit.ci_low <= fit.lambda_ <= fit.ci_high

    context = 5
    per_step = lambda_by_step(ledgers, scales, context=context)
    assert len(per_step) == max(0, len(ledgers) - 2 * context)
    steps = [s for s, _ in per_step]
    assert steps == sorted(steps)
    assert all(0.0 <= v <= 1.0 or math.isnan(v) for _, v in per_step)


def test_the_first_order_picture_does_not_carry_this_run_and_the_reading_says_so(long_run):
    """`Λ` is near zero here and that is the finding, not a failure.

    Eight rollouts per step, a 0.6M-parameter model, a toy length grader, and a fresh pair of
    prompts at every step: `Δz` is dominated by which prompts the batch happened to draw. The
    reading carries the task overlap for exactly this reason and it is zero on every pair.
    """
    samples = steps_from_run(long_run, SurfaceFeatures())
    ledgers = ledger_series(samples, eta=1.0)
    assert all(led.task_overlap == 0.0 for led in ledgers)
    assert any("share no prompt" in note for led in ledgers for note in led.notes)

    fit = fit_lambda(ledgers, feature_scales(samples), n_bootstrap=500)
    assert fit.lambda_ < 0.1
    assert not fit.is_certificate


def test_lambda_beats_its_permuted_step_null_or_says_it_does_not(long_run):
    """The baseline the catalogue names for F2, run on the real record.

    On this run `Λ` is 0.0012 and the permuted-step null median is of the same order, so the
    honest verdict is that the claim is not distinguishable from the null. Asserting that the
    comparison ran and produced a p-value is the acceptance condition; asserting that it wins would
    be asserting a result this run does not contain.
    """
    samples = steps_from_run(long_run, SurfaceFeatures())
    ledgers = ledger_series(samples, eta=1.0)
    scales = feature_scales(samples)
    fit = fit_lambda(ledgers, scales, n_bootstrap=0)
    draws = permuted_step_null(ledgers, scales, n_draws=500, seed=0)
    result = summarise("baseline.permuted_step", "Lambda", fit.lambda_, draws)
    assert result.n_draws == 500
    assert 0.0 < result.p_value <= 1.0
    assert result.baseline == "baseline.permuted_step"


def test_lambda_is_the_certificate_f1_is_checked_against_and_the_loop_closes(long_run):
    """The whole point of F2, exercised end to end on a real record.

    F2 runs on the bare record and produces `Λ`. F1 refuses while `Λ` is unknown, because unknown
    is not a pass. Feeding `Λ` back into the regime reading turns the refusal from "nobody
    measured it" into "it was measured and it fails", which are different states and must render
    differently.
    """
    featuriser = SurfaceFeatures()
    indices = sorted(long_run.steps.indices)
    window = (min(indices), max(indices) + 1)
    regime = measure_regime(long_run, window=window)
    assert regime.holds(RegimeCondition.LINEAR_RESPONSE) is None

    reading = SelectionExplainedFraction(long_run, featuriser, n_bootstrap=200).estimate(
        _ctx(regime_reading=regime)
    )
    assert isinstance(reading, Evidence)
    lam = reading.value["lambda"]

    unknown = SelectionTerm(long_run, featuriser).estimate(_ctx(regime_reading=regime))
    assert isinstance(unknown, Refusal)
    assert unknown.reason is RefusalReason.ENVELOPE_VIOLATED
    assert "Lambda is not available" in unknown.detail

    informed = measure_regime(long_run, window=window, inputs=RegimeInputs(explained_fraction=lam))
    assert informed.holds(RegimeCondition.LINEAR_RESPONSE) is False
    measured = SelectionTerm(long_run, featuriser).estimate(_ctx(regime_reading=informed))
    assert isinstance(measured, Refusal)
    assert measured.reason is RefusalReason.ENVELOPE_VIOLATED
    assert "FAIL" in measured.detail
    assert "smaller learning rate" in measured.remedy


def test_f1_emits_a_closed_ledger_when_the_envelope_holds(long_run):
    """With the envelope satisfied, both halves emit and the identity closes on every row.

    The regime is supplied rather than measured here, which is what a caller does when they have
    established the conditions another way. The assertion is the identity itself: for every feature
    at every step, `Δz` equals the selection term plus the residual to floating-point precision.
    """
    from reward_lens.core.envelope import RegimeReading

    holds = RegimeReading.of(LINEAR_RESPONSE=True, GROUP_NONDEGENERATE=True, NEAR_POLICY=True)
    ctx = _ctx(regime_reading=holds)
    term = SelectionTerm(long_run, SurfaceFeatures(), eta="schedule").estimate(ctx)
    residual = SelectionResidual(long_run, SurfaceFeatures(), eta="schedule").estimate(ctx)
    assert isinstance(term, Evidence) and isinstance(residual, Evidence)
    assert term.quantity == "selection.term"
    assert residual.quantity == "selection.residual"

    selection = np.asarray(term.value["selection"])
    delta_z = np.asarray(residual.value["delta_z"])
    rho = np.asarray(residual.value["residual"])
    assert selection.shape == delta_z.shape == rho.shape
    assert np.allclose(delta_z, selection + rho, atol=1e-12)

    assert term.value["eta_source"] == "schedule"
    assert term.value["operator"] == "within_group"
    assert term.value["advantage_source"] == "recorded"


def test_the_residual_dominates_on_this_run_which_is_f1s_own_kill_condition(long_run):
    """F1's catalogue kill condition: if `ρ` is the size of the selection term for every feature,
    the first-order picture explains nothing. On this run it does dominate, and the instrument
    reports that rather than hiding it."""
    from reward_lens.core.envelope import RegimeReading

    ctx = _ctx(
        regime_reading=RegimeReading.of(
            LINEAR_RESPONSE=True, GROUP_NONDEGENERATE=True, NEAR_POLICY=True
        )
    )
    residual = SelectionResidual(long_run, SurfaceFeatures(), eta="schedule").estimate(ctx)
    names = list(residual.value["features"])
    dominates = np.asarray(residual.value["residual_dominates"])
    # The two constant features have `Δz` and `Cov` both exactly zero, so neither half dominates
    # the other and both flags are False. Scoring the kill condition over them would dilute it with
    # rows where there was nothing to explain.
    varying = [i for i, n in enumerate(names) if n not in ("type_token_ratio", "n_turns")]
    assert dominates[:, varying].mean() > 0.95
    assert not dominates[:, [i for i in range(len(names)) if i not in varying]].any()
    assert "se_delta_z" in residual.value


def test_a_reading_under_the_frozen_study_is_registered(long_run):
    """Gate 3, on this package's own instrument: a frozen StudyID makes the reading REGISTERED."""
    from reward_lens.core.envelope import RegimeReading

    frozen = freeze_prediction()
    ctx = _ctx(
        regime_reading=RegimeReading.of(GROUP_NONDEGENERATE=True, NEAR_POLICY=True),
        study=str(frozen.study_id),
    )
    reading = SelectionExplainedFraction(long_run, SurfaceFeatures(), n_bootstrap=0).estimate(ctx)
    assert isinstance(reading, Evidence)
    assert reading.trust is TrustLevel.REGISTERED


# ---------------------------------------------------------------------------
# 3. The prediction, frozen before the labelled series was read
# ---------------------------------------------------------------------------


def test_the_prediction_is_frozen_and_names_its_comparator():
    frozen = freeze_prediction(frozen_at="2026-08-05T00:00:00+00:00")
    assert frozen.spec_hash == "spec:9551a981f5cac07f78f7890c562a24a3"
    assert frozen.study_id == "study:f2-lambda-lead-time@v1#9551a981"
    metrics = {h.prediction.metric for h in LAMBDA_LEAD_TIME_SPEC.hypotheses}
    assert metrics == {"lambda_lead_fraction", "lambda_minus_gradnorm_lead_fraction"}
    assert LAMBDA_LEAD_TIME_SPEC.kill_criteria[0].metric == "hack_rate_transition_width_steps"


# ---------------------------------------------------------------------------
# 4. The clause, on the AISI labelled series
# ---------------------------------------------------------------------------


def test_the_labelled_series_reads_as_a_step_axis_and_not_as_a_row_counter(aisi_table):
    """The recorded trap: `rollout_index` is per eval file. The guard is that the two counts agree."""
    from reward_lens.measure.ledger.labelled import check_step_axis

    n_steps, n_files = check_step_axis(aisi_table)
    assert n_steps == n_files == 401


def test_the_labelled_rate_counts_nulls_separately_from_negatives(aisi_table):
    """`reward_hacked` is `int64` with 1, 0 or null. A null is unscored, never a zero."""
    from reward_lens.measure.ledger.labelled import label_rate

    rates = label_rate(aisi_table)
    assert len(rates) == 401
    assert all(r.n_total == 64 for r in rates)
    assert all(r.n_labelled + r.n_null == r.n_total for r in rates)
    # This series happens to carry no nulls; the arithmetic that keeps them out of the denominator
    # is asserted regardless, because the companion kl0.02 series is a different length and the
    # card documents nulls as possible in both.
    assert sum(r.n_null for r in rates) == 0


def test_lambda_per_step_against_the_labelled_hack_rate(aisi_table):
    """The clause. `Λ` per step on a labelled series, placed against the hack rate.

    Whatever the relationship, it is the first such measurement, and the relationship here is that
    `Λ` **lags**. The assertions fix the sign and the orders of magnitude rather than the exact
    numbers: the magnitude of the lag moves with how degenerate groups are treated (0.93 widths
    with them included, 3.6 with them masked out) and the sign does not.
    """
    from reward_lens.measure.ledger.labelled import label_rate, rate_series, steps_from_table

    steps_h, rate_h = rate_series(label_rate(aisi_table))
    assert len(steps_h) == 401
    assert rate_h[0] < 0.05 and rate_h[-1] > 0.9

    samples = steps_from_table(aisi_table)
    assert len(samples) == 401
    assert all(s.advantage_source == "reconstructed" for s in samples)
    ledgers = ledger_series(samples, eta=1.0)
    scales = feature_scales(samples)

    fit = fit_lambda(ledgers, scales, n_bootstrap=500)
    assert fit.n_steps == 400
    assert fit.ci_low > 0.0
    assert fit.eta_eff > 0.0
    assert fit.eta_eff / fit.se_eta_eff > 3.0

    per_step = lambda_by_step(ledgers, scales, context=5)
    lam_steps = np.asarray([s for s, _ in per_step], dtype=float)
    lam_values = np.asarray([v for _, v in per_step], dtype=float)
    assert len(per_step) == 390

    hack = transition_window(rate_h, steps_h)
    assert hack.fitted
    assert 90.0 < hack.t50 < 125.0
    assert 10.0 < hack.width < 60.0

    result = score_lead_time(lam_values, lam_steps, rate_h, steps_h)
    assert result.resolved
    assert result.transition.fitted

    # The measurement the clause asks for, and it is a lag rather than a lead.
    assert result.lambda_transition is not None and result.lambda_transition.fitted
    assert result.lambda_transition.t50 > hack.t50
    assert result.metrics()[METRIC_MIDPOINT_LEAD] < 0.0

    # Before the transition, Lambda sits in the noise band; after it, it is an order of magnitude
    # larger. That is the measurement, stated without a detector.
    before = lam_values[lam_steps < hack.t50 - hack.width]
    after = lam_values[lam_steps > hack.t50 + 2 * hack.width]
    assert float(np.nanmedian(before)) < 0.05
    assert float(np.nanmedian(after)) > 0.20


def test_the_registered_cusum_metric_does_not_resolve_the_question_on_this_series(aisi_table):
    """The frozen analysis runs and its answer is an artefact, which is recorded rather than used.

    With `baseline=None` the CUSUM standardises `Λ` against a mean the post-transition regime
    raised, so every early point sits about 1.3 sigma below it, the negative accumulator crosses
    within seven points, and the reported lead of +3.97 widths is the detector using the future to
    define normal. The sign it reports is the opposite of the one the two fitted transitions agree
    on, which is what makes this worth asserting rather than quietly correcting.
    """
    from reward_lens.measure.ledger.labelled import label_rate, rate_series, steps_from_table

    steps_h, rate_h = rate_series(label_rate(aisi_table))
    samples = steps_from_table(aisi_table)
    ledgers = ledger_series(samples, eta=1.0)
    per_step = lambda_by_step(ledgers, feature_scales(samples), context=5)
    lam_steps = np.asarray([s for s, _ in per_step], dtype=float)
    lam_values = np.asarray([v for _, v in per_step], dtype=float)

    result = score_lead_time(lam_values, lam_steps, rate_h, steps_h)
    assert result.lambda_lead > 3.0  # the registered metric says Lambda leads by four widths
    assert result.midpoint_lead < 0.0  # the two fitted transitions say it lags
    assert result.lambda_lead * result.midpoint_lead < 0.0  # and they disagree in sign

    # H2 cannot be resolved here at all: a rollout table carries no optimiser telemetry.
    assert math.isnan(result.metrics()["lambda_minus_gradnorm_lead_fraction"])
    assert "no optimiser telemetry" in result.detail

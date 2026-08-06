"""Acceptance: the controls bank. M3, M4, M5 and M10.

The clauses this file discharges: *a claim without a baseline fails lint; a null without a matched
control is refused; all six baselines run through one interface and return comparable readings; a
detector that a string match matches is reported as
matched rather than as a win; the semantic placebo bank returns a coherent irrelevant direction and
selecting one is the default path; the power calculation is validated against simulation, and the
simulation is in the test; the resolution ratio q = N/N\\* is computed and a q < 1 case reports "not
resolved" rather than a verdict.*

Two of those need saying twice. **The power validation is against a simulation written here, not
against the module's own simulator**, so the check is not the module agreeing with itself: this
file draws the paired outcomes and calls `scipy.stats.binomtest` once per replicate, which is a
different code path from the vectorised binomial tail the module uses. And **the placebo default
path is tested by calling it with no contrast argument**, because a control that requires a
decision is a control that gets skipped.

The four instruments' generated invariance tests are here too, under `reward.affine` for three of
them and `units` for M10, whose registered quantities carry that group. The `reward.affine` checks
are not vacuous: each one confirms that a reading which is supposed to be a control does not move
when the reward it is controlling for is rescaled.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from reward_lens.core.invariance import (
    INVARIANT,
    InvariancePayload,
    check_invariance,
    check_unit_refusal,
)
from reward_lens.core.quantity import QUANTITIES, load_quantities
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.measure.base import Context, lint_instrument
from reward_lens.measure.controls import (
    CONTROLS,
    PLACEBO_BANK,
    ControlDesign,
    DumbBaselineBank,
    InterventionArm,
    MatchedControl,
    MatchedPositiveControl,
    NullClaim,
    PowerAndMDE,
    SemanticPlacebo,
    compare_to_placebo,
    gate_null,
    guard_null,
    register_proposed,
    resolve_row,
    semantic_placebo,
)
from reward_lens.measure.controls.quantities import PROPOSED
from reward_lens.stats.baselines import (
    ALL_SIX,
    BaselineScore,
    DetectionTask,
    compare_against_baselines,
    lint_claim,
    run_bank,
)
from reward_lens.stats.power import (
    DEFAULT_CLOSE_PAIR,
    DIMENSIONLESS,
    EFFECT,
    USES_CORRELATION,
    PairedBinaryDesign,
    PowerQuantity,
    detection_band,
    difference,
    mcnemar_p_values,
    required_n,
    resolution_ratio,
    simulate_power,
)

WORDS = ("alpha", "beta", "gamma", "delta", "epsilon")


def _ctx() -> Context:
    """No signal. These four read injected data, which is why they can run at all."""
    return Context()


def _task(n: int = 160, seed: int = 0, *, marker: str = "exit(0)") -> DetectionTask:
    """A task with a planted marker: the shape the published AUC 0.998 case had."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.5).astype(int)
    texts = tuple(
        " ".join(rng.choice(WORDS, size=int(rng.integers(20, 40))))
        + (f" {marker} " + "the same line again " * 5 if label else "")
        for label in y
    )
    return DetectionTask(
        labels=y,
        texts=texts,
        prompts=tuple(f"task {i}" for i in range(n)),
        series=rng.normal(0.0, 1.0, n) + 0.9 * y,
        markers=(marker,),
        judge=lambda prompt: float(len(prompt) % 13) / 13.0,
        name="acceptance",
    )


# ===========================================================================
# Clause: a claim without a baseline fails lint
# ===========================================================================


def test_a_claim_without_a_baseline_fails_lint():
    """The clause, verbatim. Three ways to have no baseline, all three findings."""
    assert lint_claim(SimpleNamespace(instrument="Probe"))  # no mapping at all
    assert lint_claim({"instrument": "Probe", "baselines": {}})  # an empty one
    partial = {"instrument": "Probe", "baselines": {"baseline.length": 0.61}}
    findings = lint_claim(partial)
    assert len(findings) == len(ALL_SIX) - 1
    assert all("skipped silently" in f.problem for f in findings)


def test_an_instrument_with_no_baseline_declaration_fails_instrument_lint():
    """The same rule one level up: the declaration, not the reading."""

    class Undeclared:
        name = "Undeclared"
        quantity = "study.power"
        baselines: tuple = ()
        envelope = None
        invariance = ""

    load_quantities()
    fields = {f.field for f in lint_instrument(Undeclared())}
    assert "baselines" in fields


def test_a_claim_that_ran_the_bank_passes_lint_and_records_the_baselines_it_could_not_run():
    """A refusal is a result. Five scores and one recorded refusal is a complete bank."""
    rng = np.random.default_rng(1)
    y = (rng.random(80) < 0.5).astype(int)
    no_series = DetectionTask(
        labels=y,
        texts=tuple(" ".join(rng.choice(WORDS, size=25)) for _ in range(80)),
        markers=("exit(0)",),
    )
    bank = run_bank(no_series)
    claim = {"instrument": "Probe", "baselines": bank.as_mapping()}
    assert lint_claim(claim, bank) == []
    assert "baseline.gradnorm_peak" in bank.refusals()
    assert lint_claim(claim) != []  # with no bank the refusal looks like an omission


# ===========================================================================
# Clause: all six run through one interface and return comparable readings
# ===========================================================================


def test_all_six_baselines_run_through_one_interface_and_return_comparable_readings():
    bank = run_bank(_task())
    assert set(bank.readings) == set(ALL_SIX) and len(ALL_SIX) == 6
    for bid, reading in bank.readings.items():
        assert isinstance(reading, BaselineScore), f"{bid} did not score on a complete task"
        assert 0.0 <= reading.auroc <= 1.0
        assert reading.n == 160 and reading.scores.shape == (160,)
        assert reading.wall_ms >= 0.0
    # Comparable means rankable: one call orders all six on one scale.
    best = bank.best()
    assert best is not None and best.auroc == max(s.auroc for s in bank.scored().values())


def test_the_bank_names_the_cost_of_every_comparison_it_ran():
    """A 1 ms baseline beside a one-call-per-item judge is a finding, not a footnote."""
    rendered = run_bank(_task()).render()
    for bid in ALL_SIX:
        assert bid in rendered
    assert "parameter" in rendered


# ===========================================================================
# Clause: a detector a string match matches is reported as matched, not as a win
# ===========================================================================


def test_a_detector_that_a_string_match_matches_is_reported_as_matched_not_as_a_win():
    """The published case, in miniature: a strong detector against a zero-parameter comparator."""
    task = _task()
    bank = run_bank(task, ["baseline.string_match"])
    string_match = bank.readings["baseline.string_match"]
    assert string_match.n_parameters == 0 and not string_match.fitted
    assert string_match.auroc == pytest.approx(1.0)

    # A detector that agrees with the string match on all but a handful of items.
    rng = np.random.default_rng(7)
    own = string_match.scores.astype(float) + rng.normal(0.0, 0.02, task.n)
    verdict = compare_against_baselines(own, task.labels, bank, seed=0)
    assert verdict.verdict in {"matched", "baseline_wins"}
    assert not verdict.is_win
    assert "baseline.string_match" in verdict.render()
    assert "win" in verdict.render()  # only as the thing it is not


def test_a_claim_that_beats_nothing_is_not_allowed_to_report_a_win():
    task = _task()
    bank = run_bank(task)
    rng = np.random.default_rng(8)
    noise = rng.normal(0.0, 1.0, task.n)
    assert compare_against_baselines(noise, task.labels, bank, seed=0).verdict == "baseline_wins"


# ===========================================================================
# Clause: a null without a matched control is refused
# ===========================================================================


def test_a_null_without_a_matched_control_is_refused():
    """The clause, verbatim, with the library's own 0.13-power card as the shape."""
    claim = NullClaim(
        instrument="Susceptibility",
        effect=0.01,
        p_value=0.62,
        design=ControlDesign(n=40, alpha=0.05, statistic="paired-permutation"),
        mde=0.18,
    )
    verdict = gate_null(claim, None)
    assert not verdict.ok
    refusal = verdict.refusal
    assert isinstance(refusal, Refusal)
    assert refusal.reason is RefusalReason.NO_MATCHED_CONTROL
    assert "underpowered" in refusal.detail
    # A remedy is an instruction.
    assert "run the same measurement on a matched positive control at the same n" in refusal.remedy
    assert "report this as underpowered rather than null" in refusal.remedy


def test_the_gate_is_real_rather_than_remembered():
    """An instrument that never thought about controls still cannot ship a bare null."""

    class Forgetful:
        name = "Forgetful"

        def preflight(self, ctx):
            from reward_lens.measure.base import PreflightResult

            return PreflightResult(instrument=self.name, ok=True)

        def estimate(self, ctx):
            return ctx.emit({"effect": 0.004, "p_value": 0.81, "n": 32, "alpha": 0.05})

    out = guard_null(Forgetful()).estimate(_ctx())
    assert isinstance(out, Refusal) and out.reason is RefusalReason.NO_MATCHED_CONTROL


def test_a_control_that_missed_its_own_planted_effect_does_not_license_the_null():
    claim = NullClaim(
        instrument="Susceptibility",
        effect=0.01,
        p_value=0.62,
        design=ControlDesign(n=40, alpha=0.05),
    )
    missed = MatchedControl(
        id="planted",
        design=ControlDesign(n=40, alpha=0.05),
        planted_effect=0.30,
        observed_effect=0.06,
        p_value=0.39,
    )
    verdict = gate_null(claim, missed)
    assert not verdict.ok and "cannot detect effects of that size" in verdict.refusal.detail


# ===========================================================================
# Clause: the placebo bank, and selecting one is the default path
# ===========================================================================


def test_the_semantic_placebo_bank_returns_a_coherent_irrelevant_direction_by_default():
    """No contrast argument. A control that needs a decision is a control that gets skipped."""

    def encode(phrases):
        rng = np.random.default_rng(0)
        table = {w: rng.standard_normal(48) for w in _vocabulary()}
        return np.stack(
            [np.mean([table[w] for w in p.lower().split() if w in table], axis=0) for p in phrases]
        )

    claimed = np.random.default_rng(2).standard_normal(48) * 4.0
    direction = semantic_placebo(encode, match_to=claimed)
    assert direction.contrast in {c.id for c in PLACEBO_BANK}
    assert direction.is_norm_matched
    assert direction.norm == pytest.approx(float(np.linalg.norm(claimed)))
    assert direction.vector.shape == claimed.shape


def _vocabulary() -> set[str]:
    words: set[str] = set()
    for c in PLACEBO_BANK:
        for phrase in c.positive + c.negative:
            words.update(phrase.lower().split())
    return words


def test_the_bank_is_coherent_and_argued_rather_than_a_word_list():
    assert len(PLACEBO_BANK) >= 6
    ids = [c.id for c in PLACEBO_BANK]
    assert len(set(ids)) == len(ids)
    assert "vampires_vs_werewolves" in ids  # the published one this instrument exists because of
    for c in PLACEBO_BANK:
        assert len(c.positive) >= 4 and len(c.negative) >= 4
        assert c.rationale.strip()


def test_a_steering_claim_with_no_placebo_arm_is_refused():
    out = compare_to_placebo(InterventionArm("reward-hacking direction", effect=-0.44), None)
    assert isinstance(out, Refusal) and out.reason is RefusalReason.NO_MATCHED_CONTROL
    assert "semantic_placebo(encode, match_to=your_direction)" in out.remedy


def test_a_placebo_that_suppresses_as_well_as_the_claim_reports_the_claim_as_non_specific():
    """The published failure: a vampires direction did the job the real direction claimed."""
    rng = np.random.default_rng(3)
    real = InterventionArm("hacking direction", -0.40, per_item=rng.normal(-0.40, 0.12, 240))
    placebo = InterventionArm("vampires", -0.40, per_item=rng.normal(-0.40, 0.12, 240))
    out = compare_to_placebo(real, placebo, seed=0)
    assert out.verdict == "non_specific"
    assert "NOT SPECIFIC" in out.render()


# ===========================================================================
# Clause: the power calculation is validated against simulation, in this test
# ===========================================================================


def _independent_paired_draw(design: PairedBinaryDesign, replicates: int, rng) -> np.ndarray:
    """Draw the two systems' per-item correctness directly, not through the module.

    Builds the joint from the marginals and the correlation here, samples each item's cell with a
    uniform draw against the cumulative probabilities, and returns the (b, c) discordant counts.
    This is a different code path from `stats.power.simulate_power`, which is the point.
    """
    p11, p10, p01, p00 = design.cells
    edges = np.cumsum([p11, p10, p01, p00])
    u = rng.random((replicates, design.n))
    cell = np.searchsorted(edges, u, side="right")
    b = (cell == 1).sum(axis=1)
    c = (cell == 2).sum(axis=1)
    return np.stack([b, c], axis=1)


def _independent_power(design: PairedBinaryDesign, replicates: int, seed: int) -> float:
    """Empirical rejection rate using `scipy.stats.binomtest` once per replicate."""
    from scipy.stats import binomtest

    counts = _independent_paired_draw(design, replicates, np.random.default_rng(seed))
    rejected = 0
    for b, c in counts:
        m = int(b + c)
        if m == 0:
            continue
        if binomtest(int(b), m, 0.5, alternative="two-sided").pvalue <= design.alpha:
            rejected += 1
    return rejected / replicates


def test_the_modules_exact_p_values_match_an_independent_binomtest():
    from scipy.stats import binomtest

    b = np.array([0, 1, 3, 5, 10, 12, 20, 25])
    c = np.array([0, 4, 3, 12, 10, 20, 25, 20])
    mine = mcnemar_p_values(b, c, tails=2)
    for i, (bi, ci) in enumerate(zip(b, c)):
        m = int(bi + ci)
        expected = 1.0 if m == 0 else binomtest(int(bi), m, 0.5).pvalue
        assert mine[i] == pytest.approx(expected, abs=1e-12)


def test_the_power_calculation_is_validated_against_a_simulation_written_here():
    """`required_n` returns an n; this test re-simulates that n by an independent path."""
    design = PairedBinaryDesign(n=1, accuracy_a=0.82, accuracy_b=0.85, rho=0.5)
    n_star = required_n(design, target_power=0.8, replicates=6000, seed=0)
    at_target = _independent_power(design.at_n(n_star), replicates=3000, seed=99)
    assert at_target == pytest.approx(0.80, abs=0.04), at_target
    below = _independent_power(design.at_n(int(n_star * 0.6)), replicates=3000, seed=99)
    assert below < 0.75, below


def test_the_simulated_size_at_a_zero_effect_is_the_nominal_alpha():
    """The simulator's calibration, checked where the answer is known without any formula."""
    null = PairedBinaryDesign(n=400, accuracy_a=0.80, accuracy_b=0.80, rho=0.5, alpha=0.05)
    assert _independent_power(null, replicates=3000, seed=5) <= 0.06
    assert simulate_power(null, replicates=20_000, seed=0).power <= 0.06


def test_three_of_the_five_standard_calculators_are_roughly_2x_wrong_here():
    """Validated against the simulation, not against another formula. The finding, reproduced."""
    from reward_lens.stats.power import compare_calculators

    checks = compare_calculators(DEFAULT_CLOSE_PAIR, replicates=8000, seed=0)
    wrong = {name for name, c in checks.items() if c.roughly_2x_wrong}
    assert wrong == {name for name, uses in USES_CORRELATION.items() if not uses}
    assert len(wrong) == 3
    assert all(1.6 <= checks[name].ratio <= 2.3 for name in wrong)


def test_absence_of_signal_above_the_detection_band_reads_as_not_measurable():
    band = detection_band(
        n=400,
        logit_advantage=0.5,
        replicates=2500,
        grid=tuple(np.round(np.arange(0.55, 0.995, 0.05), 4)),
        seed=0,
    )
    assert band.high < 0.98
    verdict = band.interpret(0.99, detected=False)
    assert "NOT MEASURABLE" in verdict
    assert "not as unbiased" in verdict
    assert "no bias detected" not in verdict


# ===========================================================================
# Clause: q = N/N* is computed and q < 1 reports "not resolved" rather than a verdict
# ===========================================================================


def test_the_resolution_ratio_is_computed_and_q_below_one_reports_not_resolved():
    design = PairedBinaryDesign(n=500, accuracy_a=0.82, accuracy_b=0.85, rho=0.5)
    reading = PowerAndMDE(design, replicates=3000, with_calculators=False).estimate(_ctx())
    q = reading.value["q"]
    assert q == pytest.approx(500 / reading.value["n_star"])
    assert q < 1.0 and reading.value["resolved"] is False

    resolution = resolution_ratio(500, reading.value["n_star"])
    assert "NOT RESOLVED" in resolution.render()
    assert "unresolved rather than as a result" in resolution.render()

    refusal = resolve_row("Leaderboard row 7", resolution, observed_gap=0.03)
    assert isinstance(refusal, Refusal)
    assert "not resolved" in refusal.detail
    assert refusal.statistics["q"] == pytest.approx(q, abs=1e-9)


def test_a_resolved_row_is_allowed_to_report_a_verdict():
    out = resolve_row("Leaderboard row 1", resolution_ratio(4000, 1274))
    assert not isinstance(out, Refusal) and out.resolved


# ===========================================================================
# Lint and the generated invariance tests
# ===========================================================================


def test_lint_instrument_is_empty_for_all_four_controls():
    load_quantities()
    register_proposed()
    for cls in CONTROLS:
        assert lint_instrument(cls()) == [], cls.__name__


def test_the_three_proposed_quantities_are_the_only_registration_this_package_needs():
    """M3 and M4 have no registered id (E14); M5 and M10 estimate `study.power`."""
    load_quantities()
    assert {q.id for q in PROPOSED} == {
        "baseline.best_score",
        "baseline.margin",
        "placebo.effect_ratio",
    }
    assert MatchedPositiveControl.quantity == "study.power"
    assert PowerAndMDE.quantity == "study.power"
    assert "study.power" in QUANTITIES and "study.resolution_ratio" in QUANTITIES


def test_m3_is_invariant_under_an_affine_rescaling_of_the_reward():
    """The bank reads the transcript and the label, so a reward rescaling must not move it."""
    rng = np.random.default_rng(11)
    n = 120
    labels = (rng.random(n) < 0.5).astype(int)

    def run(inst, payload):
        task = DetectionTask(labels=labels, series=np.asarray(payload.scores), name="invariance")
        bank, _ = DumbBaselineBank(task, which=["baseline.gradnorm_peak"]).compute()
        return bank.best().auroc

    payload = InvariancePayload(scores=rng.normal(0.0, 1.0, n) + 0.8 * labels)
    report = check_invariance(
        DumbBaselineBank(), "reward.affine", payload, n=12, relation=INVARIANT, run=run
    )
    assert report.passed, report.render()
    assert report.n == 12  # not vacuous: the group really was drawn and applied


def test_m4_is_invariant_under_an_affine_rescaling_of_the_reward():
    """A ratio of two effects in reward units: `a` cancels in the ratio, `b` in each difference."""

    def run(inst, payload):
        s = np.asarray(payload.scores, dtype=np.float64)
        k = s.size // 4
        claimed = InterventionArm("real", float(s[:k].mean() - s[k : 2 * k].mean()))
        placebo = InterventionArm("vampires", float(s[2 * k : 3 * k].mean() - s[3 * k :].mean()))
        return SemanticPlacebo(claimed, placebo).compute().ratio

    rng = np.random.default_rng(12)
    payload = InvariancePayload(scores=rng.normal(0.0, 1.0, 400))
    report = check_invariance(
        SemanticPlacebo(), "reward.affine", payload, n=12, relation=INVARIANT, run=run
    )
    assert report.passed, report.render()


def test_m5_is_invariant_under_an_affine_rescaling_of_the_reward():
    """A scale-free paired test gives the same p-value, so the same verdict."""
    from reward_lens.stats.effects import paired_permutation_test

    design = ControlDesign(n=50, alpha=0.05, statistic="paired-permutation")

    def run(inst, payload):
        s = np.asarray(payload.scores, dtype=np.float64)
        half = s.size // 2
        p = paired_permutation_test(s[:half], s[half:], n_permutations=400, seed=0)
        claim = NullClaim("X", float(s[:half].mean() - s[half:].mean()), p, design)
        control = MatchedControl("planted", design, 0.3, 0.31, 1e-5)
        verdict = MatchedPositiveControl(claim, control).compute()
        return 1.0 if verdict.ok else 0.0

    rng = np.random.default_rng(13)
    payload = InvariancePayload(scores=rng.normal(0.0, 1.0, 100))
    report = check_invariance(
        MatchedPositiveControl(), "reward.affine", payload, n=8, relation=INVARIANT, run=run
    )
    assert report.passed, report.render()


def test_m10_carries_the_units_group_and_its_assertion_is_a_refusal():
    """`study.power` is registered under `units`, whose assertion is a refusal, not a value."""
    load_quantities()
    assert QUANTITIES.get("study.power").invariance == "units"
    report = check_invariance(
        PowerAndMDE(), "units", InvariancePayload(), relation=INVARIANT, run=lambda i, p: 1.0
    )
    assert report.passed and "refusal" in report.skipped

    # The real assertion for this group, exercised: a power and an MDE do not subtract.
    assert check_unit_refusal(
        difference,
        PowerQuantity("study.power", 0.8, DIMENSIONLESS),
        PowerQuantity("study.mde", 0.03, EFFECT),
    )
    assert isinstance(
        difference(
            PowerQuantity("study.power", 0.8, DIMENSIONLESS),
            PowerQuantity("study.mde", 0.03, EFFECT),
        ),
        Refusal,
    )


def test_every_control_emits_a_payload_the_store_codec_can_encode():
    """`ValueCodec` raises on an unregistered payload type rather than degrading to a dict.

    So a payload carrying a dataclass nobody registered is now a loud failure at write time, which
    is the correct behaviour and worth pinning here: all four of these emit flat mappings of
    numbers, strings and lists on purpose, and a future field that quietly becomes a dataclass
    would break the store rather than the test.
    """
    design = PairedBinaryDesign(n=200, accuracy_a=0.82, accuracy_b=0.85, rho=0.5)
    readings = [
        DumbBaselineBank(_task(80), which=["baseline.string_match"]).estimate(_ctx()),
        PowerAndMDE(design, replicates=800, with_calculators=False).estimate(_ctx()),
    ]
    rng = np.random.default_rng(21)
    readings.append(
        SemanticPlacebo(
            InterventionArm("real", -0.4, per_item=rng.normal(-0.4, 0.1, 60)),
            InterventionArm("vampires", -0.05, per_item=rng.normal(-0.05, 0.1, 60)),
        ).estimate(_ctx())
    )
    claim = NullClaim("X", 0.01, 0.62, ControlDesign(n=40, alpha=0.05))
    control = MatchedControl("planted", ControlDesign(n=40, alpha=0.05), 0.3, 0.31, 1e-5)
    readings.append(MatchedPositiveControl(claim, control).estimate(_ctx()))

    for reading in readings:
        assert not isinstance(reading, Refusal), reading
        encoded = reading.envelope()["value"]
        # The codec tags a mapping payload; unwrap the tag rather than asserting its spelling.
        body = encoded.get("__map__", encoded)
        assert "baselines" in body, sorted(body)


def test_every_control_declares_an_envelope_and_a_group_rather_than_leaving_them_blank():
    for cls in CONTROLS:
        assert cls.envelope is not None, cls.__name__
        assert cls.invariance, cls.__name__
        assert cls.baselines, cls.__name__
        assert cls.faithful_to and cls.deviations, cls.__name__


def test_the_maths_behind_the_six_lineage_floor_is_the_one_documented():
    """The margin interval declines under five lineages because of resample count, not a guess."""
    assert math.comb(2 * 4 - 1, 4) < 40  # four lineages: 35 distinct resamples
    assert math.comb(2 * 5 - 1, 5) >= 40  # five: 126

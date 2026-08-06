"""Unit tests for the dumb-baseline bank (M3, `reward_lens.stats.baselines`).

Two things get checked here that are easy to get wrong and invisible when wrong. Every fitted
baseline scores every item out of fold, so a bank AUROC is a generalisation estimate rather than a
fit quality: the test for that is a random labelling, on which every baseline has to land near
0.5. And a baseline whose input is missing returns a refusal with a remedy rather than a number,
because a bank that quietly reports 0.5 for a comparator nobody ran is a gate that passes
everything.
"""

from __future__ import annotations

import numpy as np
import pytest

from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.stats.baselines import (
    ALL_SIX,
    BASELINES,
    BaselineScore,
    DetectionTask,
    claim_baselines,
    compare_against_baselines,
    lint_claim,
    run_bank,
    stratified_folds,
)
from reward_lens.stats.baselines.series import gradnorm_peak, smooth
from reward_lens.stats.baselines.text import (
    TfidfLogisticRegression,
    mine_markers,
    render_scaffold,
    scaffold_hash,
)

FILLER = ("alpha", "beta", "gamma", "delta", "epsilon", "zeta")


def _texts(labels: np.ndarray, rng: np.random.Generator, *, marker: str = "exit(0)") -> tuple:
    out = []
    for y in labels:
        body = " ".join(rng.choice(FILLER, size=int(rng.integers(20, 40))))
        if y == 1:
            body = f"{body} {marker} " + "same phrase again " * 6
        out.append(body)
    return tuple(out)


def _task(n: int = 120, *, seed: int = 0, signal: bool = True) -> DetectionTask:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.5).astype(int)
    labels_for_text = y if signal else np.zeros(n, dtype=int)
    return DetectionTask(
        labels=y,
        texts=_texts(labels_for_text, rng),
        series=rng.normal(0.0, 1.0, n) + (0.9 * y if signal else 0.0),
        prompts=tuple(f"task {i}" for i in range(n)),
        markers=("exit(0)",) if signal else (),
        judge=lambda prompt: float(len(prompt) % 11) / 11.0,
        name="unit",
    )


# ---------------------------------------------------------------------------
# The interface
# ---------------------------------------------------------------------------


def test_the_bank_holds_exactly_the_six_the_specification_names():
    assert set(ALL_SIX) == {
        "baseline.string_match",
        "baseline.length",
        "baseline.tfidf",
        "baseline.ngram_diversity",
        "baseline.scaffolded_prompt",
        "baseline.gradnorm_peak",
    }
    assert set(BASELINES) == set(ALL_SIX)


def test_every_baseline_declares_what_it_reads_and_what_it_costs():
    for bid, baseline in BASELINES.items():
        assert baseline.id == bid
        assert baseline.reads, f"{bid} declares no inputs, so its refusal cannot name one"
        assert baseline.version
        assert baseline.cost.render()


def test_all_six_run_through_one_call_and_return_the_same_reading_type():
    bank = run_bank(_task())
    assert len(bank.readings) == 6
    for reading in bank.readings.values():
        assert isinstance(reading, (BaselineScore, Refusal))
    assert len(bank.scored()) == 6
    for score in bank.scored().values():
        assert 0.0 <= score.auroc <= 1.0
        assert score.scores.shape == (120,)


def test_the_bank_mapping_is_the_shape_evidence_wants():
    mapping = run_bank(_task()).as_mapping()
    assert set(mapping) == set(ALL_SIX)
    assert all(isinstance(v, float) for v in mapping.values())


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_a_baseline_with_no_input_refuses_with_a_remedy_naming_what_to_supply():
    rng = np.random.default_rng(1)
    y = (rng.random(60) < 0.5).astype(int)
    bare = DetectionTask(labels=y, texts=_texts(y, rng))
    bank = run_bank(bare)
    assert "baseline.gradnorm_peak" in bank.refusals()
    assert "baseline.scaffolded_prompt" in bank.refusals()
    grad = bank.refusals()["baseline.gradnorm_peak"]
    assert grad.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert "series" in grad.remedy
    assert "judge=" in bank.refusals()["baseline.scaffolded_prompt"].remedy


def test_a_single_class_task_refuses_rather_than_reporting_half():
    rng = np.random.default_rng(2)
    y = np.ones(40, dtype=int)
    task = DetectionTask(labels=y, texts=_texts(y, rng))
    for reading in run_bank(task).readings.values():
        assert isinstance(reading, Refusal)
        assert reading.reason is RefusalReason.ESS_BELOW_FLOOR


def test_tfidf_refuses_rather_than_reporting_an_in_sample_fit():
    rng = np.random.default_rng(3)
    y = np.zeros(30, dtype=int)
    y[0] = 1  # one positive: no stratified hold-out exists
    task = DetectionTask(labels=y, texts=_texts(y, rng))
    reading = run_bank(task, ["baseline.tfidf"]).readings["baseline.tfidf"]
    assert isinstance(reading, Refusal)
    assert "in-sample" in reading.remedy


def test_asking_for_a_baseline_that_does_not_exist_raises():
    with pytest.raises(KeyError):
        run_bank(_task(), ["baseline.telepathy"])


# ---------------------------------------------------------------------------
# Out-of-fold honesty
# ---------------------------------------------------------------------------


def test_a_random_labelling_leaves_every_fitted_baseline_near_chance():
    """The check that the fitting is honest. In-sample TF-IDF would score near 1.0 here."""
    rng = np.random.default_rng(4)
    n = 200
    y = (rng.random(n) < 0.5).astype(int)
    task = DetectionTask(
        labels=y,
        texts=_texts(np.zeros(n, dtype=int), rng),
        series=rng.normal(0.0, 1.0, n),
        name="noise",
    )
    bank = run_bank(task, ["baseline.tfidf", "baseline.length", "baseline.gradnorm_peak"])
    for bid, score in bank.scored().items():
        assert abs(score.auroc - 0.5) < 0.15, f"{bid} found signal in a random labelling"


def test_a_supplied_marker_set_costs_zero_parameters_and_a_mined_one_says_so():
    task = _task()
    supplied = run_bank(task, ["baseline.string_match"]).readings["baseline.string_match"]
    assert supplied.n_parameters == 0 and not supplied.fitted

    rng = np.random.default_rng(5)
    y = (rng.random(120) < 0.5).astype(int)
    mined_task = DetectionTask(labels=y, texts=_texts(y, rng), name="mined")
    mined = run_bank(mined_task, ["baseline.string_match"]).readings["baseline.string_match"]
    assert mined.fitted and mined.n_parameters > 0


def test_mining_markers_finds_the_planted_substring():
    rng = np.random.default_rng(6)
    y = (rng.random(80) < 0.5).astype(int)
    texts = []
    for label in y:
        body = " ".join(rng.choice(FILLER, size=25))
        texts.append(f"{body} tomato" if label == 1 else body)
    assert any("tomato" in m for m in mine_markers(texts, y, 3))


# ---------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------


def test_a_claim_that_ties_the_string_match_is_reported_as_matched():
    task = _task()
    bank = run_bank(task, ["baseline.string_match"])
    own = bank.readings["baseline.string_match"].scores.astype(float)
    verdict = compare_against_baselines(own, task.labels, bank, seed=0)
    assert verdict.verdict == "matched"
    assert not verdict.is_win
    assert "MATCHED" in verdict.render()


def test_a_claim_worse_than_a_baseline_is_reported_as_a_loss():
    task = _task()
    bank = run_bank(task, ["baseline.string_match"])
    rng = np.random.default_rng(7)
    own = 0.4 * task.labels + rng.normal(0.0, 1.0, task.n)
    verdict = compare_against_baselines(own, task.labels, bank, seed=0)
    assert verdict.verdict == "baseline_wins"
    assert "WINS" in verdict.render()


def test_a_genuinely_better_claim_is_allowed_to_win():
    rng = np.random.default_rng(8)
    n = 300
    y = (rng.random(n) < 0.5).astype(int)
    task = DetectionTask(labels=y, texts=_texts(y, rng), series=rng.normal(0, 1, n), name="weak")
    bank = run_bank(task, ["baseline.gradnorm_peak"])
    own = 3.0 * y + rng.normal(0.0, 1.0, n)
    verdict = compare_against_baselines(own, y, bank, seed=0)
    assert verdict.verdict == "beats_baselines"
    assert verdict.ci_low > 0


def test_an_empty_bank_is_reported_as_no_baseline_not_as_a_win():
    rng = np.random.default_rng(9)
    y = np.ones(20, dtype=int)
    task = DetectionTask(labels=y, texts=_texts(y, rng))
    verdict = compare_against_baselines(np.arange(20.0), y, run_bank(task))
    assert verdict.verdict == "no_baseline"
    assert "lint failure" in verdict.render()


def test_cloning_a_seed_set_does_not_narrow_the_margin_interval():
    """Ten byte-identical copies of twenty seeds is worth twenty, and the interval must say so."""
    rng = np.random.default_rng(10)
    n_seeds, per_seed = 20, 10
    seed_y = (rng.random(n_seeds) < 0.5).astype(int)
    seed_series = rng.normal(0, 1, n_seeds) + 0.7 * seed_y
    seed_own = 1.2 * seed_y + rng.normal(0.0, 1.0, n_seeds)
    y = np.repeat(seed_y, per_seed)
    lineage = tuple(np.repeat(np.arange(n_seeds), per_seed).tolist())
    task = DetectionTask(
        labels=y,
        texts=_texts(y, rng),
        series=np.repeat(seed_series, per_seed),
        name="cloned",
    )
    bank = run_bank(task, ["baseline.gradnorm_peak"])
    own = np.repeat(seed_own, per_seed)
    by_row = compare_against_baselines(own, y, bank, seed=0)
    by_lineage = compare_against_baselines(own, y, bank, seed_labels=lineage, seed=0)
    row_width = by_row.ci_high - by_row.ci_low
    lineage_width = by_lineage.ci_high - by_lineage.ci_low
    assert lineage_width > row_width, (row_width, lineage_width)


def test_too_few_lineages_declines_the_interval_and_reports_matched():
    """Two lineages give three distinct resamples, so a percentile interval means nothing."""
    rng = np.random.default_rng(11)
    n = 100
    y = np.tile([0, 1], n // 2)
    task = DetectionTask(
        labels=y, texts=_texts(y, rng), series=rng.normal(0, 1, n) + 0.7 * y, name="two-lineages"
    )
    bank = run_bank(task, ["baseline.gradnorm_peak"])
    own = 3.0 * y + rng.normal(0.0, 0.5, n)
    verdict = compare_against_baselines(
        own, y, bank, seed_labels=tuple(i % 2 for i in range(n)), seed=0
    )
    assert verdict.verdict == "matched"
    assert "could not be formed" in verdict.detail


# ---------------------------------------------------------------------------
# The lint rule
# ---------------------------------------------------------------------------


def test_a_claim_with_no_baselines_mapping_fails_lint():
    findings = lint_claim(object())
    assert len(findings) == 1
    assert "no baselines mapping" in findings[0].problem


def test_a_claim_with_an_empty_baselines_mapping_fails_lint():
    findings = lint_claim({"instrument": "Probe", "baselines": {}})
    assert findings and "empty" in findings[0].problem


def test_a_claim_missing_one_of_the_six_fails_lint_unless_the_bank_refused_it():
    rng = np.random.default_rng(11)
    y = (rng.random(60) < 0.5).astype(int)
    bare = DetectionTask(labels=y, texts=_texts(y, rng))
    bank = run_bank(bare)
    claim = {"instrument": "Probe", "baselines": bank.as_mapping()}
    assert lint_claim(claim, bank) == []
    assert lint_claim(claim) != []  # without the bank, the two refusals look like omissions


def test_lint_finds_baselines_in_an_evidence_payload():
    class Ev:
        observable = "Probe"
        value = {"baselines": {"baseline.string_match": 1.0}}

    assert claim_baselines(Ev()) == {"baseline.string_match": 1.0}


def test_a_populated_payload_beats_an_empty_structural_field():
    class Ev:
        observable = "Probe"
        baselines: dict = {}
        value = {"baselines": {"baseline.string_match": 1.0}}

    assert claim_baselines(Ev()) == {"baseline.string_match": 1.0}


# ---------------------------------------------------------------------------
# The series baseline
# ---------------------------------------------------------------------------


def test_the_gradient_norm_peak_finds_a_planted_spike_and_reports_its_height():
    series = np.concatenate([np.ones(50), np.array([1.0, 3.0, 6.0, 3.0, 1.0]), np.ones(45)])
    peak = gradnorm_peak(series, window=3)
    assert peak.detected and 50 <= peak.index <= 55
    assert peak.strength > 2.0


def test_a_flat_series_reports_a_peak_with_no_strength():
    assert gradnorm_peak(np.ones(60)).strength == pytest.approx(0.0, abs=1e-9)


def test_smoothing_preserves_length():
    assert smooth(np.arange(31.0), 5).shape == (31,)


# ---------------------------------------------------------------------------
# The scaffold
# ---------------------------------------------------------------------------


def test_the_scaffold_is_hashed_so_a_comparison_names_the_prompt_it_used():
    task = _task(n=4)
    rendered = render_scaffold(task, 0)
    assert task.texts[0] in rendered and "task 0" in rendered
    assert scaffold_hash().startswith("scaffold:")
    assert scaffold_hash() == scaffold_hash()


# ---------------------------------------------------------------------------
# Lineage containment in the fold split (E36)
# ---------------------------------------------------------------------------


def _twinned_corpus(n_lineage: int = 120, seed: int = 0) -> DetectionTask:
    """Two items per lineage sharing a body, with arbitrary opposite labels.

    The generalisable signal is exactly zero by construction: the two members of a lineage are the
    same text, so nothing about the text predicts the label. Any baseline scoring away from 0.5 on
    this corpus is reading a lineage it memorised, not a feature.
    """
    rng = np.random.default_rng(seed)
    vocabulary = [f"w{i}" for i in range(40)]
    texts, labels, seeds = [], [], []
    for lineage in range(n_lineage):
        body = " ".join(rng.choice(vocabulary, size=25))
        for label in (0, 1):
            texts.append(body)
            labels.append(label)
            seeds.append(lineage)
    return DetectionTask(texts=tuple(texts), labels=np.asarray(labels), seed_labels=tuple(seeds))


def test_a_lineage_is_never_split_across_folds():
    task = _twinned_corpus()
    folds = stratified_folds(task.labels, 5, seed=0, groups=task.seed_labels)
    home: dict[int, int] = {}
    for f, fold in enumerate(folds):
        for i in fold:
            lineage = task.seed_labels[i]
            assert home.setdefault(lineage, f) == f, (
                f"lineage {lineage} appears in folds {home[lineage]} and {f}. A near-duplicate "
                f"across the split lets the model memorise the label instead of learning it."
            )
    assert sum(len(f) for f in folds) == task.n
    assert len({len(f) for f in folds}) == 1, (
        "grouping should still balance when lineages are equal"
    )


def test_the_grouped_split_reports_chance_where_the_ungrouped_one_reports_a_leak():
    """The regression E36 exists for, with both numbers in one assertion.

    Ungrouped, the twin in the training fold teaches the model the shared body and the twin in the
    test fold carries the opposite label, so the baseline is not merely optimistic: it is
    confidently *anti*-predictive, which reads as a finding rather than as a bug. The label study
    hit this on 1,542 real receipt transcripts and got 0.026 against a true answer of 0.5.
    """
    task = _twinned_corpus()
    leaked = (
        TfidfLogisticRegression().score(DetectionTask(texts=task.texts, labels=task.labels)).auroc
    )
    contained = TfidfLogisticRegression().score(task).auroc

    assert leaked < 0.25, f"expected the ungrouped split to leak; got {leaked}"
    assert contained == pytest.approx(0.5, abs=0.05), (
        f"a corpus with no generalisable signal must score at chance; got {contained}"
    )


def test_grouping_is_optional_and_absent_lineages_keep_the_old_split():
    labels = np.array([0, 1] * 20)
    assert [f.tolist() for f in stratified_folds(labels, 4, seed=3)] == [
        f.tolist() for f in stratified_folds(labels, 4, seed=3, groups=None)
    ]


def test_a_group_label_per_item_is_required_when_grouping():
    with pytest.raises(ValueError, match="group labels"):
        stratified_folds(np.array([0, 1, 0, 1]), 2, groups=("a", "b"))

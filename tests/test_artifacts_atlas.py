"""The Atlas-v0 leaderboard tabulates stored Evidence and flags uncalibrated cells (M7).

This is the M7 Atlas acceptance exercised on synthetic evidence: register a population, append battery
Evidence for a couple of models to a temp store, and assert that the leaderboard is a faithful view
over the store. A calibrated cell shows its value; an uncalibrated cell is flagged exactly as a card
flags an unvalidated index (gate 1 at the render layer); a model with no Evidence has empty cells and
no invented number. The leaderboard computes nothing (I5); it reads the latest Evidence per pair. The
real ten-model sweep is GPU-gated and is not run here; the plan-and-price path is what these tests
cover for it.
"""

from __future__ import annotations

from reward_lens.artifacts.atlas import (
    Atlas,
    AtlasEntry,
    ModelLineage,
    SweepGatedError,
    declared_fingerprint,
)
from reward_lens.core.evidence import Uncertainty, make_evidence
from reward_lens.core.gates import CalibrationRef
from reward_lens.core.provenance import Cost, capture_provenance
from reward_lens.core.store import EvidenceStore
from reward_lens.core.types import GaugeStatus, ModelFP, SubjectRef


def _seed_store(tmp_path) -> tuple[EvidenceStore, Atlas, str, str]:
    """A store with one calibrated and one uncalibrated observable on Skywork-v0.2."""
    atlas = Atlas.standard()
    store = EvidenceStore(tmp_path)
    v02 = atlas.by_name("Skywork-Reward-Llama-3.1-8B-v0.2")
    assert v02 is not None
    fp = v02.fingerprint
    subj = SubjectRef(signals=(ModelFP(fp),), dataset="ds:diag", readout="reward")
    # A calibrated index (has a scorecard reference), with a metered cost so total GPU-seconds is real.
    store.append(
        make_evidence(
            observable="BiasBattery",
            observable_version="1.0",
            subject=subj,
            value={"verbosity": -0.05},
            uncertainty=Uncertainty(ci_low=-0.1, ci_high=0.0, n=30, n_effective=6.0),
            gauge=GaugeStatus.INVARIANT,
            calibration=CalibrationRef("ev:scorecard", "planted-bias"),
            provenance=capture_provenance(cost=Cost(gpu_seconds=1.5)),
        )
    )
    # An uncalibrated index (no scorecard): must be flagged on the leaderboard.
    store.append(
        make_evidence(
            observable="DistortionV2",
            observable_version="1.0",
            subject=subj,
            value=0.42,
            gauge=GaugeStatus.INVARIANT,
        )
    )
    return store, atlas, fp, v02.name


def test_register_a_few_models_and_tabulate(tmp_path):
    # Build a small population by hand to exercise the register() path directly.
    atlas = Atlas()
    a = AtlasEntry(
        fingerprint=ModelFP("mfp:model-a"),
        name="Model-A",
        repo_id="org/model-a",
        lineage=ModelLineage("base-x", "data-x", "2025-01", "card-claimed"),
    )
    b = AtlasEntry(
        fingerprint=ModelFP("mfp:model-b"),
        name="Model-B",
        repo_id="org/model-b",
        lineage=ModelLineage("base-y", "data-y", "2025-02", "weights-verified"),
    )
    atlas.register(a)
    atlas.register(b)
    assert len(atlas) == 2
    assert atlas.by_name("Model-B").weights_verified

    store = EvidenceStore(tmp_path)
    store.append(
        make_evidence(
            observable="RobustnessSNR",
            observable_version="1.0",
            subject=SubjectRef(signals=(a.fingerprint,)),
            value=3.1,
            calibration=CalibrationRef("ev:sc", "planted"),
        )
    )
    store.append(
        make_evidence(
            observable="RobustnessSNR",
            observable_version="1.0",
            subject=SubjectRef(signals=(b.fingerprint,)),
            value=1.2,
        )
    )
    lb = atlas.leaderboard(store, ["RobustnessSNR"])
    assert lb.table.loc["Model-A", "RobustnessSNR"] == "3.1"
    assert lb.table.loc["Model-B", "RobustnessSNR"] == "1.2"
    # Model-A's cell is calibrated, Model-B's is flagged.
    assert lb.cell("Model-A", "RobustnessSNR").validated
    assert not lb.cell("Model-B", "RobustnessSNR").validated


def test_standard_population_is_ten_and_card_claimed():
    atlas = Atlas.standard()
    assert len(atlas) == 10
    names = {e.name for e in atlas.entries}
    for expected in (
        "Skywork-Reward-Llama-3.1-8B-v0.1",
        "Skywork-Reward-Llama-3.1-8B-v0.2",
        "ArmoRM-Llama3-8B",
        "Skywork-Reward-Gemma-2-27B",
        "Tulu-3-8B-RM",
    ):
        assert expected in names
    # The registry hashes no weights, so every entry is a declared, card-claimed fingerprint.
    for e in atlas.entries:
        assert e.lineage.provenance_tier == "card-claimed"
        assert not e.weights_verified
        assert e.lineage.base_model  # lineage is populated, not blank
        assert e.fingerprint == declared_fingerprint(e.repo_id)


def test_leaderboard_tabulates_synthetic_evidence(tmp_path):
    store, atlas, fp, name = _seed_store(tmp_path)
    lb = atlas.leaderboard(store, ["BiasBattery", "DistortionV2"])

    # The pandas comparison table has a row per registered model and a column per observable.
    table = lb.table
    assert list(table.columns) == ["BiasBattery", "DistortionV2"]
    assert len(table.index) == len(atlas)
    # The measured model shows its stored values; nothing is recomputed.
    assert table.loc[name, "BiasBattery"] != ""
    assert table.loc[name, "DistortionV2"] == "0.42"


def test_leaderboard_flags_uncalibrated(tmp_path):
    store, atlas, fp, name = _seed_store(tmp_path)
    lb = atlas.leaderboard(store, ["BiasBattery", "DistortionV2"])

    assert lb.cell(name, "BiasBattery").validated
    assert not lb.cell(name, "DistortionV2").validated
    # The only flagged (present-but-uncalibrated) cell is the distortion one.
    assert [(c.model, c.observable) for c in lb.flagged] == [(name, "DistortionV2")]
    # The parallel flag grid says the same thing.
    assert lb.calibration_table.loc[name, "BiasBattery"] == "calibrated"
    assert lb.calibration_table.loc[name, "DistortionV2"] == "uncalibrated"


def test_leaderboard_absent_cells_are_empty_not_zero(tmp_path):
    store, atlas, fp, name = _seed_store(tmp_path)
    lb = atlas.leaderboard(store, ["BiasBattery", "DistortionV2"])
    other = "ArmoRM-Llama3-8B"
    # A model with no Evidence has no cell at all, and the table shows an empty string, not a 0.
    assert lb.cell(other, "BiasBattery") is None
    assert lb.table.loc[other, "BiasBattery"] == ""


def test_leaderboard_meters_cost_and_exports(tmp_path):
    store, atlas, fp, name = _seed_store(tmp_path)
    lb = atlas.leaderboard(store, ["BiasBattery", "DistortionV2"])
    # Total metered GPU-seconds is summed from the Evidence provenance (R13), not fabricated.
    assert lb.total_gpu_seconds == 1.5

    export = atlas.export_leaderboard(
        store=store, observables=["BiasBattery", "DistortionV2"], out_dir=tmp_path / "site"
    )
    assert "uncalibrated_cells" in export["json"]
    assert "unvalidated" in export["html"]  # the uncalibrated cell renders distinctly
    assert export["json_path"].exists()
    assert export["html_path"].exists()


def test_leaderboard_default_observables_from_store(tmp_path):
    store, atlas, fp, name = _seed_store(tmp_path)
    # With no observables argument, the columns are exactly what the store measured for the population.
    lb = atlas.leaderboard(store)
    assert set(lb.observables) == {"BiasBattery", "DistortionV2"}


def test_sweep_plans_prices_and_is_gpu_gated(tmp_path):
    store, atlas, fp, name = _seed_store(tmp_path)
    battery = ["BiasBattery", "DistortionV2"]
    plan = atlas.sweep([fp], battery, Cost(gpu_seconds=100.0), store=store)
    # Two cells, both priced from the prior metered cost in the store (a view over cost, I5/R13).
    assert len(plan.cells) == 2
    assert plan.n_estimated == 2
    assert plan.estimated_total.gpu_seconds == 1.5  # 1.5 from BiasBattery + 0.0 from DistortionV2
    assert plan.within_budget

    # A model with no priors is unestimated, not invented.
    fresh = declared_fingerprint("some/never-measured-model")
    plan2 = atlas.sweep([fresh], battery, Cost(gpu_seconds=100.0), store=store)
    assert plan2.n_estimated == 0

    # Executing the real sweep is GPU-gated: it refuses rather than fabricating scores.
    import pytest

    with pytest.raises(SweepGatedError):
        atlas.sweep([fp], battery, Cost(gpu_seconds=100.0), store=store, execute=True)

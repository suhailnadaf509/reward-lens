"""The static site export writes cards, the scoreboard, and the Atlas leaderboard (M13).

This exercises ``build_site`` on a synthetic store: it must write an index, a scoreboard page, an
Atlas page carrying the population and its leaderboard, and one card page per model the store holds
Evidence about. The pages are markdown for the MkDocs Material theme, and an uncalibrated index stays
flagged in the markdown exactly as on the HTML card (gate 1). Nothing is computed here; the pages are
assembled from the existing generators, so this test asserts on files and their flagged content, not
on any fresh number.
"""

from __future__ import annotations

from reward_lens.artifacts.atlas import Atlas
from reward_lens.artifacts.site import build_site
from reward_lens.core.evidence import Uncertainty, make_evidence
from reward_lens.core.gates import CalibrationRef
from reward_lens.core.store import EvidenceStore
from reward_lens.core.types import GaugeStatus, ModelFP, SubjectRef


def _seed(tmp_path) -> tuple[EvidenceStore, Atlas, str]:
    atlas = Atlas.standard()
    store = EvidenceStore(tmp_path / "store")
    fp = atlas.by_name("Skywork-Reward-Llama-3.1-8B-v0.2").fingerprint
    subj = SubjectRef(signals=(ModelFP(fp),), dataset="ds:diag", readout="reward")
    store.append(
        make_evidence(
            observable="BiasBattery",
            observable_version="1.0",
            subject=subj,
            value={"verbosity": -0.05},
            uncertainty=Uncertainty(ci_low=-0.1, ci_high=0.0, n=30, n_effective=6.0),
            gauge=GaugeStatus.INVARIANT,
            calibration=CalibrationRef("ev:scorecard", "planted-bias"),
        )
    )
    store.append(
        make_evidence(
            observable="DistortionV2",
            observable_version="1.0",
            subject=subj,
            value=0.42,
            gauge=GaugeStatus.INVARIANT,
        )
    )
    return store, atlas, fp


def test_build_site_writes_expected_pages(tmp_path):
    store, atlas, fp = _seed(tmp_path)
    out = tmp_path / "site"
    written = build_site(store, out, atlas=atlas, observables=["BiasBattery", "DistortionV2"])

    # The core pages exist.
    for key in ("index", "scoreboard", "atlas"):
        assert key in written
        assert written[key].exists()
    # One card page for the one model the store has Evidence about.
    card_keys = [k for k in written if k.startswith("card:")]
    assert card_keys == [f"card:{fp}"]
    assert written[card_keys[0]].exists()
    # Files landed under the requested directory.
    assert (out / "index.md").exists()
    assert (out / "cards").is_dir()


def test_site_atlas_page_has_population_and_leaderboard(tmp_path):
    store, atlas, fp = _seed(tmp_path)
    out = tmp_path / "site"
    written = build_site(store, out, atlas=atlas, observables=["BiasBattery", "DistortionV2"])
    atlas_md = written["atlas"].read_text(encoding="utf-8")
    assert "## Population" in atlas_md
    assert "## Leaderboard" in atlas_md
    # The lineage table shows the card-claimed provenance tier for the population.
    assert "card-claimed" in atlas_md
    # The uncalibrated cell is flagged in the leaderboard markdown.
    assert "unvalidated" in atlas_md


def test_site_card_page_flags_uncalibrated(tmp_path):
    store, atlas, fp = _seed(tmp_path)
    out = tmp_path / "site"
    written = build_site(store, out, atlas=atlas, observables=["BiasBattery", "DistortionV2"])
    card_md = written[f"card:{fp}"].read_text(encoding="utf-8")
    assert card_md.startswith("# RM Card")
    # The uncalibrated index is marked unvalidated; the calibrated one is not flagged.
    assert "**unvalidated** DistortionV2" in card_md
    assert "Explicit gaps" in card_md


def test_build_site_empty_store_writes_pages(tmp_path):
    # With no Evidence, the site still builds: no cards, but index/scoreboard/atlas exist.
    store = EvidenceStore(tmp_path / "empty")
    out = tmp_path / "site"
    written = build_site(store, out)
    assert written["index"].exists()
    assert [k for k in written if k.startswith("card:")] == []
    index_md = written["index"].read_text(encoding="utf-8")
    assert "No model cards yet" in index_md

"""Proofs for the calibrated concept factory (probes / beliefs / diff_dict / banks).

Everything here is proven on planted or synthetic structure, so the answer is known by construction
and the passing test is the deliverable. The four constructs are held to the claims their design
makes:

- The probe factory recovers a planted concept with high held-out AUC and earns a real
  `CalibrationRef` from the answer-key scorecard; on label-shuffled data the same machinery reports
  chance (~0.5), never a fabricated high number; and the seed-grouped cross-validation never splits
  a clone across folds.
- The belief probe decodes a planted answer-is-correct latent above threshold against a known key,
  is answer-keyed, and refuses a self-labeled target.
- The preference-difference dictionary reconstructs the Bradley-Terry margin exactly (~1e-6) on
  held-out pairs when it spans the difference subspace, and reports an honest residual when it does
  not; the residual is carried on stored Evidence.
- Each standard bank loads with its documented structure and produces persisted directions, from
  synthetic captures and from the tiny model.
"""

from __future__ import annotations

import numpy as np
import pytest

from reward_lens.concepts import banks, beliefs, diff_dict
from reward_lens.concepts.probes import (
    Direction,
    SiteCaptures,
    fit_probe,
    group_kfold_indices,
    make_direction,
)
from reward_lens.core.store import EvidenceStore
from reward_lens.core.types import GaugeStatus, Site, TrustLevel
from reward_lens.data.corruptions import corrupt_step
from reward_lens.measure.indices._support import FeatureBank
from reward_lens.organisms.foundry import spurious_correlation_organism

# ---------------------------------------------------------------------------
# Planting helpers (the ground truth the proofs are graded against)
# ---------------------------------------------------------------------------


def _plant_concept_captures(
    *,
    n: int = 260,
    d: int = 24,
    seed: int = 0,
    site_strengths: tuple[float, ...] = (0.5, 1.1, 2.2),
    noise: float = 1.0,
    labels: np.ndarray | None = None,
    groups: np.ndarray | None = None,
    answer_key=None,
    name: str = "planted-concept",
) -> tuple[SiteCaptures, dict[Site, np.ndarray], np.ndarray]:
    """A multi-site capture with a planted concept direction whose SNR rises with depth.

    Each site gets its own random unit direction ``v`` and activations ``X = label * strength * v +
    noise``, so the concept is a known direction by construction and the deepest site (largest
    strength) is the most decodable. Returns the captures, the planted direction per site, and the
    labels.
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n) if labels is None else np.asarray(labels).astype(np.int64)
    n = y.shape[0]
    grp = np.arange(n) if groups is None else np.asarray(groups)
    features: dict[Site, np.ndarray] = {}
    planted: dict[Site, np.ndarray] = {}
    sites = [Site(layer, "resid_post") for layer in range(len(site_strengths))]
    for site, strength in zip(sites, site_strengths):
        v = rng.standard_normal(d)
        v = v / np.linalg.norm(v)
        planted[site] = v
        x = (y[:, None] * float(strength)) * v[None, :] + rng.standard_normal((n, d)) * noise
        features[site] = x.astype(np.float32)
    caps = SiteCaptures(
        features=features,
        labels=y,
        groups=grp,
        answer_key=answer_key,
        name=name,
    )
    return caps, planted, y


def _recovery_cosine(direction: Direction, planted: np.ndarray) -> float:
    v = np.asarray(direction.vector, dtype=np.float64)
    p = np.asarray(planted, dtype=np.float64)
    return abs(float(np.dot(v, p) / (np.linalg.norm(v) * np.linalg.norm(p))))


# ===========================================================================
# probes.py
# ===========================================================================


def test_probe_recovers_planted_concept_with_high_auc_and_real_calibration(tmp_path):
    """A probe recovers a planted concept at high held-out AUC and earns a real CalibrationRef."""
    _, key = spurious_correlation_organism(rho=0.9, n=10, seed=0)
    caps, planted, _ = _plant_concept_captures(seed=1, answer_key=key)
    store = EvidenceStore(tmp_path)

    fit = fit_probe(caps, cv=5, name="planted", solver="numpy", store=store)

    # The sweep picks the most decodable (deepest) site, and recovers the planted direction there.
    assert fit.best_site == Site(2, "resid_post")
    assert fit.held_out_auc > 0.9
    assert _recovery_cosine(fit.direction, planted[fit.best_site]) > 0.9

    # Automatic scorecard binding attached a real CalibrationRef pointing at the answer-key ROC.
    assert fit.calibration is not None
    assert fit.direction.is_calibrated
    assert fit.calibration.organism_family == key.family
    assert fit.scorecard_evidence is not None
    # The scorecard graded the held-out scores honestly: a decodable concept scores high.
    assert fit.scorecard_evidence.value.aucs[0] > 0.85

    # The direction is a persisted, calibrated, COVARIANT store citizen (R8), linked to its scorecard.
    assert fit.evidence.gauge is GaugeStatus.COVARIANT
    assert fit.evidence.trust == TrustLevel.CALIBRATED
    assert fit.scorecard_evidence.id in fit.evidence.provenance.parents
    reloaded = EvidenceStore(tmp_path)
    got = reloaded.get(fit.evidence.id)
    assert np.allclose(got.value.vector, fit.direction.vector, atol=1e-6)
    assert got.value.direction_id == str(fit.direction.id)


def test_probe_unit_direction_and_depth_curve():
    """The returned direction is unit fp32 and the per-site sweep is a real depth curve."""
    caps, _, _ = _plant_concept_captures(seed=2)
    fit = fit_probe(caps, cv=5, name="planted", solver="numpy")
    assert fit.direction.vector.dtype == np.float32
    assert abs(float(np.linalg.norm(fit.direction.vector)) - 1.0) < 1e-5
    # Held-out AUC increases with the planted SNR across the swept sites.
    aucs = [p.held_out_auc for p in fit.per_site]
    assert aucs[0] < aucs[-1]
    assert fit.per_site[-1].held_out_auc > 0.9


def test_probe_on_shuffled_labels_reports_chance_not_fabricated(tmp_path):
    """A probe on label-shuffled data honestly reports calibration at chance, not a high number."""
    _, key = spurious_correlation_organism(rho=0.9, n=10, seed=0)
    caps, _, y = _plant_concept_captures(seed=3, answer_key=key)

    rng = np.random.default_rng(99)
    shuffled = y.copy()
    rng.shuffle(shuffled)
    caps_shuf = SiteCaptures(
        features=caps.features, labels=shuffled, groups=caps.groups, answer_key=key, name="shuffled"
    )

    fit = fit_probe(caps_shuf, cv=5, name="shuffled", solver="numpy", store=EvidenceStore(tmp_path))

    # The concept is gone: held-out AUC is at chance, and the scorecard reports that, not a fake high.
    assert 0.4 < fit.held_out_auc < 0.6
    assert fit.calibration is not None  # a real scorecard was run
    assert abs(fit.scorecard_evidence.value.aucs[0] - 0.5) < 0.1
    # No detection at the operating point on shuffled data: the honest outcome.
    assert fit.calibration.operating_point is None


def test_seed_grouped_cv_never_splits_a_clone():
    """The seed-grouped CV keeps every clone of a seed in one fold (no leakage across folds)."""
    # Groups with deliberate clones: seed 0 has three rows, seed 1 has two, and so on.
    groups = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5, 6, 7, 7])
    splits = group_kfold_indices(groups, n_splits=4, seed=0)

    # Every fold's test set is a union of whole groups: no group appears in both train and test.
    for train_idx, test_idx in splits:
        train_groups = set(groups[train_idx].tolist())
        test_groups = set(groups[test_idx].tolist())
        assert train_groups.isdisjoint(test_groups), "a seed leaked across the train/test split"

    # Every group lands in exactly one test fold, and all its clones with it.
    for g in np.unique(groups):
        rows = np.where(groups == g)[0]
        folds_hit = [f for f, (_, te) in enumerate(splits) if np.any(np.isin(rows, te))]
        assert len(folds_hit) == 1, f"clones of seed {g} were split across folds {folds_hit}"
        te = splits[folds_hit[0]][1]
        assert set(rows.tolist()).issubset(set(te.tolist()))


def test_probe_captures_reject_misaligned_shapes():
    """SiteCaptures refuses features whose row count does not match the labels (a data-integrity bite)."""
    with pytest.raises(ValueError):
        SiteCaptures(
            features={Site(0, "resid_post"): np.zeros((5, 8), dtype=np.float32)},
            labels=np.array([0, 1, 0]),
            groups=np.array([0, 1, 2]),
        )


def test_make_direction_id_is_content_derived():
    """Identical directions share a content-derived DirectionID; a different vector changes it."""
    v = np.array([1.0, 2.0, 2.0, 0.0], dtype=np.float32)
    site = Site(1, "resid_post")
    a = make_direction(name="c", site=site, vector=v, method="probe_lr", train_data=None)
    b = make_direction(name="c", site=site, vector=v, method="probe_lr", train_data=None)
    c = make_direction(name="c", site=site, vector=-v, method="probe_lr", train_data=None)
    assert a.id == b.id
    assert a.id != c.id
    assert str(a.id).startswith("dir:")


# ===========================================================================
# beliefs.py
# ===========================================================================


def _corruption_keyed_labels(n_solutions: int = 130, seed: int = 0):
    """Verifiable answer-is-correct labels from the corruption generator's known edit.

    For each templated solution the clean version is correct (label 1); corrupting step 1 with the
    mechanical swap makes it incorrect (label 0). The label is the corruption key, produced by the
    generator and not by any model, which is exactly what a belief target requires.
    """
    rng = np.random.default_rng(seed)
    labels: list[int] = []
    groups: list[str] = []
    for i in range(n_solutions):
        a, b, c, dd = (int(x) for x in rng.integers(2, 40, size=4))
        solution = f"x = {a} + {b}\ny = x * {c}\nz = y - {dd}"
        # The clean solution's step is correct.
        labels.append(1)
        groups.append(f"sol{i}-clean")
        # Corrupting step 1 plants a known error: the step is now incorrect. The edit script is the key.
        corrupted, edits, _span = corrupt_step(solution, 1, mode="swap_number")
        assert corrupted != solution and len(edits) == 1  # the key exists and is exact
        labels.append(0)
        groups.append(f"sol{i}-corrupt")
    return np.array(labels, dtype=np.int64), np.array(groups, dtype=object)


def test_belief_probe_decodes_answer_correct_latent_answer_keyed(tmp_path):
    """A belief probe decodes a planted answer-is-correct latent above threshold, answer-keyed."""
    labels, groups = _corruption_keyed_labels(seed=1)
    _, key = spurious_correlation_organism(rho=0.9, n=10, seed=0)  # a real family to grade against
    caps, planted, _ = _plant_concept_captures(
        seed=5,
        labels=labels,
        groups=groups,
        site_strengths=(0.6, 1.6),
        answer_key=key,
        name="answer-correct",
    )
    belief = beliefs.answer_key_target("answer-correct", lambda item, side: None)
    store = EvidenceStore(tmp_path)

    probe = beliefs.fit_belief_probe(
        caps,
        belief=belief,
        answer_key=key,
        cv=5,
        solver="numpy",
        store=store,
        decode_threshold=0.75,
    )

    # The verifiable latent decodes above threshold, and its calibration is answer-keyed (gate 1).
    assert probe.decodes_above(0.75)
    assert probe.is_calibrated
    assert probe.key_source == "answer_key"
    assert probe.organism_family == key.family
    assert _recovery_cosine(probe.direction, planted[probe.fit.best_site]) > 0.85
    # The belief direction persisted with CALIBRATED trust (strictest standard met).
    assert probe.fit.evidence.trust == TrustLevel.CALIBRATED


def test_belief_probe_refuses_self_labeled_target():
    """Swapping to a self-labeled (non-verifiable) target is refused before any fitting."""
    labels, groups = _corruption_keyed_labels(n_solutions=40, seed=2)
    _, key = spurious_correlation_organism(rho=0.9, n=10, seed=0)
    caps, _, _ = _plant_concept_captures(
        seed=6, labels=labels, groups=groups, site_strengths=(1.0,), answer_key=key
    )
    # A target sourced from the model's own output is not a valid belief target.
    self_target = beliefs.self_labeled_target("model-verdict", lambda item, side: 1)
    with pytest.raises(beliefs.SelfLabeledBeliefError):
        beliefs.fit_belief_probe(caps, belief=self_target, answer_key=key)


def test_belief_probe_requires_calibration():
    """A belief probe with no answer key is refused, not returned EXPLORATORY (strictest standard)."""
    labels, groups = _corruption_keyed_labels(n_solutions=40, seed=3)
    caps, _, _ = _plant_concept_captures(
        seed=7, labels=labels, groups=groups, site_strengths=(1.0,)
    )
    belief = beliefs.answer_key_target("answer-correct", lambda item, side: None)
    with pytest.raises(beliefs.UncalibratedBeliefError):
        beliefs.fit_belief_probe(caps, belief=belief, answer_key=None)


def test_belief_target_verifiability_flags():
    """The verifiable/self-labeled target constructors carry the provenance the factory checks."""
    ok = beliefs.answer_key_target("x", lambda i, s: 1)
    bad = beliefs.self_labeled_target("y", lambda i, s: 1)
    ok.check_verifiable()  # does not raise
    assert ok.source == "answer_key" and ok.verifiable
    with pytest.raises(beliefs.SelfLabeledBeliefError):
        bad.check_verifiable()


# ===========================================================================
# diff_dict.py
# ===========================================================================


def _plant_diff_dict(*, d=28, k0=6, n_train=320, n_ho=140, seed=0, full_rank=False):
    """Plant delta_h from a known dictionary (rank ``k0``) or as full-rank noise, with a random w_r."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(d)
    w = w / np.linalg.norm(w)
    if full_rank:
        delta_train = rng.standard_normal((n_train, d))
        delta_ho = rng.standard_normal((n_ho, d))
        return delta_train, delta_ho, w, None
    atoms = np.linalg.qr(rng.standard_normal((d, k0)))[0].T  # (k0, d), orthonormal rows
    delta_train = rng.standard_normal((n_train, k0)) @ atoms
    delta_ho = rng.standard_normal((n_ho, k0)) @ atoms
    return delta_train, delta_ho, w, atoms


def test_diff_dict_exact_margin_decomposition_on_heldout(tmp_path):
    """When the dictionary spans the difference subspace, the margin identity is exact to ~1e-6."""
    delta_train, delta_ho, w, _atoms = _plant_diff_dict(seed=1, k0=6)
    store = EvidenceStore(tmp_path)

    result = diff_dict.fit_and_verify(delta_train, delta_ho, w, n_atoms=6, store=store)

    # The reconstructed margin equals the true Bradley-Terry margin on held-out pairs to precision.
    assert result.exact
    assert result.max_abs_residual < 1e-6
    assert result.verification.reconstruction_r2 > 1.0 - 1e-9

    # The stored verification Evidence carries that residual (the certificate travels with the dict).
    assert result.verification_evidence.value.max_abs_residual == result.max_abs_residual
    assert result.verification_evidence.gauge is GaugeStatus.INVARIANT
    assert result.dict_evidence.id in result.verification_evidence.provenance.parents

    # Independent recompute of the identity: margin == sum_i f_i (w_r . d_i), pair by pair.
    margin_recon, contributions = diff_dict.decompose_margin(result.dictionary, delta_ho)
    margin_true = diff_dict.true_margin(delta_ho, w)
    assert np.max(np.abs(margin_recon - margin_true)) < 1e-6
    assert np.allclose(contributions.sum(axis=1), margin_recon, atol=1e-9)

    # Round-trip the dictionary and its certificate through the store (R8).
    reloaded = EvidenceStore(tmp_path)
    got_dict = reloaded.get(result.dict_evidence.id)
    assert np.allclose(got_dict.value.atoms, result.dictionary.atoms, atol=1e-6)
    assert reloaded.get(result.verification_evidence.id).value.exact is True


def test_diff_dict_reports_honest_residual_when_truncated():
    """A dictionary too small to span the difference reports a real residual, not a fake zero."""
    delta_train, delta_ho, w, _ = _plant_diff_dict(seed=2, full_rank=True, d=28)
    result = diff_dict.fit_and_verify(delta_train, delta_ho, w, n_atoms=5)  # 5 << d = 28

    assert not result.exact
    assert result.max_abs_residual > 1e-3
    assert result.verification.reconstruction_r2 < 0.9
    # The honest residual is what is stored; nothing rounds it to zero.
    assert result.verification.max_abs_residual == result.max_abs_residual


def test_diff_dict_delta_h_from_pairs_matches_margin():
    """delta_h_from_pairs and true_margin compose to the linear reward difference w_r . (h_c - h_r)."""
    rng = np.random.default_rng(4)
    d = 16
    h_c = rng.standard_normal((20, d))
    h_r = rng.standard_normal((20, d))
    w = rng.standard_normal(d)
    delta = diff_dict.delta_h_from_pairs(h_c, h_r)
    margin = diff_dict.true_margin(delta, w)
    assert np.allclose(margin, (h_c - h_r) @ w)


# ===========================================================================
# banks.py
# ===========================================================================


def test_standard_banks_load_with_documented_structure():
    """Each named bank loads with its documented category and feature-grounded specs."""
    assert set(banks.STANDARD_BANKS) == {"style", "safety", "quality", "belief"}
    for name, specs in banks.STANDARD_BANKS.items():
        assert len(specs) >= 1
        for spec in specs:
            assert spec.category == name
            # Every concept is tied to an exact feature marker in the controlled substrate.
            from reward_lens.organisms._features import ALL_FEATURES

            assert spec.feature in ALL_FEATURES
    # The belief bank's concept is marked verifiable (belief-probe standard).
    assert all(s.verifiable for s in banks.BELIEF_BANK)
    assert banks.bank("quality") is banks.QUALITY_BANK
    with pytest.raises(KeyError):
        banks.bank("nonsense")


def test_build_bank_produces_persisted_directions_from_synthetic_captures(tmp_path):
    """A bank built from synthetic captures produces persisted directions and satisfies FeatureBank."""
    rng = np.random.default_rng(7)
    d = 20
    site = Site(1, "resid_post")
    specs = banks.STYLE_BANK
    sides: banks.ConceptSides = {}
    planted: dict[str, np.ndarray] = {}
    for spec in specs:
        v = rng.standard_normal(d)
        v = v / np.linalg.norm(v)
        planted[spec.name] = v
        pos = 1.6 * v[None, :] + rng.standard_normal((12, d)) * 0.5
        neg = rng.standard_normal((12, d)) * 0.5
        sides[spec.name] = (pos.astype(np.float32), neg.astype(np.float32))

    store = EvidenceStore(tmp_path)
    built = banks.build_bank(specs, sides, site, bank_name="style", store=store)

    # Documented structure: one direction per style concept, all persisted, plus a manifest.
    assert built.bank.names == tuple(s.name for s in specs)
    assert isinstance(built.bank, FeatureBank)
    assert built.bank.directions().shape == (len(specs), d)
    assert built.bank.featurize(np.zeros((4, d))).shape == (4, len(specs))
    assert len(built.evidence) == len(specs) + 1  # directions + manifest

    # Extraction reused the mean-difference estimator and recovered each planted style direction.
    for direction in built.directions:
        assert abs(float(np.linalg.norm(direction.vector)) - 1.0) < 1e-5
        assert direction.method == "contrast_mean"
        assert _recovery_cosine(direction, planted[direction.name]) > 0.85

    # Directions and the manifest are reconstructible from the store (R8). Direction evidences come
    # first in declared order, the bank manifest last.
    reloaded = EvidenceStore(tmp_path)
    got = reloaded.get(built.evidence[0].id)
    assert np.allclose(got.value.vector, built.directions[0].vector, atol=1e-6)
    manifest_ev = reloaded.get(built.evidence[-1].id)
    assert manifest_ev.value.direction_ids == [str(dd.id) for dd in built.directions]


def test_build_bank_skips_uncaptured_concepts_rather_than_fabricate(tmp_path):
    """A concept with no captured sides is skipped, not padded with a fabricated direction."""
    rng = np.random.default_rng(8)
    d = 12
    site = Site(0, "resid_post")
    specs = banks.QUALITY_BANK  # factual, cites
    # Only supply captures for one of the two concepts.
    only = specs[0]
    v = rng.standard_normal(d)
    sides = {
        only.name: (1.5 * v[None, :] + rng.standard_normal((8, d)), rng.standard_normal((8, d)))
    }
    built = banks.build_bank(specs, sides, site, bank_name="quality")
    assert built.bank.names == (only.name,)
    assert len(built.directions) == 1


def test_default_feature_bank_from_tiny_model_persists_directions(tmp_path):
    """The tiny-model capture path builds a real, persisted ConceptBank end to end (torch)."""
    from reward_lens.signals.loaders import from_tiny

    signal = from_tiny(d_model=32, n_layers=2, n_heads=4, seed=0)
    store = EvidenceStore(tmp_path)
    concept_bank = banks.default_feature_bank(signal, category="quality", store=store)

    # A real bank of unit directions that satisfies the FeatureBank protocol the indices consume.
    assert isinstance(concept_bank, FeatureBank)
    assert concept_bank.names == tuple(s.name for s in banks.QUALITY_BANK)
    d_model = int(signal.meta.d_model)
    assert concept_bank.directions().shape == (len(banks.QUALITY_BANK), d_model)
    for entry in concept_bank.entries:
        assert abs(float(np.linalg.norm(entry.vector)) - 1.0) < 1e-4
        assert entry.method == "contrast_mean"
    # Directions were persisted from the tiny-model captures (R8).
    assert len(store) >= len(banks.QUALITY_BANK)

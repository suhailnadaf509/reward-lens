"""The claims checker is the structural fix for the PAPER_DISCREPANCIES failure class.

A document may not claim a number the store does not contain. These tests build a store with a known
Evidence, then check documents that (a) cite the number correctly, (b) cite a wrong number, (c) cite
a nonexistent evidence id, and (d) reference a dangling ev id, and assert the checker passes only the
correct claim. This is the M7 acceptance criterion "the claims checker fails a doc with an unbound
number", exercised in isolation.
"""

from __future__ import annotations

from reward_lens.artifacts.claims import check_text
from reward_lens.core.evidence import make_evidence
from reward_lens.core.store import EvidenceStore
from reward_lens.core.types import ModelFP, SubjectRef


def _store_with_e04(tmp_path) -> tuple[EvidenceStore, str]:
    store = EvidenceStore(tmp_path)
    ev = make_evidence(
        observable="E04Faithfulness",
        observable_version="1.0",
        subject=SubjectRef(signals=(ModelFP("mfp:skywork"),)),
        value={"per_model_mean_rho": {"Skywork": -0.1709}},
    )
    store.append(ev)
    return store, ev.id


def test_correct_claim_passes(tmp_path):
    store, eid = _store_with_e04(tmp_path)
    doc = f"The per-model mean rho is [[claim value=-0.171 ev={eid} field=per_model_mean_rho.Skywork tol=0.01]]."
    rep = check_text(doc, store)
    assert rep.ok
    assert rep.n_failures == 0
    assert rep.results[0].ok


def test_wrong_value_fails(tmp_path):
    store, eid = _store_with_e04(tmp_path)
    # Claim a number the store contradicts: the stored value is -0.171, not -0.30.
    doc = f"[[claim value=-0.30 ev={eid} field=per_model_mean_rho.Skywork tol=0.01]]"
    rep = check_text(doc, store)
    assert not rep.ok
    assert rep.n_failures == 1
    assert "but stored" in rep.results[0].message


def test_nonexistent_evidence_fails(tmp_path):
    store, _ = _store_with_e04(tmp_path)
    doc = "[[claim value=-0.171 ev=ev:deadbeefdeadbeef field=x tol=0.01]]"
    rep = check_text(doc, store)
    assert not rep.ok
    assert "not in the store" in rep.results[0].message


def test_dangling_bare_reference_flagged(tmp_path):
    store, _ = _store_with_e04(tmp_path)
    doc = "As shown in ev:0000000000000000, the effect holds."
    rep = check_text(doc, store)
    assert not rep.ok
    assert "ev:0000000000000000" in rep.unresolved_refs


def test_bare_reference_resolves(tmp_path):
    store, eid = _store_with_e04(tmp_path)
    doc = f"See {eid} for the measurement."
    rep = check_text(doc, store)
    assert rep.ok
    assert not rep.unresolved_refs

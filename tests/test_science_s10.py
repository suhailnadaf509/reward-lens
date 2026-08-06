"""S10 runs end to end as a frozen study, recovering the planted legibility knee and tacit residual.

The calibration arm builds a synthetic reward that is a known k-predicate rubric plus a known tacit
remainder, then shows the legibility frontier knees at the true k and that the residual left at the
knee matches the planted tacit fraction. The production legibility fit rides an in-flight subsystem
and a real reward population, so it is recorded as an honestly gated (inconclusive) follow-on.
"""

from __future__ import annotations

from reward_lens.core.store import EvidenceStore
from reward_lens.core.types import TrustLevel
from reward_lens.studies import render_report, run_study
from studies.s10_decompiling.analysis import build_spec


def test_s10_recovers_knee_and_tacit_residual(tmp_path):
    store = EvidenceStore(tmp_path)
    frozen, result = run_study(build_spec(), store=store)

    # The frontier knees at the true k and the residual matches the planted tacit fraction.
    assert result.outcomes["H1-knee-at-k"] == "confirmed", result.metrics
    assert result.outcomes["H2-tacit-residual"] == "confirmed", result.metrics
    assert not result.killed
    assert result.metrics["recovered_knee"] == 5.0
    assert result.metrics["tacit_fraction_error"] < 0.05
    # A substantial tacit remainder was recovered, not erased by overfitting.
    assert 0.25 < result.metrics["tacit_fraction"] < 0.45

    # The headline Evidence is REGISTERED and traces back to the fitted frontier (a real DAG).
    residual = store.find(observable="S10.TacitResidual")
    assert residual and residual[0].trust is TrustLevel.REGISTERED
    parents = store.parents(residual[0])
    assert parents and parents[0].observable == "S10.LegibilityFrontier"


def test_s10_gates_the_production_fit_honestly(tmp_path):
    store = EvidenceStore(tmp_path)
    frozen, result = run_study(build_spec(), store=store)

    assert result.outcomes["H3-real-legibility"] == "void"
    assert "real_tacit_fraction" not in result.metrics

    gate = store.find(observable="S10.RealLegibilityGate")
    assert gate and gate[0].trust is TrustLevel.REGISTERED
    assert gate[0].value["status"] == "gated"
    assert "reward_lens.measure.indices" in gate[0].value["need"]

    report = render_report(frozen, result, store)
    assert "CONFIRMED" in report

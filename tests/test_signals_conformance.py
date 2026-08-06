"""M1 acceptance: the conformance suite passes on the tiny ``ClassifierRM`` (M1).

``run_conformance`` is the gate every signal adapter must clear before it is trusted: determinism,
batch-vs-single parity, left-pad invariance, readout-matches-head, prefix consistency, the
dtype-policy matrix, and template round-trip. This test asserts the tiny classifier passes it, which
is the structural fix for the InternLM2/QRM silent-exclusion class made executable. The four campaign
models run the same suite in a GPU job (they cannot be loaded here); passing on the tiny model with
the real module tree is the M1 deliverable.
"""

from __future__ import annotations

import pytest

from reward_lens.signals.conformance import run_conformance
from reward_lens.signals.loaders import from_tiny

# The five checks the M1 acceptance criterion names explicitly.
_CORE_CHECKS = {
    "determinism",
    "batch_vs_single",
    "left_pad_invariance",
    "readout_matches_head",
    "prefix_consistency",
}


@pytest.fixture(scope="module")
def tiny_signal():
    return from_tiny(seed=3, conformance_quickcheck=False)


def test_conformance_report_passes(tiny_signal):
    report = run_conformance(tiny_signal)
    assert report.passed, "conformance failed:\n" + report.summary()


def test_core_checks_all_present_and_green(tiny_signal):
    report = run_conformance(tiny_signal)
    by_name = {c.name: c for c in report.checks}
    for name in _CORE_CHECKS:
        assert name in by_name, f"missing conformance check: {name}"
        check = by_name[name]
        assert check.passed and not check.skipped, f"{name} failed: {check.detail}"


def test_determinism_is_exact(tiny_signal):
    report = run_conformance(tiny_signal)
    determinism = next(c for c in report.checks if c.name == "determinism")
    assert determinism.passed, determinism.detail


def test_batch_parity_within_tolerance(tiny_signal):
    report = run_conformance(tiny_signal)
    parity = next(c for c in report.checks if c.name == "batch_vs_single")
    assert parity.passed, parity.detail


def test_readout_matches_native_head(tiny_signal):
    report = run_conformance(tiny_signal)
    check = next(c for c in report.checks if c.name == "readout_matches_head")
    assert check.passed, check.detail


def test_prefix_last_equals_score(tiny_signal):
    report = run_conformance(tiny_signal)
    check = next(c for c in report.checks if c.name == "prefix_consistency")
    assert check.passed, check.detail


def test_dtype_matrix_no_nan(tiny_signal):
    """The bf16/fp16/fp32-trunk x fp32-head matrix produces no NaN wherever the dtype runs."""
    report = run_conformance(tiny_signal)
    check = next(c for c in report.checks if c.name == "dtype_matrix")
    # fp32 must always run finite; a device that cannot run bf16/fp16 marks those skipped, not NaN.
    assert check.passed, f"dtype matrix produced NaN: {check.detail}"
    assert "float32:finite" in check.detail, check.detail


def test_template_round_trip_preserves_span(tiny_signal):
    report = run_conformance(tiny_signal)
    check = next(c for c in report.checks if c.name == "template_round_trip")
    assert check.passed, check.detail

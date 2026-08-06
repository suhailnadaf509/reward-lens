"""Acceptance: every record-only instrument returns Evidence or a Refusal, none an exception.

Run against the real campaign store, which is the point: a converter proved on a fixture it also
wrote has proved nothing about whether the record format can hold data it was not designed
for. The synthetic tests are in ``tests/test_record_convert.py`` and they are the fast ones.

The store lives outside the repository because it is a 313 MB archive of a $17.73 experiment.
``REWARD_LENS_CAMPAIGN_STORE`` is the only way to point at it and there is no default, so the
module skips naming that variable rather than guessing at a path.

Which store matters: ``campaign-results/runs/campaign``, not ``campaign-results/store``. The
second one holds no adjudication rows, and a converter pointed at it can reproduce no verdict.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from reward_lens.core.evidence import Evidence
from reward_lens.core.reading import Refusal
from reward_lens.core.types import Access, Capability, Component
from reward_lens.record.convert import (
    CampaignStore,
    convert_campaign,
    reader_access,
    shipped_instruments,
    sweep,
)
from reward_lens.record.convert.instruments import (
    access_declaration_findings,
    capabilities_in_record,
    is_record_only,
    uninstantiable,
    unnamed_bases,
)

#: The campaign evidence store, which is not in this repository. There is no default: point
#: ``REWARD_LENS_CAMPAIGN_STORE`` at the store directory or the tests that need it skip.
_CAMPAIGN_ENV = os.environ.get("REWARD_LENS_CAMPAIGN_STORE")
CAMPAIGN_STORE = Path(_CAMPAIGN_ENV) if _CAMPAIGN_ENV else None

pytestmark = pytest.mark.skipif(
    CAMPAIGN_STORE is None or not (CAMPAIGN_STORE / "evidence.jsonl").exists(),
    reason=(
        "no campaign evidence store. It is the 313 MB archive the 2.0 campaign produced and it "
        "is not in the repository; set REWARD_LENS_CAMPAIGN_STORE to its directory to run this "
        "file."
    ),
)


@pytest.fixture(scope="module")
def converted():
    """The converted run, its report, and the full capability scan, computed once.

    The capability scan decodes every one of the thousand banks and takes about 25 seconds. It is
    not bounded, because the eight ProcessBench banks sit at step indices 597 to 606 and a bounded
    scan would report that the record holds no per-step scores, which is false.
    """
    run, report = convert_campaign(CAMPAIGN_STORE)
    caps, scanned, total = capabilities_in_record(run)
    return run, report, caps, scanned, total


@pytest.fixture(scope="module")
def swept(converted):
    run, _, caps, scanned, _ = converted
    return sweep(run, shipped_instruments(), limit=8, caps=caps, caps_steps=scanned)


# ---------------------------------------------------------------------------
# The clause
# ---------------------------------------------------------------------------


def test_no_instrument_raises_against_the_converted_record(swept) -> None:
    """The clause, stated as the sweep's exception list being empty.

    Asserted on the recorded list rather than by letting an exception escape, because a failure
    then names the instrument and carries its exception instead of being one line of traceback in
    the middle of thirty-three calls.
    """
    assert swept.exceptions == (), "\n".join(o.render() for o in swept.exceptions)


def test_every_record_only_instrument_returns_evidence_or_a_refusal(swept) -> None:
    """And the record-only ones specifically, which is what the clause names."""
    assert swept.record_only, "no instrument was found to be satisfiable at RECORD access"
    for name in swept.record_only:
        outcome = swept.for_instrument(name)
        assert outcome is not None
        assert outcome.kind in ("evidence", "note", "refusal"), outcome.render()
        assert isinstance(outcome.reading, (Evidence, Refusal)), outcome.render()


def test_every_refusal_carries_a_reason_and_a_remedy(swept) -> None:
    """A refusal is a value the reader acts on, so the remedy is the part that has to be there."""
    refusals = swept.of_kind("refusal")
    assert refusals
    for outcome in refusals:
        refusal = outcome.reading
        assert refusal.reason is not None
        assert refusal.detail.strip()
        assert refusal.remedy.strip()
        assert refusal.meaning


def test_the_record_only_set_is_the_one_the_access_matrix_decides(swept, converted) -> None:
    """Membership is computed from `declared_access`, not from a list somebody kept up to date."""
    run = converted[0]
    access = reader_access(run)
    assert access == {
        Component.TASK: Access.RECORD,
        Component.GRADER: Access.RECORD,
        Component.RECORD: Access.RECORD,
    }
    recomputed = {
        getattr(i, "name", type(i).__name__)
        for i in shipped_instruments()
        if is_record_only(i, access)
    }
    assert recomputed == set(swept.record_only)


# ---------------------------------------------------------------------------
# What the sweep found, recorded so a change to it is visible
# ---------------------------------------------------------------------------


def test_the_shipped_battery_is_discovered_rather_than_listed() -> None:
    """Discovery is by import, and what it skips it names.

    Deliberately not an exact count. The battery grew from 33 to 38 while this package was being
    built, because other series were landing in the same wave, and a test that pins the number
    turns somebody else's merge into this package's failure. What is asserted is the rule:
    everything discovered declares a name and constructs bare, and the two kinds of skip are
    enumerated rather than silent.

    `ControlInstrument` and `FrontierInstrument` inherit `BaseObservable.name`, so both satisfy the
    runtime-checkable `Instrument` protocol without being instruments. That is the signal discovery
    uses, and it is why an abstract base does not appear in the sweep as a `NotImplementedError`.
    """
    instruments = shipped_instruments()
    assert len(instruments) >= 33
    for instrument in instruments:
        assert instrument.name and instrument.name != "observable"

    assert "reward_lens.measure.controls._base.ControlInstrument" in unnamed_bases()
    assert "reward_lens.measure.rate.regime.RunRegime" in uninstantiable()
    assert not set(unnamed_bases()) & {type(i).__module__ for i in instruments}


def test_the_record_holds_scores_a_readout_and_step_scores_and_no_activations(converted) -> None:
    """Capabilities measured from the record, not declared by anyone.

    The twenty capture manifests reference 324 tensors and every one of them is an `AbsentRef`:
    the campaign wrote activations to a Modal volume and shipped the manifests. So `ACTIVATIONS` is
    absent from the record even though a manifest for it is in the store, which is exactly the
    distinction `AbsentRef` exists to keep.
    """
    _, report, caps, scanned, total = converted
    assert caps & Capability.SCORES
    assert caps & Capability.STEP_SCORES
    assert caps & Capability.LINEAR_READOUT
    assert not (caps & Capability.ACTIVATIONS)
    assert not (caps & Capability.SPAN_TYPES)
    assert (scanned, total) == (1000, 1000)
    assert report.capture_manifests == 20
    assert report.absent_tensors == 324
    assert report.readout_vectors == 11


def test_the_converted_run_is_the_shape_the_store_says_it_is(converted) -> None:
    """Every number here was counted off the store rather than quoted from the specification.

    A second directory, ``campaign-results/store``, was recorded as holding 1,315 rows and 143
    payload arrays. This store has 1,363 rows, 82 observables and 1,076 payload arrays, and the
    1,315 is the merge count in ``close_summary.json`` taken before 48 close-out rows were
    appended.
    """
    run, report, _, _, _ = converted
    assert run.kind == "eval"
    assert report.store_rows == 1363
    assert report.observables == 82
    assert run.n_steps == 1000  # 992 score banks plus 8 ProcessBench banks
    assert len(report.graders) == 14
    assert len(report.banks) == 13
    assert report.unrepresented_rows == 332

    integrity = CampaignStore(CAMPAIGN_STORE).sidecar_report()
    assert integrity["referenced"] == 1076
    assert integrity["unresolved"] == 0
    assert integrity["orphaned_in_primary"] == 0


def test_the_run_declares_no_regime_because_the_campaign_declared_none(converted) -> None:
    """Absent is not a pass, and an empty declaration is how the record says nobody declared."""
    run = converted[0]
    assert run.regime.declared == {}
    assert run.regime.disagreements(None) == {}


def test_the_schema_findings_travel_with_the_conversion(converted) -> None:
    """A finding that lives only in a side report is a finding nobody reads twice."""
    report = converted[1]
    assert len(report.findings) == 6
    for finding in report.findings:
        assert finding.strip()
    assert any("Step.index" in f for f in report.findings)
    assert any("readout vector" in f for f in report.findings)


def test_no_new_instrument_declares_its_access_under_the_wrong_name() -> None:
    """A defect in another package, recorded here because this sweep is what surfaced it.

    The instrument contract and `declared_access` read `requires`. Three of the control
    instruments declare `access`, so preflight never checks their access matrix and the
    `ACCESS_INSUFFICIENT` refusal it would produce never fires.

    Asserted as a subset of the three known offenders rather than as an exact count, so fixing them
    upstream passes and a fourth appearing fails. Measured at three when this was written.
    """
    known = {"DumbBaselineBank", "MatchedPositiveControl", "SemanticPlacebo"}
    findings = access_declaration_findings(shipped_instruments())
    offenders = {f.split(" ")[0] for f in findings}
    assert offenders <= known, (
        f"a new instrument declares its access matrix under another name: {offenders - known}"
    )


def test_the_layout_name_does_not_pin_the_group_size_and_the_shape_does(converted) -> None:
    """Two of the 992 banks declare `bank` and carry a (2000, 4) array.

    Fourteen other `bank` rows carry (N, 32). Nothing is corrupt: the campaign's layout vocabulary
    does not determine K. A converter that had trusted the name would have built two thousand
    groups of thirty-two out of four columns.
    """
    report = converted[1]
    assert len(report.layout_mismatches) == 2
    for line in report.layout_mismatches:
        assert "layout 'bank' says K=32, shape says K=4" in line

    run = converted[0]
    from reward_lens.record.convert.campaign import layout_audit

    banks, mismatched = layout_audit(CampaignStore(CAMPAIGN_STORE))
    assert banks == 992
    assert len(mismatched) == 2
    assert run.n_steps == banks + 8  # the eight ProcessBench banks


def test_the_full_walk_counts_what_the_store_says_it_holds(converted) -> None:
    """Counted by walking every bank, against the store's own item and score totals.

    616,023 items and 1,062,908 scalar scores across the 992 score banks, plus 6,800 ProcessBench
    solutions carrying 58,194 reasoning steps between them. Zero abstentions: no grader in this
    campaign returned a non-finite score.
    """
    from reward_lens.record.convert import count_run

    run, report, _, _, _ = converted
    counted = count_run(run, report)
    assert counted.counted is True
    assert counted.groups == 622823  # 616,023 bank items + 6,800 ProcessBench solutions
    assert counted.trajectories == 1069708  # 1,062,908 bank scores + 6,800
    assert counted.turns == 1121102  # one per bank response, one per reasoning step
    assert counted.abstentions == 0
    assert counted.labels_blinded == 6800
    assert counted.labels_declined == 0


def test_the_held_out_error_step_labels_are_blinded_and_carry_no_error_rate(converted) -> None:
    """A held-out label reaches `Trajectory.labels` only inside a `Blind`, and unaudited.

    ProcessBench's earliest-error-step annotation is an oracle. Nobody measured its error rate for
    this campaign, so `LabelQuality.error_rate` is None and the library refuses any scoring read
    against it, which is the correct outcome rather than an obstacle.
    """
    run = converted[0]
    prm = next(s for s in run.steps.slice(597, 598))
    traj = prm.groups[0].trajectories[0]
    assert traj.n_turns >= 1
    assert traj.turns[0].step_score is not None

    label = traj.labels["earliest_error_step"]
    assert type(label).__name__ == "Blind"
    assert not hasattr(label, "unwrap")
    assert label.quality is not None
    assert label.quality.error_rate is None
    assert label.quality.is_measured is False

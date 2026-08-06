"""Acceptance: `record/labels.py` and `Blind[T]`.

The clause, in full: *a function annotated to take features cannot be passed a `Blind`, checked by
the type checker in CI, plus the three runtime tests from `flight-recorder`: disjoint field sets, a
name blocklist, and introspection of the detector protocol's annotation.*

Two halves and they are asserted separately, because they fail separately and they are enforced by
different machinery. The static half needs a type checker, so it runs one, on a fixture that must be
rejected: `tests/w2_3_typing/leaks.py`, with the precise line-by-line assertions in
`tests/test_record_labels_typing.py` and the CI job in `.github/workflows/tests.yml` named
`blind-types`. The runtime half needs no tools and holds on a base install.

A fifth test walks the whole path once, because the four clause tests each check one wall and the
thing being built is a room: a record goes in, a detector reads the visible half and never the
oracle, the scorer opens the oracle through `adjudicate`, and the store ends up holding a row that
says so. If any wall were fake that walk would still pass, which is why it is last rather than
first.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.store import EvidenceStore
from reward_lens.record.labels import (
    Blind,
    LabelLeak,
    LabelQuality,
    OracleFrame,
    ReadPurpose,
    RolloutFrame,
    adjudicate_frame,
    blind,
    check_detector,
    detector_findings,
    label_reads,
    split_trajectory,
)
from reward_lens.record.schema import FeatureID, make_trajectory
from reward_lens.record.turns import Turn

ROOT = Path(__file__).resolve().parent.parent.parent
LEAKS = ROOT / "tests" / "w2_3_typing" / "leaks.py"

AUDITED = LabelQuality(
    error_rate=0.03,
    n_audited=200,
    method="two raters on a stratified sample, third adjudicating disagreements",
    measured_by="pool-a",
)


# ---------------------------------------------------------------------------
# The static half
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    importlib.util.find_spec("mypy") is None,
    reason=(
        "mypy is not installed. The static half of the clause is unverified here and is "
        "verified by the blind-types job in CI, which installs the [dev] extra."
    ),
)
def test_a_features_function_cannot_be_passed_a_blind_under_the_type_checker(tmp_path) -> None:
    """The static half of the clause, as the clause words it.

    `MYPYPATH` points at `src/` because the package ships no `py.typed` marker, and without it
    mypy resolves every reward_lens name to `Any` and reports none of this. That is asserted, and
    argued, in `tests/test_record_labels_typing.py`.
    """
    env = dict(os.environ, MYPYPATH=str(ROOT / "src"))
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--follow-imports=silent",
            "--no-error-summary",
            "--no-color-output",
            f"--cache-dir={tmp_path / 'mypy-cache'}",
            str(LEAKS),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert proc.returncode != 0, (
        "mypy accepted the leakage fixture. A clean type check there is a failure of this clause, "
        "not a pass: the fixture exists to be rejected."
    )
    assert 'Argument 1 to "rate_by_length" has incompatible type "Blind' in proc.stdout
    assert 'expected "Mapping[FeatureID, float]"' in proc.stdout
    # There is no `.unwrap()`, and that is a static fact rather than a docstring.
    assert '"Blind[bool | int | float | str]" has no attribute "unwrap"' in proc.stdout


# ---------------------------------------------------------------------------
# The three runtime tests
# ---------------------------------------------------------------------------


def test_1_the_frames_have_disjoint_field_sets() -> None:
    rollout = set(RolloutFrame.__dataclass_fields__)
    oracle = set(OracleFrame.__dataclass_fields__)
    assert rollout and oracle
    assert rollout & oracle == set()


def test_2_a_label_cannot_be_carried_in_under_a_blocked_name() -> None:
    """`features` is `Mapping[FeatureID, float]`, so this leak is invisible to the type checker."""
    with pytest.raises(LabelLeak, match="answer key in float clothing"):
        RolloutFrame(
            trajectory_id="t1",
            task_id="task-1",
            turns=("perhaps",),
            n_tokens=1,
            advantage=None,
            features={FeatureID("is_hack"): 1.0},
        )


def hedging_rate(frame: RolloutFrame) -> float:
    """A detector: the visible half in, a score out."""
    return sum(text.count("perhaps") for text in frame.turns) / max(frame.n_tokens, 1)


def peeks_at_the_oracle(frame: OracleFrame) -> float:
    return float(len(frame.labels))


def peeks_at_one_label(label: Blind[bool]) -> float:
    return 0.0


def test_3_the_detector_protocols_annotation_is_introspected() -> None:
    check_detector(hedging_rate)
    for leaky in (peeks_at_the_oracle, peeks_at_one_label):
        assert len(detector_findings(leaky)) == 1
        with pytest.raises(LabelLeak, match="can reach the held-out labels"):
            check_detector(leaky)


# ---------------------------------------------------------------------------
# The walk through
# ---------------------------------------------------------------------------


def a_trajectory(index: int, hacked: bool) -> object:
    return make_trajectory(
        id=f"t{index}",
        task_ref="task-1",
        turns=[
            Turn(
                index=0,
                role="assistant",
                text="perhaps the tests pass" if hacked else "here is the proof",
                token_ids=(1, 2, 3, 4),
                logprobs_sampling=None,
                logprobs_train=None,
                loss_mask=None,
                tool_call=None,
            )
        ],
        advantage=0.5 if hacked else -0.25,
        labels={"hacked": blind(hacked, key="hacked", quality=AUDITED)},
        features={"len_tokens": 4.0},
    )


def test_a_scoring_pass_reads_the_oracle_once_and_the_store_says_so(tmp_path) -> None:
    """Four trajectories, one detector, one scoring pass, four rows.

    The row is the whole design. The point was never that a label can never be read, since a label
    that can never be read cannot score anything. The point is that reading one leaves a trace a
    reviewer can find, and this is that reviewer's query.
    """
    store = EvidenceStore(tmp_path)
    truth = [True, False, True, False]

    correct = 0
    for index, hacked in enumerate(truth):
        rollout, oracle = split_trajectory(a_trajectory(index, hacked))

        # The detector sees the visible half. It has no path to the oracle: not through an
        # attribute, and not through the type checker.
        predicted = hedging_rate(rollout) > 0.0
        assert not hasattr(rollout, "labels")

        opened = adjudicate_frame(
            oracle,
            instrument="w2.3.acceptance",
            purpose=ReadPurpose.SCORING,
            why="scoring the hedging-rate detector against the held-out hack labels",
            store=store,
        )
        assert not isinstance(opened, Refusal)
        correct += int(opened["hacked"] == predicted)

    assert correct == 4  # the planted signal is authored, so detecting it proves plumbing only

    rows = label_reads(store)
    assert len(rows) == 4
    assert {row.value.instrument for row in rows} == {"w2.3.acceptance"}
    assert {row.value.purpose for row in rows} == {"scoring"}
    assert all(row.value.n_labels == 1 for row in rows)
    assert all(row.value.error_rate == 0.03 for row in rows)

    # And the rows carry fingerprints rather than labels, so the audit trail is not itself a
    # second copy of the answer key sitting at RECORD access.
    on_disk = (tmp_path / "evidence.jsonl").read_text(encoding="utf-8")
    assert all(row.value.fingerprint in on_disk for row in rows)
    assert '"_value"' not in on_disk


def test_scoring_against_an_unaudited_answer_key_is_refused(tmp_path) -> None:
    """A refusal is a value, and the reason the barrier is not only a type.

    A `Blind` keeps the oracle out of the detector. It says nothing about whether the oracle is
    right, and scoring against a one-third-wrong answer key measures the answer key.
    """
    store = EvidenceStore(tmp_path)
    frame = OracleFrame(trajectory_ref="t0", labels={"hacked": blind(True, key="hacked")})

    reading = adjudicate_frame(
        frame,
        instrument="w2.3.acceptance",
        purpose=ReadPurpose.SCORING,
        why="scoring the hedging-rate detector",
        store=store,
    )

    assert isinstance(reading, Refusal)
    assert reading.reason is RefusalReason.LABEL_QUALITY_UNKNOWN
    assert "n_audited=0" in reading.detail
    assert "purpose=ReadPurpose.AUDIT" in reading.remedy
    assert label_reads(store) == ()

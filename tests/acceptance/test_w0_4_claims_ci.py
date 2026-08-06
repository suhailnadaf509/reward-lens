"""Acceptance: an unbound number in a docs page fails the workflow.

The clause this file discharges: *a deliberately unbound number in a docs page fails the workflow.*

`artifacts/claims.py` has been documented as the CI entry point since 2.0 and grep over
`.github/workflows/` for "claims" returned nothing, so the guard against numbers that came from a
draft rather than from a measurement spent two releases not running. It also had no runnable
entry point and no notion of an *unbound* number at all: it verified tagged claims and dangling
evidence references, which catches a number that disagrees with the store and misses the commoner
failure of a number that was never tagged.

Both halves are tested here, and so is the fact that the workflow calls it.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

from reward_lens.artifacts.claims import baseline_key, check_files, find_unbound_numbers, main

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_WORKFLOW = _ROOT / ".github" / "workflows" / "tests.yml"


def test_a_bare_measurement_in_prose_is_unbound():
    found = find_unbound_numbers("Your effective group size is 4.2, not 16.\n")
    assert [u.value for u in found] == ["4.2"]
    assert "effective group size" in found[0].context


@pytest.mark.parametrize(
    "text",
    [
        "An illustrative value of 0.75 shows the shape.\n",
        "For example, a curl mass of 0.21 would mean this.\n",
        "The margin is 0.31 [[claim value=0.31 ev=ev:abcd1234]].\n",
        "Set `tol=0.05` in the config.\n",
        "```\nmean = 0.62\n```\n",
        "reward-lens 3.0.0a1 supersedes v2.0 today.\n",
        "See section 3.1 and appendix 4.7 for the derivation.\n",
        "hypothesis is MPL-2.0 and cosmic-ray is MIT.\n",
    ],
)
def test_what_is_not_an_unbound_number(text):
    """False positives are what kill a checker: people learn to ignore it, then it is decoration."""
    assert find_unbound_numbers(text) == []


def test_an_illustrative_marker_only_covers_its_own_sentence():
    """Otherwise one disclaimer at the top of a page launders every number below it."""
    text = "An illustrative value of 0.75 shows the shape.\nThe measured attenuation is 0.71.\n"
    assert [u.value for u in find_unbound_numbers(text)] == ["0.71"]


def test_a_deliberately_unbound_number_in_a_docs_page_exits_nonzero(tmp_path):
    """The clause, run the way CI runs it."""
    page = tmp_path / "page.md"
    page.write_text("# A page\n\nThe measured attenuation factor is 0.71.\n", encoding="utf-8")
    assert main([str(page)]) == 1

    page.write_text(
        "# A page\n\nThe measured attenuation factor is 0.71 [[claim value=0.71 ev=ev:deadbeef]].\n",
        encoding="utf-8",
    )
    # Still nonzero, now for the right reason: the evidence id does not resolve. A number bound to
    # a measurement that does not exist is not better than an unbound one.
    assert main([str(page)]) == 1


def test_a_malformed_claim_tag_fails_rather_than_crashing_the_run(tmp_path):
    """A checker whose job is to catch bad numbers must not be stoppable by one.

    The docs carry a template tag with a placeholder value, and until this was fixed it took the
    whole run down with `ValueError: could not convert string to float`.
    """
    page = tmp_path / "page.md"
    page.write_text("[[claim value=… ev=… field=…]]\n", encoding="utf-8")
    report = check_files([page])
    assert report.n_failures == 1
    assert "malformed claim tag" in report.render()


def test_the_baseline_ratchet_exempts_the_known_and_fails_the_new(tmp_path):
    """A permanently red gate is one people learn to ignore, which is worse than no gate."""
    page = tmp_path / "page.md"
    page.write_text("Old number 0.41 here.\n", encoding="utf-8")
    baseline = tmp_path / "baseline.txt"

    assert main([str(page), "--baseline", str(baseline), "--write-baseline"]) == 0
    assert main([str(page), "--baseline", str(baseline)]) == 0

    page.write_text("Old number 0.41 here.\nAnd a new one, 0.87.\n", encoding="utf-8")
    assert main([str(page), "--baseline", str(baseline)]) == 1


def test_the_baseline_key_survives_an_edit_above_it(tmp_path):
    """Keyed on the sentence, not the line, so unrelated edits do not re-fire the whole backlog."""
    page = tmp_path / "page.md"
    page.write_text("Old number 0.41 here.\n", encoding="utf-8")
    before = [baseline_key(u) for u in check_files([page]).unbound]

    page.write_text("A new heading paragraph.\n\nOld number 0.41 here.\n", encoding="utf-8")
    after = [baseline_key(u) for u in check_files([page]).unbound]
    assert before == after


def test_the_workflow_actually_calls_the_checker():
    """The defect was never that the checker was wrong. It was that nothing ran it."""
    text = _WORKFLOW.read_text(encoding="utf-8")
    assert "reward-lens-claims" in text, "the claims checker is not wired into CI"
    assert "claims" in text


def test_the_docs_gate_is_green_as_committed():
    """Whatever the baseline exempts, the repository must pass its own gate right now."""
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "reward_lens.artifacts.claims",
            str(_ROOT / "docs" / "content"),
            "--baseline",
            str(_ROOT / "docs" / "claims-baseline.txt"),
            *[
                arg
                for name in (
                    "cards-and-claims.md",
                    "artifacts-operate.md",
                    "cli.md",
                    "measurements-you-can-trust.md",
                    "evidence-store.md",
                )
                for arg in ("--exclude", name)
            ],
        ],
        capture_output=True,
        text=True,
        cwd=_ROOT,
    )
    assert r.returncode == 0, r.stdout + r.stderr

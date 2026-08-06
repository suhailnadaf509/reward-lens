"""The static half of the barrier, driven from pytest: mypy must reject the leakage fixture.

The clause is *a function annotated to take features cannot be passed a `Blind`,
checked by the type checker in CI*. That is a claim about a checker rejecting something, so the
test has to run the checker and assert on what it rejected. A test that only asserted the good
path type checks would pass just as happily against a `Blind` that was an alias for `Any`.

Two fixtures under `tests/w2_3_typing/`, neither collected by pytest and neither imported by
anything:

- `leaks.py` must be rejected, on exactly the lines carrying an `# EXPECT: <code>` marker and on no
  others. Both directions matter. An expected error that stops firing means the barrier is gone; an
  unexpected one usually means the fixture failed to import, which would otherwise let "mypy exited
  non-zero" stand in for "mypy caught the leak".
- `clean.py` must be accepted, so the barrier is a barrier rather than a wall.

`MYPYPATH` points at `src/`, so these run against the checkout rather than whatever is installed.
It used to be load-bearing rather than tidiness: the package shipped no `py.typed` marker, so mypy
treated the installed `reward_lens` as untyped and, with the `ignore_missing_imports = true` in
this project's mypy config, resolved every name in it to `Any`. Under that resolution `leaks.py`
produced one unrelated error and none of the eight real ones, which is a barrier that reads as
enforced and enforces nothing.

The marker now ships, so `MYPYPATH` is belt and braces here and the guard has moved into
`test_the_type_check_is_actually_reading_our_types`, which runs with no `MYPYPATH` at all and
asserts mypy resolves the real signature. That test is what catches the marker being lost again to
an edit in `[tool.setuptools.package-data]`.
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "w2_3_typing"
LEAKS = FIXTURES / "leaks.py"
CLEAN = FIXTURES / "clean.py"

#: `path:line: error: message  [code]`
_ERROR = re.compile(
    r"^(?P<path>[^:]+):(?P<line>\d+): error: (?P<message>.*?)\s+\[(?P<code>[^\]]+)\]$"
)
#: `# EXPECT: code` at the end of a fixture line.
_MARKER = re.compile(r"#\s*EXPECT:\s*(?P<code>[a-z-]+)\s*$")

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("mypy") is None,
    reason=(
        "mypy is not installed, so the static half of the barrier cannot be checked here. It is "
        "installed by the [dev] extra and it runs in the blind-types job in CI."
    ),
)


@pytest.fixture(scope="module")
def cache(tmp_path_factory) -> Path:
    """One mypy cache for the whole module.

    Every invocation here type checks the same import graph, and building it cold costs about nine
    seconds. Sharing the cache takes the module from 46 seconds to 14, measured. The one run that
    must not share it is the `py.typed` probe, which deliberately resolves the package differently.
    """
    return tmp_path_factory.mktemp("mypy-cache")


def run_mypy(target: Path, cache: Path) -> tuple[int, list[tuple[int, str]]]:
    """Run mypy on one file and return its exit code and every (line, error-code) it reported.

    ``--follow-imports=silent`` still type checks everything the fixture imports; it only declines
    to report errors found in those files. Without it a sibling package mid-edit could turn this
    test red for a reason that has nothing to do with `Blind`.
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
            f"--cache-dir={cache}",
            str(target),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    reported = []
    for line in proc.stdout.splitlines():
        match = _ERROR.match(line.strip())
        if match:
            reported.append((int(match["line"]), match["code"]))
    return proc.returncode, reported


def expected_from_markers(fixture: Path) -> list[tuple[int, str]]:
    """The `# EXPECT: <code>` markers in a fixture, as (line number, error code)."""
    out = []
    for number, text in enumerate(fixture.read_text(encoding="utf-8").splitlines(), start=1):
        match = _MARKER.search(text)
        if match:
            out.append((number, match["code"]))
    return out


def test_mypy_rejects_the_leakage_fixture(cache) -> None:
    """The clause. A clean type check here is a failure, not a pass."""
    expected = expected_from_markers(LEAKS)
    assert len(expected) == 8, "the fixture lost its markers"

    code, reported = run_mypy(LEAKS, cache)

    assert code != 0, (
        "mypy accepted tests/w2_3_typing/leaks.py, which passes a Blind to a function annotated "
        "to take features. The static half of the barrier is not being enforced."
    )
    assert sorted(reported) == sorted(expected)


def test_the_features_function_is_the_one_the_clause_names(cache) -> None:
    """Pinned separately because it is the clause's own sentence.

    The other seven markers are the surrounding barrier. This one line is what the clause says has
    to be a type error, so it is asserted by message rather than only by code.
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
            f"--cache-dir={cache}",
            str(LEAKS),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert 'Argument 1 to "rate_by_length" has incompatible type "Blind' in proc.stdout
    assert 'expected "Mapping[FeatureID, float]"' in proc.stdout


def test_mypy_accepts_the_clean_fixture(cache) -> None:
    """Detecting, scoring and auditing all type check on the intended path.

    Without this, the cheapest way to pass the test above would be a type nothing can be written
    against, and the barrier would be routed around within a week while CI stayed green.
    """
    code, reported = run_mypy(CLEAN, cache)
    assert reported == []
    assert code == 0


def test_the_label_module_itself_type_checks(cache) -> None:
    code, reported = run_mypy(ROOT / "src" / "reward_lens" / "record" / "labels.py", cache)
    assert reported == []
    assert code == 0


def test_the_type_check_is_actually_reading_our_types(tmp_path) -> None:
    """Without `py.typed`, every reward_lens type resolves to `Any` and this whole file is theatre.

    A type checker treats a package with no `py.typed` marker as untyped, and this project's
    `ignore_missing_imports = true` then resolves every name in it to `Any`. Under that resolution
    `Blind` accepts every call and the eight expected errors in `leaks.py` drop to one unrelated
    one, so the barrier reads as enforced and enforces nothing.

    This was originally written the other way round, asserting the marker was still *missing* so
    that landing it would fail loudly rather than leaving a redundant workaround behind. It landed,
    so the assertion is inverted: the marker is present, and mypy resolves the real signature with
    no `MYPYPATH` on the environment. That makes this a standing guard rather than a one-shot
    alarm, because the marker can be lost again by an edit to `[tool.setuptools.package-data]`
    without anything else in the suite noticing.
    """
    marker = ROOT / "src" / "reward_lens" / "py.typed"
    assert marker.exists(), (
        "src/reward_lens/py.typed is gone. Without it a downstream type checker resolves every "
        "reward_lens name to Any, which makes Blind[T] unenforceable at exactly the boundary it "
        "guards. Restore the file and keep it in pyproject's package-data."
    )

    probe = tmp_path / "probe.py"
    probe.write_text(
        "from reward_lens.record.labels import Blind\nreveal_type(Blind)\n", encoding="utf-8"
    )
    env = {k: v for k, v in os.environ.items() if k != "MYPYPATH"}
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--no-error-summary",
            "--no-color-output",
            f"--cache-dir={tmp_path / 'mypy-cache'}",
            str(probe),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert 'Revealed type is "Any"' not in proc.stdout, (
        "mypy resolved Blind to Any with py.typed present. The marker is not reaching the "
        "installed package: check that it is still listed in pyproject's package-data."
    )
    assert "Blind" in proc.stdout, proc.stdout

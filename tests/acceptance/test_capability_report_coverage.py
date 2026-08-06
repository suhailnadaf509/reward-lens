"""The capability report must not tell a reader that a built library is unbuilt.

This is the regression for the corrected E58. The report decides what to put under
`SPECIFIED, NOT YET BUILT` by asking which quantities have a registered estimator, and estimators
register at module import rather than centrally. In a process that had only done `import
reward_lens`, that set was empty, so **every** catalogue row rendered as an instrument nobody had
written yet. Measured on the 3.0.0rc1 wheel in a clean environment: 0 registered, 95 of 95 rows
reported unbuilt.

**Every assertion here runs in a subprocess, and that is the whole design of the file.** In-process
the defect is invisible: by the time a test module has imported its fixtures, something has already
pulled the leaves and `ESTIMATORS` is full. The bug only exists in a process that has not done that,
which is exactly the process a user gets. A test that checked this in-process would have passed
throughout the period the front door was broken.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest


def _in_fresh_process(body: str) -> str:
    """Run `body` in a subprocess that has imported nothing, and return its stdout.

    `-I` isolates from the environment and the user site directory, so the child cannot inherit a
    partially imported tree or a stray `PYTHONPATH` that would mask the condition under test.
    """
    result = subprocess.run(
        [sys.executable, "-I", "-c", textwrap.dedent(body)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"the probe process failed, which is itself the finding:\n{result.stderr[-2000:]}"
    )
    return result.stdout.strip()


def test_a_fresh_process_registers_the_estimator_ladder() -> None:
    """The hook exists and fills a registry that starts empty.

    Asserted as a transition rather than as a final count, because the count moves every time a
    package lands and a pinned number would make an unrelated wave's arrival look like a failure
    here. What must not change is that it starts at zero and does not stay there.
    """
    out = _in_fresh_process(
        """
        from reward_lens.core.quantity import ESTIMATORS
        before = len(ESTIMATORS)
        from reward_lens.access.estimators import load_estimator_ladder
        report = load_estimator_ladder()
        print(before, len(ESTIMATORS), report.walked, report.ok)
        """
    )
    before, after, walked, ok = out.split()

    assert int(before) == 0, (
        "this test is only meaningful in a process where the leaves are unimported, and this one "
        "already had estimators registered before the loader ran"
    )
    assert int(after) > 0, "the loader ran and registered nothing"
    assert int(walked) > 100, (
        f"the walk only reached {walked} modules, so it did not cover the tree"
    )
    assert ok == "True", "a module failed to import for a reason that is not a missing extra"


def test_the_capability_report_does_not_report_a_built_library_as_unbuilt() -> None:
    """The regression proper: the front door, from a process that has imported nothing.

    The bound is deliberately loose and one-sided. Pinning the exact number of covered quantities
    would make this test fail every time a package lands, which trains people to update the number
    rather than to read the failure. What it must catch is the categorical error: a report in which
    *no* row resolves, which is what shipped.
    """
    out = _in_fresh_process(
        """
        from reward_lens.access.report import capability_report, load_instrument_catalogue
        from reward_lens.core.quantity import ESTIMATORS
        from reward_lens.core.types import Access, Component

        capability_report({Component.GRADER: Access.QUERY, Component.RECORD: Access.RECORD}, None, None)

        rows = load_instrument_catalogue()
        covered = {e.quantity for e in ESTIMATORS.values()}
        resolved = [r for r in rows if any(q in covered for q in r.quantities)]
        print(len(rows), len(covered), len(resolved))
        """
    )
    rows, covered, resolved = (int(v) for v in out.split())

    assert rows > 0, "the catalogue did not load at all"
    assert covered > 0, (
        "the capability report ran and no quantity has a registered estimator, so every catalogue "
        "row renders under SPECIFIED, NOT YET BUILT. That is the corrected E58: the report is "
        "telling a reader that a library of shipped instruments contains none of them"
    )
    assert resolved > 0, (
        f"all {rows} catalogue rows render as not yet built. The likely cause is that the "
        "estimator ladder is no longer loaded before the report computes its covered set"
    )


def test_the_front_door_survives_a_base_install() -> None:
    """A missing extra is named, never mistaken for a missing instrument, and never fatal.

    The base install is the case that matters most here, because it is what `pip install
    reward-lens` gives and it is the configuration where the most modules cannot import. The
    report's whole job is to answer honestly under partial access, so a module behind an extra must
    be reported as out of reach rather than as unwritten, and must not take the command down.
    """
    out = _in_fresh_process(
        """
        from reward_lens.access.estimators import load_estimator_ladder
        report = load_estimator_ladder()
        print(report.ok, len(report.broken), len(report.skipped))
        """
    )
    ok, broken, _skipped = out.split()

    assert ok == "True" and int(broken) == 0, (
        "a module failed to import for a reason that is not a missing optional extra. That is a "
        "defect in the tree, and it makes every quantity that module estimates look like an open "
        "research target rather than a broken import"
    )


@pytest.mark.parametrize("module", ["torch", "transformers"])
def test_the_ladder_stays_opt_in_so_importing_the_package_stays_torch_free(module: str) -> None:
    """The walk must never be triggered by importing the package.

    This assertion was written the other way round first, as "loading the ladder must not import
    torch", and it failed. **The first version was a property stated too strongly rather than a
    defect**: the walk imports every module it can, so in any environment where torch is installed
    it imports torch, and that is the walk working. Asserting otherwise would have forced the
    loader to special-case the modules it exists to load.

    The property that actually matters is narrower and is the one that could really be broken here.
    `import reward_lens` must not import torch, which its own CI job asserts, and the obvious wrong
    way to fix the corrected E58 would have been to call `load_estimator_ladder` from
    `reward_lens/__init__.py`. That would have made the front door correct and made every base
    install pull the deep-learning stack at import. It is called from `capability_report` instead,
    and this is what holds that line.
    """
    out = _in_fresh_process(
        f"""
        import sys
        import reward_lens
        import reward_lens.access
        print({module!r} in sys.modules)
        """
    )
    assert out == "False", (
        f"importing reward_lens pulled in {module}. The estimator ladder is meant to load when the "
        f"capability report asks for it, not when the package is imported; check that nothing "
        f"calls load_estimator_ladder at module scope"
    )

"""The catalogue's `status` field, enumerated against what the tree actually registers.

`spec/CATALOGUE.yaml` is the operative catalogue and `status` is its statement about whether an
instrument exists. Nothing recomputes that statement, so it went stale the moment the first package
closed and stayed stale from then on: 89 of 95 rows claimed an instrument had not been written
while 83 of them shipped, and `reward-lens capabilities` told a stranger so. E58.

The repair is a data edit and the data will drift again. This file is the part that does not: it
walks the installed tree, collects every class that declares a quantity, and compares that against
the field, naming the rows that disagree and in which direction. A row claiming `built` with no
instrument and a row claiming `planned` with a shipped one are different defects with different
remedies, and the messages say which.

**Why an enumeration rather than a count.** Three times an enumeration-as-a-test has caught a hole
nobody was looking for: E19's 28 unregistered retrofit rows, the docs build's quantity registered
only in-process, and E56's four contract-layer instruments that shipped failing lint rule 1 while
their package read `done`, found by enumerating the registry for an unrelated reason. The registry
enumeration is what stops a large build developing holes nobody notices. A count would have passed
every one of them.

**The two directions are not equally establishable, and that decides the shape of this file.** The
walk can only ever see fewer instruments than exist, because a subsystem behind an optional extra
does not import. So finding an instrument for a row that denies having one is sound under any
partial install and is asserted unconditionally. Failing to find one for a row that claims one is
only sound when the walk was complete, so a module it could not import is parsed instead: one that
assigns no `quantity` in any class body could not have carried a row, and does not weaken the
verdict. Only a module that could have carried one and could not be read downgrades the direction to
a skip, which names the rows and the modules rather than guessing at either. That is the same rule
the instruments follow: say what was not checked instead of reporting it as checked.

**What this file does not fix.** `status` being true is necessary and is not sufficient. The
capability report decides what to group under SPECIFIED, NOT YET BUILT from the estimator registry
rather than from this field, and an instrument that never called `register_estimator` is invisible
to it however correct its row is. Measured at the time of writing: 43 quantities carry an estimator
entry and 145 classes declare one, so 62 rows repaired here are still reported as not yet built. The
remaining half of E58 is in `access/report.py` and is not this file's to close.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import pathlib
import pkgutil

import pytest

import reward_lens
from reward_lens.core.extras import ExtraRequiredError
from reward_lens.measure.base import lint_instrument

CATALOGUE = pathlib.Path(reward_lens.__file__).parent / "spec" / "CATALOGUE.json"

#: Rows whose `quantities` field is the bare string OPEN, so no quantity id links them to code.
#: Both are deliberate and both carry the reason on the row. M6 has no instrument and nobody can
#: write its definition; M7 has one, `measure/meta/gum.py`, whose measurand is per-instance because
#: a combined standard uncertainty is in the units of whatever it is a budget for. Pinned by id
#: rather than counted, so a third row going OPEN is visible instead of silently undecidable.
UNLINKABLE = {"M6", "M7"}


def _declares_a_quantity_in_source(module_name: str) -> bool:
    """Whether an unimportable module assigns `quantity` in a class body, read rather than run.

    A module behind a missing extra can still be parsed, and parsing settles the only question that
    matters about it here: could it have carried a catalogue row. `reward_lens.sae` cannot, so the
    `[dict]` extra being absent does not have to cost this file its second direction. Returning True
    on a file that will not parse is the safe answer, because it only ever weakens a verdict.
    """
    path = pathlib.Path(reward_lens.__file__).parent / (
        module_name[len("reward_lens.") :].replace(".", "/")
    )
    for candidate in (path.with_suffix(".py"), path / "__init__.py"):
        if not candidate.exists():
            continue
        try:
            tree = ast.parse(candidate.read_text(encoding="utf-8"))
        except SyntaxError:
            return True
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for stmt in node.body:
                targets = (
                    [stmt.target]
                    if isinstance(stmt, ast.AnnAssign)
                    else getattr(stmt, "targets", [])
                )
                if any(isinstance(t, ast.Name) and t.id == "quantity" for t in targets):
                    return True
        return False
    return True


def _rows() -> list[dict]:
    with open(CATALOGUE, encoding="utf-8") as fh:
        return list(json.load(fh)["instruments"])


def _quantities(row: dict) -> list[str]:
    """A row's quantity ids, with the bare OPEN string normalised to nothing.

    The polymorphic-field trap of E14: iterating the string ``OPEN`` yields four single-character
    ids that resolve to nothing and get reported as nothing.
    """
    value = row.get("quantities")
    if value is None or isinstance(value, str):
        return []
    return [str(v) for v in value]


def _walk() -> tuple[dict[str, list[str]], list[str], list[tuple[str, str]]]:
    """Import every module and collect the quantity each class declares.

    Returns the quantity index, the modules skipped because an extra is not installed, and any
    module that failed for some other reason. `reward_lens.sae` is the skip on a full dev install:
    It sits behind the `[dict]` extra deliberately, so its `ExtraRequiredError` is the guard
    working rather than a defect. On a base install the white-box subsystems join it.
    """
    declared: dict[str, list[str]] = {}
    skipped: list[str] = []
    broken: list[tuple[str, str]] = []
    modules = []
    for found in pkgutil.walk_packages(reward_lens.__path__, "reward_lens."):
        try:
            modules.append(importlib.import_module(found.name))
        except ExtraRequiredError:
            skipped.append(found.name)
        except BaseException as exc:  # noqa: BLE001 - the point is to report it, not to raise it
            broken.append((found.name, f"{type(exc).__name__}: {exc}"))

    seen: set[tuple[str, str]] = set()
    for module in modules:
        for obj in vars(module).values():
            if not inspect.isclass(obj) or obj.__module__ != module.__name__:
                continue
            key = (obj.__module__, obj.__qualname__)
            if key in seen:
                continue
            seen.add(key)
            quantity = getattr(obj, "quantity", None)
            if isinstance(quantity, str) and quantity:
                declared.setdefault(quantity, []).append(f"{obj.__module__}.{obj.__qualname__}")
    return declared, skipped, broken


@pytest.fixture(scope="module")
def walk():
    return _walk()


def test_the_walk_reaches_the_tree_and_the_only_modules_it_misses_are_behind_an_extra(walk):
    """Establish the instrument first, because everything below reads its output.

    An import error that is not an extras guard would silently shrink the index and turn this whole
    file into a test that passes by seeing nothing.
    """
    declared, skipped, broken = walk
    assert not broken, (
        "modules failed to import for a reason that is not a missing extra:\n"
        + "\n".join(f"  {name}: {why}" for name, why in broken)
    )
    assert len(declared) > 100, (
        f"only {len(declared)} quantities are declared by any class, which is too few for this "
        f"comparison to mean anything. Something stopped the walk."
    )
    for name in skipped:
        assert name.startswith("reward_lens."), name


def test_no_catalogue_row_denies_an_instrument_that_ships(walk):
    """The E58 direction, and the one a partial install cannot get wrong.

    Seeing a class that declares a row's quantity is proof the instrument exists. Whether other
    modules imported is irrelevant to that, so this direction is asserted unconditionally.
    """
    declared, _, _ = walk
    wrong = []
    for row in _rows():
        if row.get("status") == "built":
            continue
        carriers = sorted({c for q in _quantities(row) for c in declared.get(q, [])})
        if carriers:
            wrong.append((row["id"], row.get("status"), carriers))
    assert not wrong, (
        "catalogue rows say an instrument has not been written and the tree ships one. The "
        "capability report groups these under SPECIFIED, NOT YET BUILT and tells a reader they do "
        "not exist. Set `status: built` on each row in spec/CATALOGUE.yaml, then run "
        "`python tools/regen_spec_json.py`. A Phase 6 row reaching this list means its module "
        "landed and its reading has still never been produced, which is the case for a status "
        "value distinct from both `built` and `planned`; no field records that today:\n"
        + "\n".join(
            f"  {rid} (status: {status}) is carried by {', '.join(carriers)}"
            for rid, status, carriers in wrong
        )
    )


def test_no_catalogue_row_claims_an_instrument_that_does_not_exist(walk):
    """The E56 direction: a row asserting a capability the tree cannot back.

    A row with no carrier is a real defect on a full install and is indistinguishable from an
    uninstalled extra on a base one. Two things keep this from being useless in either: a module
    that could not be imported is parsed instead, and one that assigns no `quantity` in any class
    body could not have carried a row whatever else it does, so it does not block the verdict; and
    when a module that could have carried one is genuinely unreadable, the verdict is downgraded to
    a skip naming both the rows and the modules rather than dropped or guessed.
    """
    declared, skipped, _ = walk
    blocking = sorted(m for m in skipped if _declares_a_quantity_in_source(m))
    wrong = []
    for row in _rows():
        if row.get("status") != "built" or row["id"] in UNLINKABLE:
            continue
        if not any(declared.get(q) for q in _quantities(row)):
            wrong.append((row["id"], _quantities(row)))
    detail = "\n".join(f"  {rid} names {q or ['no quantities at all']}" for rid, q in wrong)
    if wrong and blocking:
        pytest.skip(
            "these rows claim `status: built` and no class the walk could read declares their "
            "quantities, and the walk could not read "
            + ", ".join(blocking)
            + ", which do declare a quantity in source, so an instrument of theirs would be "
            "invisible here. Install the extras those modules need to settle it:\n" + detail
        )
    assert not wrong, (
        "catalogue rows claim `status: built` and no class in the tree declares any of their "
        "quantities. Either the instrument was never written, or it was written and its quantity "
        "id does not match the row:\n" + detail
    )


def test_the_rows_no_quantity_id_can_link_are_exactly_the_two_that_declare_none(walk):
    """`quantities: OPEN` makes a row undecidable here, so the set of them is pinned by name.

    Left as a count this would be the failure mode the enumeration exists to prevent: a third row
    going OPEN would move a number nobody reads instead of naming a row somebody has to look at.
    """
    unlinkable = {row["id"] for row in _rows() if not _quantities(row)}
    assert unlinkable == UNLINKABLE, (
        f"the rows this file cannot decide by quantity id have changed. Expected {sorted(UNLINKABLE)}, "
        f"found {sorted(unlinkable)}. A row with `quantities: OPEN` has no link to code, so its "
        f"`status` is set by hand and nothing checks it. If the change is deliberate, add the row "
        f"to UNLINKABLE with the reason it declares no measurand; if it is not, give the row its "
        f"quantity ids."
    )


def test_every_instrument_in_the_tree_passes_its_own_lint(walk):
    """E56's rule, standing: an acceptance test that renders a reading does not lint the declaration.

    Four instruments discharged their clause completely and did not exist by the architecture's
    own definition, because the clause tested the measurement and the lint tests the declaration. Those fail independently, so this checks the second over
    everything the walk can see rather than package by package.
    """
    declared, _, _ = walk
    modules = {}
    for quantity, carriers in declared.items():
        for carrier in carriers:
            modules[carrier] = quantity

    findings = []
    for carrier in sorted(modules):
        module_name, _, qualname = carrier.rpartition(".")
        obj = getattr(importlib.import_module(module_name), qualname)
        for finding in lint_instrument(obj):
            findings.append(f"  {carrier}: {finding.render()}")
    assert not findings, (
        "instruments in the tree fail the instrument lint. An instrument that cannot pass "
        "`lint_instrument` does not exist:\n" + "\n".join(findings)
    )


def test_the_pinned_counts_in_the_catalogue_header_match_the_rows(walk):
    """Every pinned count moves in the same edit as the rows it counts.

    E10 and E14 are both this, and E58's own scale paragraph is a third: it gives 63 rows in closed
    packages against a file that has 69, because 63 was derived by subtraction from the header's
    stale 89 rather than counted.
    """
    source = pathlib.Path(__file__).resolve().parents[2] / "spec" / "CATALOGUE.yaml"
    if not source.exists():
        pytest.skip("not a source checkout, so the YAML header is not on disk")
    # The leading comment block only. Every row in the file carries its own provenance comments, and
    # sweeping those in would let a sentence four thousand lines down satisfy a count pinned here.
    lines: list[str] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.startswith("#"):
            break
        lines.append(line)
    header = "\n".join(lines)
    rows = _rows()
    assert f"{len(rows)} instrument records" in header, (
        f"the header does not pin {len(rows)} instrument records. Recount it in the same edit as "
        f"the rows."
    )
    wedge = sum(1 for row in rows if row.get("wedge") is True)
    assert f"{wedge} carry the wedge marker" in header, (
        f"the header does not pin {wedge} wedge rows."
    )
    # Each status keyword opens a block of the header's status table and the block runs until the
    # next one. Scoping the search to the block is what stops "5 rows" satisfying two statuses at
    # once, which a search over the whole header would allow and which would make this vacuous.
    for status in ("built", "planned", "OPEN"):
        n = sum(1 for row in rows if row.get("status") == status)
        start = next(i for i, line in enumerate(lines) if line.startswith(f"#   {status} "))
        end = next(
            (
                i
                for i, line in enumerate(lines[start + 1 :], start + 1)
                if line.rstrip() == "#" or (line.startswith("#   ") and line[4:5].strip())
            ),
            len(lines),
        )
        block = "\n".join(lines[start:end])
        assert f"{n} rows" in block, (
            f"the header's `{status}` block does not pin {n} rows. Recount it in the same edit as "
            f"the rows it counts:\n{block}"
        )

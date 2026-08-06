"""Render the catalogue, the quantity registry and the refusal reference from the code.

Every page this module emits is a transcription of something the library already holds, so a
page cannot say the catalogue has 89 instruments while the catalogue has 92. Three sources, in
order of authority:

- ``reward_lens.core.quantity``: ``QUANTITIES`` after ``load_quantities()``, and ``ladder()`` for
  the estimators registered against each one. This is the live registry, not a file.
- ``reward_lens.access.report.load_instrument_catalogue()``: the loader the capability report
  itself uses, which normalises the catalogue's polymorphic fields (six instruments store
  ``quantities`` as the bare string ``OPEN``, and iterating that yields four single-character ids
  that resolve to nothing).
- ``spec/CATALOGUE.json``, reached through ``reward_lens.core.quantity.catalogue_path``, for the
  narrative fields the loader has no reason to carry: the headline, the ladder rungs, the
  baselines, the kill condition, and what a reading would say. Same file the loader reads, through
  the library's own path resolution, and ``_crosscheck`` asserts the two agree on every field they
  both hold. A disagreement fails the build naming the field, because a catalogue page that
  disagrees with the loader is worse than no catalogue page.

**Lint rule two lives here.** A `Quantity` with no estimator fails the docs build with a message
naming it as an open research target rather than a bug. It is enforced as a ratchet, the same
shape as ``docs/claims-baseline.txt``: ``LEDGER_PATH``
records the quantities that are open today, the build fails on any open quantity that is not in
it, and the list may shrink and must not grow. A ratchet rather than a hard failure because most
of the registered quantities are specified-and-not-built, so a hard failure would make the docs
permanently red, and a permanently red gate is one people learn to ignore.

The ledger is written from the leanest environment the docs can build in, base dependencies only.
That direction matters: a richer environment imports more estimator modules, so more quantities
are closed, so the observed-open set can only shrink. Generating it from a fat environment would
make the gate fire spuriously on any thinner one.

Run it directly:

    python docs/gen_catalogue.py --check          # fail if the ledger is stale or has grown
    python docs/gen_catalogue.py --write-ledger   # re-record the ledger, then exit 0
    python docs/gen_catalogue.py --out DIR        # write the pages to disk, for inspection
"""

from __future__ import annotations

import argparse
import importlib
import json
import pkgutil
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

DOCS_DIR = Path(__file__).resolve().parent
LEDGER_PATH = DOCS_DIR / "open-quantities.txt"

#: The catalogue rows carry only the series letter, so the titles live here, trimmed from the
#: series headings. Series N has no heading of its own: the frontier and the contract layer are
#: developed together, so N1 to N4 take their title from the section that develops them.
SERIES: dict[str, str] = {
    "A": "Signal metrology: the grader as a measurement device",
    "B": "Grader structure: is this thing a scalar, and what is it made of?",
    "C": "Grader white-box: the existing battery, re-typed",
    "D": "Verifier and environment science",
    "E": "The estimator: where a good reward becomes a bad gradient",
    "F": "The four books",
    "G": "Credit geometry",
    "H": "Rate and regime",
    "I": "Adversarial pressure, and the monitor as a target",
    "J": "Monitoring",
    "K": "Transfer and survival",
    "L": "Reference materials and label metrology",
    "M": "Meta-instruments: measurements about measurements",
    "N": "The frontier: what would happen if we optimised",
}


class LintFailure(RuntimeError):
    """A docs-build lint failure. Carries the message the build prints and stops on."""


# ---------------------------------------------------------------------------
# Reading the library
# ---------------------------------------------------------------------------


@dataclass
class RegistryView:
    """One snapshot of what the library knows, taken once per build."""

    quantities: dict[str, Any]
    ladders: dict[str, list[Any]]
    open_quantities: list[str]
    instruments: tuple[Any, ...]
    rows: dict[str, dict[str, Any]]
    source: str
    skipped_modules: list[tuple[str, str]]
    n_estimators: int

    @property
    def n_instruments(self) -> int:
        return len(self.instruments)

    @property
    def n_wedge(self) -> int:
        return sum(1 for i in self.instruments if i.wedge)

    @property
    def n_wedge_quantities(self) -> int:
        return sum(1 for q in self.quantities.values() if q.wedge)


def _is_optional_extra(exc: BaseException) -> bool:
    """Whether an import failure is a missing optional dependency rather than a defect.

    Two shapes count as expected. `ExtraRequiredError` is the library saying so itself. A
    `ModuleNotFoundError` naming something outside `reward_lens` is a third-party package the docs
    environment deliberately does not install, torch above all.

    Everything else is a defect in the tree: a `NameError` from a missing import, a `SyntaxError`,
    a `ModuleNotFoundError` naming a `reward_lens` module that should exist. Those are reported as
    what they are rather than folded in with the extras, because a broken module makes its
    quantities look unbuilt, and diagnosing that as an open research target would send somebody
    looking in exactly the wrong place.
    """
    if type(exc).__name__ == "ExtraRequiredError":
        return True
    name = getattr(exc, "name", None)
    return isinstance(exc, ModuleNotFoundError) and not str(name or "").startswith("reward_lens")


def _import_estimator_modules() -> list[tuple[str, str]]:
    """Import every submodule so its ``_register()`` runs, and report what would not import.

    Estimators register at module import: ``verifier/coverage.py`` ends in a bare ``_register()``
    call. Nothing registers them centrally, so a walk is the only way to see the whole ladder. A
    module that needs an extra the docs build does not have is skipped and named, never swallowed:
    its quantities then read as open, and a reader who is told which extras were missing can tell
    the difference between "nobody has built this" and "this build could not see it".

    `walk_packages` imports each package itself in order to recurse into it, and that import is
    outside any `try` a caller can write, so `onerror` is not optional here. Without it one
    half-written module anywhere in the tree takes the whole documentation build down with a
    traceback that says nothing about documentation.
    """
    import reward_lens

    skipped: list[tuple[str, str]] = []
    broken: list[str] = []

    def _record(name: str, exc: BaseException) -> None:
        detail = f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"
        if _is_optional_extra(exc):
            skipped.append((name, detail))
        else:
            broken.append(f"{name}: {detail}")

    def _on_walk_error(name: str) -> None:
        exc = sys.exc_info()[1]
        if exc is not None:
            _record(name, exc)

    for mod in pkgutil.walk_packages(reward_lens.__path__, "reward_lens.", _on_walk_error):
        try:
            importlib.import_module(mod.name)
        except Exception as exc:
            _record(mod.name, exc)

    if broken:
        raise LintFailure(
            "these modules do not import, and it is not a missing optional extra:\n  "
            + "\n  ".join(sorted(set(broken)))
            + "\nEstimators register when their module is imported, so a module that will not "
            "import makes every quantity it estimates look like an open research target. Fix the "
            "import; the catalogue cannot report on a tree that does not load."
        )
    return skipped


def _crosscheck(instruments: Iterable[Any], rows: dict[str, dict[str, Any]]) -> None:
    """Assert the catalogue loader and the catalogue file agree on every shared field."""
    problems: list[str] = []
    loader_ids = {i.id for i in instruments}
    file_ids = set(rows)
    if loader_ids != file_ids:
        problems.append(
            f"the loader sees {sorted(loader_ids - file_ids)} that the file does not, and the "
            f"file has {sorted(file_ids - loader_ids)} that the loader drops"
        )
    for inst in instruments:
        row = rows.get(inst.id)
        if row is None:
            continue
        if tuple(_as_list(row.get("quantities"))) != inst.quantities:
            problems.append(f"{inst.id}: quantities differ between the loader and the file")
        if bool(row.get("wedge", False)) != inst.wedge:
            problems.append(f"{inst.id}: wedge differs between the loader and the file")
        for field, members in (("substrates", inst.substrates), ("phases", inst.phases)):
            declared = _as_list(row.get(field))
            if len(declared) != len(members):
                dropped = sorted(set(n.upper() for n in declared) - {m.name for m in members})
                problems.append(
                    f"{inst.id}: {field} {dropped} in the catalogue file resolve to no enum "
                    f"member, so the loader silently drops them"
                )
    if problems:
        raise LintFailure(
            "the catalogue loader and spec/CATALOGUE.json disagree:\n  "
            + "\n  ".join(problems)
            + "\nA catalogue page that disagrees with the loader is worse than no catalogue page."
        )


def read_registry() -> RegistryView:
    """Take one snapshot of the registry, the catalogue loader and the catalogue file."""
    try:
        from reward_lens.access.report import load_instrument_catalogue
        from reward_lens.core.quantity import (
            ESTIMATORS,
            QUANTITIES,
            catalogue_path,
            ladder,
            load_quantities,
            open_quantities,
        )
    except ImportError as exc:  # pragma: no cover - the message is the whole point
        raise LintFailure(
            f"the docs build cannot import reward_lens ({exc}). The catalogue, the quantity "
            f"registry and the refusal reference are rendered from the code rather than "
            f"transcribed, so this build has nothing to render. Install the package into the "
            f"docs environment: pip install -e ."
        ) from exc

    report = load_quantities()
    skipped = _import_estimator_modules()
    instruments = load_instrument_catalogue()

    path = catalogue_path("CATALOGUE.json")
    if path is None:
        raise LintFailure("spec/CATALOGUE.json is not on any search path the library knows.")
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = {str(r["id"]): r for r in doc.get("instruments", [])}
    _crosscheck(instruments, rows)

    quantities = dict(QUANTITIES.items())
    return RegistryView(
        quantities=quantities,
        ladders={qid: ladder(qid) for qid in quantities},
        open_quantities=open_quantities(),
        instruments=instruments,
        rows=rows,
        source=report.source,
        skipped_modules=skipped,
        n_estimators=len(ESTIMATORS),
    )


# ---------------------------------------------------------------------------
# Lint rule two
# ---------------------------------------------------------------------------


def load_ledger(path: Path = LEDGER_PATH) -> set[str]:
    if not path.exists():
        return set()
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }


#: The preamble a ledger gets when there is no file to take one from.
DEFAULT_LEDGER_HEADER = """\
# Quantities registered with an estimator ladder nobody has built yet.
# Lint rule two: a quantity with no estimator is an open research target, and the docs
# build says so by name rather than treating it as a bug. This list may shrink and must
# not grow. Register an estimator and delete its line; docs/gen_catalogue.py --check
# enforces it, and --write-ledger rewrites it.
#
# Recorded from a base install, which is the leanest environment the docs build in. A
# richer environment imports more estimator modules and so sees fewer open quantities,
# which the ratchet allows.
"""


def _existing_header(path: Path) -> str | None:
    """The comment block at the top of an existing ledger, or None if there is no file.

    The header is hand-maintained: it carries the argument for why the current entries are open,
    which is the only part of this file a reader learns anything from. An earlier `--write-ledger`
    rewrote it from a constant, so regenerating the ledger silently deleted whatever the last
    wave had written there. Keeping it means the command is safe to run at any point in a wave,
    which is the point of having a command at all.
    """
    if not path.exists():
        return None
    lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("#"):
            break
        lines.append(line)
    return "\n".join(lines) + "\n" if lines else None


def write_ledger(view: RegistryView, path: Path = LEDGER_PATH) -> int:
    ids = sorted(view.open_quantities)
    header = _existing_header(path) or DEFAULT_LEDGER_HEADER
    path.write_text(header + "\n".join(ids) + "\n", encoding="utf-8")
    return len(ids)


def check_ledger(view: RegistryView, ledger: set[str]) -> list[str]:
    """Lint rule two. Returns the lines the build should print; raises when the ratchet grew."""
    observed = set(view.open_quantities)
    new = sorted(observed - ledger)
    closed = sorted(ledger - observed)
    if new:
        listed = "\n  ".join(f"{qid}: {_would_close(view, qid)}" for qid in new)
        raise LintFailure(
            f"{len(new)} registered quantit{'y has' if len(new) == 1 else 'ies have'} no "
            f"estimator and no entry in {LEDGER_PATH.name}:\n  {listed}\n"
            f"This is lint rule two. A quantity with no estimator is an open "
            f"research target rather than a bug, and the docs build names it as one. If that is "
            f"what this is, record it: python docs/gen_catalogue.py --write-ledger. If it is not, "
            f"register the estimator."
        )
    out = [f"lint rule two: {len(observed)} open quantities, all recorded."]
    if closed:
        out.append(
            f"  {len(closed)} ledger entries now have an estimator and can be deleted: "
            f"{', '.join(closed[:8])}{' ...' if len(closed) > 8 else ''}"
        )
    return out


def _would_close(view: RegistryView, qid: str) -> str:
    """What would close an open quantity, for the failure message and the roadmap page."""
    owners = [i for i in view.instruments if qid in i.quantities]
    if not owners:
        return "no catalogue instrument claims it"
    inst = owners[0]
    return f"{inst.id} {inst.name}"


# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------

_OPEN = ("OPEN", "open", "", None)


def _as_list(value: Any) -> list[str]:
    """A list field, with the catalogue's bare ``OPEN`` normalised to nothing (E14)."""
    if value is None or isinstance(value, str):
        return []
    if isinstance(value, dict):
        return [str(k) for k in value]
    return [str(v) for v in value]


def _text(value: Any) -> str:
    """A prose field, with ``OPEN`` rendered as the absence it is."""
    if value in _OPEN:
        return ""
    return str(value).strip()


def _cell(value: Any) -> str:
    """One markdown table cell: pipes escaped, newlines flattened."""
    s = _text(value)
    if not s:
        return "not stated"
    return " ".join(s.replace("|", "\\|").split())


def _joined(values: Iterable[str]) -> str:
    items = sorted(values)
    return ", ".join(f"`{v}`" for v in items) if items else "not stated"


def _ordered(members: Iterable[Any]) -> str:
    """Enum members in declaration order, which for `Phase` is the order a run passes through."""
    items = list(members)
    if not items:
        return "not stated"
    order = list(type(items[0]))
    return ", ".join(f"`{m.name}`" for m in sorted(items, key=order.index))


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out.extend("| " + " | ".join(r) + " |" for r in rows)
    return out


# ---------------------------------------------------------------------------
# The catalogue pages
# ---------------------------------------------------------------------------


def _instrument_section(view: RegistryView, inst: Any) -> list[str]:
    row = view.rows[inst.id]
    out = [f"### {inst.id}. {inst.name} {{ #{_slug(inst.id)} }}", ""]

    headline = _text(row.get("headline"))
    if headline:
        out += [headline, ""]

    facts = [
        ["Quantity", _joined(inst.quantities)],
        ["Access", _cell(row.get("access_min"))],
        ["Substrates", _joined(s.name for s in inst.substrates)],
        ["Phases", _ordered(inst.phases)],
        ["Invariance group", _cell(row.get("invariance_group"))],
        ["Wedge", "yes" if inst.wedge else "no"],
        ["Status", _text(row.get("status")) or "not scheduled"],
    ]
    requires = _as_list(row.get("envelope_requires"))
    measured_by = row.get("envelope_measured_by")
    if requires:
        env = _joined(requires)
        if isinstance(measured_by, dict) and measured_by:
            env += ", measured by " + ", ".join(
                f"`{k}` through {v}" for k, v in sorted(measured_by.items())
            )
        facts.insert(4, ["Envelope", env])
    else:
        justification = _text(row.get("envelope_unconditional_justification"))
        facts.insert(
            4,
            ["Envelope", "unconditional. " + justification if justification else "unconditional"],
        )
    out += _table(["", ""], facts) + [""]

    rungs = _as_list(row.get("ladder"))
    ladder_rows = row.get("ladder") if isinstance(row.get("ladder"), list) else []
    if ladder_rows:
        out += ["**The ladder.**", ""]
        out += _table(
            ["Rung", "Estimator", "Access", "Bias", "Cost"],
            [
                [
                    str(r.get("rung", "")),
                    _cell(r.get("estimator")),
                    _cell(r.get("access")),
                    _cell(r.get("bias")),
                    _cell(r.get("cost")),
                ]
                for r in ladder_rows
            ],
        )
        out += [""]
    del rungs

    baselines = _as_list(row.get("baselines"))
    if len(baselines) == 1:
        out += ["**Baselines.** " + baselines[0], ""]
    elif baselines:
        out += ["**Baselines.**", ""] + [f"- {b}" for b in baselines] + [""]

    kill = _text(row.get("kill_condition"))
    if kill:
        out += ["**Kill condition.** " + kill[0].upper() + kill[1:], ""]

    says = _text(row.get("says"))
    if says:
        out += ["**What a reading would say.** " + says, ""]

    built = sorted({e.impl for qid in inst.quantities for e in view.ladders.get(qid, [])})
    if built:
        out += ["**Registered estimators.** " + ", ".join(f"`{b}`" for b in built), ""]
    else:
        out += [
            "**Registered estimators.** None yet. Every quantity above is an "
            "[open research target](open.md).",
            "",
        ]
    return out


_CATALOGUE_PREAMBLE = """The catalogue is the registry's own rows, rendered from the code that
reads them rather than retyped. Each entry names one instrument, the quantity or quantities it
estimates, the access it needs, the substrates and phases it applies to, the envelope conditions
under which its answer means anything, the invariance group it declares, the baselines a claim from
it must be reported against, and the condition under which the instrument gets deleted.

Two of those fields deserve a word before you read a page of them.

**The kill condition** is what would make the instrument not worth having. Writing one down before
building is the cheapest discipline in the catalogue and it is the field most often missing
elsewhere. When you read "if r0 and r3 agree within their intervals on five graders, the ladder is
decoration and only r0 ships", that is the author saying in advance which result would retire the
work.

**What a reading would say** is a specimen sentence, not a measurement. It is there so you can tell
at a glance whether an instrument answers a question you have. The numbers in those sentences are
written to show the shape of the answer and none of them came from a run."""


def _series_page(view: RegistryView, letter: str, title: str) -> str:
    members = [i for i in view.instruments if i.id[0] == letter]
    members.sort(key=lambda i: (len(i.id), i.id))
    wedge = sum(1 for i in members if i.wedge)
    out = [
        f"# Series {letter}. {title}",
        "",
        f"{len(members)} instruments, {wedge} of them in the wedge. The wedge is the subset the "
        f"library builds first, chosen because those instruments answer a question somebody is "
        f"already asking with a worse tool.",
        "",
    ]
    out += _table(
        ["Instrument", "Quantity", "Access", "Wedge"],
        [
            [
                f"[{i.id}](#{_slug(i.id)}) {i.name}",
                _joined(i.quantities),
                _cell(view.rows[i.id].get("access_min")),
                "yes" if i.wedge else "",
            ]
            for i in members
        ],
    )
    out += [""]
    for inst in members:
        out += _instrument_section(view, inst)
    return "\n".join(out) + "\n"


def _catalogue_index(view: RegistryView) -> str:
    counts = {letter: sum(1 for i in view.instruments if i.id[0] == letter) for letter in SERIES}
    out = [
        "# The instrument catalogue",
        "",
        f"{view.n_instruments} instruments in {len(SERIES)} series, {view.n_wedge} of them in the "
        f"wedge. Every field on every page below is read at build time from "
        f"`spec/CATALOGUE.json` through the same loader the capability report uses, so this "
        f"catalogue and the library cannot disagree about what is in it.",
        "",
        _CATALOGUE_PREAMBLE,
        "",
        "## The series",
        "",
    ]
    out += _table(
        ["Series", "What it measures", "Instruments", "Wedge"],
        [
            [
                f"[{letter}](series-{letter.lower()}.md)",
                title,
                str(counts[letter]),
                str(sum(1 for i in view.instruments if i.id[0] == letter and i.wedge)),
            ]
            for letter, title in SERIES.items()
        ],
    )
    out += [
        "",
        "## What is catalogued and what is built",
        "",
        f"A catalogue entry is a specification, not an implementation, and the gap between the "
        f"two is the honest measure of where this library is. {len(view.quantities)} quantities "
        f"are registered from `{Path(view.source).name}` and "
        f"{view.n_wedge_quantities} of them are in the wedge. Of the whole registry, "
        f"{len(view.quantities) - len(view.open_quantities)} have an estimator this build could "
        f"see and load. Every one of the rest is named on the "
        f"[open research targets](open.md) page rather than quietly omitted, which is the whole "
        f"of lint rule two.",
        "",
        "The [quantity registry](quantities.md) is the other axis on the same data: one row per "
        "quantity, with its unit, its invariance group, the access its cheapest rung needs, and "
        "how many rungs its ladder has.",
        "",
    ]
    return "\n".join(out) + "\n"


def _quantities_page(view: RegistryView) -> str:
    by_quantity: dict[str, list[str]] = {}
    for inst in view.instruments:
        for qid in inst.quantities:
            by_quantity.setdefault(qid, []).append(inst.id)

    out = [
        "# The quantity registry",
        "",
        f"{len(view.quantities)} quantities, {view.n_wedge_quantities} of them in the wedge. A "
        f"quantity is what you want to know; an estimator is one way to get it, at a stated "
        f"access level, with a stated bias, at a stated cost. Keeping them apart is what lets a "
        f"closed lab get a real answer at rung 0 with an honest bias direction instead of a "
        f"refusal, and it is what makes two labs' numbers comparable when they used different "
        f"methods.",
        "",
        "`Estimators` counts the registered estimators visible to this build. A zero there means "
        "the quantity is specified and unbuilt, and every one of them is named on the "
        "[open research targets](open.md) page.",
        "",
        "The unit column is the three axes of `Unit`: what is counted, what it is counted over, "
        "and the convention it is expressed in. `OPEN` on an axis is a decomposition nobody has "
        "settled, and a unit with an undecided axis is incomparable with everything including "
        "another undecided one, which is deliberate.",
        "",
    ]
    rows = []
    for qid in sorted(view.quantities):
        q = view.quantities[qid]
        estimators = view.ladders.get(qid, [])
        rows.append(
            [
                f"`{qid}`",
                f"`{q.unit}`" if str(q.unit) else "not stated",
                f"`{q.invariance}`",
                ", ".join(by_quantity.get(qid, [])) or "none",
                "yes" if q.wedge else "",
                str(len(estimators)),
            ]
        )
    out += _table(
        ["Quantity", "Unit", "Invariance group", "Instrument", "Wedge", "Estimators"], rows
    )
    out += [""]
    return "\n".join(out) + "\n"


#: The extra name inside an `ExtraRequiredError` detail line. The message is written for a human
#: ("needs the optional 'white-box' extra, which is not installed") and this is the one machine-
#: readable thing in it. Worth pulling out rather than guessing: the obvious guess is the second
#: component of the module path, which yields `attribution`, `concepts`, `measure` and `policy`,
#: none of which is an extra anybody can install, and a page that tells a reader to install
#: `pip install reward-lens[measure]` has sent them somewhere that does not exist.
_EXTRA_RE = re.compile(r"optional '([A-Za-z0-9_.-]+)' extra")


def _extras_named(skipped: Iterable[tuple[str, str]]) -> list[str]:
    """The optional extras named by the import failures, deduplicated and sorted."""
    found = {m.group(1) for _, detail in skipped if (m := _EXTRA_RE.search(detail))}
    return sorted(found)


def _open_page(view: RegistryView, ledger: set[str]) -> str:
    open_ids = sorted(view.open_quantities)
    out = [
        "# Open research targets",
        "",
        f"{len(open_ids)} of {len(view.quantities)} registered quantities have no estimator this "
        f"build could resolve. They are listed here by name because that is what lint rule two "
        f"asks for: a quantity with no estimator is an open research target rather than a bug, and a "
        f"docs build that quietly omitted it would make the library look finished.",
        "",
        "This page is generated, and the docs build fails if a quantity turns up open without an "
        "entry in `docs/open-quantities.txt`. The list may shrink and must not grow.",
        "",
        "**Read the list with its three shapes in mind, because they are not the same problem.**",
        "",
        "The first is the one the page is named for: nobody has built an estimator, and often no "
        "instrument in the catalogue is even aimed at the quantity. That is a research target.",
        "",
        "The second is an environment artifact. An estimator registers when its module is "
        "imported, so a quantity whose only estimator lives behind an optional extra reads as open "
        "in a build that does not have that extra. This page is rendered from the leanest "
        "environment the documentation builds in, which makes the list as long as it can honestly "
        "be; a richer environment sees strictly fewer.",
        "",
        "The third is a bookkeeping gap and it is the one worth knowing about. Several quantities "
        "here have a **shipped instrument that computes them** and no registered `EstimatorEntry`, "
        "which is the ladder. The reading exists; what does not exist is the row that tells "
        "`best_estimator` which rung it is, what access it needs and what it costs, so the "
        "capability report cannot cost it and the ladder cannot resolve it. The right column below "
        "names the instrument when there is one, which is how you tell this case from the first.",
        "",
    ]

    if view.skipped_modules:
        extras = _extras_named(view.skipped_modules)
        subpackages = sorted({m.split(".")[1] for m, _ in view.skipped_modules})
        if extras:
            named = (
                "the "
                + ", ".join(f"`{e}`" for e in extras)
                + (" extra" if len(extras) == 1 else " extras")
            )
        else:
            named = "optional extras a documentation build does not install"
        out += [
            '!!! note "What this build could not see"',
            "",
            f"    {len(view.skipped_modules)} modules would not import in the environment that "
            f"built this page, because they need {named}. They are under "
            f"{', '.join('`reward_lens.' + s + '`' for s in subpackages)}. Estimators register "
            f"when their module is imported, so a quantity whose only estimator lives in one of "
            f"those modules is listed below as open when it is in fact built. Install the extras "
            f"and run `python docs/gen_catalogue.py --check` to see the shorter list.",
            "",
        ]

    out += [
        "## What would close each one",
        "",
        "The right column is the catalogue instrument that claims the quantity. A quantity no "
        "instrument claims is the more interesting case: it means the registry knows a thing is "
        "worth measuring and nothing in the catalogue is aimed at it.",
        "",
    ]
    out += _table(
        ["Quantity", "Unit", "What would close it"],
        [
            [f"`{qid}`", f"`{view.quantities[qid].unit}`", _would_close(view, qid)]
            for qid in open_ids
        ],
    )
    out += [""]

    unclaimed = [q for q in open_ids if not any(q in i.quantities for i in view.instruments)]
    if unclaimed:
        out += [
            f"{len(unclaimed)} of those are claimed by no catalogue instrument at all: "
            + ", ".join(f"`{q}`" for q in unclaimed[:20])
            + ("." if len(unclaimed) <= 20 else ", and others in the table above."),
            "",
        ]
    del ledger
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# The refusal reference
# ---------------------------------------------------------------------------
#
# `REASON_MEANING` in reward_lens/core/reading.py says what each reason means. It does not say what
# to do, and what to do is the thing a user holding a refusal actually wants. The instruction lives
# here, keyed by the same enum, and `_refusals_page` fails the build if the two key sets differ.
# That is deliberate, and it has already earned itself twice: `RECORD_INCOMPLETE` and
# `QUANTITY_UNDEFINED` both landed as amendments, and on each the docs build stopped until somebody
# wrote down what a user holding one should do about it. Nothing about the count is hardcoded below;
# the page counts the enum.

WHAT_TO_DO: dict[str, str] = {
    "ACCESS_INSUFFICIENT": (
        "Ask what the cheaper rung would cost. Every quantity has a ladder, and the refusal "
        "carries the rung that would work and what it needs, so `what_would_it_take` turns this "
        "into a shopping list: one more checkpoint, or logprobs on the sampling policy, or "
        "permission to call the grader twice on the same input. If none of that is available, the "
        "quantity is out of reach on this run and the honest move is to say so in the write-up "
        "rather than substitute something adjacent."
    ),
    "QUANTITY_UNDEFINED": (
        "Nothing to fix and nothing to buy: this question does not apply to this object, so no "
        "amount of access and no rewriting of the record will produce an answer. The commonest "
        "case is asking a mean-centred estimator for its amplification, when the amplification is "
        "a property of dividing by the group standard deviation and this estimator does not do "
        "that. Read the remedy, which names the question that **does** apply to what you have, and "
        "ask that one instead. If the named alternative is not the question you wanted, the honest "
        "conclusion is that the thing you wanted to know is not a property of this object, which "
        "is worth writing down rather than working around."
    ),
    "RECORD_INCOMPLETE": (
        "Do not go looking for more access; it will not help. The field was never written, so "
        "nothing you do to this record recovers it. Fix it upstream: turn on the dump in whatever "
        "produced the run, or record the missing field on the next run. The refusal names both "
        'the field and what it is missing from, so "no `logprobs_sampling` on 412 of 512 '
        'trajectories" tells you whether this is a configuration problem or a partial write.'
    ),
    "SUBSTRATE_MISMATCH": (
        "Nothing to fix. You asked a question that does not apply to this kind of grader, most "
        "often an activation question of a program. Reach for the instrument built for this "
        "substrate: the capability report lists them, and for a program the verifier series "
        "answers the structural questions the white-box series answers for a network."
    ),
    "PHASE_MISMATCH": (
        "Either the run is over and you asked an in-run question, or it has not started and you "
        "asked a post-run one. If the run is over, the question has to be answered from the "
        "record, and the instrument that does that is a different one. If it has not started, "
        "this is the moment to record what the in-run instrument will need, because a phase you "
        "have passed cannot be revisited."
    ),
    "ENVELOPE_VIOLATED": (
        "Read which condition failed and what its statistic was; the refusal carries both. Then "
        "pick one of three: restrict the analysis to a window where the condition holds, switch "
        "to a rung whose envelope does not require it, or accept that the quantity is not "
        "estimable on this run. An instrument that is available and invalid is worse than one "
        "that is unavailable, so the option that is not on the list is running it anyway."
    ),
    "BELOW_LOD": (
        "The effect is smaller than the measurement's disagreement with itself, so there is "
        "nothing here to interpret in either direction. This is not a negative result and must "
        "not be written up as one. To go further you need a smaller limit of detection: more "
        "replicates, a lower-variance readout, or a stimulus set that separates the conditions "
        "more sharply. The refusal carries the limit, so you can compute how much more."
    ),
    "ABOVE_LOD_BELOW_LOQ": (
        "Use the bound. It is real, it is in `partial`, and an upper bound is a usable answer for "
        "most decisions a point estimate would have been used for. Report it as a bound and say "
        "so. If the decision genuinely needs the point estimate, the limit of quantification "
        "tells you how much more data would get you there."
    ),
    "ESS_BELOW_FLOOR": (
        "The importance weights have degenerated, so you are extrapolating past the point where "
        "the data constrains anything. Shorten the extrapolation: ask for the quantity at a "
        "smaller distance from the sampling policy, where the effective sample size is still "
        "above the floor. Widening the interval instead is the mistake this refusal exists to "
        "prevent, because past the horizon the interval is not wide, it is undefined."
    ),
    "NO_MATCHED_CONTROL": (
        "You asked for a null and there is no positive control at the same power, so a real "
        "absence and an underpowered experiment look identical. Run the matched positive control "
        "from `stats/baselines`: it is a case where the effect is known to exist, at the same n "
        "and the same readout. If the control also comes back null, the experiment is "
        "underpowered and the number of samples that would fix it is a power calculation away."
    ),
    "GAUGE_MISMATCH": (
        "You compared a covariant quantity across two frames with no shared basis, which makes "
        "the difference a coordinate artifact rather than a finding. Fit a shared frame and "
        "compare in it, or compare an invariant of the two quantities instead. Comparing raw "
        "coordinates across models is the specific error this gate exists to catch, and it "
        "produces numbers that look reasonable, which is why it needs a gate rather than a "
        "warning."
    ),
    "UNIT_MISMATCH": (
        "Two quantities in incompatible units met, most often per-token against per-sequence. "
        "Decide which unit the question is actually in and get both sides into it, which needs "
        "data the comparison did not have: how many tokens, or how many sequences. The library "
        "will not do that conversion for you, because the factor is a property of your data "
        "rather than of the unit."
    ),
    "REFERENCE_UNCERTIFIED": (
        "The reference material you calibrated against has no uncertainty of its own, so the "
        "calibrated number would inherit an error nobody has measured. Either certify the "
        "reference, which means measuring its own uncertainty and recording it, or use a "
        "certified one. If neither is possible, report the reading as uncalibrated and let it sit "
        "at the trust level that implies."
    ),
    "LABEL_QUALITY_UNKNOWN": (
        "Scoring against labels whose error rate nobody measured measures the labels. Get a "
        "measured error rate: a doubly-labelled subset is usually enough, and the label metrology "
        "series exists to turn one into an error rate with an interval. Until then any score "
        "against these labels is bounded above by their quality and the bound is unknown."
    ),
    "PLAN_NOT_CLOSED": (
        "A registered prediction names a metric that no arc in this plan produces, so the "
        "prediction could never have been graded. Either add the arc that produces the metric or "
        "change the prediction to one the plan answers. This fires before anything runs, which is "
        "the only useful time for it to fire."
    ),
    "BUDGET_EXCEEDED": (
        "The costed plan is more expensive than the budget you declared. Cut the plan, raise the "
        "budget deliberately, or drop to a cheaper rung and accept its bias, which the ladder "
        "states. The one thing not to do is run it and find out, because a plan that runs out of "
        "budget half way produces an arc nobody can interpret."
    ),
    "VOID": (
        "The run is not readable, which is a different thing from a negative result and must "
        "never be written up as one. Find out why: a truncated write, a missing manifest, a "
        "schema the reader does not recognise. A void run contributes nothing in either "
        "direction, and counting it as evidence of no effect is how a broken pipeline becomes a "
        "published null."
    ),
}


_REFUSAL_PREAMBLE = """Every instrument in this library returns a `Reading`, and a `Reading` is
either `Evidence` or a `Refusal`. A refusal is a value, not an exception. It is never a `None`,
never a zero, and never a silent fall back to a worse estimator, because a confident wrong number
is the only output this design treats as unforgivable.

So a refusal is not a failure to be worked around. It is the measurement coming back and telling
you that the number you asked for would not have meant anything, along with the numbers that led it
to that conclusion and an instruction for what to do next.

Here is the shape of one. The numbers in it are invented; the four parts are not.

```
A1  ENVELOPE_VIOLATED
    GROUP_NONDEGENERATE fails: 14 of 16 rollouts in this group scored identically
    Remedy: Widen the sampling temperature, or read the group-size reading as a bound.
```

Those four parts are always there. The instrument, so you know who is speaking. The reason, which is
one of the ones listed below. The detail, which carries the statistics that produced the refusal so
you can audit it rather than take its word. And the remedy, which a `Refusal` cannot be constructed
without, because a refusal with no remedy is a tool that looks broken instead of a tool that looks
careful.

A fifth part is often there and is easy to miss. `Refusal.partial` carries a bound when one is
available, and `is_bounded` tells you when you have one. A refusal that says it cannot give you the
effective group size but can bound it above is still an answer, and if the decision in front of you
can be made from a bound, it has already been made."""


def _refusals_page() -> str:
    from reward_lens.core.reading import REASON_MEANING, RefusalReason

    declared = {r.name for r in RefusalReason}
    written = set(WHAT_TO_DO)
    meanings = {r.name for r in REASON_MEANING}
    if declared != meanings:
        raise LintFailure(
            f"RefusalReason and REASON_MEANING disagree: "
            f"{sorted(declared ^ meanings)} appears in one and not the other. Every reason a user "
            f"can be handed needs a sentence written for someone holding it."
        )
    if declared != written:
        missing = sorted(declared - written)
        extra = sorted(written - declared)
        raise LintFailure(
            f"the refusal reference does not cover every reason. Undocumented: {missing}. "
            f"Documented but not declared: {extra}. Add the instruction to WHAT_TO_DO in "
            f"docs/gen_catalogue.py. A reason a user can be handed with no page telling them what "
            f"to do about it is a dead end with a name."
        )

    order = list(RefusalReason)
    out = [
        "# Every refusal, and what to do about it",
        "",
        _REFUSAL_PREAMBLE,
        "",
        f"There are {len(order)} reasons. This page is generated from the enum and from "
        "`REASON_MEANING`, so it cannot fall behind the code.",
        "",
        "## The reasons at a glance",
        "",
    ]
    out += _table(
        ["Reason", "What happened"],
        [[f"[`{r.name}`](#{_slug(r.name)})", " ".join(REASON_MEANING[r].split())] for r in order],
    )
    out += [
        "",
        "## The one question that sorts them: where is the remedy answerable?",
        "",
        "A list this long is only navigable if it has an organising idea, and this one does. The "
        "three refusals that look most alike send you in three different directions, and the test "
        "that separates them is a single question about the remedy rather than anything about the "
        "reason's name.",
        "",
    ]
    out += _table(
        ["Reason", "The remedy", "Answerable"],
        [
            [
                "[`ACCESS_INSUFFICIENT`](#access-insufficient)",
                "get more access, or drop a rung and accept its stated bias",
                "**where you are standing**",
            ],
            [
                "[`RECORD_INCOMPLETE`](#record-incomplete)",
                "write the field and run again",
                "**upstream**, where the record was produced",
            ],
            [
                "[`QUANTITY_UNDEFINED`](#quantity-undefined)",
                "ask the question that does apply",
                "**nowhere**",
            ],
        ],
    )
    out += [
        "",
        'A remedy that says "get more access" when the honest answer is "your framework does not '
        'dump this" costs somebody an afternoon and then still does not work. That is why the '
        "first two are two reasons and not one.",
        "",
        "The third is the one worth sitting with, because nothing fixes it. `QUANTITY_UNDEFINED` "
        "means the question does not apply to the object in front of you. An estimator that does "
        "not z-score has no amplification to measure, and no access and no rewriting of the record "
        "will give it one. So the only useful sentence available is the name of the question that "
        "*does* apply, which is why `refuse_undefined` takes `instead` as a **required** argument "
        "rather than an optional courtesy: making it optional would let the refusal degrade to "
        '"no" with no forward path.',
        "",
        "It is also not `SUBSTRATE_MISMATCH`, which is about the grader's kind (a program has no "
        "activations) rather than the estimator's. The two live on different axes and an "
        "instrument can be refused on either.",
        "",
        "The line between the second and the third is genuinely close in places, and one site was "
        "argued and deliberately left where it was. `record/tensors.py` refuses a compacted "
        "trajectory's importance ratio with `RECORD_INCOMPLETE`, not `QUANTITY_UNDEFINED`, on the "
        "reasoning that a run recorded *without* that compaction does carry the tensor. The "
        "quantity is not undefined for the object; it is missing from this particular record, and "
        "the remedy really is answerable upstream. When you are deciding between these two, that "
        "is the question to ask: would a differently recorded run of the same thing have had it?",
        "",
        "## Every reason, one at a time",
        "",
    ]
    for reason in order:
        out += [
            f"### `{reason.name}` {{ #{_slug(reason.name)} }}",
            "",
            "**What happened.** " + " ".join(REASON_MEANING[reason].split()),
            "",
            "**What to do.** " + " ".join(WHAT_TO_DO[reason.name].split()),
            "",
        ]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# The build
# ---------------------------------------------------------------------------


def build_pages(view: RegistryView, ledger: set[str]) -> dict[str, str]:
    """Every generated page, keyed by its path under the docs directory."""
    pages = {
        "catalogue/index.md": _catalogue_index(view),
        "catalogue/quantities.md": _quantities_page(view),
        "catalogue/open.md": _open_page(view, ledger),
        "refusals.md": _refusals_page(),
    }
    for letter, title in SERIES.items():
        pages[f"catalogue/series-{letter.lower()}.md"] = _series_page(view, letter, title)
    return pages


def generate(*, check: bool = True) -> tuple[dict[str, str], list[str]]:
    """Read the registry, run lint rule two, and render. Raises `LintFailure` on either."""
    view = read_registry()
    ledger = load_ledger()
    notes = check_ledger(view, ledger) if check else []
    return build_pages(view, ledger), notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gen_catalogue", description=__doc__)
    parser.add_argument(
        "--write-ledger",
        action="store_true",
        help="re-record docs/open-quantities.txt from the registry as it is now, and exit 0",
    )
    parser.add_argument("--check", action="store_true", help="run lint rule two and render")
    parser.add_argument("--out", type=Path, default=None, help="write the pages to this directory")
    args = parser.parse_args(argv)

    try:
        if args.write_ledger:
            view = read_registry()
            n = write_ledger(view)
            print(f"recorded {n} open quantities in {LEDGER_PATH}")
            return 0
        pages, notes = generate(check=True)
    except LintFailure as exc:
        print(f"docs lint failed: {exc}", file=sys.stderr)
        return 1

    for note in notes:
        print(note)
    print(f"rendered {len(pages)} pages, {sum(len(v) for v in pages.values()):,} characters")
    if args.out:
        for uri, text in pages.items():
            target = args.out / uri
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        print(f"wrote to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

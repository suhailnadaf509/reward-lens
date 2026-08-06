"""Load the estimator ladder before anyone asks what can be measured.

Estimators register at module import. `verifier/coverage.py` ends in a bare `_register()` call and
`policy/quantities.py` ends in `register_estimators()`; nothing registers them centrally. So
`ESTIMATORS` is empty in a process that has only done `import reward_lens`, and it stays empty
after `import reward_lens.measure`, because a package's `__init__` does not pull its leaves.

That matters here more than anywhere else in the library. The capability report decides what to put
under `SPECIFIED, NOT YET BUILT` by asking which quantities have a registered estimator: a catalogue
row whose quantities are all unregistered is reported as an instrument nobody has written yet. In a
process where nothing has imported the leaves, that is every row in the catalogue, and the report
tells a reader that a library of ninety-five instruments contains none.

The documentation build already had this exactly right, in `docs/gen_catalogue.py`, whose own
docstring says a broken module "makes its quantities look unbuilt, and diagnosing that as an open
research target would send somebody looking in exactly the wrong place". It said so about modules
that fail to import. The same sentence is true, and was unhandled, for modules that were simply
never imported. `docs/` does not ship in the wheel, so the fix lived where no user could reach it.

Three properties this has to have, and each is here because leaving it out produces a worse failure
than the one being fixed:

- **A missing optional extra is not a missing instrument.** A module behind `[white-box]` or
  `[dict]` raises `ExtraRequiredError` on a base install. Its quantities genuinely cannot be
  estimated in that process, so reporting them as unavailable is correct, but the reason is the
  extra and not the absence of an implementation. Those are collected and named.
- **A module that is actually broken is not silently folded in with the extras.** A `NameError`, a
  `SyntaxError`, or a `ModuleNotFoundError` naming a `reward_lens` module is a defect in the tree,
  and swallowing it would present a broken instrument as an unbuilt one.
- **`walk_packages` imports each package itself in order to recurse into it**, and that import
  happens outside any `try` a caller can write, so `onerror` is not optional. Without it, one
  half-written module anywhere in the tree takes down the cheapest command the library offers.

The walk is done once per process and the result is cached, because the capability report is the
command a user runs repeatedly while changing arguments and importing the tree twice buys nothing.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
from dataclasses import dataclass, field

__all__ = ["LadderLoad", "load_estimator_ladder"]


@dataclass(frozen=True)
class LadderLoad:
    """What a walk of the tree found, so a caller can say why a quantity is missing.

    `skipped` is the honest half of a capability report on a base install: those modules exist and
    their instruments are written, and this process cannot see them because an extra is not
    installed. A reader told which extras were missing can tell the difference between "nobody has
    built this" and "this install cannot reach it", which is the same distinction the refusal
    vocabulary draws between a quantity that is undefined and one that is out of reach.
    """

    registered: int
    skipped: tuple[tuple[str, str], ...] = ()
    broken: tuple[str, ...] = ()
    walked: int = 0
    extras_wanted: frozenset[str] = field(default_factory=frozenset)

    @property
    def ok(self) -> bool:
        """True when nothing in the tree failed for a reason other than a missing extra."""
        return not self.broken


_CACHE: LadderLoad | None = None


def _is_optional_extra(exc: BaseException) -> bool:
    """Distinguish "this install lacks an extra" from "this module is broken".

    `ExtraRequiredError` is the library's own typed error and names an installable extra, so it is
    unambiguous. A bare `ModuleNotFoundError` is only an extra if the module it names belongs to
    somebody else: one naming a `reward_lens` module is a defect in this tree and is reported as
    one, because presenting it as a missing dependency would send a reader to `pip` for a package
    that does not exist.
    """
    if type(exc).__name__ == "ExtraRequiredError":
        return True
    name = getattr(exc, "name", None)
    return isinstance(exc, ModuleNotFoundError) and not str(name or "").startswith("reward_lens")


def _extra_named(exc: BaseException) -> str | None:
    """Pull the extra's name off the library's typed error, where it is a declared field."""
    for attr in ("extra", "extra_name"):
        value = getattr(exc, attr, None)
        if isinstance(value, str) and value:
            return value
    return None


def load_estimator_ladder(*, force: bool = False) -> LadderLoad:
    """Import every `reward_lens` submodule so its estimator registrations run.

    Returns what the walk found rather than raising, because the capability report's whole job is
    to answer honestly under partial access and a base install is a legitimate place to run it. A
    caller that needs the stricter contract reads `LadderLoad.ok` and the `broken` list; the
    documentation build does exactly that and fails on it, which is the right behaviour there and
    the wrong one here.

    Idempotent and cached. `force=True` re-walks, which is only useful to a test that has
    registered something itself.
    """
    global _CACHE
    if _CACHE is not None and not force:
        return _CACHE

    import reward_lens
    from reward_lens.core.quantity import ESTIMATORS

    skipped: list[tuple[str, str]] = []
    broken: list[str] = []
    extras: set[str] = set()
    walked = 0

    def _record(name: str, exc: BaseException) -> None:
        detail = f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"
        if _is_optional_extra(exc):
            skipped.append((name, detail))
            extra = _extra_named(exc)
            if extra:
                extras.add(extra)
        else:
            broken.append(f"{name}: {detail}")

    def _on_walk_error(name: str) -> None:
        exc = sys.exc_info()[1]
        if exc is not None:
            _record(name, exc)

    for mod in pkgutil.walk_packages(reward_lens.__path__, "reward_lens.", _on_walk_error):
        walked += 1
        try:
            importlib.import_module(mod.name)
        except Exception as exc:  # noqa: BLE001 - classified by _record, never swallowed
            _record(mod.name, exc)

    _CACHE = LadderLoad(
        registered=len(ESTIMATORS),
        skipped=tuple(sorted(set(skipped))),
        broken=tuple(sorted(set(broken))),
        walked=walked,
        extras_wanted=frozenset(extras),
    )
    return _CACHE

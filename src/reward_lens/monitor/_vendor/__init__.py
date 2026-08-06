"""Third-party source vendored into this package, unmodified, with its licence attached.

One file so far, `cif.py`, from `AsiaeeLab/certified-interventional-fidelity` under MIT. It is a
module inside a research repository rather than a package: no `pyproject.toml`, no release, no name
on any index. There is nothing to put in a dependency list, so it is copied in with its provenance
and its licence in the file header, and a test re-hashes the vendored body against the sha256 of
what was fetched.

The rule for anything that lands here: **do not edit it.** Everything this library needs on top of
a vendored file lives outside it, so the hash check stays meaningful and an upstream update is a
re-copy rather than a merge.
"""

from __future__ import annotations

__all__: list[str] = []

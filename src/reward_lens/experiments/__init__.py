"""``reward_lens.experiments`` — analyses that are studies rather than instruments.

An instrument measures a quantity on a subject and lives under the series it belongs to. What lives
here is the other thing: a self-contained piece of confirmatory work with a frozen spec, a
registered prediction, a mandatory baseline, a kill criterion and a power calculation, whose output
is a section of a write-up rather than a reading a caller composes with.

Each module here is expected to be runnable as ``python -m reward_lens.experiments.<name>`` and to
produce the same numbers every time from the same inputs, so a claim in a write-up can be traced to
a command. Nothing here is imported by the library's runtime paths.

Torch-free.
"""

from __future__ import annotations

__all__: list[str] = []

"""Versioned builtin datasets that ship in the package wheel.

These are the small, human-authored (and, where marked, mechanically authored) seed sets that seed
the data plane: the v1 diagnostic triples imported with honest lineage, plus the two new dimensions
authored for v3. Importing this package registers their dataset cards so ``registry.load_dataset``
can serve them by name. Everything here is torch-free and cheap to import.

``diagnostic_seeds`` holds the 65 hand-written triples themselves. It was
``reward_lens.diagnostic_data_v2`` in v1, where it sat at the package root beside the v1 analysis
modules and got counted as one of them. It never was: it is a dataset, its only consumer is
``diagnostic_v3`` here, and this is where the datasets live.
"""

from __future__ import annotations

from reward_lens.data.builtin.diagnostic_seeds import (
    ALL_DIMENSIONS_V2,
    PreferencePair,
    get_pairs_by_dim_v2,
    get_pairs_v2,
)
from reward_lens.data.builtin.diagnostic_v3 import (
    ALL_DIMENSIONS_V3,
    all_pairs,
    load_diagnostic_v3,
    matched_prompt_views,
)

__all__ = [
    "ALL_DIMENSIONS_V3",
    "load_diagnostic_v3",
    "all_pairs",
    "matched_prompt_views",
    # the seed corpus v3 is built from, addressable in its own right
    "ALL_DIMENSIONS_V2",
    "PreferencePair",
    "get_pairs_v2",
    "get_pairs_by_dim_v2",
]

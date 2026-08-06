"""Stimulus lineage and effective sample size.

This is the module that makes v1's worst failure class structurally impossible. In v1 the
"n = 30 pairs/dimension" behind every confidence interval was five to seven hand-written stimuli
expanded by prompt-prefix mutations (and, in the bias battery, padded with byte-identical
duplicates); the bootstrap then resampled those manufactured rows and reported an interval as if
they were independent observations. The E19 repetition headline traced back to a single stimulus.

The fix is to make effective sample size a *tracked property of the data* rather than a claim in
prose. Every item carries a `Lineage`: the seed it descends from, the builder that produced it, and
the explicit list of mutation operations applied. A builder that mutates a seed keeps the seed id
and records the op, so the statistics engine can resample at the seed level and refuse to inflate
n across clones. Exact-duplicate content is detected at ingest and collapsed to weighted items.

The canonical effective-sample-size implementation lives in `reward_lens.stats.ess`
(Kish size over seed-label multiplicities). This module calls it through a lazy import, so
importing the data plane never pulls the stats engine in, and it falls back to the identical Kish
formula locally if that import fails, so a `DataView` is never non-functional. `stats.ess` is
authoritative wherever it is importable, and the fallback computes the same quantity, so no number
changes either way.
"""

from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Sequence, TypeVar

from reward_lens.core import content_hash

T = TypeVar("T")


@dataclass(frozen=True)
class Lineage:
    """The provenance of a single stimulus.

    ``seed_id`` names the hand-authored (or externally sourced) seed this item descends from;
    resampling happens at this level by default, so clones of one seed count as one. ``builder_id``
    names the builder that produced the item. ``ops`` is the explicit, ordered list of mutation
    operations applied to the seed to reach this item (empty for an unmutated seed). ``content_hash``
    is the content-derived id of the item's payload, which is what clone detection groups on.
    """

    seed_id: str
    builder_id: str
    ops: tuple[str, ...]
    content_hash: str

    def with_op(self, op: str, content: Any) -> "Lineage":
        """Return a new lineage with ``op`` appended and the content hash recomputed.

        A builder that mutates an item calls this: the seed id is preserved (so effective n is
        unchanged by the mutation), the op is recorded, and the content hash is refreshed to the
        mutated payload so clone detection and the dataset checksum see the new content.
        """
        return make_lineage(self.seed_id, self.builder_id, (*self.ops, op), content)

    def __canonical__(self) -> dict[str, Any]:
        return {
            "seed_id": self.seed_id,
            "builder_id": self.builder_id,
            "ops": list(self.ops),
            "content_hash": self.content_hash,
        }


def make_lineage(
    seed_id: str,
    builder_id: str,
    ops: Sequence[str],
    content: Any,
) -> Lineage:
    """Build a `Lineage`, computing the content hash from ``content``.

    ``content`` is any canonicalizable representation of the item's payload (the schema builders
    pass the same canonical content tuple used for the dataset checksum, so the hash here and the
    checksum agree). The hash carries the ``ch`` prefix ("content hash").
    """
    return Lineage(
        seed_id=seed_id,
        builder_id=builder_id,
        ops=tuple(ops),
        content_hash=content_hash(content, "ch"),
    )


def _default_content_key(item: Any) -> str:
    """Group key for clone detection: the item's lineage content hash.

    Every schema item carries a ``lineage`` whose ``content_hash`` is derived from the item's
    payload (never from the seed id or ops), so two items with identical content share a key even
    if their nominal seeds differ. That is the correct notion for ingest-time deduplication: the
    v1 bias battery's byte-identical padding must collapse regardless of how it was labelled.
    """
    lineage = getattr(item, "lineage", None)
    if lineage is None or not isinstance(lineage, Lineage):
        raise TypeError(
            f"cannot deduplicate {type(item).__name__}: it carries no Lineage; "
            "pass an explicit key= to collapse_duplicates for lineage-free items"
        )
    return lineage.content_hash


def collapse_duplicates(
    items: Sequence[T],
    key: Callable[[T], str] | None = None,
    *,
    warn: bool = True,
) -> tuple[list[T], list[int]]:
    """Detect exact-duplicate content and collapse it to weighted unique items.

    Returns ``(unique_items, weights)`` where ``unique_items`` preserves first-occurrence order and
    ``weights[i]`` is the number of raw items that collapsed onto ``unique_items[i]``. When any
    duplicates are found and ``warn`` is set, a warning names the collapse so an inflated dataset
    cannot pass through ingest silently. ``key`` defaults to the item's lineage content hash.

    The weights are what downstream statistics use to keep a collapsed dataset honest: a weight of
    forty means one unique stimulus, not forty independent observations.
    """
    key_fn = key or _default_content_key
    seen: dict[str, int] = {}
    unique: list[T] = []
    weights: list[int] = []
    for item in items:
        k = key_fn(item)
        if k in seen:
            weights[seen[k]] += 1
        else:
            seen[k] = len(unique)
            unique.append(item)
            weights.append(1)
    n_collapsed = len(items) - len(unique)
    if n_collapsed and warn:
        warnings.warn(
            f"collapse_duplicates: {n_collapsed} exact-duplicate item(s) collapsed onto "
            f"{len(unique)} unique item(s); resampling will use content weights, not the raw count",
            stacklevel=2,
        )
    return unique, weights


def _kish_effective_size(seed_labels: Sequence[str]) -> float:
    """Kish effective sample size over seed-label multiplicities (the local fallback).

    ``n_eff = (sum of counts)^2 / sum(counts^2)`` where counts are the per-seed multiplicities.
    N clones of one seed give 1.0; N distinct seeds give N. This is the same quantity
    `reward_lens.stats.ess.effective_sample_size` computes; it is duplicated here only so the data
    plane is usable before the stats engine exists, and the two agree by construction.
    """
    labels = list(seed_labels)
    if not labels:
        return 0.0
    counts = Counter(labels)
    total = float(sum(counts.values()))
    sum_sq = float(sum(c * c for c in counts.values()))
    if sum_sq == 0.0:
        return 0.0
    return (total * total) / sum_sq


def effective_sample_size(seed_labels: Sequence[str]) -> float:
    """Effective sample size from seed labels, preferring the canonical stats implementation.

    Calls `reward_lens.stats.ess.effective_sample_size` (the authoritative Kish-over-seed-labels
    implementation) through a lazy import, so importing this module never pulls the stats engine
    in. If that import fails, it falls back to the identical local computation. The lazy import is
    deliberate: the data plane must not import-fail on account of a subsystem it borrows one
    formula from.
    """
    try:
        from reward_lens.stats.ess import effective_sample_size as canonical

        return float(canonical(list(seed_labels)))
    except ImportError:
        return _kish_effective_size(seed_labels)


__all__ = [
    "Lineage",
    "make_lineage",
    "collapse_duplicates",
    "effective_sample_size",
]

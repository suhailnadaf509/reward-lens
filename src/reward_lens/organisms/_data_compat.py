"""Data-plane compatibility shim for the organism foundry.

The foundry produces `Pair` and `Tournament` objects wrapped in a `DataView`, with a `Lineage` on
every item. Those types are owned by the data plane (`reward_lens.data.schema` /
`reward_lens.data.lineage`). This module prefers importing the real types and, only if that import
fails, falls back to a minimal locally-defined compatible set with the same field names.

This module is that single point of indirection: every other file in `organisms` imports its data
types from here, so if the real schema is present we use it verbatim, and if it is momentarily
absent the foundry still builds and its pure tests still pass. ``USING_SHIM`` records which path was
taken so a caller can flag it. The shim mirrors the real field names exactly
(`Pair.prompt/chosen/rejected/axis/lineage/meta`, `Response.text/spans/meta`,
`Lineage.seed_id/builder_id/ops/content_hash`) so swapping back to the real types changes nothing
downstream.
"""

from __future__ import annotations

from typing import Any

from reward_lens.core import content_hash

USING_SHIM: bool
_SHIM_REASON: str | None = None

try:  # Prefer the real data plane; this is the expected path.
    from reward_lens.data.lineage import Lineage, make_lineage
    from reward_lens.data.schema import (
        DataView,
        EdgeObs,
        Pair,
        Response,
        Tournament,
        response_content,
    )

    USING_SHIM = False
except Exception as _exc:  # pragma: no cover - exercised only when the data plane is absent
    _SHIM_REASON = f"{type(_exc).__name__}: {_exc}"
    USING_SHIM = True

    from collections.abc import Callable, Iterator, Sequence
    from dataclasses import dataclass, field

    @dataclass(frozen=True)
    class Response:  # type: ignore[no-redef]
        """Minimal stand-in for `reward_lens.data.schema.Response` (same field names)."""

        text: str
        spans: tuple[Any, ...] = ()
        meta: dict[str, Any] = field(default_factory=dict)

    @dataclass(frozen=True)
    class Lineage:  # type: ignore[no-redef]
        """Minimal stand-in for `reward_lens.data.lineage.Lineage` (same field names)."""

        seed_id: str
        builder_id: str
        ops: tuple[str, ...]
        content_hash: str

        def __canonical__(self) -> dict[str, Any]:
            return {
                "seed_id": self.seed_id,
                "builder_id": self.builder_id,
                "ops": list(self.ops),
                "content_hash": self.content_hash,
            }

    def make_lineage(  # type: ignore[no-redef]
        seed_id: str, builder_id: str, ops: Sequence[str], content: Any
    ) -> Lineage:
        return Lineage(
            seed_id=seed_id,
            builder_id=builder_id,
            ops=tuple(ops),
            content_hash=content_hash(content, "ch"),
        )

    def response_content(r: "Response") -> Any:  # type: ignore[no-redef]
        return ["Response", r.text, []]

    @dataclass(frozen=True)
    class Pair:  # type: ignore[no-redef]
        """Minimal stand-in for `reward_lens.data.schema.Pair` (same field names)."""

        prompt: Any
        chosen: Response
        rejected: Response
        axis: str
        lineage: Lineage
        meta: dict[str, Any] = field(default_factory=dict)

        @property
        def prompt_text(self) -> str:
            return self.prompt if isinstance(self.prompt, str) else self.prompt.text

        @property
        def seed_id(self) -> str:
            return self.lineage.seed_id

    @dataclass(frozen=True)
    class EdgeObs:  # type: ignore[no-redef]
        """Minimal stand-in for `reward_lens.data.schema.EdgeObs` (same field names)."""

        i: int
        j: int
        wins_i: int
        wins_j: int
        annotator_id: str | None = None
        judge_id: str | None = None

        def __canonical__(self) -> dict[str, Any]:
            return {
                "i": self.i,
                "j": self.j,
                "wins_i": self.wins_i,
                "wins_j": self.wins_j,
                "annotator_id": self.annotator_id,
                "judge_id": self.judge_id,
            }

    @dataclass(frozen=True)
    class Tournament:  # type: ignore[no-redef]
        """Minimal stand-in for `reward_lens.data.schema.Tournament` (same field names)."""

        prompt: Any
        responses: tuple[Response, ...]
        edges: tuple[EdgeObs, ...]
        lineage: Lineage
        meta: dict[str, Any] = field(default_factory=dict)

        @property
        def prompt_text(self) -> str:
            return self.prompt if isinstance(self.prompt, str) else self.prompt.text

        @property
        def seed_id(self) -> str:
            return self.lineage.seed_id

    class DataView:  # type: ignore[no-redef]
        """Minimal stand-in for `reward_lens.data.schema.DataView` (same API surface used here)."""

        __slots__ = ("_items", "name")

        def __init__(self, items: Sequence[Any], *, name: str | None = None) -> None:
            self._items = tuple(items)
            self.name = name

        def __len__(self) -> int:
            return len(self._items)

        def __iter__(self) -> Iterator[Any]:
            return iter(self._items)

        def __getitem__(self, index: Any) -> Any:
            if isinstance(index, slice):
                return DataView(self._items[index], name=self.name)
            return self._items[index]

        def __repr__(self) -> str:
            label = f" {self.name!r}" if self.name else ""
            return f"<DataView{label} n={len(self._items)}>"

        @property
        def items(self) -> tuple[Any, ...]:
            return self._items

        def filter(self, predicate: "Callable[[Any], bool]") -> "DataView":
            return DataView([it for it in self._items if predicate(it)], name=self.name)

        def seed_ids(self) -> list[str]:
            return [it.lineage.seed_id for it in self._items]

        def effective_n(self) -> float:
            labels = self.seed_ids()
            if not labels:
                return 0.0
            from collections import Counter

            counts = Counter(labels)
            total = float(sum(counts.values()))
            sum_sq = float(sum(c * c for c in counts.values()))
            return (total * total) / sum_sq if sum_sq else 0.0

        def checksum(self) -> str:
            return content_hash([getattr(it, "lineage").content_hash for it in self._items], "ds")


def pair_content(prompt_text: str, chosen: Response, rejected: Response, axis: str) -> Any:
    """The canonical content tuple of a `Pair`, mirroring `schema.content_of` exactly.

    Used by the foundry to stamp a pair's `Lineage` *before* the frozen `Pair` is constructed, so the
    lineage content hash equals ``content_hash(content_of(pair), "ch")`` and the dataset checksum and
    clone detection agree. Mirrors the schema's structure
    ``["Pair", prompt, response_content(chosen), response_content(rejected), axis]``.
    """
    return ["Pair", prompt_text, response_content(chosen), response_content(rejected), axis]


def tournament_content(
    prompt_text: str, responses: tuple[Response, ...], edges: tuple[Any, ...]
) -> Any:
    """The canonical content tuple of a `Tournament`, mirroring `schema.content_of` exactly."""
    return [
        "Tournament",
        prompt_text,
        [response_content(r) for r in responses],
        [e.__canonical__() for e in edges],
    ]


__all__ = [
    "USING_SHIM",
    "Pair",
    "Response",
    "Lineage",
    "make_lineage",
    "EdgeObs",
    "Tournament",
    "DataView",
    "response_content",
    "pair_content",
    "tournament_content",
]

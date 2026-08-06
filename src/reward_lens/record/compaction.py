"""Compaction: where the prefix was rewritten, and what that does to every ratio.

Long-horizon agents run out of context and the standard fix is to rewrite the prefix: summarise
the first forty turns into one, drop the tool output nobody referenced again, re-render the
history. After that, `pi_old(y_t | prefix)` is not defined on the same prefix, because the prefix
is a different string. The importance ratio is therefore **undefined, not merely stale**, and any
per-token importance-weighted quantity crossing the boundary is not a number.

That is a stronger claim than staleness and it needs a stronger response. A stale ratio is a
biased estimate you can bound; an undefined one is not an estimate. So `NO_COMPACTION` is one of
the twelve regime conditions and its violation behaviour is `refuse`, and `CompactionEvent` is a
first-class record rather than metadata, because metadata is what gets dropped by the third
converter in a chain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class CompactionEvent:
    """After this, the importance ratio is UNDEFINED, not merely stale.

    ``at_turn`` is the index of the first turn generated against the rewritten prefix. Turns
    before it were generated against the old prefix, turns from it onward against the new one, and
    a window containing the boundary spans two different conditioning contexts.

    ``tokens_before`` and ``tokens_after`` are the prefix lengths on either side. They are recorded
    because the compression ratio is the only cheap signal of how much conditioning context was
    destroyed, and because a "compaction" that removed nothing is a recorded event that changed
    nothing and should be visible as such rather than triggering a refusal.
    """

    at_turn: int
    tokens_before: int
    tokens_after: int
    method: str
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.at_turn < 0:
            raise ValueError(f"compaction at_turn cannot be negative; got {self.at_turn}")
        if self.tokens_before < 0 or self.tokens_after < 0:
            raise ValueError(
                f"compaction token counts cannot be negative; got before={self.tokens_before}, "
                f"after={self.tokens_after}"
            )
        if not self.method.strip():
            raise ValueError(
                "a CompactionEvent must name its method. 'summarise', 'drop_tool_output', "
                "'rerender' and 'truncate_head' destroy different things, and an instrument that "
                "wants to know whether a receipt survived cannot ask an unnamed method."
            )

    @property
    def tokens_removed(self) -> int:
        return self.tokens_before - self.tokens_after

    @property
    def is_lossless(self) -> bool:
        """A recorded compaction that removed nothing. Rare, and worth not refusing over."""
        return self.tokens_removed <= 0

    @property
    def compression_ratio(self) -> float | None:
        """``tokens_after / tokens_before``, or None when the prefix was empty."""
        if self.tokens_before <= 0:
            return None
        return self.tokens_after / self.tokens_before

    def __canonical__(self) -> dict[str, Any]:
        return {
            "at_turn": self.at_turn,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "method": self.method,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_canonical(cls, obj: Mapping[str, Any]) -> "CompactionEvent":
        return cls(
            at_turn=obj["at_turn"],
            tokens_before=obj["tokens_before"],
            tokens_after=obj["tokens_after"],
            method=obj["method"],
            extra=dict(obj.get("extra", {})),
        )


def crosses(events: Iterable[CompactionEvent], *, first_turn: int, last_turn: int) -> bool:
    """Whether any lossy compaction lands strictly inside the turn window ``[first, last]``.

    An event at ``first_turn`` itself does not cross the window: everything in the window was then
    generated against the same, post-rewrite prefix. The boundary case is the one that matters,
    because a window pinned to a compaction boundary is exactly the analysis that stays valid, and
    refusing it would push users toward ignoring the flag.
    """
    return any(first_turn < e.at_turn <= last_turn and not e.is_lossless for e in events)


def events_in_window(
    events: Iterable[CompactionEvent], *, first_turn: int, last_turn: int
) -> tuple[CompactionEvent, ...]:
    """The compactions inside a turn window, in turn order."""
    return tuple(
        sorted(
            (e for e in events if first_turn < e.at_turn <= last_turn),
            key=lambda e: e.at_turn,
        )
    )


@dataclass(frozen=True)
class CompactionSummary:
    """What `NO_COMPACTION` is measured from. Counts and totals, no verdict.

    ``n_lossy`` is the number the condition turns on. ``n_events`` includes the lossless ones so
    the difference between "no compaction happened" and "compaction happened and removed nothing"
    survives into the reading.
    """

    n_events: int
    n_lossy: int
    tokens_removed: int
    methods: tuple[str, ...]

    def __canonical__(self) -> dict[str, Any]:
        return {
            "n_events": self.n_events,
            "n_lossy": self.n_lossy,
            "tokens_removed": self.tokens_removed,
            "methods": list(self.methods),
        }


def summarise(events: Iterable[CompactionEvent]) -> CompactionSummary:
    """Summarise compaction events for the regime reading."""
    items = list(events)
    return CompactionSummary(
        n_events=len(items),
        n_lossy=sum(1 for e in items if not e.is_lossless),
        tokens_removed=sum(max(0, e.tokens_removed) for e in items),
        methods=tuple(sorted({e.method for e in items})),
    )


def check_ordered(events: Sequence[CompactionEvent], n_turns: int, *, where: str = "") -> None:
    """Raise unless every event sits inside the trajectory and no two share a turn.

    Two compactions at the same turn are not a hard case, they are a recording bug: the prefix was
    rewritten once and something double-counted it, or twice and the intermediate state was never
    recorded. Either way the ``tokens_before``/``tokens_after`` chain no longer composes, and a
    chain that does not compose is worse than a missing one.
    """
    prefix = f"{where}: " if where else ""
    seen: set[int] = set()
    for e in events:
        if e.at_turn > n_turns:
            raise ValueError(
                f"{prefix}compaction at turn {e.at_turn} on a trajectory with {n_turns} turn(s). "
                f"at_turn is the index of the first turn generated against the rewritten prefix, "
                f"so it may equal n_turns (a rewrite after the last turn) but not exceed it."
            )
        if e.at_turn in seen:
            raise ValueError(
                f"{prefix}two compaction events at turn {e.at_turn}. The token counts on either "
                f"side no longer compose into a single prefix history; record the intermediate "
                f"state or merge them."
            )
        seen.add(e.at_turn)


__all__ = [
    "CompactionEvent",
    "CompactionSummary",
    "check_ordered",
    "crosses",
    "events_in_window",
    "summarise",
]

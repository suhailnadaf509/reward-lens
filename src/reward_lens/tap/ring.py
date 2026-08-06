"""The bounded buffer the tap writes into, so the hot path never does blocking I/O.

Plane A is allowed to reduce a value and append it to a ring buffer. It is not allowed to write a
file, take a lock it might wait on, or let a queue grow until the host runs out of memory. That
leaves exactly one design: a fixed-capacity buffer whose write either succeeds immediately or
fails immediately, and a drain that somebody else's thread runs later.

**The drop count is the whole point.** A buffer that silently discards records produces a record
with holes in it that nobody can see, and a measurement taken from a record with invisible holes is
the failure mode this library exists to prevent. So every write that does not land is counted, by
cause, and the counts travel with the data rather than into a log nobody reads. A run whose ring
dropped a third of its records is still readable; a run whose ring dropped a third of its records
without saying so is not.

Two causes, kept separate because they have different fixes. ``dropped_full`` means the drain is
not keeping up, and the fix is to drain more often or raise the capacity. ``dropped_contended``
means two threads reached the ring at the same instant and the loser refused to wait, and the fix
is a ring per producer. Folding them into one number would hide which one you have.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from reward_lens.tap.contract import RECORD_FOOTPRINT_BYTES, GraderCall


@dataclass(frozen=True)
class RingStats:
    """What the ring did, as numbers that travel with the record.

    ``high_water`` is the largest occupancy the ring ever reached. It is the number that says
    whether a capacity was comfortable or lucky: a ring that peaked at 4,090 of 4,096 did not drop
    anything and is one slow drain away from dropping a lot.
    """

    capacity: int
    size: int
    accepted: int = 0
    dropped_full: int = 0
    dropped_contended: int = 0
    dropped_closed: int = 0
    drained: int = 0
    high_water: int = 0
    #: Bytes the ring's own containers hold. Not the host objects the records point at.
    resident_bytes: int = 0
    #: Host objects the ring is currently keeping alive by holding a reference to them.
    retained_refs: int = 0

    @property
    def dropped(self) -> int:
        return self.dropped_full + self.dropped_contended + self.dropped_closed

    @property
    def offered(self) -> int:
        return self.accepted + self.dropped

    @property
    def drop_rate(self) -> float:
        """The fraction of records that never made it. Zero is the only comfortable value."""
        return self.dropped / self.offered if self.offered else 0.0

    def render(self) -> str:
        parts = [
            f"{self.size}/{self.capacity} held",
            f"{self.accepted} accepted",
            f"{self.drained} drained",
            f"high water {self.high_water}",
        ]
        if self.dropped:
            parts.append(
                f"DROPPED {self.dropped} ({self.drop_rate:.2%}): {self.dropped_full} full, "
                f"{self.dropped_contended} contended, {self.dropped_closed} closed"
            )
        else:
            parts.append("0 dropped")
        return ", ".join(parts)

    def __canonical__(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "size": self.size,
            "accepted": self.accepted,
            "dropped_full": self.dropped_full,
            "dropped_contended": self.dropped_contended,
            "dropped_closed": self.dropped_closed,
            "drained": self.drained,
            "high_water": self.high_water,
            "resident_bytes": self.resident_bytes,
            "retained_refs": self.retained_refs,
        }


class TapRing:
    """A fixed-capacity, non-blocking, drop-and-count buffer.

    **It drops the newest, not the oldest.** A ring that evicts its oldest entry to make room keeps
    the most recent history, which sounds right and is wrong here for two reasons. Eviction has to
    touch an entry the drain may be reading, which is the one place a lock-free design gets hard;
    and refusing the write leaves the buffer holding a contiguous run of records with a counted gap
    after it, which is analysable, where eviction leaves a gap in the middle that is not. The
    alternative is defensible and this one is a decision rather than an oversight.

    **The lock is never waited on.** ``acquire(blocking=False)`` costs about 51 ns uncontended on
    the reference machine and returns immediately when it cannot be had. A producer that loses the
    race counts a contended drop and returns; it does not wait, because waiting inside somebody
    else's training step is the behaviour that ends an adoption.

    Capacity may be given directly or derived from a byte budget with ``for_bytes``, which is how
    ``TapBudget.max_resident_bytes`` becomes something the ring can enforce by construction rather
    than by checking after the fact.
    """

    __slots__ = (
        "_capacity",
        "_items",
        "_lock",
        "_accepted",
        "_dropped_full",
        "_dropped_contended",
        "_dropped_closed",
        "_drained",
        "_high_water",
        "_closed",
    )

    def __init__(self, capacity: int = 8192) -> None:
        if capacity < 1:
            raise ValueError(
                f"a ring needs at least one slot; got capacity {capacity}. A zero-capacity ring "
                f"drops everything, which is a switched-off tap wearing a buffer."
            )
        self._capacity = int(capacity)
        self._items: list[GraderCall] = []
        self._lock = threading.Lock()
        self._accepted = 0
        self._dropped_full = 0
        self._dropped_contended = 0
        self._dropped_closed = 0
        self._drained = 0
        self._high_water = 0
        self._closed = False

    @classmethod
    def for_bytes(
        cls,
        max_resident_bytes: int,
        *,
        record_bytes: int = RECORD_FOOTPRINT_BYTES,
        reserve: int = 0,
    ):
        """A ring sized so the whole tap's containers cannot exceed a byte budget.

        ``record_bytes`` is the measured footprint of one record's own storage, not of the host
        objects it points at. Sizing on anything else would be sizing on memory the tap did not
        allocate and cannot release.

        ``reserve`` is what the rest of the tap holds outside the ring, which in practice is the
        guard's two fixed latency windows. Leaving it out is the mistake that makes a ring sized
        from a budget breach that same budget on its first full step, and it is exactly the kind of
        off-by-a-constant that a byte ceiling exists to catch. ``TapGuard.fixed_bytes`` supplies
        the number; ``tap`` and ``instrument_grader`` pass it.
        """
        usable = int(max_resident_bytes) - int(reserve)
        capacity = max(1, usable // max(1, int(record_bytes)))
        return cls(capacity=capacity)

    # -- the hot path ------------------------------------------------------

    def offer(self, record: GraderCall) -> bool:
        """Append if there is room and the lock is free. Never blocks, never raises, never grows.

        Returns True when the record landed. The return value exists so the caller can count its
        own losses; the ring counts them too, and the two should agree.
        """
        if not self._lock.acquire(False):
            self._dropped_contended += 1
            return False
        try:
            if self._closed:
                self._dropped_closed += 1
                return False
            n = len(self._items)
            if n >= self._capacity:
                self._dropped_full += 1
                return False
            self._items.append(record)
            self._accepted += 1
            if n + 1 > self._high_water:
                self._high_water = n + 1
            return True
        finally:
            self._lock.release()

    # -- the drain, which is somebody else's thread -------------------------

    def drain(self, max_items: int | None = None) -> list[GraderCall]:
        """Take records out, oldest first, and release the references they hold.

        This is not hot-path code: it waits for the lock, because a drain that gave up would leave
        the ring to fill and start dropping, and the drain has no host to protect. What it must not
        do is any I/O while holding the lock, so it swaps the buffer out and returns; writing the
        records to a store is the caller's business and happens after this returns.
        """
        with self._lock:
            if max_items is None or max_items >= len(self._items):
                out = self._items
                self._items = []
            else:
                out = self._items[:max_items]
                del self._items[:max_items]
            self._drained += len(out)
        return out

    def drain_to(self, sink: Callable[[GraderCall], Any], max_items: int | None = None) -> int:
        """Drain and hand each record to ``sink``, returning how many were handed over.

        The sink runs outside the lock. An exception from the sink propagates: this is the drain
        thread, not the hot path, and a store that cannot be written to is a real failure that
        should be loud. The records already taken are lost in that case, which is why a sink that
        can fail should buffer rather than raise.
        """
        records = self.drain(max_items)
        for r in records:
            sink(r)
        return len(records)

    def extend_from(self, records: Iterable[GraderCall]) -> int:
        """Push records back, for a drain that failed after taking them. Counts drops as usual."""
        landed = 0
        for r in records:
            if self.offer(r):
                landed += 1
        return landed

    def close(self) -> None:
        """Stop accepting. Further offers are counted as ``dropped_closed`` rather than lost."""
        with self._lock:
            self._closed = True

    # -- what it did -------------------------------------------------------

    def stats(self) -> RingStats:
        """A snapshot. Cheap enough to take per step and not cheap enough for the hot path."""
        items = self._items
        size = len(items)
        retained = 0
        for r in items:
            if r.args is not None:
                retained += len(r.args)
            if r.kwargs is not None:
                retained += len(r.kwargs)
            if r.value is not None:
                retained += 1
        return RingStats(
            capacity=self._capacity,
            size=size,
            accepted=self._accepted,
            dropped_full=self._dropped_full,
            dropped_contended=self._dropped_contended,
            dropped_closed=self._dropped_closed,
            drained=self._drained,
            high_water=self._high_water,
            resident_bytes=size * RECORD_FOOTPRINT_BYTES,
            retained_refs=retained,
        )

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def resident_bytes(self) -> int:
        """The ring's own footprint, in constant time, for the per-call budget check."""
        return len(self._items) * RECORD_FOOTPRINT_BYTES

    def peek(self) -> Sequence[GraderCall]:
        """The live records, without taking them. For tests and for a card that reads mid-run."""
        return tuple(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        s = self.stats()
        return f"<TapRing {s.size}/{s.capacity}, {s.accepted} accepted, {s.dropped} dropped>"


__all__ = ["RingStats", "TapRing"]

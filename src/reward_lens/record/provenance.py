"""Segment provenance: which policy generated which part of a trajectory.

A trajectory does not have a generating policy. It has several. Kimi K3 partial rollouts resume a
trajectory under a policy several optimizer steps newer than the one that started it, so the turns
of a single trajectory carry different `PolicyVersionID` values, and the off-policy correction that
would fix it needs old logits the infrastructure discards.

That is why `Trajectory.provenance` is a tuple and not a field, and why the tuple has to tile: a
gap means some turns have no recorded generating policy, and an overlap means two policies claim
the same turns. Either one makes every importance-weighted quantity on that trajectory undefined
rather than approximate, so the invariant is enforced at construction and not checked later.

`NEAR_POLICY`, one of the twelve regime conditions, reads exactly this field: "staleness below a
bound; segment provenance is singular". The statistics it needs are at the bottom of this module.
The thresholds are not, because a threshold in an envelope is not a decision this module makes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, NewType, Sequence

from reward_lens.record.tensors import Engine

#: A policy checkpoint's identity. Content-derived where the tap can see the weights, otherwise
#: whatever the framework calls it; either way it must be stable within a run, because every
#: staleness number is a difference between two of these.
PolicyVersionID = NewType("PolicyVersionID", str)

#: `Engine.id`. The canonical schema types `SegmentProvenance.engine` as this; the field holds the
#: `Engine` structure so the components survive, and `Engine.id` is this string. See
#: `record.tensors`.
EngineID = NewType("EngineID", str)


class TilingError(ValueError):
    """The segment provenance does not tile the trajectory.

    A distinct type because converters catch it: a framework whose record cannot produce a tiling
    is telling you something real about that framework, and the converter should say which
    framework and which trajectory rather than dying inside a dataclass constructor.
    """


@dataclass(frozen=True)
class SamplingMeta:
    """How the tokens in a segment were drawn.

    ``batch_composition`` is recorded because it changes the answer. Qwen3-235B at temperature 0
    produces 80 unique completions in 1000 samples, identical for 102 tokens and diverging at
    token 103, entirely because of what else was in the batch. A record that says "temperature 0,
    therefore deterministic" is wrong on current serving stacks, and this field is where the
    counter-evidence goes when the recorder can see it.
    """

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None
    max_tokens: int | None = None
    batch_composition: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __canonical__(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
            "batch_composition": self.batch_composition,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_canonical(cls, obj: Mapping[str, Any]) -> "SamplingMeta":
        return cls(
            temperature=obj.get("temperature"),
            top_p=obj.get("top_p"),
            top_k=obj.get("top_k"),
            seed=obj.get("seed"),
            max_tokens=obj.get("max_tokens"),
            batch_composition=obj.get("batch_composition"),
            extra=dict(obj.get("extra", {})),
        )


@dataclass(frozen=True)
class SegmentProvenance:
    """A trajectory does not have A generating policy. It has several.

    ``turn_range`` is half-open, ``[start, end)``, matching `core.types.Span`. The convention is
    stated explicitly because the two readings differ by exactly one turn at every boundary, which
    is the kind of off-by-one that produces a plausible wrong number rather than an error.

    ``staleness_steps`` is how many optimizer steps behind the current policy this segment's
    generating policy was at generation time. Zero is on-policy. It is an int and not a float
    because it counts updates.
    """

    turn_range: tuple[int, int]
    policy_version: PolicyVersionID
    staleness_steps: int
    engine: Engine
    sampling: SamplingMeta = field(default_factory=SamplingMeta)

    def __post_init__(self) -> None:
        start, end = self.turn_range
        if start < 0:
            raise TilingError(f"segment turn_range start cannot be negative; got {start}")
        if end <= start:
            raise TilingError(
                f"segment turn_range {self.turn_range} is empty or reversed. The range is "
                f"half-open [start, end), so a one-turn segment is (i, i+1)."
            )
        if self.staleness_steps < 0:
            raise TilingError(
                f"staleness_steps counts optimizer steps behind and cannot be negative; got "
                f"{self.staleness_steps}"
            )

    @property
    def n_turns(self) -> int:
        return self.turn_range[1] - self.turn_range[0]

    def covers(self, turn_index: int) -> bool:
        return self.turn_range[0] <= turn_index < self.turn_range[1]

    def __canonical__(self) -> dict[str, Any]:
        return {
            "turn_range": list(self.turn_range),
            "policy_version": str(self.policy_version),
            "staleness_steps": self.staleness_steps,
            "engine": self.engine.__canonical__(),
            "sampling": self.sampling.__canonical__(),
        }

    @classmethod
    def from_canonical(cls, obj: Mapping[str, Any]) -> "SegmentProvenance":
        return cls(
            turn_range=(obj["turn_range"][0], obj["turn_range"][1]),
            policy_version=PolicyVersionID(obj["policy_version"]),
            staleness_steps=obj["staleness_steps"],
            engine=Engine.from_canonical(obj["engine"]),
            sampling=SamplingMeta.from_canonical(obj.get("sampling", {})),
        )


def check_tiling(segments: Sequence[SegmentProvenance], n_turns: int, *, where: str = "") -> None:
    """Raise `TilingError` unless the segments tile ``[0, n_turns)`` with no gap and no overlap.

    Called from `Trajectory.__post_init__`. The error names the first gap or overlap and both
    segments involved, because a converter that produces an untiled trajectory has a bug at one
    specific boundary and "provenance does not tile" is not enough to find it.

    A trajectory with no turns tiles with no segments, and only with no segments. That case is
    real: an empty rollout (an immediate refusal, an environment that failed to start) is a
    recordable event and it has no generating policy for any turn.
    """
    prefix = f"{where}: " if where else ""
    if n_turns == 0:
        if segments:
            raise TilingError(
                f"{prefix}a trajectory with no turns carries {len(segments)} provenance "
                f"segment(s); there is nothing for them to cover."
            )
        return
    if not segments:
        raise TilingError(
            f"{prefix}a trajectory with {n_turns} turn(s) carries no SegmentProvenance. It is "
            f"mandatory: without it there is no way to know which policy generated these tokens, "
            f"and every importance-weighted quantity on this trajectory is undefined rather than "
            f"stale. Record one segment covering (0, {n_turns}) if there was only one policy."
        )
    ordered = sorted(segments, key=lambda s: s.turn_range)
    if ordered[0].turn_range[0] != 0:
        raise TilingError(
            f"{prefix}segment provenance starts at turn {ordered[0].turn_range[0]}, so turns "
            f"0 to {ordered[0].turn_range[0] - 1} have no recorded generating policy."
        )
    for left, right in zip(ordered, ordered[1:]):
        if right.turn_range[0] < left.turn_range[1]:
            raise TilingError(
                f"{prefix}segments {left.turn_range} ({left.policy_version}) and "
                f"{right.turn_range} ({right.policy_version}) overlap on turns "
                f"{right.turn_range[0]} to {min(left.turn_range[1], right.turn_range[1]) - 1}. "
                f"Two policies cannot both have generated the same turn."
            )
        if right.turn_range[0] > left.turn_range[1]:
            raise TilingError(
                f"{prefix}gap between segments {left.turn_range} and {right.turn_range}: turns "
                f"{left.turn_range[1]} to {right.turn_range[0] - 1} have no recorded generating "
                f"policy."
            )
    if ordered[-1].turn_range[1] != n_turns:
        raise TilingError(
            f"{prefix}segment provenance ends at turn {ordered[-1].turn_range[1]} but the "
            f"trajectory has {n_turns} turn(s); turns {ordered[-1].turn_range[1]} to "
            f"{n_turns - 1} have no recorded generating policy."
        )


def single_segment(
    n_turns: int,
    policy_version: str,
    *,
    engine: Engine,
    staleness_steps: int = 0,
    sampling: SamplingMeta | None = None,
) -> tuple[SegmentProvenance, ...]:
    """The one-policy tiling, for the common case and for tests.

    A convenience, not a default. There is deliberately no default value for
    `Trajectory.provenance`: a trajectory that does not say which policy generated it should be
    impossible to construct, because the alternative is a run where the question was never asked
    and the record cannot tell you so.
    """
    if n_turns == 0:
        return ()
    return (
        SegmentProvenance(
            turn_range=(0, n_turns),
            policy_version=PolicyVersionID(policy_version),
            staleness_steps=staleness_steps,
            engine=engine,
            sampling=sampling or SamplingMeta(),
        ),
    )


# ---------------------------------------------------------------------------
# Statistics NEAR_POLICY is measured from
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyMixture:
    """What `NEAR_POLICY` needs, measured. Two statistics and neither is a verdict.

    ``singular`` is the "segment provenance is singular" half of the condition: one policy version
    across every segment of every trajectory considered. ``max_staleness`` is the other half, and
    the bound it is compared against lives in the envelope, not here.
    """

    n_segments: int
    n_policy_versions: int
    max_staleness: int
    mean_staleness: float
    n_engines: int

    @property
    def singular(self) -> bool:
        return self.n_policy_versions <= 1

    def __canonical__(self) -> dict[str, Any]:
        return {
            "n_segments": self.n_segments,
            "n_policy_versions": self.n_policy_versions,
            "max_staleness": self.max_staleness,
            "mean_staleness": self.mean_staleness,
            "n_engines": self.n_engines,
        }


def policy_mixture(segments: Iterable[SegmentProvenance]) -> PolicyMixture:
    """Summarise a set of segments for the `NEAR_POLICY` reading.

    Weighting is by segment and not by turn, deliberately. A one-turn segment generated by a
    forty-step-stale policy breaks the singular-policy assumption exactly as thoroughly as a
    hundred-turn one, because the assumption is about whether a single importance ratio is
    defined at all, not about how much of the trajectory it covers.
    """
    versions: set[str] = set()
    engines: set[str] = set()
    stalenesses: list[int] = []
    n = 0
    for seg in segments:
        n += 1
        versions.add(str(seg.policy_version))
        engines.add(seg.engine.id)
        stalenesses.append(seg.staleness_steps)
    return PolicyMixture(
        n_segments=n,
        n_policy_versions=len(versions),
        max_staleness=max(stalenesses) if stalenesses else 0,
        mean_staleness=(sum(stalenesses) / len(stalenesses)) if stalenesses else 0.0,
        n_engines=len(engines),
    )


def policy_versions(segments: Iterable[SegmentProvenance]) -> tuple[str, ...]:
    """Distinct policy versions in generation order of their first segment."""
    seen: list[str] = []
    for seg in sorted(segments, key=lambda s: s.turn_range):
        v = str(seg.policy_version)
        if v not in seen:
            seen.append(v)
    return tuple(seen)


def segment_for_turn(
    segments: Sequence[SegmentProvenance], turn_index: int
) -> SegmentProvenance | None:
    """Which segment generated a given turn, or None if none does.

    On a tiled trajectory None is unreachable for an in-range index, which is the point of
    enforcing the tiling: callers do not have to handle a case the constructor made impossible.
    """
    for seg in segments:
        if seg.covers(turn_index):
            return seg
    return None


__all__ = [
    "EngineID",
    "PolicyMixture",
    "PolicyVersionID",
    "SamplingMeta",
    "SegmentProvenance",
    "TilingError",
    "check_tiling",
    "policy_mixture",
    "policy_versions",
    "segment_for_turn",
    "single_segment",
]

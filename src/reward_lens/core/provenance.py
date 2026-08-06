"""Provenance capture.

Every Evidence records where it came from: the git commit of the code that produced it, a hash
of the configuration, the seeds, the compute it cost, any oracle calls that fed it,
and the ids of the parent Evidence it was derived from (I5). The parent list is what makes the
store a DAG; a forecast regression that consumes stored KUI and chi Evidence names them as
parents, and that is what lets a card or a paper trace every number back to leaf measurements.

Provenance is cheap to collect at creation and impossible to reconstruct afterwards, so it is a
mandatory field, not optional metadata. The store rejects Evidence whose declared parents
it cannot resolve.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from reward_lens.core.types import EvidenceID, content_hash


@lru_cache(maxsize=8)
def git_sha(repo_dir: str | None = None) -> str:
    """Return the current git commit sha, or ``"unknown"`` if not in a repo.

    Cached per directory because it shells out; the sha does not change within a run. A dirty
    working tree is flagged with a ``"+dirty"`` suffix so a number produced from uncommitted
    code is never mistaken for a reproducible one.
    """
    cwd = repo_dir or os.getcwd()
    try:
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"
    try:
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=cwd, stderr=subprocess.DEVNULL
        ).decode()
        if dirty.strip():
            sha += "+dirty"
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass
    return sha


@dataclass(frozen=True)
class Cost:
    """The compute a measurement consumed.

    Metered on every Evidence so studies and cards can report their total cost, and so the
    Atlas at population scale is budgetable. ``gpu_seconds`` is wall-clock GPU time; ``tokens``
    counts forward-pass tokens; ``wall_seconds`` is total wall time including CPU work.
    """

    gpu_seconds: float = 0.0
    tokens: int = 0
    wall_seconds: float = 0.0

    def __add__(self, other: "Cost") -> "Cost":
        return Cost(
            gpu_seconds=self.gpu_seconds + other.gpu_seconds,
            tokens=self.tokens + other.tokens,
            wall_seconds=self.wall_seconds + other.wall_seconds,
        )

    def __canonical__(self) -> dict[str, Any]:
        return {
            "gpu_seconds": self.gpu_seconds,
            "tokens": self.tokens,
            "wall_seconds": self.wall_seconds,
        }


@dataclass(frozen=True)
class Provenance:
    """The full provenance envelope carried by every Evidence.

    ``config_hash`` is a content hash of whatever configuration object produced the run;
    ``seeds`` records every RNG seed used; ``oracle_calls`` lists ``OracleCall`` ids for any
    LLM-derived input; ``parents`` lists the Evidence ids this quantity was derived from
    (I5). ``study`` is the frozen StudyID when the run was registered (gate 3), else None.
    """

    git_sha: str = "unknown"
    config_hash: str | None = None
    seeds: tuple[int, ...] = ()
    cost: Cost = field(default_factory=Cost)
    oracle_calls: tuple[str, ...] = ()
    parents: tuple[EvidenceID, ...] = ()
    study: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __canonical__(self) -> dict[str, Any]:
        return {
            "git_sha": self.git_sha,
            "config_hash": self.config_hash,
            "seeds": list(self.seeds),
            "cost": self.cost.__canonical__(),
            "oracle_calls": list(self.oracle_calls),
            "parents": list(self.parents),
            "study": self.study,
            "extra": self.extra,
        }


def capture_provenance(
    config: Any = None,
    seeds: tuple[int, ...] = (),
    cost: Cost | None = None,
    oracle_calls: tuple[str, ...] = (),
    parents: tuple[EvidenceID, ...] = (),
    study: str | None = None,
    repo_dir: str | None = None,
    **extra: Any,
) -> Provenance:
    """Build a Provenance, hashing ``config`` and reading the git sha automatically."""
    return Provenance(
        git_sha=git_sha(repo_dir),
        config_hash=content_hash(config, "cfg") if config is not None else None,
        seeds=tuple(seeds),
        cost=cost or Cost(),
        oracle_calls=tuple(oracle_calls),
        parents=tuple(parents),
        study=study,
        extra=extra,
    )


__all__ = ["Cost", "Provenance", "capture_provenance", "git_sha"]

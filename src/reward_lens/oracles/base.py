"""The Oracle protocol: LLM assistance with mandatory provenance.

Oracles are the LLM judges and labelers that power tournaments, corruption assistance, feature
glossing, and the auditing game's agent operators. The discipline on them is strict, and the reason
is oracle contamination: every oracle output records the model id, the prompt
hash, and the date, and is cached, so a judgment can be traced and never silently recomputed with a
different model. Oracle-derived ground truth is typed distinctly from human or mechanical ground
truth and is never silently mixed into an answer key.

This module provides the protocol, the on-disk cache keyed by (model id, prompt hash), a cost
ledger, and a deterministic `MockOracle` for tests and offline development. Concrete LLM-backed
oracles (an Anthropic or local-model judge pool) live in `oracles/judges.py` and are gated on API
access; they are marked as needing credentials, never faked.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from reward_lens.core.config import get_settings
from reward_lens.core.types import content_hash

# The provenance tier of a piece of ground truth. Oracle-derived truth is never silently mixed with
# human or mechanical truth; the tier is carried so an answer key knows where its labels came from.
GroundTruthTier = str  # "mechanical" | "human" | "oracle"


@dataclass(frozen=True)
class OracleCall:
    """A single oracle interaction, stamped with provenance.

    ``model_id`` is the exact model that produced ``response``; ``prompt_hash`` is the content hash
    of the prompt; ``date`` is when it was made; ``cost`` records tokens and money. The id is
    content-derived from the model and prompt, so an identical call resolves to the same record and
    the cache is exact.
    """

    id: str
    model_id: str
    prompt_hash: str
    response: str
    date: str
    tokens: int = 0
    usd: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


class OracleCache:
    """A content-addressed on-disk cache of oracle calls.

    Keyed by ``(model_id, prompt_hash)`` so the same question to the same model is asked once. The
    cache is a directory of JSON files under the configured home, kept trivially inspectable like
    the evidence store. Caching is not an optimization here; it is what makes oracle-derived ground
    truth reproducible.
    """

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else (get_settings().home / "oracle_cache")
        self.path.mkdir(parents=True, exist_ok=True)

    def _key(self, model_id: str, prompt: str) -> str:
        return content_hash({"model": model_id, "prompt": prompt}, "orc").split(":")[1]

    def get(self, model_id: str, prompt: str) -> OracleCall | None:
        f = self.path / f"{self._key(model_id, prompt)}.json"
        if not f.exists():
            return None
        d = json.loads(f.read_text())
        return OracleCall(**d)

    def put(self, call: OracleCall, prompt: str) -> None:
        key = self._key(call.model_id, prompt)
        (self.path / f"{key}.json").write_text(json.dumps(call.__dict__, default=str))


@dataclass
class CostLedger:
    """A running tally of oracle spend.

    Every oracle call adds to the ledger so a study can report exactly what it spent on LLM
    assistance and a budget can be enforced. The ledger is queried by the study runner before an
    oracle-heavy step to refuse work that would exceed the study's oracle budget.
    """

    calls: int = 0
    tokens: int = 0
    usd: float = 0.0

    def add(self, call: OracleCall) -> None:
        self.calls += 1
        self.tokens += call.tokens
        self.usd += call.usd


@runtime_checkable
class Oracle(Protocol):
    """An LLM judge or labeler.

    ``ask`` takes a prompt and returns an `OracleCall`, consulting the cache first and stamping
    provenance on a miss. Implementations must record the model id, prompt hash, and date on every
    output. The ``tier`` is always "oracle", which is how downstream answer keys keep oracle
    labels typed distinctly from human and mechanical ones.
    """

    model_id: str
    tier: GroundTruthTier

    def ask(self, prompt: str, **kwargs: Any) -> OracleCall: ...


class MockOracle:
    """A deterministic oracle for tests and offline development.

    Returns a stable, prompt-derived response so pipelines that consume oracle output can be tested
    without network or credentials. It stamps full provenance exactly as a real oracle would, so
    code that records oracle lineage is exercised. It is clearly not a real judge; anything that
    needs a real judgment gates on `oracles/judges.py` and its credentials.
    """

    tier: GroundTruthTier = "oracle"

    def __init__(
        self,
        model_id: str = "mock-oracle-v1",
        cache: OracleCache | None = None,
        date: str = "1970-01-01",
    ):
        self.model_id = model_id
        self.cache = cache
        self.date = date
        self.ledger = CostLedger()

    def ask(self, prompt: str, **kwargs: Any) -> OracleCall:
        if self.cache is not None:
            hit = self.cache.get(self.model_id, prompt)
            if hit is not None:
                return hit
        prompt_hash = content_hash(prompt, "ph")
        # A deterministic pseudo-judgment: the parity of the hash, stable across runs.
        verdict = "A" if int(prompt_hash.split(":")[1][:2], 16) % 2 == 0 else "B"
        call = OracleCall(
            id=content_hash({"model": self.model_id, "prompt": prompt}, "call"),
            model_id=self.model_id,
            prompt_hash=prompt_hash,
            response=verdict,
            date=self.date,
            tokens=len(prompt.split()),
            usd=0.0,
            meta={"mock": True},
        )
        self.ledger.add(call)
        if self.cache is not None:
            self.cache.put(call, prompt)
        return call


__all__ = [
    "GroundTruthTier",
    "OracleCall",
    "OracleCache",
    "CostLedger",
    "Oracle",
    "MockOracle",
]

"""``reward_lens.artifacts`` — the delivery layer.

Cards, the Atlas, safety cases, the claims checker, and the site export are views over the evidence
store. They assemble stored Evidence and never compute fresh numbers, which is what guarantees a
card and a paper cite identical values (I5, liability 4). The claims checker is the enforcement
mechanism: a document may not claim a number the store does not contain. The safety case is the
strictest view: it refuses to assemble unless every component is calibrated and registered.
"""

from __future__ import annotations

from reward_lens.artifacts.atlas import (
    Atlas,
    AtlasEntry,
    Leaderboard,
    LeaderboardCell,
    ModelLineage,
    SweepGatedError,
    SweepPlan,
    declared_fingerprint,
)
from reward_lens.artifacts.card import Card, CardEntry, build_card
from reward_lens.artifacts.claims import ClaimReport, ClaimResult, check_files, check_text
from reward_lens.artifacts.safety_case import (
    SafetyCase,
    SafetyCaseRefusal,
    SafetyComponent,
    assemble_safety_case,
)
from reward_lens.artifacts.site import build_site

__all__ = [
    # claims
    "ClaimReport",
    "ClaimResult",
    "check_text",
    "check_files",
    # card
    "Card",
    "CardEntry",
    "build_card",
    # atlas
    "Atlas",
    "AtlasEntry",
    "ModelLineage",
    "Leaderboard",
    "LeaderboardCell",
    "SweepPlan",
    "SweepGatedError",
    "declared_fingerprint",
    # safety case
    "SafetyCase",
    "SafetyComponent",
    "SafetyCaseRefusal",
    "assemble_safety_case",
    # site
    "build_site",
]

"""Envelope schema versions, and the migrations between them.

2.0.1 shipped without a schema version, so every envelope written before this module existed is
unversioned and the first migration has to sniff. That sniff is written once, here, and never
again: an envelope with no ``schema_version`` key is version 0, because there is no other version
it could be.

A migration is a pure function on the envelope dict, registered against the exact pair of versions
it bridges, and `migrate` walks the chain. Chained rather than direct so that adding version 4
means writing one function, not three.

**The trap, stated once because it will not be obvious later.** An envelope's ``value`` is not
plain JSON. `ValueCodec` inlines small arrays and writes larger ones to content-addressed ``.npy``
sidecars, so in a stored envelope a large array is the dict
``{"__ndarray__": {"sidecar": "<hash>.npy", ...}}`` and the numbers are on disk somewhere else. A
migration that rewrites a scalar can work on the dict directly. A migration that has to change an
array must decode through the codec, transform, and re-encode, or it will silently rewrite the
*reference* and leave the payload untouched. `payload_of` and `with_payload` below exist so that
is one call rather than a thing to remember.
"""

from __future__ import annotations

from typing import Any, Callable

#: The version this build writes. Bump it in the same commit that adds the migration into it.
SCHEMA_VERSION = 1

Migration = Callable[[dict[str, Any]], dict[str, Any]]

#: (from_version, to_version) -> a pure function on the envelope dict. Steps of one, walked in
#: order by `migrate`.
MIGRATIONS: dict[tuple[int, int], Migration] = {}


class MigrationError(Exception):
    """No path from the envelope's version to the target version."""


def register(from_version: int, to_version: int) -> Callable[[Migration], Migration]:
    """Register a migration. Steps must be adjacent, so the chain is unambiguous."""

    def deco(fn: Migration) -> Migration:
        if to_version != from_version + 1:
            raise ValueError(
                f"migrations move one version at a time; got {from_version} -> {to_version}. "
                f"Register the intermediate steps and let migrate() chain them."
            )
        MIGRATIONS[(from_version, to_version)] = fn
        return fn

    return deco


def sniff_version(env: dict[str, Any]) -> int:
    """The schema version of an envelope.

    Absent means 0. Every envelope written by 2.0.1 or earlier is unversioned, including the 1,363
    rows of the campaign store, and there is no ambiguity to resolve: the key did not exist.
    """
    v = env.get("schema_version")
    return 0 if v is None else int(v)


def migrate(env: dict[str, Any], *, to: int = SCHEMA_VERSION) -> dict[str, Any]:
    """Bring an envelope up (or down) to version ``to``, applying each registered step in turn.

    Returns the input unchanged when it is already at ``to``, so this is safe to call on every
    read. Raises `MigrationError` when a step is missing rather than returning a partly-migrated
    envelope, because a half-migrated record read as a whole one is exactly the confident wrong
    answer this library exists to refuse.
    """
    version = sniff_version(env)
    if version == to:
        return env
    if version > to:
        raise MigrationError(
            f"envelope is at schema version {version} and this build reads {to}. "
            f"Downgrading is not supported; read it with a newer reward-lens."
        )
    out = env
    while version < to:
        step = MIGRATIONS.get((version, version + 1))
        if step is None:
            raise MigrationError(
                f"no migration from schema version {version} to {version + 1}; "
                f"the chain from {sniff_version(env)} to {to} is broken."
            )
        out = step(dict(out))
        out["schema_version"] = version + 1
        version += 1
    return out


# ---------------------------------------------------------------------------
# Payload access, so a migration cannot accidentally rewrite a reference
# ---------------------------------------------------------------------------


def payload_of(env: dict[str, Any], sidecar_dir: Any = None) -> Any:
    """Decode an envelope's ``value`` through the codec, resolving any sidecar."""
    from reward_lens.core.evidence import _CODEC

    return _CODEC.decode(env["value"], sidecar_dir)


def with_payload(env: dict[str, Any], value: Any, sidecar_dir: Any = None) -> dict[str, Any]:
    """Return a copy of ``env`` whose ``value`` is ``value``, re-encoded through the codec."""
    from reward_lens.core.evidence import _CODEC

    out = dict(env)
    out["value"] = _CODEC.encode(value, sidecar_dir)
    return out


# ---------------------------------------------------------------------------
# The migrations themselves
# ---------------------------------------------------------------------------


@register(0, 1)
def _v0_to_v1(env: dict[str, Any]) -> dict[str, Any]:
    """Version 0 to 1: stamp the version, and nothing else.

    Deliberately the identity. Version 1 is the same envelope shape 2.0.1 wrote, and pretending
    otherwise would mean rewriting 1,363 rows of a published evidence store to no purpose. What
    version 1 buys is that every envelope written from here on says what it is, so the *next*
    migration, which will change something, has a version to key on.
    """
    return env


__all__ = [
    "MIGRATIONS",
    "SCHEMA_VERSION",
    "Migration",
    "MigrationError",
    "migrate",
    "payload_of",
    "register",
    "sniff_version",
    "with_payload",
]

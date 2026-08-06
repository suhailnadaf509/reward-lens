"""Kernel configuration.

One pydantic settings object resolves the paths and policies the kernel needs: where the
evidence store lives, where the activation cache lives, the disk budget for that cache, and the
device policy. Defaults put the store under ``~/.reward_lens`` and honour environment overrides
prefixed ``REWARD_LENS_`` so a study can be pointed at a repo-local store without code changes.

The store must remain trivially inspectable and diffable, which is why the
defaults are plain directories of files, never a database server.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_root() -> Path:
    return Path(os.environ.get("REWARD_LENS_HOME", str(Path.home() / ".reward_lens")))


class Settings(BaseSettings):
    """Resolved kernel settings, overridable by environment or constructor.

    ``store_path`` is the evidence store root; studies may point it at a repo-local
    ``./evidence``. ``cache_path`` is the activation store root. ``cache_disk_cap_gb`` bounds the
    activation cache under LRU eviction. ``device`` is the torch device policy, ``"auto"`` by
    default (CUDA if present, else CPU); the runtime resolves it at load.
    """

    model_config = SettingsConfigDict(env_prefix="REWARD_LENS_", extra="ignore")

    home: Path = Field(default_factory=_default_root)
    store_path: Path | None = None
    cache_path: Path | None = None
    cache_disk_cap_gb: float = 50.0
    device: str = "auto"
    default_dtype: str = "float32"

    def resolved_store(self) -> Path:
        path = self.store_path or (self.home / "store")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def resolved_cache(self) -> Path:
        path = self.cache_path or (self.home / "cache")
        path.mkdir(parents=True, exist_ok=True)
        return path


_SETTINGS: Settings | None = None


def get_settings() -> Settings:
    """Return the process-wide settings singleton, constructed on first use."""
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = Settings()
    return _SETTINGS


def set_settings(settings: Settings) -> None:
    """Override the process-wide settings (used by tests and studies to redirect the store)."""
    global _SETTINGS
    _SETTINGS = settings


__all__ = ["Settings", "get_settings", "set_settings"]

"""Plugin registries.

One mechanism for extension across the whole kernel: string-keyed registries with decorator
registration and, optionally, ``importlib.metadata`` entry-point discovery. Signal adapters,
observables, interventions, organism generators, dataset builders, oracles, and card sections
all register here. This generalizes the v1 experiment registry pattern and is the sanctioned way
new capabilities enter the system: a new observable is a decorated registration plus its tests,
not an edit to a dispatch table.
"""

from __future__ import annotations

from importlib import metadata
from typing import Any, Callable, Generic, Iterator, TypeVar

from reward_lens.core.errors import RegistryError

T = TypeVar("T")


class Registry(Generic[T]):
    """A named, string-keyed registry with decorator registration.

    ``@registry.register("name")`` registers a class or factory under a key; ``registry.get`` and
    ``registry.create`` retrieve it. ``entry_point_group`` names a setuptools entry-point group
    whose members are discovered lazily on first access, so third-party packages can contribute
    adapters without this package importing them eagerly.
    """

    def __init__(self, kind: str, entry_point_group: str | None = None):
        self.kind = kind
        self.entry_point_group = entry_point_group
        self._items: dict[str, T] = {}
        self._discovered = False

    def register(self, name: str, obj: T | None = None) -> Callable[[T], T] | T:
        """Register ``obj`` under ``name``; usable as a decorator or a direct call."""

        def _do(target: T) -> T:
            if name in self._items:
                raise RegistryError(f"{self.kind} '{name}' is already registered")
            self._items[name] = target
            return target

        if obj is not None:
            return _do(obj)
        return _do

    def _discover(self) -> None:
        if self._discovered or self.entry_point_group is None:
            self._discovered = True
            return
        try:
            eps = metadata.entry_points(group=self.entry_point_group)
        except TypeError:  # pragma: no cover - older importlib API
            legacy: Any = metadata.entry_points()
            eps = legacy.get(self.entry_point_group, [])
        for ep in eps:
            if ep.name not in self._items:
                try:
                    self._items[ep.name] = ep.load()
                except Exception:  # pragma: no cover - a broken plugin must not break the kernel
                    continue
        self._discovered = True

    def get(self, name: str) -> T:
        self._discover()
        if name not in self._items:
            raise RegistryError(f"unknown {self.kind} '{name}'; registered: {sorted(self._items)}")
        return self._items[name]

    def create(self, name: str, *args: object, **kwargs: object) -> object:
        """Instantiate a registered class/factory with the given arguments."""
        factory = self.get(name)
        return factory(*args, **kwargs)  # type: ignore[operator]

    def names(self) -> list[str]:
        self._discover()
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        self._discover()
        return name in self._items

    def __iter__(self) -> Iterator[str]:
        self._discover()
        return iter(sorted(self._items))


# The kernel's standard registries. Subsystems import and register into these.
SIGNALS: Registry = Registry("signal adapter", "reward_lens.signals")
OBSERVABLES: Registry = Registry("observable", "reward_lens.observables")
INTERVENTIONS: Registry = Registry("intervention", "reward_lens.interventions")
ORGANISMS: Registry = Registry("organism generator", "reward_lens.organisms")
DATASETS: Registry = Registry("dataset builder", "reward_lens.datasets")
ORACLES: Registry = Registry("oracle", "reward_lens.oracles")
CARD_SECTIONS: Registry = Registry("card section", "reward_lens.card_sections")


__all__ = [
    "Registry",
    "SIGNALS",
    "OBSERVABLES",
    "INTERVENTIONS",
    "ORGANISMS",
    "DATASETS",
    "ORACLES",
    "CARD_SECTIONS",
]

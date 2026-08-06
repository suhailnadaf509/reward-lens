"""Dataset cards and loaders: where the limit/subset bug class dies.

A v1 loader applied ``limit`` before ``subset`` and silently shrank "200 held-out pairs" to about
thirty; two model loads failed and simply dropped their models from the campaign; a network hiccup
could have returned an empty cell that flowed straight into a published mean. Every one of those is
the same failure: a loader that returns a plausible-looking wrong collection with no contract to
violate.

The fix is a typed loader boundary. A `DatasetCard` declares the dataset's identity, its expected
count, and (optionally) its content checksum. `load_dataset` runs the loader and then *asserts* both
before returning: a count mismatch or a checksum mismatch raises `DataError`, and any loader
exception (an IO or network failure) is re-raised as `DataError` rather than returning silently
empty. A loader that returns the wrong rows cannot pass this boundary, so the bug class dies here
rather than three layers up inside a bootstrap.

Loaders for the builtin diagnostic set are pure functions into schema types, registered by the
builtin package at import. External datasets (RewardBench, RM-Bench, HelpSteer2, and so on) are
registered as *cards* with loader stubs that raise a clear "requires the `datasets` extra" error, so
the catalogue is complete and honest while the heavyweight loaders arrive with their dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from reward_lens.core.errors import DataError
from reward_lens.data.schema import DataView

# A loader is a pure function from a card to a DataView.
Loader = Callable[["DatasetCard"], DataView]


@dataclass(frozen=True)
class DatasetCard:
    """The declared contract of a dataset.

    ``declared_count`` is asserted at load: this is the primary defense against the limit/subset
    class of bug. ``checksum`` is the expected `DataView.checksum` (a ``ds:...`` id); when set it is
    verified at load, catching a loader that returned the right *number* of wrong rows. ``checksum``
    may be None for a card whose content is not yet pinned (an external set behind its dependency),
    in which case only the count is enforced.

    ``annotator_linked`` records whether the set preserves per-annotator ratings (needed for the
    pluralism and topology sciences); ``contamination_note`` and ``license_note`` are free text the
    card renderer surfaces so a benchmark's caveats travel with its numbers. ``loader_key`` selects
    the registered loader (defaulting to ``name``).
    """

    name: str
    builder_version: str
    declared_count: int
    checksum: str | None
    license_note: str
    annotator_linked: bool
    contamination_note: str
    loader_key: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def resolved_loader_key(self) -> str:
        return self.loader_key or self.name


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

_LOADERS: dict[str, Loader] = {}
_CARDS: dict[str, DatasetCard] = {}


def dataset_loader(key: str) -> Callable[[Loader], Loader]:
    """Register a loader under ``key`` (decorator).

    A duplicate key raises, so two builders cannot silently shadow each other's loader.
    """

    def register(fn: Loader) -> Loader:
        if key in _LOADERS:
            raise DataError(f"dataset loader {key!r} is already registered")
        _LOADERS[key] = fn
        return fn

    return register


def register_card(card: DatasetCard, *, replace: bool = False) -> DatasetCard:
    """Register a `DatasetCard` by name. Refuses to overwrite unless ``replace`` is set."""
    if card.name in _CARDS and not replace:
        raise DataError(f"dataset card {card.name!r} is already registered")
    _CARDS[card.name] = card
    return card


def get_card(name: str) -> DatasetCard:
    """Look up a registered card, raising `DataError` if it is unknown."""
    try:
        return _CARDS[name]
    except KeyError:
        raise DataError(f"no dataset card named {name!r}; registered: {sorted(_CARDS)}") from None


def list_cards() -> list[str]:
    """The names of all registered cards, sorted."""
    return sorted(_CARDS)


def has_loader(key: str) -> bool:
    return key in _LOADERS


# ---------------------------------------------------------------------------
# load_dataset: the enforcing boundary
# ---------------------------------------------------------------------------


def load_dataset(card: DatasetCard, *, loader: Loader | None = None) -> DataView:
    """Load a dataset through its card, asserting the declared count and checksum.

    Resolution order for the loader: the explicit ``loader`` argument (used by tests and by callers
    who build a loader inline), then the registry keyed by ``card.resolved_loader_key``. The loader is
    called inside a guard that converts any exception into `DataError`, so an IO or network failure is
    a loud typed error and never a silently empty view. After loading:

    1. ``len(view) == card.declared_count`` or `DataError` (the limit/subset defense).
    2. if ``card.checksum`` is set, ``view.checksum() == card.checksum`` or `DataError`.

    Returns the verified view. Both checks are the reason this function exists; skipping either would
    reopen the failure class it closes.
    """
    fn = loader or _LOADERS.get(card.resolved_loader_key)
    if fn is None:
        raise DataError(
            f"no loader registered for card {card.name!r} (key {card.resolved_loader_key!r}); "
            f"registered loaders: {sorted(_LOADERS)}"
        )
    try:
        view = fn(card)
    except DataError:
        raise  # already the right type; let the loader's own message through
    except Exception as exc:  # IO, network, parse: never return an empty cell (liability 7)
        raise DataError(
            f"loader for dataset {card.name!r} failed: {type(exc).__name__}: {exc}"
        ) from exc

    if not isinstance(view, DataView):
        raise DataError(f"loader for {card.name!r} returned {type(view).__name__}, not a DataView")

    if len(view) != card.declared_count:
        raise DataError(
            f"dataset {card.name!r} declared {card.declared_count} items but the loader returned "
            f"{len(view)}; this is exactly the limit/subset loader bug and it stops here"
        )

    if card.checksum is not None:
        actual = view.checksum()
        if actual != card.checksum:
            raise DataError(
                f"dataset {card.name!r} checksum mismatch: card declares {card.checksum} but the "
                f"loaded content hashes to {actual}; the loader returned the wrong content"
            )

    return view


def make_card_from_view(
    view: DataView,
    *,
    name: str,
    builder_version: str,
    license_note: str = "",
    annotator_linked: bool = False,
    contamination_note: str = "",
    loader_key: str | None = None,
    meta: dict[str, Any] | None = None,
) -> DatasetCard:
    """Build a card whose declared count and checksum are taken from an already-loaded view.

    This is how a builtin builder pins its own contract: it constructs the view, then stamps the card
    with the view's actual count and checksum, so a later `load_dataset` verifies the loader still
    produces byte-identical content. It is the honest direction of the dependency (the data defines
    the card), not a hand-typed count that can drift from reality.
    """
    return DatasetCard(
        name=name,
        builder_version=builder_version,
        declared_count=len(view),
        checksum=str(view.checksum()),
        license_note=license_note,
        annotator_linked=annotator_linked,
        contamination_note=contamination_note,
        loader_key=loader_key,
        meta=dict(meta or {}),
    )


# ---------------------------------------------------------------------------
# External dataset cards (registered; loaders stubbed until the `datasets` extra)
# ---------------------------------------------------------------------------


def _external_stub_loader(card: DatasetCard) -> DataView:
    raise DataError(
        f"loading {card.name!r} requires the optional `datasets` extra (pip install "
        "'reward-lens[datasets]') and its loader, which is not implemented. "
        "The card is registered so the catalogue is complete."
    )


# One external loader stub, shared by every external card. Registering the cards (not just the
# loader) is what makes `list_cards()` an honest catalogue of what the library will read.
_EXTERNAL_CARDS = (
    DatasetCard(
        name="rewardbench",
        builder_version="external-stub-v0",
        declared_count=2985,
        checksum=None,
        license_note="RewardBench (Allen AI); see upstream license.",
        annotator_linked=False,
        contamination_note="Benchmark; check for train-set overlap before using as held-out.",
        loader_key="_external_stub",
        meta={"upstream": "allenai/reward-bench", "track": "RewardBench 1"},
    ),
    DatasetCard(
        name="rmbench",
        builder_version="external-stub-v0",
        declared_count=1327,
        checksum=None,
        license_note="RM-Bench; see upstream license.",
        annotator_linked=False,
        contamination_note="Benchmark; sensitivity-focused, style-controlled subsets.",
        loader_key="_external_stub",
        meta={"upstream": "RM-Bench"},
    ),
    DatasetCard(
        name="helpsteer2",
        builder_version="external-stub-v0",
        declared_count=21362,
        checksum=None,
        license_note="HelpSteer2 (NVIDIA), CC-BY-4.0.",
        annotator_linked=True,
        contamination_note="Multi-attribute ratings; per-annotator spread available.",
        loader_key="_external_stub",
        meta={"upstream": "nvidia/HelpSteer2"},
    ),
    DatasetCard(
        name="ultrafeedback",
        builder_version="external-stub-v0",
        declared_count=63967,
        checksum=None,
        license_note="UltraFeedback; see upstream license.",
        annotator_linked=False,
        contamination_note="GPT-4-annotated; oracle-labelled, treat as machine preference.",
        loader_key="_external_stub",
        meta={"upstream": "openbmb/UltraFeedback"},
    ),
    DatasetCard(
        name="processbench",
        builder_version="external-stub-v0",
        declared_count=3400,
        checksum=None,
        license_note="ProcessBench (Qwen); see upstream license.",
        annotator_linked=False,
        contamination_note="Step-level error labels; the answer key for the verification science.",
        loader_key="_external_stub",
        meta={"upstream": "Qwen/ProcessBench"},
    ),
)


def _register_external_cards() -> None:
    if not has_loader("_external_stub"):
        dataset_loader("_external_stub")(_external_stub_loader)
    for card in _EXTERNAL_CARDS:
        if card.name not in _CARDS:
            register_card(card)


_register_external_cards()


__all__ = [
    "DatasetCard",
    "Loader",
    "dataset_loader",
    "register_card",
    "get_card",
    "list_cards",
    "has_loader",
    "load_dataset",
    "make_card_from_view",
]

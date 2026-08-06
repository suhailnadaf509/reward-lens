"""Dataset registry tests: the limit/subset bug dies at the typed loader boundary.

The v1 loader shrank "200 held-out pairs" to about thirty by applying a limit before a subset, and
nothing caught it. `load_dataset` asserts the declared count and the content checksum after every
load, so the synthetic version of that bug (a loader that returns the wrong number of rows, or the
right number of wrong rows) raises `DataError` here instead of flowing into a bootstrap. These tests
plant exactly those two bugs and assert the boundary catches them, and confirm the honest paths (a
correct loader, the builtin diagnostic set) load cleanly.
"""

from __future__ import annotations

import pytest

from reward_lens.core import DataError
from reward_lens.data import (
    DatasetCard,
    DataView,
    get_card,
    load_dataset,
    make_card_from_view,
    make_pair,
)


def _sample_view(n: int, *, tag: str = "x") -> DataView:
    return DataView(
        [
            make_pair(
                f"prompt {tag} {i}",
                f"chosen {i}",
                f"rejected {i}",
                axis="test",
                seed_id=f"{tag}:{i}",
                builder_id="test",
            )
            for i in range(n)
        ]
    )


def test_declared_count_mismatch_raises() -> None:
    """The synthetic limit/subset bug: a loader returns fewer rows than the card declares."""
    card = DatasetCard(
        name="synthetic_wrong_count",
        builder_version="t",
        declared_count=200,
        checksum=None,
        license_note="",
        annotator_linked=False,
        contamination_note="",
    )
    # The loader "accidentally" returns 30 rows, exactly the E17 shape.
    buggy_loader = lambda _card: _sample_view(30)  # noqa: E731
    with pytest.raises(DataError, match="declared 200"):
        load_dataset(card, loader=buggy_loader)


def test_checksum_mismatch_raises() -> None:
    """A loader returns the right count but the wrong content: the checksum catches it."""
    view = _sample_view(10, tag="right")
    card = make_card_from_view(view, name="synthetic_checksum", builder_version="t")
    # The loader returns a different 10-row set, so the count passes but the checksum fails.
    wrong_loader = lambda _card: _sample_view(10, tag="wrong")  # noqa: E731
    with pytest.raises(DataError, match="checksum mismatch"):
        load_dataset(card, loader=wrong_loader)


def test_correct_loader_roundtrips() -> None:
    """A card stamped from a view verifies against a loader that reproduces that view."""
    view = _sample_view(12, tag="ok")
    card = make_card_from_view(view, name="synthetic_ok", builder_version="t")
    loaded = load_dataset(card, loader=lambda _card: _sample_view(12, tag="ok"))
    assert len(loaded) == 12
    assert loaded.checksum() == card.checksum


def test_loader_exception_becomes_data_error() -> None:
    """An IO/parse failure inside a loader surfaces as a typed DataError, never an empty view."""
    card = DatasetCard(
        name="synthetic_raises",
        builder_version="t",
        declared_count=5,
        checksum=None,
        license_note="",
        annotator_linked=False,
        contamination_note="",
    )

    def broken_loader(_card: DatasetCard) -> DataView:
        raise OSError("simulated network failure")

    with pytest.raises(DataError, match="failed"):
        load_dataset(card, loader=broken_loader)


def test_missing_loader_raises() -> None:
    card = DatasetCard(
        name="synthetic_no_loader",
        builder_version="t",
        declared_count=1,
        checksum=None,
        license_note="",
        annotator_linked=False,
        contamination_note="",
        loader_key="does_not_exist",
    )
    with pytest.raises(DataError, match="no loader"):
        load_dataset(card)


def test_external_card_stub_raises_datasets_extra() -> None:
    """External sets are registered as cards; their loaders raise a clear dependency error."""
    card = get_card("rewardbench")
    assert card.declared_count > 0
    with pytest.raises(DataError, match="datasets"):
        load_dataset(card)


def test_builtin_diagnostic_v3_loads_and_verifies() -> None:
    """The builtin diagnostic set loads through its card with count and checksum verified."""
    card = get_card("diagnostic_v3")
    view = load_dataset(card)
    assert len(view) == card.declared_count
    assert view.checksum() == card.checksum
    # It is lineage-honest: every item is an unmutated distinct seed, so effective n equals the count
    # (this holds for both the canonical stats.ess and the local Kish fallback).
    assert view.effective_n() == pytest.approx(float(len(view)), abs=1e-6)

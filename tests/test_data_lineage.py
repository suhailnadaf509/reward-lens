"""Lineage and effective-sample-size tests (the R7 property test).

The single most important thing this subsystem does is make v1's fake-n failure class impossible to
hide. These tests pin the two mechanisms: a view of clones reports an effective n of about one and a
view of distinct seeds reports about its length; and exact-duplicate content is detected at ingest and
collapsed to weights with a warning. The `DataView.effective_n` test routes through the canonical
`reward_lens.stats.ess` and skips if the stats engine is not importable yet (it is built
concurrently); the local Kish fallback is tested directly so the behaviour is covered either way.
"""

from __future__ import annotations

import warnings

import pytest

from reward_lens.data import (
    DataView,
    Lineage,
    collapse_duplicates,
    effective_sample_size,
    make_lineage,
    make_pair,
)


class _LineageItem:
    """A minimal lineage-carrying item, to test seed-level ESS without full Pair construction."""

    def __init__(self, lineage: Lineage) -> None:
        self.lineage = lineage


def _clone_view(seed_id: str, n: int) -> DataView:
    """A view of n pairs that all descend from one seed (distinct content, same seed id)."""
    pairs = [
        make_pair(
            f"prompt {i}",
            f"chosen {i}",
            f"rejected {i}",
            axis="test",
            seed_id=seed_id,
            builder_id="test",
        )
        for i in range(n)
    ]
    return DataView(pairs)


def _distinct_view(n: int) -> DataView:
    pairs = [
        make_pair(
            f"prompt {i}",
            f"chosen {i}",
            f"rejected {i}",
            axis="test",
            seed_id=f"seed:{i}",
            builder_id="test",
        )
        for i in range(n)
    ]
    return DataView(pairs)


def _stats_ess_available() -> bool:
    try:
        import reward_lens.stats.ess  # noqa: F401

        return True
    except ImportError:
        return False


def test_effective_n_of_clones_is_about_one() -> None:
    if not _stats_ess_available():
        pytest.skip("reward_lens.stats.ess not importable yet; canonical ESS unavailable")
    view = _clone_view("one-seed", 40)
    assert view.effective_n() == pytest.approx(1.0, abs=1e-6)


def test_effective_n_of_distinct_seeds_is_about_n() -> None:
    if not _stats_ess_available():
        pytest.skip("reward_lens.stats.ess not importable yet; canonical ESS unavailable")
    view = _distinct_view(30)
    assert view.effective_n() == pytest.approx(30.0, abs=1e-6)


def test_effective_sample_size_fallback_matches_kish() -> None:
    """The local fallback (used when stats.ess is absent) computes the Kish size exactly."""
    assert effective_sample_size(["s"] * 40) == pytest.approx(1.0, abs=1e-9)
    assert effective_sample_size([f"s{i}" for i in range(30)]) == pytest.approx(30.0, abs=1e-9)
    # A half-clone mixture: 20 of one seed and 20 distinct seeds -> Kish size strictly between.
    labels = ["clone"] * 20 + [f"u{i}" for i in range(20)]
    ess = effective_sample_size(labels)
    assert 1.0 < ess < 40.0
    # Closed form: (40)^2 / (20^2 + 20*1^2) = 1600 / 420.
    assert ess == pytest.approx(1600.0 / 420.0, rel=1e-9)


def test_collapse_duplicates_detects_clones_and_warns() -> None:
    """Byte-identical content collapses to one weighted item with a warning (the bias-battery bug)."""
    dup = make_pair(
        "same prompt", "same chosen", "same rejected", axis="test", seed_id="a", builder_id="test"
    )
    # Five items with identical content (built independently, same payload).
    identical = [
        make_pair(
            "same prompt",
            "same chosen",
            "same rejected",
            axis="test",
            seed_id="a",
            builder_id="test",
        )
        for _ in range(5)
    ]
    view = DataView(identical)
    with pytest.warns(UserWarning, match="duplicate"):
        unique, weights = view.collapse_duplicates()
    assert len(unique) == 1
    assert weights == [5]
    # The single unique item is content-equal to the duplicate template.
    assert unique.items[0].lineage.content_hash == dup.lineage.content_hash


def test_collapse_duplicates_keeps_distinct_content() -> None:
    view = _distinct_view(6)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any spurious collapse warning fails the test
        unique, weights = view.collapse_duplicates()
    assert len(unique) == 6
    assert weights == [1] * 6


def test_collapse_duplicates_mixed_weights() -> None:
    a1 = make_pair("p", "c", "r", axis="t", seed_id="s", builder_id="b")
    a2 = make_pair("p", "c", "r", axis="t", seed_id="s", builder_id="b")  # clone of a1
    b1 = make_pair("q", "c", "r", axis="t", seed_id="s2", builder_id="b")  # distinct
    with pytest.warns(UserWarning, match="duplicate"):
        unique, weights = collapse_duplicates([a1, a2, b1], key=lambda it: it.lineage.content_hash)
    assert len(unique) == 2
    assert weights == [2, 1]


def test_make_lineage_is_content_deterministic() -> None:
    a = make_lineage("seed", "builder", (), ["Pair", "p", "c", "r", "axis"])
    b = make_lineage("seed", "builder", (), ["Pair", "p", "c", "r", "axis"])
    c = make_lineage("seed", "builder", (), ["Pair", "p", "c", "different", "axis"])
    assert a.content_hash == b.content_hash
    assert a.content_hash != c.content_hash
    assert a.content_hash.startswith("ch:")


def test_lineage_with_op_preserves_seed_and_records_op() -> None:
    """A mutation keeps the seed id (so ESS is unchanged) and records the op and new content hash."""
    base = make_lineage("seed:7", "builder", (), ["orig"])
    mutated = base.with_op("corrupt_step:swap_number", ["mutated"])
    assert mutated.seed_id == "seed:7"  # resampling still happens at the seed level
    assert mutated.ops == ("corrupt_step:swap_number",)
    assert mutated.content_hash != base.content_hash  # content changed, so the hash refreshed

    # A view of mutations of one seed still has effective n about one: mutation does not inflate n.
    view = DataView(
        [
            _LineageItem(base),
            _LineageItem(base.with_op("m1", ["a"])),
            _LineageItem(base.with_op("m2", ["b"])),
        ]
    )
    assert effective_sample_size(view.seed_ids()) == pytest.approx(1.0, abs=1e-9)

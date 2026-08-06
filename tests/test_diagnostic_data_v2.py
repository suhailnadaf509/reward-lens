"""Tests for the hand-written diagnostic seed corpus and its mutation expansion.

The module was `reward_lens.diagnostic_data_v2` through 2.0.x and is now
`reward_lens.data.builtin.diagnostic_seeds`, next to the only thing that consumes it. The
assertions are unchanged; only the import moved.
"""

from reward_lens.data.builtin.diagnostic_seeds import (
    ALL_DIMENSIONS_V2,
    PreferencePair,
    get_pairs_by_dim_v2,
    get_pairs_v2,
)


class TestDiagnosticDataV2:
    def test_all_dimensions_present(self):
        """12 dimensions as documented in the module docstring."""
        expected = {
            "helpfulness",
            "safety",
            "verbosity",
            "sycophancy",
            "formatting",
            "confidence",
            "correctness",
            "refusal_quality",
            "factuality",
            "instruction_following",
            "code_correctness",
            "math_correctness",
        }
        assert set(ALL_DIMENSIONS_V2.keys()) == expected

    def test_at_least_10_dimensions(self):
        """The dataset contract requires ≥10 dimensions."""
        assert len(ALL_DIMENSIONS_V2) >= 10

    def test_get_pairs_returns_list(self):
        pairs = get_pairs_v2()
        assert isinstance(pairs, list)
        assert len(pairs) > 0

    def test_pair_structure(self):
        pairs = get_pairs_v2()
        for p in pairs[:5]:
            assert isinstance(p, PreferencePair)
            assert len(p.prompt) > 0
            assert len(p.preferred) > 0
            assert len(p.dispreferred) > 0
            assert p.dimension in ALL_DIMENSIONS_V2
            assert len(p.description) > 0

    def test_at_least_30_per_dimension(self):
        """The dataset contract: ≥30 pairs per dimension."""
        pairs_by_dim = get_pairs_by_dim_v2(n_per_dim=30)
        for dim, pairs in pairs_by_dim.items():
            assert len(pairs) >= 30, f"dimension '{dim}' has only {len(pairs)} pairs (need ≥30)"

    def test_filter_by_dimension(self):
        pairs = get_pairs_v2(dimensions=["safety", "helpfulness"])
        dims = {p.dimension for p in pairs}
        assert dims == {"safety", "helpfulness"}

    def test_single_dimension_filter(self):
        pairs = get_pairs_v2(dimensions=["verbosity"])
        assert all(p.dimension == "verbosity" for p in pairs)

    def test_n_per_dim_parameter(self):
        pairs = get_pairs_v2(dimensions=["helpfulness"], n_per_dim=10)
        assert len(pairs) >= 7  # at least seed count
        assert len(pairs) <= 15  # shouldn't wildly overshoot

    def test_seed_pairs_are_human_written(self):
        """The dataset contract: ≥5 human-reviewed seed pairs per dimension."""
        from reward_lens.data.builtin.diagnostic_seeds import _SEEDS

        for dim, seeds in _SEEDS.items():
            assert len(seeds) >= 5, f"dimension '{dim}' has only {len(seeds)} seed pairs (need ≥5)"

    def test_preferred_differs_from_dispreferred(self):
        """Each pair should have distinct preferred and dispreferred."""
        pairs = get_pairs_v2()
        for p in pairs:
            assert p.preferred != p.dispreferred, (
                f"pair '{p.prompt[:50]}...' has identical preferred/dispreferred"
            )

    def test_get_pairs_by_dim(self):
        result = get_pairs_by_dim_v2(n_per_dim=5)
        assert isinstance(result, dict)
        assert set(result.keys()) == set(ALL_DIMENSIONS_V2.keys())
        for dim, pairs in result.items():
            assert all(p.dimension == dim for p in pairs)

"""The E07 acceptance test: no cross-dimension cascade on honest matched data.

E07 correlated arbitrarily paired per-dimension delta vectors and reported the noise floor as "no
cascade." The problem was never the conclusion; it was that the pairing was arbitrary, so the number
meant nothing either way. The v3 fix is matched-prompt construction: the matched-prompt block builds
each surface dimension on one shared set of prompts, so pair i of every dimension shares prompt i and a
cross-dimension correlation compares like with like.

This test exercises exactly that machinery. Because M2 has no reward signal yet, the per-pair delta is
a clearly labelled placeholder: a deterministic standard-normal draw seeded from the pair's content
hash, which is independent across dimensions by construction. The point is to prove the DATA design and
the cross-dimension ANALYSIS are correct: on matched prompts with independently constructed dimensions,
the cross-dimension cascade sits at the noise floor (zero within a bootstrap CI). A real signal gets
wired in later; if a genuine cross-dimension coupling exists, this same machinery will show it
surviving on honest data instead of being an artifact of construction order.
"""

from __future__ import annotations

import itertools

import numpy as np

from reward_lens.data import Pair, matched_prompt_views

# This is a PLACEHOLDER measurement, not a model score. It stands in for a real per-pair reward delta
# so the cross-dimension machinery can be tested end to end. It is deterministic (seeded from content)
# and independent across dimensions (different content -> different seed), which is the honest null:
# no dimension's placeholder delta carries information about another's.
_PLACEHOLDER = True


def _placeholder_delta(pair: Pair) -> float:
    seed = int(pair.lineage.content_hash.split(":")[1][:12], 16)
    return float(np.random.default_rng(seed).standard_normal())


def _delta_vectors() -> dict[str, np.ndarray]:
    views = matched_prompt_views()
    return {
        dim: np.array([_placeholder_delta(p) for p in views[dim].items]) for dim in sorted(views)
    }


def _pairwise_correlations(deltas: dict[str, np.ndarray]) -> dict[tuple[str, str], float]:
    dims = sorted(deltas)
    return {
        (a, b): float(np.corrcoef(deltas[a], deltas[b])[0, 1])
        for a, b in itertools.combinations(dims, 2)
    }


def test_matched_block_is_actually_matched() -> None:
    """Every matched dimension has the same count and pair i shares prompt i across dimensions."""
    views = matched_prompt_views()
    dims = sorted(views)
    counts = {len(views[d]) for d in dims}
    assert len(counts) == 1  # all equal length
    n = counts.pop()
    assert n >= 8  # enough matched stimuli for the correlation to mean anything
    for i in range(n):
        prompts = {views[d].items[i].prompt_text for d in dims}
        assert len(prompts) == 1, f"matched index {i} does not share one prompt: {prompts}"


def test_cross_dimension_cascade_is_at_noise_floor() -> None:
    """On matched prompts with independent dimensions, cross-dimension correlation brackets zero."""
    deltas = _delta_vectors()
    corrs = _pairwise_correlations(deltas)
    values = np.array(list(corrs.values()))

    # No individual cross-dimension correlation is large: there is no spurious cascade.
    assert np.abs(values).max() < 0.75

    # The mean cross-dimension correlation is near zero.
    assert abs(float(values.mean())) < 0.15

    # And zero lies inside a bootstrap CI of the mean pairwise correlation (the E07 claim, made
    # honestly): resample the matched indices, recompute the mean correlation, take the 95% interval.
    dims = sorted(deltas)
    n = len(next(iter(deltas.values())))
    rng = np.random.default_rng(0)
    boot = np.empty(3000)
    for b in range(boot.size):
        idx = rng.integers(0, n, n)
        cs = [
            np.corrcoef(deltas[x][idx], deltas[y][idx])[0, 1]
            for x, y in itertools.combinations(dims, 2)
        ]
        boot[b] = float(np.nanmean(cs))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    assert lo <= 0.0 <= hi, f"cross-dimension cascade CI [{lo:.3f}, {hi:.3f}] excludes zero"


def test_delta_is_labelled_placeholder() -> None:
    """Guard: the delta used here is the documented placeholder, not a real measurement.

    This is a tripwire so a later edit that wires in a real signal cannot silently reuse the noise-floor
    threshold that only makes sense for the independent placeholder null.
    """
    assert _PLACEHOLDER is True
    # Determinism: the same pair yields the same placeholder delta on every call.
    pair = matched_prompt_views()["helpfulness"].items[0]
    assert _placeholder_delta(pair) == _placeholder_delta(pair)

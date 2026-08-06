"""C2's numerics: `h² = G_ii/C_ii`, the Hansen-Houle indices, and realised heritability.

Every property asserted here is one the derivation guarantees, checked against a planted
`G` where the answer is known in closed form rather than against a reference implementation. The
three that matter are the bound `C ⪰ G` (Cauchy-Schwarz on the score-function directions), the
bound `a(β) = c/e ∈ [0, 1]` (Cauchy-Schwarz again, on `G` and `G⁻¹`), and the exactness of the
univariate breeder's equation on a diagonal `G`, which is what licenses the rung-0 estimator at all.

The refusals are tested as hard as the numbers, because on this build they fire more often than the
numbers land: a heritability above 1 is an instrument bug rather than a large heritability, and an
estimator that returned 12,130 rather than refusing would have been believed.
"""

from __future__ import annotations

import numpy as np
import pytest

from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.measure.indices.chi import feature_covariance
from reward_lens.measure.indices.heritability import (
    Autonomy,
    Evolvability,
    FeatureHeritability,
    GConditioning,
    HeritabilityReading,
    RealisedHeritability,
    RealisedHeritabilityReading,
    evolvability_indices,
    heritability,
    ledger_arrays,
    realised_heritability,
    refuse_out_of_bounds,
)


def psd(rng: np.random.Generator, k: int, scale: float = 1.0) -> np.ndarray:
    """A random symmetric positive-definite `k x k`, well away from singular."""
    a = rng.standard_normal((k, k + 4))
    return scale * (a @ a.T) / (k + 4)


# ---------------------------------------------------------------------------
# h2 = G_ii / C_ii
# ---------------------------------------------------------------------------


def test_h2_recovers_a_planted_ratio_exactly() -> None:
    """`G = ρC` plants `h² = ρ` on every feature, whatever the off-diagonal structure."""
    rng = np.random.default_rng(0)
    C = psd(rng, 5)
    for rho in (0.0, 0.25, 0.5, 1.0):
        got = heritability(rho * C, C, tuple("abcde"))
        assert isinstance(got, HeritabilityReading)
        assert np.allclose(got.h2, rho)
        assert got.bound_violations == ()


def test_h2_is_bounded_in_zero_one_for_every_admissible_G() -> None:
    """`C ⪰ G` is the theorem; sample `N ⪰ 0` and check the ratio never leaves [0, 1]."""
    rng = np.random.default_rng(1)
    for _ in range(40):
        N = psd(rng, 4)
        G = psd(rng, 4)
        C = G + N  # C - G = N is PSD by construction, which is exactly the derivation's conclusion
        got = heritability(G, C, tuple("abcd"))
        assert isinstance(got, HeritabilityReading)
        assert np.all(got.h2 >= -1e-12)
        assert np.all(got.h2 <= 1.0 + 1e-12)


def test_h2_above_one_refuses_and_names_the_features() -> None:
    """`h² > 1` means `N_ii < 0`, which is a bug in G or C. It is refused, not clipped."""
    C = np.diag([1.0, 1.0, 1.0])
    G = np.diag([0.5, 1.5, 0.25])
    got = heritability(G, C, ("ok", "broken", "fine"))
    assert isinstance(got, Refusal)
    assert got.reason is RefusalReason.ENVELOPE_VIOLATED
    assert got.statistics["bound_violations"] == ["broken"]
    assert pytest.approx(got.statistics["max_h2"]) == 1.5
    # The remedy is an instruction, which is what a remedy is for.
    assert "same rollouts" in got.remedy and "damping" in got.remedy


def test_a_constant_feature_is_undefined_rather_than_zero() -> None:
    """`C_ii = 0` makes `h²` a 0/0. That is not a zero heritability and it is not dropped."""
    C = np.diag([2.0, 0.0, 1.0])
    G = np.diag([1.0, 0.0, 0.5])
    got = heritability(G, C, ("varies", "constant", "also_varies"))
    assert isinstance(got, HeritabilityReading)
    assert got.undefined == ("constant",)
    assert np.isnan(got.h2[1])
    assert got.h2[0] == pytest.approx(0.5)
    assert any("zero observed variance" in n for n in got.notes)


def test_every_feature_constant_refuses() -> None:
    got = heritability(np.zeros((2, 2)), np.zeros((2, 2)), ("a", "b"))
    assert isinstance(got, Refusal)
    assert got.reason is RefusalReason.RECORD_INCOMPLETE


def test_off_diagonal_violation_is_a_note_not_a_silent_pass() -> None:
    """`N` can fail to be PSD with a clean diagonal, and a per-feature check cannot see it."""
    C = np.array([[1.0, 0.0], [0.0, 1.0]])
    G = np.array([[0.5, 0.9], [0.9, 0.5]])  # diag is fine; C - G has a negative eigenvalue
    got = heritability(G, C, ("a", "b"))
    assert isinstance(got, HeritabilityReading)
    assert got.psd_residual_min_eigenvalue < 0
    assert any("negative eigenvalue" in n for n in got.notes)


def test_mismatched_basis_raises_rather_than_ratioing_the_wrong_pairs() -> None:
    with pytest.raises(ValueError, match="one basis in one order"):
        heritability(np.eye(3), np.eye(2), ("a", "b"))


# ---------------------------------------------------------------------------
# The Hansen and Houle indices
# ---------------------------------------------------------------------------


def test_autonomy_is_bounded_in_zero_one() -> None:
    """`(βᵀGβ)(βᵀG⁻¹β) ≥ 1`, so `c ≤ e` and `a = c/e` never leaves [0, 1]."""
    rng = np.random.default_rng(2)
    for _ in range(50):
        G = psd(rng, 5)
        beta = rng.standard_normal(5)
        got = evolvability_indices(G, beta)
        assert not isinstance(got, Refusal)
        assert 0.0 <= got.autonomy <= 1.0 + 1e-12
        assert got.conditional_evolvability <= got.evolvability + 1e-12


def test_autonomy_is_one_exactly_when_G_is_isotropic() -> None:
    """A `G` proportional to the identity drags nothing along, so every direction is autonomous."""
    got = evolvability_indices(3.0 * np.eye(4), np.array([1.0, -2.0, 0.5, 0.0]))
    assert not isinstance(got, Refusal)
    assert got.autonomy == pytest.approx(1.0)
    assert got.evolvability == pytest.approx(3.0)
    assert got.conditional_evolvability == pytest.approx(3.0)


def test_the_indices_are_scale_free_in_beta_only_for_autonomy() -> None:
    """`e` and `c` are homogeneous of degree two; the unit normalisation is why `a` is not."""
    rng = np.random.default_rng(3)
    G = psd(rng, 3)
    beta = rng.standard_normal(3)
    a = evolvability_indices(G, beta)
    b = evolvability_indices(G, 17.0 * beta)
    assert not isinstance(a, Refusal) and not isinstance(b, Refusal)
    assert a.autonomy == pytest.approx(b.autonomy)
    assert a.evolvability == pytest.approx(b.evolvability)


def test_a_direction_outside_the_range_of_G_has_zero_autonomy() -> None:
    """`G` singular and `β` with mass in its null space: `c = 0` exactly, and that is the answer."""
    G = np.diag([1.0, 0.0])
    got = evolvability_indices(G, np.array([1.0, 1.0]))
    assert not isinstance(got, Refusal)
    assert got.rank_G == 1
    assert got.conditional_evolvability == 0.0
    assert got.autonomy == 0.0
    assert got.beta_outside_G == pytest.approx(0.5)
    assert any("outside the range of G" in n for n in got.notes)


def test_zero_beta_and_zero_G_refuse() -> None:
    assert isinstance(evolvability_indices(np.eye(2), np.zeros(2)), Refusal)
    assert isinstance(evolvability_indices(np.zeros((2, 2)), np.ones(2)), Refusal)


# ---------------------------------------------------------------------------
# Rung 0: realised heritability
# ---------------------------------------------------------------------------


def test_realised_h2_recovers_a_planted_breeders_equation() -> None:
    """Plant `Δz = h²·(η·S)` per step and the cumulative regression returns `h²` on the nose."""
    rng = np.random.default_rng(4)
    truth = np.array([0.8, 0.3, 0.05])
    sel = rng.gamma(2.0, 1.0, size=(60, 3))
    dz = sel * truth
    got = realised_heritability(dz, sel, ("a", "b", "c"))
    assert isinstance(got, RealisedHeritabilityReading)
    assert np.allclose(got.h2, truth)
    assert np.all(got.in_bounds)
    assert refuse_out_of_bounds(got) is None


def test_realised_h2_survives_noise_and_keeps_its_bound_check() -> None:
    rng = np.random.default_rng(5)
    truth = np.array([0.6])
    sel = rng.gamma(3.0, 1.0, size=(200, 1))
    dz = sel * truth + 0.01 * rng.standard_normal((200, 1))
    got = realised_heritability(dz, sel, ("a",))
    assert isinstance(got, RealisedHeritabilityReading)
    assert got.h2[0] == pytest.approx(0.6, abs=0.02)
    assert bool(got.in_bounds[0])


def test_realised_h2_out_of_bounds_refuses_with_the_share_that_explains_it() -> None:
    """A response 10,000x its selection term is not a heritability, and the refusal says why."""
    sel = np.full((50, 1), 1e-4)
    dz = np.full((50, 1), 1.0)
    got = realised_heritability(dz, sel, ("runaway",))
    assert isinstance(got, RealisedHeritabilityReading)
    assert got.h2[0] == pytest.approx(1e4)
    assert not got.in_bounds[0]
    refusal = refuse_out_of_bounds(got)
    assert isinstance(refusal, Refusal)
    assert refusal.reason is RefusalReason.ENVELOPE_VIOLATED
    assert refusal.statistics["min_selection_share"] == pytest.approx(1e-4)
    assert "breeder's equation" in refusal.detail
    assert "Λ" in refusal.remedy


def test_realised_h2_needs_more_than_one_step() -> None:
    got = realised_heritability(np.ones((1, 2)), np.ones((1, 2)), ("a", "b"))
    assert isinstance(got, Refusal)
    assert got.reason is RefusalReason.RECORD_INCOMPLETE


def test_a_feature_with_no_selection_carries_nan_and_is_named() -> None:
    sel = np.column_stack([np.ones(10), np.zeros(10)])
    dz = np.column_stack([0.5 * np.ones(10), np.ones(10)])
    got = realised_heritability(dz, sel, ("selected", "untouched"))
    assert isinstance(got, RealisedHeritabilityReading)
    assert got.h2[0] == pytest.approx(0.5)
    assert np.isnan(got.h2[1])
    assert any("untouched" in n for n in got.notes)


def test_undefined_is_not_out_of_bounds() -> None:
    """A feature with no estimate has produced no claim, so it cannot have violated a bound.

    Found by pointing the rung-0 estimator at the 200-step GRPO record, where two of the five
    surface features are constant: `out_of_bounds` named all four of the features that were not
    cleanly inside [0, 1], so a refusal about the three that overshot read as a refusal about five.
    """
    sel = np.column_stack([np.full(20, 1e-6), np.zeros(20)])
    dz = np.column_stack([np.ones(20), np.ones(20)])
    got = realised_heritability(dz, sel, ("overshoots", "no_selection"))
    assert isinstance(got, RealisedHeritabilityReading)
    assert got.out_of_bounds == ("overshoots",)
    assert got.undefined == ("no_selection",)
    refusal = refuse_out_of_bounds(got)
    assert isinstance(refusal, Refusal)
    assert refusal.statistics["out_of_bounds"] == ["overshoots"]
    assert "no_selection" not in refusal.detail


def test_a_reading_with_only_undefined_features_does_not_refuse_on_the_bound() -> None:
    """Nothing overshot, so there is nothing for the bound check to object to."""
    got = realised_heritability(np.ones((10, 2)), np.zeros((10, 2)), ("a", "b"))
    assert isinstance(got, RealisedHeritabilityReading)
    assert got.undefined == ("a", "b")
    assert got.out_of_bounds == ()
    assert refuse_out_of_bounds(got) is None


def test_ledger_arrays_rejects_two_bases() -> None:
    class Row:
        def __init__(self, name: str) -> None:
            self.feature, self.delta_z, self.selection = name, 1.0, 1.0

    class Led:
        def __init__(self, names: tuple[str, ...]) -> None:
            self.step = 0
            self.rows = tuple(Row(n) for n in names)

    with pytest.raises(ValueError, match="regression of one quantity on another"):
        ledger_arrays([Led(("a", "b")), Led(("a", "c"))])


# ---------------------------------------------------------------------------
# The instruments
# ---------------------------------------------------------------------------


def test_instruments_pass_the_section_4_2_lint() -> None:
    """Rule three of the build: an instrument that cannot pass lint does not exist."""
    from reward_lens.measure.base import lint_instrument

    for inst in (
        FeatureHeritability(np.eye(2), np.eye(2), ("a", "b")),
        Evolvability(np.eye(2), np.ones(2)),
        Autonomy(np.eye(2), np.ones(2)),
        GConditioning(np.eye(2)),
        RealisedHeritability(np.ones((3, 2)), np.ones((3, 2)), ("a", "b")),
    ):
        assert lint_instrument(inst) == [], inst.name


def test_every_instrument_declares_a_registered_quantity() -> None:
    from reward_lens.core.quantity import QUANTITIES

    for inst in (FeatureHeritability, Evolvability, Autonomy, GConditioning, RealisedHeritability):
        assert inst.quantity in QUANTITIES, inst.name


def test_g_conditioning_reports_the_rank_and_the_effective_dimension() -> None:
    got = GConditioning(np.diag([4.0, 1.0, 1.0, 0.0])).compute()
    assert not isinstance(got, Refusal)
    d = got.as_dict()
    assert d["conditioning_n_D"] == pytest.approx(6.0 / 4.0)
    assert d["rank"] == 3


def test_realised_heritability_instrument_refuses_on_the_bound_by_default() -> None:
    inst = RealisedHeritability(np.ones((20, 1)), np.full((20, 1), 1e-6), ("f",))
    assert isinstance(inst.compute(), Refusal)
    # The bound check is the instrument's, not the estimator's, so it can be switched off for a
    # caller that wants the raw slope. The default is on.
    loose = RealisedHeritability(
        np.ones((20, 1)), np.full((20, 1), 1e-6), ("f",), enforce_bounds=False
    )
    assert isinstance(loose.compute(), RealisedHeritabilityReading)


def test_h2_is_invariant_under_a_rescaling_of_the_features() -> None:
    """`f → Df` sends `G → DGD` and `C → DCD`, so the ratio of diagonals does not move.

    This is `policy.reparam` seen from the feature side, and it is why `h²` is the comparable
    quantity and `e(β)` is not: `e` picks up `D` and `h²` does not.
    """
    rng = np.random.default_rng(6)
    N, G = psd(rng, 4), psd(rng, 4)
    C = G + N
    D = np.diag([1.0, 1e3, 1e-2, 7.0])
    base = heritability(G, C, tuple("abcd"))
    scaled = heritability(D @ G @ D, D @ C @ D, tuple("abcd"))
    assert isinstance(base, HeritabilityReading) and isinstance(scaled, HeritabilityReading)
    assert np.allclose(base.h2, scaled.h2)


def test_the_feature_covariance_C_is_the_one_h2_divides_by() -> None:
    """C1's `feature_covariance` and C2's denominator are one object, checked on real arrays."""
    rng = np.random.default_rng(7)
    f = rng.standard_normal((80, 3))
    gids = np.repeat(np.arange(20), 4)
    C = feature_covariance(f, ("a", "b", "c"), group_ids=gids, operator="within_group")
    got = heritability(0.4 * C.matrix, C.matrix, C.names)
    assert isinstance(got, HeritabilityReading)
    assert np.allclose(got.h2, 0.4)
    assert np.allclose(got.C_diagonal, np.diag(C.matrix))

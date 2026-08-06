"""The GUM budget's coverage factor, the Welch numerator, and two missing guards.

Five defects in `core/budget.py`, `core/reference.py` and the `ci_level` stamp in
`core/evidence.py`, each pinned here with the number it moved. They are recorded together because
three of them push in different directions and a reader who sees only the net will conclude that
none of them mattered.

    the coverage factor    k = 2 was applied whatever the degrees of freedom. At the reference
                           module's own three-dose floor the fit leaves one residual degree of
                           freedom, `t_0.975(1) = 12.706`, and the expanded uncertainty was
                           6.35 times too small. On the campaign's own cal-transfer organisms
                           that is U = 0.1466 where the honest number is 1.0881, once the
                           inverse-prediction fix below is in as well.

    the ci_level stamp     `Uncertainty.from_budget` wrote 0.95 on every interval it built. At
                           four effective degrees of freedom `k = 2` covers 88.39%. With no
                           degrees of freedom at all it covers 95.45%, which is also not 95%.

    the Welch numerator    Welch-Satterthwaite is defined on the uncorrelated combination and the
                           correlated `u_c` was being fed in. This one had to be fixed in the same
                           commit as the coverage factor: at r = -0.9 between two four-dof terms
                           the shipped `nu_eff` is 0.08, and applying `t_0.975(0.08) = 1.2e15` to
                           it would have shipped an interval fifteen orders of magnitude wide.

    the LOD/LOQ order      nothing enforced LOQ > LOD, so an override deleted the middle outcome
                           of a three-outcome decision rule in silence.

    the correlation range  `r = -3.0` composed to a negative variance which `sqrt(max(v, 0))`
                           turned into a combined uncertainty of exactly zero.

Two further defects in the same area live in `measure/labels/reference.py` and were fixed there
rather than here. They are the same defect seen from two sides, and the tests for them are at the
bottom of this file because the reproduction was built with the budget arithmetic beside it.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import norm as normal_dist
from scipy.stats import t as student_t

from reward_lens.core.budget import (
    BudgetLintError,
    BudgetTerm,
    LimitOfDetection,
    UncertaintyBudget,
)
from reward_lens.core.evidence import Uncertainty
from reward_lens.core.reference import (
    CalibrationChain,
    MatrixDescription,
    ReferenceMaterial,
    Transfer,
)

ORGANISM = MatrixDescription(system="planted 0.6B organism", scale="0.6B")


# ---------------------------------------------------------------------------
# The coverage factor
# ---------------------------------------------------------------------------


def test_the_coverage_factor_is_the_t_quantile_at_the_three_dose_floor():
    """Three doses, two fitted parameters, one residual degree of freedom.

    Before: `U = 2 * 0.06 = 0.12`, and the interval was stamped 95% while covering 70.48%.
    After:  `U = t_0.975(1) * 0.06 = 12.7062 * 0.06 = 0.762372`, a factor of 6.3531.
    """
    b = UncertaintyBudget(terms=(BudgetTerm(name="u_char", value=0.06, kind="A", dof=1),))
    assert b.effective_dof() == pytest.approx(1.0)
    assert b.coverage_factor == pytest.approx(12.70620, rel=1e-5)
    assert b.expanded == pytest.approx(0.7623720, rel=1e-6)
    # the number this replaces, and the coverage it really had
    assert 2.0 * b.combined == pytest.approx(0.12)
    assert b.expanded / (2.0 * b.combined) == pytest.approx(6.35310, rel=1e-5)
    assert 2.0 * student_t.cdf(2.0, 1) - 1.0 == pytest.approx(0.704833, rel=1e-5)
    assert b.coverage_achieved == pytest.approx(0.95)


def test_k_two_at_four_effective_degrees_of_freedom_covers_88_4_percent():
    """R8's number, and the reason the stamp had to move with the factor."""
    b = UncertaintyBudget(
        terms=(
            BudgetTerm(name="a", value=1.0, kind="A", dof=2),
            BudgetTerm(name="b", value=1.0, kind="A", dof=2),
        )
    )
    assert b.effective_dof() == pytest.approx(4.0)
    # what k = 2 would have delivered on this table, against the 0.95 it was stamped with
    assert 2.0 * student_t.cdf(2.0, 4) - 1.0 == pytest.approx(0.883883, rel=1e-5)
    assert b.coverage_factor == pytest.approx(2.776445, rel=1e-6)
    assert b.expanded == pytest.approx(2.776445 * math.sqrt(2.0), rel=1e-6)
    assert b.coverage_achieved == pytest.approx(0.95)


def test_the_reference_module_floors_reproduce_the_reviewed_4_2x():
    """R7 reported 4.2x and it is reachable from this module's own two minimums.

    A three-dose calibration line leaves one residual degree of freedom and a three-seed
    homogeneity pool leaves two. When the homogeneity term is about a third the size of the
    characterisation term, `nu_eff` lands at 1.22 and `t_0.975(1.22)/2` is 4.21. The factor is not
    a constant: it runs from 1.71x when the two terms are equal to 6.35x when characterisation
    stands alone, and 4.2x sits inside that range rather than bounding it.
    """

    def factor(ratio: float) -> float:
        b = UncertaintyBudget(
            terms=(
                BudgetTerm(name="u_char", value=1.0, kind="A", dof=1),
                BudgetTerm(name="u_bb", value=ratio, kind="A", dof=2),
            )
        )
        return b.coverage_factor / 2.0

    assert factor(0.3268) == pytest.approx(4.2062, rel=1e-3)
    assert factor(1.0) == pytest.approx(1.7099, rel=1e-3)
    assert factor(0.3) == pytest.approx(4.4405, rel=1e-3)


def test_the_conventional_factor_is_a_floor_so_a_well_replicated_budget_does_not_move():
    """`t_0.975` is below 2 for every nu above 60, and narrowing an interval is not a fix.

    M7's real reading on eleven reward models has `nu_eff = 772.98`, where `t_0.975 = 1.96304`.
    Substituting the quantile unconditionally would have narrowed its expanded uncertainty from
    2.275262 to 2.233212, which is -1.85% that no reviewer asked for. The factor is a floor, so
    that budget is untouched and only the small-sample case moves.
    """
    b = UncertaintyBudget(
        terms=(
            BudgetTerm(name="a", value=1.0, kind="A", dof=400),
            BudgetTerm(name="b", value=1.0, kind="A", dof=400),
        )
    )
    assert b.effective_dof() == pytest.approx(800.0)
    assert student_t.ppf(0.975, 800.0) < 2.0
    assert b.coverage_factor == pytest.approx(2.0)
    assert b.expanded == pytest.approx(2.0 * b.combined)
    # and the level it really has is the t level of k = 2 at 800 dof, which is a shade under the
    # normal-theory 0.954500 because the t is the heavier-tailed of the two
    assert b.coverage_achieved == pytest.approx(0.9541621, rel=1e-6)
    assert b.coverage_achieved >= b.coverage_level


def test_a_budget_with_no_declared_dof_keeps_the_conventional_factor():
    """The fallback path, which is most of the library. Nothing here may move."""
    b = UncertaintyBudget(terms=(BudgetTerm(name="a", value=3.0), BudgetTerm(name="b", value=4.0)))
    assert b.effective_dof() is None
    assert b.coverage_factor == 2.0
    assert b.expanded == pytest.approx(10.0)
    assert b.coverage_achieved == pytest.approx(2.0 * normal_dist.cdf(2.0) - 1.0)
    assert b.coverage_achieved == pytest.approx(0.9545, abs=1e-4)


# ---------------------------------------------------------------------------
# The Welch numerator, which had to move in the same commit
# ---------------------------------------------------------------------------


def test_the_welch_numerator_is_the_independent_combination():
    """Welch-Satterthwaite has no correlated form, and feeding `u_c` in is not a small error.

    Before, on two four-dof unit terms:

        r = +0.9  ->  nu_eff = 28.88  (correct 8)
        r = -0.5  ->  nu_eff =  2.00  (correct 8)
        r = -0.9  ->  nu_eff =  0.08  (correct 8), and t_0.975(0.08) = 1.2e15

    The last line is why this could not wait for a second commit: correcting the coverage factor
    while leaving the numerator alone turns a perfectly ordinary anticorrelated budget into an
    expanded uncertainty of 1.2e15 times the combined one.
    """
    for r in (0.0, 0.5, 0.9, -0.5, -0.9):
        b = UncertaintyBudget(
            terms=(
                BudgetTerm(name="a", value=1.0, kind="A", dof=4),
                BudgetTerm(name="b", value=1.0, kind="A", dof=4),
            ),
            correlations={("a", "b"): r},
        )
        assert b.effective_dof() == pytest.approx(8.0), r
        assert b.coverage_factor == pytest.approx(student_t.ppf(0.975, 8.0), rel=1e-9)
        # the correlation is in u_c and only in u_c
        assert b.combined == pytest.approx(math.sqrt(2.0 + 2.0 * r))
        assert b.independent_combined == pytest.approx(math.sqrt(2.0))

    # what the shipped numerator produced at r = -0.9, kept as the number that was wrong
    shipped_nu = math.sqrt(2.0 + 2.0 * -0.9) ** 4 / (2.0 * (1.0 / 4.0))
    assert shipped_nu == pytest.approx(0.08)
    assert student_t.ppf(0.975, shipped_nu) > 1e14


# ---------------------------------------------------------------------------
# The stamp
# ---------------------------------------------------------------------------


def test_the_ci_level_stamp_is_the_level_the_interval_actually_has():
    """It was 0.95 unconditionally. Both branches were wrong and in opposite directions."""
    small = UncertaintyBudget(
        terms=(
            BudgetTerm(name="a", value=1.0, kind="A", dof=2),
            BudgetTerm(name="b", value=1.0, kind="A", dof=2),
        )
    )
    u = Uncertainty.from_budget(small, n=9)
    assert u.ci_level == pytest.approx(0.95)
    assert u.ci_high == pytest.approx(small.expanded)
    assert u.ci_low == pytest.approx(-small.expanded)
    # before: ci_high was 2 * u_c = 2.828427 and ci_level said 0.95 for a real 0.883883
    assert 2.0 * small.combined == pytest.approx(2.828427, rel=1e-6)
    assert u.ci_high == pytest.approx(3.926485, rel=1e-6)

    plain = UncertaintyBudget(terms=(BudgetTerm(name="a", value=1.0),))
    assert Uncertainty.from_budget(plain).ci_level == pytest.approx(0.9545, abs=1e-4)


@settings(max_examples=200, deadline=None)
@given(
    values=st.lists(st.floats(0.01, 10.0), min_size=1, max_size=6),
    dofs=st.lists(st.floats(1.0, 500.0), min_size=1, max_size=6),
    level=st.sampled_from([0.90, 0.95, 0.99]),
)
def test_the_achieved_level_is_never_below_the_declared_one(values, dofs, level):
    """The guarantee the floor buys, as a property. An interval may over-cover; it may not
    under-cover the level printed beside it."""
    n = min(len(values), len(dofs))
    b = UncertaintyBudget(
        terms=tuple(
            BudgetTerm(name=f"t{i}", value=values[i], kind="A", dof=dofs[i]) for i in range(n)
        ),
        coverage_level=level,
    )
    assert b.effective_dof() is not None
    assert b.coverage_achieved >= level - 1e-9
    assert b.expanded >= 2.0 * b.combined - 1e-12


# ---------------------------------------------------------------------------
# The reference carries its degrees of freedom, or the fix above never fires
# ---------------------------------------------------------------------------


def test_the_reference_carries_its_degrees_of_freedom_into_the_budget():
    """Without this the coverage fix is inert on the one path it was found on.

    Every term `ReferenceMaterial.as_terms()` produced had `dof=None`, so every chain built on a
    reference returned `effective_dof() is None` and fell back to k = 2 whatever the reference was
    characterised from.
    """
    bare = ReferenceMaterial(
        id="ref:x",
        kind="planted_organism",
        assigned_value=0.75,
        u_characterisation=0.06,
        u_homogeneity=0.14,
        u_stability=0.03,
        matrix=ORGANISM,
    )
    assert [t.dof for t in bare.as_terms()] == [None, None, None]
    assert UncertaintyBudget(terms=bare.as_terms()).effective_dof() is None

    floored = ReferenceMaterial(
        id="ref:x",
        kind="planted_organism",
        assigned_value=0.75,
        u_characterisation=0.06,
        u_homogeneity=0.14,
        u_stability=0.03,
        matrix=ORGANISM,
        dof_characterisation=1,
        dof_homogeneity=2,
        dof_stability=1,
    )
    assert [t.dof for t in floored.as_terms()] == [1, 2, 1]
    b = UncertaintyBudget(terms=floored.as_terms())
    assert b.effective_dof() is not None
    assert b.coverage_factor > 2.0


def test_degrees_of_freedom_must_be_a_positive_count():
    """Zero is not "unknown". None is unknown, and the difference is a division."""
    with pytest.raises(ValueError, match="positive count"):
        ReferenceMaterial(
            id="r",
            kind="planted_organism",
            assigned_value=0.5,
            u_characterisation=0.1,
            matrix=ORGANISM,
            dof_characterisation=0,
        )


def test_the_chain_passes_its_own_scalar_terms_degrees_of_freedom_too():
    chain = CalibrationChain(
        reference=ReferenceMaterial(
            id="r",
            kind="planted_organism",
            assigned_value=0.75,
            u_characterisation=0.06,
            u_homogeneity=0.14,
            u_stability=0.03,
            matrix=ORGANISM,
            dof_characterisation=1,
            dof_homogeneity=2,
            dof_stability=1,
        ),
        transfers=(Transfer(from_level="reference_method", to_level="primary", value=0.419, n=64),),
        u_working=0.05,
        u_instrument=0.02,
        dof_working=31,
        dof_instrument=29,
    )
    b = chain.as_budget()
    assert {t.name: t.dof for t in b.terms} == {
        "u_working": 31,
        "t32": 63,
        "u_char": 1,
        "u_bb": 2,
        "u_stab": 1,
        "u_instrument": 29,
    }
    # t32 carries 91% of the variance on 63 degrees of freedom, so the small-dof reference terms
    # barely move nu_eff and the factor stays at the convention. The fix is targeted, not a
    # blanket inflation, and this is the case that shows it.
    assert b.effective_dof() == pytest.approx(59.0127, rel=1e-4)
    assert b.coverage_factor == pytest.approx(2.00105, rel=1e-4)


# ---------------------------------------------------------------------------
# The two guards
# ---------------------------------------------------------------------------


def test_loq_must_sit_above_lod_or_the_rule_has_two_outcomes():
    """Before: `lod_k=10, loq_k=3.3` constructed fine and no reading anywhere returned the middle
    verdict. A detected-but-not-quantifiable effect at 0.25 came back `below_lod` and was refused
    instead of bounded."""
    with pytest.raises(ValueError, match="three outcomes"):
        LimitOfDetection(sigma_blank=0.02, sensitivity=0.5, lod_k=10.0, loq_k=3.3)
    with pytest.raises(ValueError, match="three outcomes"):
        LimitOfDetection(sigma_blank=0.02, sensitivity=0.5, lod_k=3.3, loq_k=3.3)
    with pytest.raises(ValueError, match="detection multiplier"):
        LimitOfDetection(sigma_blank=0.02, sensitivity=0.5, lod_k=0.0, loq_k=10.0)

    ok = LimitOfDetection(sigma_blank=0.02, sensitivity=0.5)
    reachable = {ok.verdict(float(x)) for x in np.linspace(0.0, 5.0, 5001)}
    assert reachable == {"below_lod", "above_lod_below_loq", "quantifiable"}
    assert ok.verdict(0.25) == "above_lod_below_loq"


def test_a_correlation_outside_minus_one_to_one_raises_rather_than_clamping():
    """Before: `r = -3.0` composed to a variance of -4.0 which the square root clamped to a
    combined uncertainty of exactly 0.0, and `from_budget` published `[-0.0, 0.0]` at 95%."""
    with pytest.raises(BudgetLintError, match=r"lies in \[-1, 1\]"):
        UncertaintyBudget(
            terms=(BudgetTerm(name="a", value=1.0), BudgetTerm(name="b", value=1.0)),
            correlations={("a", "b"): -3.0},
        )
    with pytest.raises(BudgetLintError, match=r"lies in \[-1, 1\]"):
        UncertaintyBudget(
            terms=(BudgetTerm(name="a", value=1.0), BudgetTerm(name="b", value=1.0)),
            correlations={("a", "b"): 1.5},
        )
    # the boundaries are legal and compose the way they should
    assert UncertaintyBudget(
        terms=(BudgetTerm(name="a", value=1.0), BudgetTerm(name="b", value=1.0)),
        correlations={("a", "b"): -1.0},
    ).combined == pytest.approx(0.0, abs=1e-12)


def test_a_correlation_set_that_is_not_a_correlation_matrix_raises():
    """Every coefficient inside [-1, 1] and the set of them still impossible. Three terms cannot
    all be mutually at -1, and before this the composed variance of -3 was clamped to zero."""
    with pytest.raises(BudgetLintError, match="positive semi-definite"):
        UncertaintyBudget(
            terms=(
                BudgetTerm(name="a", value=1.0),
                BudgetTerm(name="b", value=1.0),
                BudgetTerm(name="c", value=1.0),
            ),
            correlations={("a", "b"): -1.0, ("a", "c"): -1.0, ("b", "c"): -1.0},
        )


def test_shares_sum_to_one_against_the_variance_the_budget_reports():
    """Before: two fully correlated unit terms composed to `u_c = 2`, so `u_c^2 = 4`, and the
    shares were 50% each against an independent total of 2. The percentages described a variance
    the table did not print."""
    b = UncertaintyBudget(
        terms=(BudgetTerm(name="a", value=1.0), BudgetTerm(name="b", value=1.0)),
        correlations={("a", "b"): 1.0},
    )
    shares = b.shares()
    assert sum(shares.values()) == pytest.approx(1.0)
    assert shares["a"] * b.combined**2 == pytest.approx(2.0)
    assert shares["b"] * b.combined**2 == pytest.approx(2.0)
    # and with no correlations nothing moved
    plain = UncertaintyBudget(
        terms=(BudgetTerm(name="a", value=3.0), BudgetTerm(name="b", value=4.0))
    )
    assert plain.shares() == pytest.approx({"a": 9 / 25, "b": 16 / 25})


def test_a_negative_share_is_reported_rather_than_clipped():
    """A term anticorrelated with the rest of the table reduces the combined variance. -26.9% is
    a fact about the apparatus and a floor at zero would hide it."""
    b = UncertaintyBudget(
        terms=(BudgetTerm(name="a", value=3.0), BudgetTerm(name="b", value=1.0)),
        correlations={("a", "b"): -0.8},
    )
    shares = b.shares()
    assert b.combined == pytest.approx(2.280351, rel=1e-6)
    assert shares["a"] == pytest.approx(1.269231, rel=1e-5)
    assert shares["b"] == pytest.approx(-0.269231, rel=1e-5)
    assert sum(shares.values()) == pytest.approx(1.0)


@settings(max_examples=200, deadline=None)
@given(
    values=st.lists(st.floats(0.1, 5.0), min_size=2, max_size=5),
    r=st.floats(-0.95, 0.95),
)
def test_shares_always_sum_to_one_against_the_combined_variance(values, r):
    terms = tuple(BudgetTerm(name=f"t{i}", value=v) for i, v in enumerate(values))
    b = UncertaintyBudget(terms=terms, correlations={("t0", "t1"): r})
    assert sum(b.shares().values()) == pytest.approx(1.0)
    total = sum(share * b.combined**2 for share in b.shares().values())
    assert total == pytest.approx(b.combined**2)


# ---------------------------------------------------------------------------
# The real subject: the campaign's own cal-transfer organisms
# ---------------------------------------------------------------------------

#: The campaign evidence store, which is not in this repository. There is no default: point
#: ``REWARD_LENS_CAMPAIGN_STORE`` at the store directory or the tests that need it skip.
_CAMPAIGN_ENV = os.environ.get("REWARD_LENS_CAMPAIGN_STORE")
STORE = Path(_CAMPAIGN_ENV) if _CAMPAIGN_ENV else None


@pytest.mark.skipif(
    STORE is None or not (STORE / "evidence.jsonl").exists(),
    reason="no campaign evidence store; set REWARD_LENS_CAMPAIGN_STORE",
)
def test_the_campaign_organisms_are_the_three_dose_floor_and_the_factor_is_635():
    """Three doses, one seed, one residual degree of freedom. The real case, not a fixture.

    The `real` arm's calibration line has a slope of 0.3888 and a residual scatter of 0.0285, so
    the sensitivity-scaled residual is 0.07331 in dose units. Two fixes compose here and the test
    carries both, because they were found separately and land together.

    The coverage factor moves from an unconditional `k = 2` to `t_0.975(1) = 12.7062`, which is
    6.353x on its own. And `u_characterisation` stops being the bare ratio: at rung 0 nothing else
    carries the individual-plant term, so the inverse-prediction form applies and at an assigned
    dose of 0.75 against a sweep centred at 0.8 it is 0.08564, which is 1.168x the ratio.

    Together the expanded uncertainty on the assigned dose of an organism nominally planted at
    rho = 0.75 goes from 0.14662 at a stamped 95% to 1.08810 at a true 95%, a factor of 7.421.
    That number says the realised strength of these organisms is not pinned down at all by the
    sweep that was run, and that is the honest reading of three points.
    """
    from reward_lens.core.quantity import load_quantities
    from reward_lens.measure.labels import campaign_plants, certify

    load_quantities()
    arms = campaign_plants(str(STORE))
    plants = arms["real"]
    cert = certify(plants, reference_id="campaign:real", assigned_dose=0.75, matrix=ORGANISM)
    fit = cert.fit
    assert cert.n_plants == 3 and cert.n_dose_levels == 3 and cert.n_seeds == 1
    assert fit.dof == 1
    assert fit.slope == pytest.approx(0.388781, rel=1e-4)
    assert fit.sensitivity_scaled_residual == pytest.approx(0.0733081, rel=1e-4)
    assert cert.u_characterisation == pytest.approx(0.0856352, rel=1e-4)

    # The state before either fix: the bare ratio, and no degrees of freedom on the material.
    before = UncertaintyBudget(
        terms=ReferenceMaterial(
            id="campaign:real",
            kind=cert.kind,  # type: ignore[arg-type]
            assigned_value=0.75,
            u_characterisation=fit.sensitivity_scaled_residual,
            matrix=ORGANISM,
        ).as_terms()
    )
    assert before.effective_dof() is None
    assert before.coverage_factor == pytest.approx(2.0)
    assert before.expanded == pytest.approx(0.146616, rel=1e-4)

    # And after both. `material()` now carries the degrees of freedom, so nothing has to be
    # reconstructed by hand to see the corrected factor.
    material = cert.material()
    assert material.dof_characterisation == pytest.approx(1.0)
    after = UncertaintyBudget(terms=material.as_terms())
    assert after.effective_dof() == pytest.approx(1.0)
    assert after.coverage_factor == pytest.approx(12.70620, rel=1e-5)
    assert after.expanded == pytest.approx(1.088098, rel=1e-4)
    assert after.expanded / before.expanded == pytest.approx(7.42140, rel=1e-4)

    # The coverage factor alone is 6.353x of that; the rest is the inverse-prediction term.
    assert after.coverage_factor / before.coverage_factor == pytest.approx(6.35310, rel=1e-4)


# ---------------------------------------------------------------------------
# The same defect from two sides, fixed in measure/labels/reference.py
# ---------------------------------------------------------------------------


def test_u_char_and_u_bb_do_not_double_count_the_seed_spread():
    """`certify` fits the calibration line through every plant including the replicates, so the
    residual absorbs the between-seed spread that `between_seed_homogeneity` measures again.

    Under the calibration model the inverse-prediction variance splits exactly into the two terms
    ISO Guide 35 names:

        u_CRM^2 = (s/|b|)^2 * [1/n + (x0 - xbar)^2/Sxx]   +   (s/|b|)^2 * 1
                  |________________ u_char ___________|       |__ u_bb __|

    so `u_bb` owns the `1` and `u_char` must be a standard error of the fitted line. Passing the
    raw residual scatter as `u_char` hands the `1` over twice. Measured over 4,000 simulated
    certifications at 5 doses by 3 seeds: shipped `u_CRM` is 1.3938 sigma_seed where the correct
    value is 1.0328 sigma_seed, an inflation of 1.3693x. The two terms correlate 0.84 to 0.93
    across trials depending on the design, because they are two estimates of one variance.
    """
    from reward_lens.measure.labels.reference import Plant, certify

    rng = np.random.default_rng(20260805)
    sigma = 0.05
    plants = [
        Plant(nominal_dose=d, response=d + float(rng.normal(0.0, sigma)), seed=s, pipeline="p")
        for d in (0.2, 0.35, 0.5, 0.65, 0.8)
        for s in range(3)
    ]
    cert = certify(plants, reference_id="sim", assigned_dose=0.5, matrix=ORGANISM)
    s_over_b = cert.fit.s_resid / abs(cert.fit.slope)
    # u_char must be the standard error of the line, not the scatter about it
    assert cert.u_characterisation == pytest.approx(
        s_over_b * math.sqrt(1.0 / cert.fit.n), rel=1e-6
    )


def test_u_char_carries_the_inverse_prediction_factor_when_no_homogeneity_term_exists():
    """At rung 0 there is no `u_bb`, so `u_char` is the whole inverse-prediction uncertainty and
    the `1 + 1/n` factor belongs to it.

    Shipped `u_char = s_resid/|slope|` omits it entirely. At the module's own three-dose floor the
    factor is `sqrt(1 + 1/3) = 1.1547`, so the shipped value is 13.3975% below the correct one at
    the sweep centre. The instrument's declared deviation says the omitted term is the one "that
    grows as the assigned dose moves away from the centre of the sweep", which implies the value
    is exact at the centre. It is 13.4% low there.
    """
    from reward_lens.measure.labels.reference import dose_response_characterisation

    rng = np.random.default_rng(7)
    xs = np.array([0.25, 0.5, 0.75])
    ys = xs + rng.normal(0.0, 0.05, size=3)
    fit = dose_response_characterisation(xs, ys)
    s_over_b = fit.s_resid / abs(fit.slope)
    assert fit.u_char == pytest.approx(s_over_b * math.sqrt(1.0 + 1.0 / fit.n), rel=1e-6)

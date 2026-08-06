"""Debt M acceptance: four numbers that were wrong, each asserted with the number it changed.

Every clause here is a measurement taken before the fix and again after it, and the before-value is
written into the test so a later reader can see what moved rather than being told that something
did. The four:

1. The collapse projection's interval moved 62.4-fold under a shift of the step index alone. It is
   Fieller's now and is invariant to where the step index starts. The `ci=None` case turns out to
   be exactly Fieller's genuinely-unbounded case rather than a second failure wearing the same
   return value, so the fix there is to say so, and the note now reaches the reader.

2. `GStudy.declare_fixed` returns `E rho^2 = Phi = 1.0000` exactly on data whose true residual
   variance is 2.0. The arithmetic is not changed, because it is the answer of a model that has
   assumed the residual away and the instrument layer above it already renders that honestly. What
   is added is the identification statement at the layer below: the coefficient is an upper bound,
   the interval it is identified to travels with it, and truncated components and a degenerate
   decomposition are flagged rather than reported as measured zeros.

3. `test_property_brier_is_proper` failed at 0.2244 against 0.2078 on a fresh `hypothesis` seed
   after a long run of passes. Two things were wrong and only one of them was the test's
   reconstruction of the recalibration map. The property itself is too strong under equal-width
   binning, and the general identity that replaces it is checked here.

4. The claims gate's baseline carried 189 entries, of which 20 were this library's own release
   names read as measurements. The gate exists to make a fabricated number visible, and a fabricated
   number is harder to see among noise.
"""

from __future__ import annotations

import json
import math
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from reward_lens.artifacts.claims import baseline_key, check_files, find_unbound_numbers
from reward_lens.forecast.score import (
    brier,
    murphy_decomposition,
    recalibrate,
    reliability_diagram,
)
from reward_lens.measure.estimator.amplifier import _project_collapse
from reward_lens.stats.gtheory import crossed_pr, crossed_pro

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_AISI = _ROOT / "tests" / "fixtures" / "aisi_olmo_kl0_seed2_series.json"

#: The exclusions the workflow passes, kept in one place so the gate is measured the way it runs.
_CLAIMS_EXCLUDE = (
    "cards-and-claims.md",
    "artifacts-operate.md",
    "cli.md",
    "measurements-you-can-trust.md",
    "evidence-store.md",
)


def _docs_pages() -> list[pathlib.Path]:
    """The pages the committed baseline was written against: tracked, minus the CI exclusions.

    Tracked rather than everything on disk, because a page added but not yet committed has numbers
    nobody has ratcheted and baselining them on its author's behalf is exactly the growth the
    ratchet forbids.
    """
    tracked = subprocess.run(
        ["git", "ls-files", "docs/content"], capture_output=True, text=True, cwd=_ROOT
    ).stdout.split()
    return [
        _ROOT / f
        for f in sorted(tracked)
        if f.endswith(".md") and not any(s in f for s in _CLAIMS_EXCLUDE)
    ]


def _aisi_series() -> tuple[np.ndarray, np.ndarray]:
    """The AISI run's per-step fraction of groups with no reward spread, over 400 steps.

    `frac_reward_zero_std` is the recorded fraction of groups whose 16 rollouts all scored the
    same, which is the quantity the collapse projection extrapolates. The real subject rather than
    a planted line: the interval has to be invariant on data that was not made to be invariant on.
    """
    d = json.loads(_AISI.read_text())
    return (
        np.asarray(d["steps"], dtype=float),
        np.asarray(d["frac_reward_zero_std"], dtype=float),
    )


# ---------------------------------------------------------------------------
# 1. The collapse projection's interval
# ---------------------------------------------------------------------------

#: A rising all-fail fraction over eight steps. The numbers are the ones the before-measurement was
#: taken on, so the 62.4-fold figure below is reproducible from this file alone.
_RISING = np.array([0.05, 0.08, 0.12, 0.19, 0.28, 0.35, 0.44, 0.52])


def _old_interval(steps: np.ndarray, fractions: np.ndarray) -> tuple[float, float] | None:
    """What `_project_collapse` computed before this change, kept so the before-number is checkable.

    Two standard errors on the slope, the intercept held fixed at its fitted value. The intercept
    is defined at `x = 0`, which is why moving `x = 0` away from the data moves this and why it is
    not a confidence statement about the crossing.
    """
    y = np.log(np.clip(fractions, 1e-6, 1 - 1e-6) / (1 - np.clip(fractions, 1e-6, 1 - 1e-6)))
    x = steps.astype(float)
    slope, intercept = np.polyfit(x, y, 1)
    target = 0.0  # logit(0.5)
    resid = y - (slope * x + intercept)
    sxx = float(((x - x.mean()) ** 2).sum())
    se = math.sqrt(float((resid**2).sum()) / max(x.size - 2, 1) / sxx)
    ends = [(target - intercept) / s for s in (slope - 2 * se, slope + 2 * se) if s > 0]
    return (min(ends), max(ends)) if len(ends) == 2 else None


def test_the_old_interval_inflated_62_fold_under_a_shift_of_the_step_index():
    """The before-number. An interval that moves when you renumber the x-axis is not an interval."""
    at_zero = _old_interval(np.arange(8, dtype=float), _RISING)
    at_400 = _old_interval(np.arange(8, dtype=float) + 400.0, _RISING)
    assert at_zero is not None and at_400 is not None
    w0, w400 = at_zero[1] - at_zero[0], at_400[1] - at_400[0]
    assert w0 == pytest.approx(0.9344, abs=5e-4)
    assert w400 == pytest.approx(58.3399, abs=5e-4)
    assert w400 / w0 == pytest.approx(62.44, abs=0.02)


@pytest.mark.parametrize("shift", [0.0, 1.0, 10.0, 100.0, 400.0, 1000.0])
def test_the_fieller_interval_only_slides_when_the_step_index_shifts(shift):
    """The after-number: the width is the same to machine precision and the crossing slides by exactly
    the shift."""
    base_at, base_ci, _ = _project_collapse(np.arange(8, dtype=float), _RISING)
    at, ci, note = _project_collapse(np.arange(8, dtype=float) + shift, _RISING)
    assert base_ci is not None and ci is not None
    assert at == pytest.approx(base_at + shift, abs=1e-9)
    assert ci[0] == pytest.approx(base_ci[0] + shift, abs=1e-9)
    assert ci[1] == pytest.approx(base_ci[1] + shift, abs=1e-9)
    assert (ci[1] - ci[0]) == pytest.approx(base_ci[1] - base_ci[0], rel=1e-12)
    assert note == ""


def test_the_fieller_interval_is_origin_invariant_on_the_real_aisi_series():
    """Not a planted line: 400 steps of a real run, in every 40-step window that projects at all."""
    steps, frac = _aisi_series()
    checked = 0
    for lo in range(0, 360, 20):
        mask = (steps >= lo) & (steps < lo + 40)
        if mask.sum() < 4:
            continue
        at, ci, _ = _project_collapse(steps[mask], frac[mask])
        at0, ci0, _ = _project_collapse(steps[mask] - steps[mask][0], frac[mask])
        if at is None:
            assert at0 is None
            continue
        assert at == pytest.approx(at0 + steps[mask][0], abs=1e-6)
        assert (ci is None) == (ci0 is None)
        if ci is not None and ci0 is not None:
            assert (ci[1] - ci[0]) == pytest.approx(ci0[1] - ci0[0], rel=1e-9)
            checked += 1
    assert checked == 4, "four of the 40-step windows carry a bounded interval"


def test_the_none_case_is_fiellers_unbounded_case_and_now_says_so():
    """The question E42 item 10 left open, answered: it is the same case, not a second failure.

    Fieller's set is unbounded exactly when the slope is not `z` standard errors clear of zero, and
    the old code dropped an endpoint exactly when `slope - 2*se <= 0`. Those are the same condition
    at `z = 2`. So `ci=None` was already right and what was missing was saying so, which the old
    code could not do because `collapse_note` was rendered only when there was no projected step.
    """
    steps, frac = _aisi_series()
    agreed = unbounded = 0
    for lo in range(0, 388, 4):
        mask = (steps >= lo) & (steps < lo + 12)
        if mask.sum() < 4:
            continue
        at, ci, note = _project_collapse(steps[mask], frac[mask])
        if at is None:
            continue
        old = _old_interval(steps[mask], frac[mask])
        assert (ci is None) == (old is None), f"the None boundary moved at step {lo}"
        agreed += 1
        if ci is None:
            unbounded += 1
            assert "unbounded" in note and "standard errors from zero" in note
    assert agreed == 56, "56 twelve-step windows project a crossing at all"
    assert unbounded == 51, "51 of those 56 have an interval Fieller leaves unbounded"


def test_the_note_reaches_the_reader_when_the_interval_is_unbounded():
    """It did not before: `_says` printed the note only when there was no projected step, so the one
    case the note exists for was the one case it never reached."""
    from reward_lens.measure.estimator.amplifier import AmplifierReading, _says

    reading = AmplifierReading(
        safety={"grader": 1.5},
        detail={"grader": {"magnitude": 0.4}},
        verdicts={"grader": "live"},
        ranking=["grader"],
        magnitude_ranking=["grader"],
        n_groups=10,
        n_allfail_groups=3,
        n_mixed_groups=7,
        predicted_collapse_step=120.0,
        collapse_note="no bounded crossing interval: the logit slope is 0.71 standard errors "
        "from zero, against the 2.0 this interval needs.",
    )
    says = _says(reading)
    assert "Projected all-fail dominance at step 120" in says
    assert "no bounded crossing interval" in says

    reading.predicted_collapse_ci = (110.0, 141.0)
    reading.collapse_note = ""
    assert "Fieller interval [110, 141]" in _says(reading)


# ---------------------------------------------------------------------------
# 2. declare_fixed, the confounded term, and the two missing flags
# ---------------------------------------------------------------------------


def _pr_design_with_residual_two():
    """40 objects by 8 raters, true sigma2(p) = 4.0, sigma2(r) = 1.0, sigma2(pr,e) = 2.0."""
    rng = np.random.default_rng(7)
    p = rng.normal(0, 2.0, 40)[:, None]
    r = rng.normal(0, 1.0, 8)[None, :]
    e = rng.normal(0, math.sqrt(2.0), (40, 8))
    return 10.0 + p + r + e


def test_declare_fixed_reports_one_point_oh_and_now_says_it_is_an_upper_bound():
    """The number that fires this: a residual of 2.0 and a reliability of exactly 1.0000.

    A reliability of exactly 1.0000 is the signature of an arithmetic identity, and here it is one:
    `pr,e` is the interaction and the residual added together, and fixing the only facet it carries
    moves the residual into universe score with the interaction. The point value is left alone,
    because it is the answer of the model as stated and the instrument layer above already renders
    it as arithmetic. What is new is that the layer below can no longer be quoted without the
    bound.
    """
    g = crossed_pr(_pr_design_with_residual_two())
    assert g.components.value("pr,e") == pytest.approx(1.73819, abs=1e-5)

    random = g.d_study()
    assert random.generalizability == pytest.approx(0.9159, abs=1e-4)
    assert random.identified
    assert random.confounded_credited == ()

    fixed = g.declare_fixed("r").d_study()
    assert fixed.generalizability == 1.0
    assert fixed.dependability == 1.0
    assert fixed.relative_error == 0.0

    # The part that was missing.
    assert not fixed.identified
    assert fixed.confounded_credited == ("pr,e",)
    lo, hi = fixed.generalizability_bounds
    assert (lo, hi) == pytest.approx((0.91594, 1.0), abs=1e-5)
    assert fixed.dependability_bounds == pytest.approx((0.91594, 1.0), abs=1e-5)
    text = fixed.render()
    assert "upper bounds" in text
    assert "the residual with it" in text
    assert "[0.9159, 1.0000]" in text


def test_a_free_facet_left_in_the_residual_term_keeps_the_coefficient_identified():
    """The three-facet design does not degenerate, and that is why Brennan's trade is real.

    `pro,e` carries both facets. Fixing only the rater leaves the occasion free, so the term keeps
    its error share and the reliability rises from 0.9073 to 0.9222 rather than to 1. Fixing both
    facets does degenerate, and is flagged.
    """
    rng = np.random.default_rng(11)
    n_p, n_r, n_o = 40, 8, 3
    x = (
        10.0
        + rng.normal(0, 1.0, (n_p, 1, 1))
        + rng.normal(0, 0.7, (1, n_r, 1))
        + rng.normal(0, 0.5, (1, 1, n_o))
        + rng.normal(0, 0.6, (n_p, n_r, 1))  # the object-by-rater term fixing r moves
        + rng.normal(0, math.sqrt(2.0), (n_p, n_r, n_o))
    )
    g = crossed_pro(x)
    one = g.declare_fixed("r").d_study()
    assert one.identified
    assert one.generalizability > g.d_study().generalizability
    assert one.generalizability_bounds[0] == one.generalizability_bounds[1]

    both = g.declare_fixed("r").declare_fixed("o").d_study()
    assert not both.identified
    assert both.confounded_credited == ("pro,e",)
    assert both.generalizability == 1.0
    assert both.generalizability_bounds[0] < 1.0


def test_a_degenerate_decomposition_no_longer_reports_a_measured_zero():
    """The defect class E41 fixed in `gauge_rr`, which `GStudy` had no equivalent for.

    Every score identical: every component is zero, the coefficient has no denominator, and 0.0 was
    reported beside 0.9159 from a real design as though the two were the same kind of number.
    """
    d = crossed_pr(np.full((5, 4), 3.0)).d_study()
    assert d.generalizability == 0.0 and d.dependability == 0.0
    assert not d.determined
    assert "is a convention" in d.render()

    live = crossed_pr(_pr_design_with_residual_two()).d_study()
    assert live.determined


def test_the_d_study_carries_the_truncation_flag():
    """A component silently zeroed is a lie about which facet dominates, and small designs truncate
    routinely."""
    rng = np.random.default_rng(3)
    d = crossed_pr(rng.normal(0, 1.0, (4, 3))).d_study()
    assert d.truncated == ("p", "r")
    assert "Truncated at zero: p, r" in d.render()
    assert crossed_pr(_pr_design_with_residual_two()).d_study().truncated == ()


# ---------------------------------------------------------------------------
# 3. The Brier properness property
# ---------------------------------------------------------------------------

#: The draw that broke the property, from `tests/test_forecast_units.py`. Thirteen forecasts,
#: seven occupied bins of ten equal-width ones, outcomes from `default_rng(250)`.
_BRIER_DRAW = [
    0.28125,
    0.125,
    0.75,
    0.25,
    0.3125,
    0.5,
    0.5,
    0.5,
    0.625,
    0.375,
    0.390625,
    0.6875,
    0.40625,
]


def _brier_draw_outcomes() -> list[bool]:
    return [bool(b) for b in np.random.default_rng(250).integers(0, 2, size=len(_BRIER_DRAW))]


def test_the_failing_draw_is_pinned_and_the_old_assertion_read_0_2244_against_0_2078():
    """The before-numbers, and the first of the two things that were wrong.

    The old test rebuilt the recalibration map by taking, for each forecast, the diagram row whose
    `bin_probability` was nearest. Under equal-width bins the rows are labelled by their within-bin
    mean, so a forecast near a bin edge can be nearer its neighbour's label than its own. Two of the
    thirteen were sent to the wrong row, and 0.2244 is what that costs.
    """
    outcomes = _brier_draw_outcomes()
    m = murphy_decomposition(_BRIER_DRAW, outcomes)
    assert m.n == 13 and m.n_bins == 7
    assert m.binning == "10 equal-width bins (approximate)"
    assert m.brier == pytest.approx(0.20780123197115385, abs=1e-12)

    diagram = reliability_diagram(_BRIER_DRAW, outcomes)
    lookup = dict(zip(diagram.bin_probability, diagram.observed_frequency))
    by_nearest = [lookup[min(lookup, key=lambda b: abs(b - p))] for p in _BRIER_DRAW]
    assert brier(by_nearest, outcomes) == pytest.approx(0.2243589743589744, abs=1e-12)

    mis = sum(
        1
        for p, row in zip(_BRIER_DRAW, diagram.assignment)
        if min(range(len(lookup)), key=lambda k: abs(diagram.bin_probability[k] - p)) != row
    )
    assert mis == 2

    # With each forecast sent to its own bin, the property holds on this draw with room to spare.
    own = recalibrate(_BRIER_DRAW, outcomes)
    assert brier(own, outcomes) == pytest.approx(0.12820512820512822, abs=1e-12)
    assert brier(own, outcomes) < m.brier


def test_recalibration_can_genuinely_hurt_under_approximate_binning():
    """The second thing that was wrong, and the answer to which of the two it was.

    Even with every forecast sent to its own bin, recalibration can raise the Brier score when the
    bins are equal-width, because a bin holding several distinct forecasts that track their outcomes
    has a positive within-bin covariance and flattening throws it away. So the property was stated
    too strongly, and the fresh seed found a case rather than a bug. Eleven forecasts, drawn from a
    perfectly calibrated generator, 0.16484 becoming 0.17424.
    """
    p = [0.963, 0.549, 0.762, 0.403, 0.477, 0.408, 0.794, 0.511, 0.744, 0.766, 0.04]
    y = [True, True, False, False, True, False, True, False, True, True, False]
    m = murphy_decomposition(p, y)
    assert m.binning == "10 equal-width bins (approximate)"
    assert m.brier == pytest.approx(0.16484227272727273, abs=1e-12)
    assert brier(recalibrate(p, y), y) == pytest.approx(0.17424242424242425, abs=1e-12)


def test_under_exact_binning_recalibration_buys_exactly_the_reliability_term():
    """Which is a stronger claim than the inequality it replaces, and the one that is actually true.

    `BS(recalibrated) = BS - REL` to machine precision whenever the bins are the distinct forecast
    values, because then the within-bin variance and covariance are both identically zero. The
    campaign's own sixteen calls take five distinct probabilities and land here.
    """
    rng = np.random.default_rng(4)
    for _ in range(200):
        p = rng.choice([0.1, 0.3, 0.5, 0.7, 0.9], size=int(rng.integers(6, 30))).tolist()
        y = (rng.random(len(p)) < np.asarray(p)).tolist()
        m = murphy_decomposition(p, y)
        assert m.binning.endswith("(exact)")
        assert brier(recalibrate(p, y), y) == pytest.approx(m.brier - m.reliability, abs=1e-12)
        assert brier(recalibrate(p, y), y) <= m.brier + 1e-12


def test_the_general_identity_holds_under_both_binnings():
    """`BS - BS(recal) = REL + (1/n) sum_k n_k [var_k(p) - 2 cov_k(p, y)]`, which is what the
    property test now asserts. Worst deviation over 2,000 mixed draws."""
    rng = np.random.default_rng(5)
    worst = 0.0
    approximate = 0
    for _ in range(2000):
        n = int(rng.integers(2, 31))
        p = np.round(rng.uniform(0.01, 0.99, size=n), int(rng.integers(1, 4)))
        y = rng.random(n) < p
        m = murphy_decomposition(p.tolist(), y.tolist())
        diagram = reliability_diagram(p.tolist(), y.tolist())
        rows = np.asarray(diagram.assignment)
        yy = y.astype(float)
        within = 0.0
        for row in range(len(diagram.count)):
            mask = rows == row
            pk, yk = p[mask], yy[mask]
            within += mask.sum() * (
                float(np.mean((pk - pk.mean()) ** 2))
                - 2.0 * float(np.mean((pk - pk.mean()) * (yk - yk.mean())))
            )
        within /= n
        gap = m.brier - brier(recalibrate(p.tolist(), y.tolist()), y.tolist())
        worst = max(worst, abs(gap - (m.reliability + within)))
        approximate += 0 if m.binning.endswith("(exact)") else 1
    assert approximate > 400, "the sweep has to reach the approximate branch to mean anything"
    assert worst < 1e-12


# ---------------------------------------------------------------------------
# 4. The claims ratchet
# ---------------------------------------------------------------------------

#: Unique baseline keys before and after the release-reference guard. The contract is that this
#: number may fall and may not rise.
_BASELINE_BEFORE = 189
_BASELINE_AFTER = 169


def test_the_baseline_shrank_by_twenty_and_every_line_it_lost_was_a_release_name():
    """The ratchet's contract, measured. 189 unique keys before, 169 after.

    `_VERSION_RE` masks `v2.0` and `3.0.0` and deliberately not a bare `1.0`, because `1.0` is a
    perfect AUC far more often than it is a release. What separates them is grammar: a release name
    sits in a noun phrase and a measurement sits in a predicate. Everything the new guard removed
    carried the value 1.0 or 2.0 and named a release of this library.
    """
    from reward_lens.artifacts.claims import load_baseline

    baseline = load_baseline(_ROOT / "docs" / "claims-baseline.txt")
    assert len(baseline) == _BASELINE_AFTER
    assert _BASELINE_AFTER < _BASELINE_BEFORE

    report = check_files(_docs_pages())
    keys = {baseline_key(u) for u in report.unbound}
    assert keys - baseline == set(), "the baseline must cover everything the checker finds"
    assert baseline - keys == set(), "and must carry no line the checker no longer produces"


def test_the_guard_masks_release_names_and_not_measurements():
    """The line-by-line check, as assertions. A guard that ate a real number would be worse than the
    noise it removes."""
    masked = [
        "If you are porting an existing 1.0 workflow, the map is here.",
        "The 2.0 API is the same shape as the tiny one.",
        "A 1.0 script that traces a preference, next to its 2.0 equivalent.",
        "1.0 spoke one reward dialect.",
        "2.0 does.",
        "The preserved 1.0 compatibility layer and the submodule-only tools.",
        "This is the successor to the analysis in the 1.0 toolkit.",
    ]
    for text in masked:
        assert find_unbound_numbers(text + "\n") == [], text

    kept = [
        ("The detector recovers the planted rule at AUC 1.0 on the split.", ["1.0"]),
        ("The Jaccard is \\(1.0\\), and that is the lesson.", ["1.0"]),
        ("Near \\(1.0\\) means the model decides at the very end.", ["1.0"]),
        ("A real erase drops that AUC to chance (1.0 to 0.5056).", ["1.0", "0.5056"]),
        ("A raw cosine reports a change from 1.0 down to 0.37.", ["1.0", "0.37"]),
        ("A sham erase leaves the AUC at 1.0 and the evidence stays EXPLORATORY.", ["1.0"]),
        ("The reliability came out at exactly 1.0, which is arithmetic.", ["1.0"]),
    ]
    for text, expected in kept:
        assert [u.value for u in find_unbound_numbers(text + "\n")] == expected, text


def test_a_fabricated_number_is_still_caught_after_the_shrink():
    """The whole point of the gate, exercised the way CI runs it plus one invented measurement."""
    report = check_files(_docs_pages())
    assert report.unbound, "the docs backlog is not empty; the gate has something to ratchet"

    invented = find_unbound_numbers(
        "The amplifier safety ratio on the held-out run came out at 3.47.\n"
    )
    assert [u.value for u in invented] == ["3.47"]

    # And the release guard does not launder a measurement that happens to sit near release prose.
    near = find_unbound_numbers("The 2.0 API reports an attenuation of 0.71 on this design.\n")
    assert [u.value for u in near] == ["0.71"]


def test_the_gate_is_green_over_the_pages_the_baseline_covers():
    """Run as the workflow runs it, restricted to the tracked pages the baseline was written for.

    `tests/acceptance/test_w0_4_claims_ci.py` owns the whole-tree version of this assertion; this
    one is the ratchet's own, and it stays measurable while a page is being written next door.
    """
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "reward_lens.artifacts.claims",
            *[str(p) for p in _docs_pages()],
            "--baseline",
            str(_ROOT / "docs" / "claims-baseline.txt"),
        ],
        capture_output=True,
        text=True,
        cwd=_ROOT,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "Unbound numbers: 0" in r.stdout

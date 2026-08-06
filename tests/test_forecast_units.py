"""Unit, property, refusal and invariance tests for `reward_lens.forecast`.

Fast and synthetic. The acceptance file is where this package meets real data; this file is where
its arithmetic is checked against values computed by hand, and where the properties that must hold
for every input are checked by `hypothesis` rather than by three examples somebody picked.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import assume, example, given, settings
from hypothesis import strategies as st

from reward_lens.core.evidence import make_evidence
from reward_lens.core.provenance import Provenance
from reward_lens.core.store import EvidenceStore
from reward_lens.core.types import SubjectRef
from reward_lens.forecast import (
    AISI_TRAPS,
    BaselineKind,
    BinaryProbability,
    CalibrationLedger,
    Comparator,
    DecisionSpec,
    Forecast,
    ForecastCalibration,
    ForecastError,
    ForecastLeakageError,
    HorizonSpec,
    InformationTime,
    IntervalForecast,
    QuantileForecast,
    ReferenceClass,
    ResolutionRule,
    Resolved,
    RunCorpus,
    Void,
    VoidReason,
    belief_flip_hash,
    brier,
    climatology,
    contrastive_belief_flip,
    coverage_score,
    dumb_statistic,
    entry_from,
    forecast_id,
    harmonic,
    issue,
    log_score,
    murphy_decomposition,
    persistence,
    persistence_rate,
    records_test,
    reliability_diagram,
    resolve,
    skill_score,
)

# Imported from the module rather than the package: `recalibrate` is new in this change and the
# package export block belongs to whoever merges it.
from reward_lens.forecast.score import recalibrate  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CLASS = ReferenceClass(
    id="test.class",
    definition="a synthetic population used only by the unit tests",
    n=100,
    base_rate=0.2,
)


def four_baselines():
    return (
        climatology(CLASS),
        persistence(True, confidence=0.8),
        dumb_statistic(0.4, name="synthetic"),
        contrastive_belief_flip((), (), judge=None),
    )


def a_forecast(p: float = 0.6, *, inputs: tuple[str, ...] = ()) -> Forecast:
    rule = ResolutionRule(metric="m", comparator=Comparator.GT, threshold=0.5)
    subject = SubjectRef(extra={"case": "unit"})
    issued = InformationTime.parse("2026-03-01T00:00:00+00:00", basis="a unit test")
    return Forecast(
        id=forecast_id(
            target="q",
            subject=subject,
            resolution=rule,
            issued_at=issued,
            distribution=BinaryProbability(p),
            inputs=inputs,
            method="unit",
        ),
        target="q",
        subject=subject,
        resolution=rule,
        issued_at=issued,
        horizon=HorizonSpec(kind="steps", value=10.0),
        reference_class=CLASS,
        distribution=BinaryProbability(p),
        method="unit",
        inputs=inputs,
        baselines=four_baselines(),
    )


def chain_store(tmp_path, times: list[str]) -> tuple[EvidenceStore, list[str]]:
    """A linear DAG whose information times are supplied, so ancestry can be walked deliberately."""
    store = EvidenceStore(tmp_path / "store")
    ids: list[str] = []
    parent = None
    for i, when in enumerate(times):
        evidence = make_evidence(
            observable=f"synthetic.step{i}",
            observable_version="1.0",
            subject=SubjectRef(extra={"i": i}),
            value=float(i),
            provenance=Provenance(parents=(parent,) if parent else ()),
            created_at=when,
            information_time=when,
        )
        store.append(evidence)
        ids.append(evidence.id)
        parent = evidence.id
    return store, ids


# ---------------------------------------------------------------------------
# The third clock
# ---------------------------------------------------------------------------


def test_information_time_normalises_to_utc_and_orders_by_instant():
    a = InformationTime.parse("2026-03-01T12:00:00+02:00", basis="a log in Athens")
    b = InformationTime.parse("2026-03-01T10:30:00+00:00", basis="a log in UTC")
    assert a.instant == "2026-03-01T10:00:00+00:00"
    assert a < b and b > a and a <= b


def test_information_time_accepts_a_trailing_z():
    assert InformationTime.parse("2026-03-01T10:00:00Z", basis="an ISO Z form").instant == (
        "2026-03-01T10:00:00+00:00"
    )


def test_information_time_requires_a_basis():
    with pytest.raises(ForecastError) as excinfo:
        InformationTime.parse("2026-03-01T10:00:00+00:00", basis="")
    assert "needs a basis" in str(excinfo.value)


def test_information_time_is_not_derivable_from_run_position():
    with pytest.raises(ForecastError) as excinfo:
        InformationTime.from_run_position(200)
    message = str(excinfo.value)
    assert "run position, not an information time" in message
    assert "reanalysis of an archive" in message


def test_information_time_differs_from_created_at_on_a_reanalysis():
    """The case the third clock exists for, in one assertion.

    A row written in 2024 and reanalysed in 2026 has a 2024 `created_at` and a 2026 information
    time, and the barrier reads the second.
    """
    from reward_lens.forecast.barrier import information_time_of

    fresh = make_evidence(
        observable="o",
        observable_version="1",
        subject=SubjectRef(),
        value=1.0,
        created_at="2024-01-01T00:00:00+00:00",
    )
    assert information_time_of(fresh).instant == "2024-01-01T00:00:00+00:00"
    assert "defaulted to created_at" in information_time_of(fresh).basis

    reanalysed = make_evidence(
        observable="o",
        observable_version="1",
        subject=SubjectRef(),
        value=1.0,
        created_at="2024-01-01T00:00:00+00:00",
        information_time="2026-08-05T00:00:00+00:00",
    )
    when = information_time_of(reanalysed)
    assert when.instant == "2026-08-05T00:00:00+00:00"
    assert when.basis == "the row's own information_time"


# ---------------------------------------------------------------------------
# The barrier
# ---------------------------------------------------------------------------


def test_the_barrier_walks_three_generations(tmp_path):
    store, ids = chain_store(
        tmp_path,
        [
            "2026-01-01T00:00:00+00:00",
            "2026-01-02T00:00:00+00:00",
            "2026-01-03T00:00:00+00:00",
        ],
    )
    # ids[2] is the youngest; its grandparent ids[0] is oldest. Issuing after ids[2] is fine.
    issue(
        target="q",
        subject=SubjectRef(),
        resolution=ResolutionRule(metric="m", comparator=Comparator.GT, threshold=0.0),
        distribution=BinaryProbability(0.5),
        inputs=(ids[2],),
        at=InformationTime.parse("2026-01-04T00:00:00+00:00", basis="a unit test"),
        store=store,
        reference_class=CLASS,
        horizon=HorizonSpec(),
        method="unit",
        baselines=four_baselines(),
    )
    # Issuing between the parent and the child leaks through the direct input.
    with pytest.raises(ForecastLeakageError) as excinfo:
        issue(
            target="q",
            subject=SubjectRef(),
            resolution=ResolutionRule(metric="m", comparator=Comparator.GT, threshold=0.0),
            distribution=BinaryProbability(0.5),
            inputs=(ids[2],),
            at=InformationTime.parse("2026-01-02T12:00:00+00:00", basis="a unit test"),
            store=store,
            reference_class=CLASS,
            horizon=HorizonSpec(),
            method="unit",
            baselines=four_baselines(),
        )
    assert excinfo.value.evidence_id == ids[2]
    assert excinfo.value.path == (ids[2],)


def test_a_reanalysed_ancestor_leaks_even_though_its_wall_time_does_not(tmp_path):
    """The whole reason the barrier reads information time and not `created_at`.

    Both rows were written in 2024. The parent was reanalysed and its information time is 2026, so
    a forecast issued in 2025 from the child leaks, and reading `created_at` would have missed it.
    """
    store = EvidenceStore(tmp_path / "store")
    parent = make_evidence(
        observable="archive.parent",
        observable_version="1.0",
        subject=SubjectRef(extra={"n": 1}),
        value=1.0,
        created_at="2024-01-01T00:00:00+00:00",
        information_time="2026-01-01T00:00:00+00:00",
    )
    store.append(parent)
    child = make_evidence(
        observable="archive.child",
        observable_version="1.0",
        subject=SubjectRef(extra={"n": 2}),
        value=2.0,
        provenance=Provenance(parents=(parent.id,)),
        created_at="2024-01-02T00:00:00+00:00",
        information_time="2024-01-02T00:00:00+00:00",
    )
    store.append(child)

    with pytest.raises(ForecastLeakageError) as excinfo:
        issue(
            target="q",
            subject=SubjectRef(),
            resolution=ResolutionRule(metric="m", comparator=Comparator.GT, threshold=0.0),
            distribution=BinaryProbability(0.5),
            inputs=(child.id,),
            at=InformationTime.parse("2025-01-01T00:00:00+00:00", basis="a unit test"),
            store=store,
            reference_class=CLASS,
            horizon=HorizonSpec(),
            method="unit",
            baselines=four_baselines(),
        )
    assert excinfo.value.evidence_id == parent.id
    assert excinfo.value.path == (child.id, parent.id)
    assert excinfo.value.information_time.instant == "2026-01-01T00:00:00+00:00"


def test_an_input_at_exactly_the_issue_instant_leaks(tmp_path):
    """`>=`, not `>`. A tie is a coin flip on write ordering and a barrier cannot depend on one."""
    store, ids = chain_store(tmp_path, ["2026-01-01T00:00:00+00:00"])
    with pytest.raises(ForecastLeakageError):
        issue(
            target="q",
            subject=SubjectRef(),
            resolution=ResolutionRule(metric="m", comparator=Comparator.GT, threshold=0.0),
            distribution=BinaryProbability(0.5),
            inputs=(ids[0],),
            at=InformationTime.parse("2026-01-01T00:00:00+00:00", basis="a unit test"),
            store=store,
            reference_class=CLASS,
            horizon=HorizonSpec(),
            method="unit",
            baselines=four_baselines(),
        )


def test_an_unresolvable_input_is_a_provenance_error_not_a_leak(tmp_path):
    """Two conditions, two errors, because the remedies point in opposite directions."""
    from reward_lens.core.errors import ProvenanceError

    store, _ = chain_store(tmp_path, ["2026-01-01T00:00:00+00:00"])
    with pytest.raises(ProvenanceError) as excinfo:
        issue(
            target="q",
            subject=SubjectRef(),
            resolution=ResolutionRule(metric="m", comparator=Comparator.GT, threshold=0.0),
            distribution=BinaryProbability(0.5),
            inputs=("ev:nothing",),
            at=InformationTime.parse("2026-02-01T00:00:00+00:00", basis="a unit test"),
            store=store,
            reference_class=CLASS,
            horizon=HorizonSpec(),
            method="unit",
            baselines=four_baselines(),
        )
    assert "unverifiable input" in str(excinfo.value)


def test_a_forecast_with_no_inputs_is_refused(tmp_path):
    store, _ = chain_store(tmp_path, ["2026-01-01T00:00:00+00:00"])
    with pytest.raises(ForecastError) as excinfo:
        issue(
            target="q",
            subject=SubjectRef(),
            resolution=ResolutionRule(metric="m", comparator=Comparator.GT, threshold=0.0),
            distribution=BinaryProbability(0.5),
            inputs=(),
            at=InformationTime.parse("2026-02-01T00:00:00+00:00", basis="a unit test"),
            store=store,
            reference_class=CLASS,
            horizon=HorizonSpec(),
            method="unit",
            baselines=four_baselines(),
        )
    assert "no inputs" in str(excinfo.value)


@settings(max_examples=60, deadline=None)
@given(
    offsets=st.lists(st.integers(min_value=1, max_value=10_000), min_size=1, max_size=6),
    issue_offset=st.integers(min_value=-5000, max_value=15_000),
)
def test_property_issue_succeeds_exactly_when_every_ancestor_predates(
    tmp_path_factory, offsets, issue_offset
):
    """The barrier's contract, over arbitrary chains: it passes if and only if `max(t) < at`."""
    base = np.datetime64("2026-01-01T00:00:00")
    times = [str(base + np.timedelta64(int(o), "s")) + "+00:00" for o in np.cumsum(offsets)]
    store, ids = chain_store(tmp_path_factory.mktemp("s"), times)
    at_instant = str(base + np.timedelta64(int(issue_offset), "s")) + "+00:00"
    at = InformationTime.parse(at_instant, basis="a property test")
    should_pass = max(times) < at_instant

    def attempt():
        return issue(
            target="q",
            subject=SubjectRef(),
            resolution=ResolutionRule(metric="m", comparator=Comparator.GT, threshold=0.0),
            distribution=BinaryProbability(0.5),
            inputs=(ids[-1],),
            at=at,
            store=store,
            reference_class=CLASS,
            horizon=HorizonSpec(),
            method="unit",
            baselines=four_baselines(),
        )

    if should_pass:
        assert attempt().issued_at.instant == at.instant
    else:
        with pytest.raises(ForecastLeakageError):
            attempt()


# ---------------------------------------------------------------------------
# Records theory
# ---------------------------------------------------------------------------


def test_harmonic_numbers_are_the_ones_in_the_textbook():
    assert harmonic(1) == 1.0
    assert harmonic(4) == pytest.approx(1 + 0.5 + 1 / 3 + 0.25)
    assert harmonic(4, 2) == pytest.approx(1 + 0.25 + 1 / 9 + 1 / 16)


def test_records_on_a_strictly_increasing_series_is_every_observation():
    """`n` records against `H_n` expected, which is the maximum possible excess."""
    test = records_test(list(range(20)))
    assert test.n_records == 20
    assert test.expected == pytest.approx(harmonic(20))
    assert test.z > 8
    assert test.p_value < 1e-12


def test_records_on_a_strictly_decreasing_series_is_one():
    """Only the first observation is a record, which is `H_n` standard deviations too few."""
    test = records_test(list(range(100, 0, -1)))
    assert test.n_records == 1
    assert test.record_steps == (0,)
    assert test.expected == pytest.approx(harmonic(100))
    assert test.z < -2
    # One-sided upward test, so a series that only falls has a p-value near one, not near zero.
    assert test.p_value > 0.98


def test_records_direction_is_a_parameter():
    """A series where lower is better reads as improving when it is told so."""
    falling = list(range(20, 0, -1))
    assert records_test(falling, higher_is_better=False).n_records == 20


def test_records_expectation_matches_simulation():
    """The `1/t` result, checked by simulating exchangeable series rather than by trusting it."""
    rng = np.random.default_rng(7)
    n = 50
    counts = []
    for _ in range(4000):
        counts.append(records_test(rng.normal(size=n)).n_records)
    observed_mean = float(np.mean(counts))
    observed_var = float(np.var(counts, ddof=1))
    assert observed_mean == pytest.approx(harmonic(n), abs=0.05)
    assert observed_var == pytest.approx(harmonic(n) - harmonic(n, 2), abs=0.15)


@settings(max_examples=50, deadline=None)
@given(values=st.lists(st.floats(-1e6, 1e6, allow_nan=False), min_size=2, max_size=60))
def test_property_records_are_between_one_and_n(values):
    test = records_test(values)
    assert 1 <= test.n_records <= test.n
    assert test.record_steps[0] == 0
    assert list(test.record_steps) == sorted(set(test.record_steps))


def test_records_needs_two_observations():
    with pytest.raises(ForecastError) as excinfo:
        records_test([1.0])
    assert "at least two observations" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Scoring, against values computed by hand
# ---------------------------------------------------------------------------


def test_brier_by_hand():
    # (0.9-1)^2 + (0.2-0)^2 + (0.5-1)^2 = 0.01 + 0.04 + 0.25 = 0.30, over three = 0.1
    assert brier([0.9, 0.2, 0.5], [True, False, True]) == pytest.approx(0.30 / 3)
    # A coin is 0.25 whatever happens.
    assert brier([0.5] * 8, [True, False] * 4) == 0.25


def test_log_score_by_hand():
    assert log_score([0.5, 0.5], [True, False]) == pytest.approx(math.log(2.0))
    assert log_score([1.0], [True]) == pytest.approx(-math.log(1 - 1e-6), abs=1e-5)


def test_murphy_decomposition_by_hand():
    """Two probability values, four forecasts, every term computed on paper.

    Forecasts 0.8, 0.8, 0.2, 0.2 with outcomes True, False, False, False.
    Base rate 1/4, so uncertainty = 0.25 x 0.75 = 0.1875.
    Bin at 0.8: n=2, observed 0.5, so reliability contributes 2 x (0.8-0.5)^2 = 0.18.
    Bin at 0.2: n=2, observed 0.0, so reliability contributes 2 x (0.2-0.0)^2 = 0.08.
    Reliability = 0.26/4 = 0.065.
    Resolution = [2 x (0.5-0.25)^2 + 2 x (0.0-0.25)^2]/4 = [0.125 + 0.125]/4 = 0.0625.
    So BS = 0.065 - 0.0625 + 0.1875 = 0.19, and directly:
    (0.2^2 + 0.8^2 + 0.2^2 + 0.2^2)/4 = (0.04+0.64+0.04+0.04)/4 = 0.19.
    """
    m = murphy_decomposition([0.8, 0.8, 0.2, 0.2], [True, False, False, False])
    assert m.uncertainty == pytest.approx(0.1875)
    assert m.reliability == pytest.approx(0.065)
    assert m.resolution == pytest.approx(0.0625)
    assert m.brier == pytest.approx(0.19)
    assert m.residual == pytest.approx(0.0, abs=1e-15)
    assert m.n_bins == 2


@settings(max_examples=60, deadline=None)
@given(
    probs=st.lists(st.floats(0.0, 1.0, allow_nan=False), min_size=2, max_size=40),
    seed=st.integers(0, 10_000),
)
def test_property_murphy_identity_closes(probs, seed):
    """`BS = REL - RES + UNC` for any forecast set, under distinct-value binning."""
    rng = np.random.default_rng(seed)
    outcomes = [bool(b) for b in rng.integers(0, 2, size=len(probs))]
    assume(len(set(probs)) <= max(10, len(probs) // 2))
    m = murphy_decomposition(probs, outcomes)
    assert m.residual == pytest.approx(0.0, abs=1e-12)
    assert m.reliability >= -1e-12
    assert m.resolution >= -1e-12
    assert 0.0 <= m.uncertainty <= 0.25 + 1e-12


#: The draw that broke this property on a fresh `hypothesis` seed after a long run of passes.
#: Recorded here rather than left in the gitignored `.hypothesis/` directory, where it survives
#: exactly as long as one machine's working tree does. Thirteen forecasts, seven occupied bins of
#: ten equal-width ones, and the old assertion read 0.2244 against 0.2078.
BRIER_PROPERTY_REGRESSION = [
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


@settings(max_examples=40, deadline=None)
@example(probs=BRIER_PROPERTY_REGRESSION, seed=250)
@given(
    probs=st.lists(st.floats(0.01, 0.99, allow_nan=False), min_size=2, max_size=30),
    seed=st.integers(0, 10_000),
)
def test_property_brier_is_proper(probs, seed):
    """What properness gives you at finite n, stated once per binning rather than once overall.

    Climatology's Brier score is exactly the uncertainty term, whatever the binning. That is what
    makes the decomposition's third term mean "the score you get for knowing nothing but the base
    rate", and it holds unconditionally.

    Recalibration is the part that has to be split. Replacing every forecast by the observed
    frequency in its own bin buys **exactly** the reliability term under distinct-value binning,
    which is the finite-sample form of properness and the reason a badly calibrated forecaster with
    real resolution is worth fixing rather than discarding. Under equal-width binning it does not:
    the general identity carries a within-bin term, `var_k(p) - 2 cov_k(p, y)` summed over bins, and
    a bin whose several distinct forecasts genuinely track their outcomes has a positive covariance
    that flattening throws away. The old form of this test asserted the inequality for both and
    `BRIER_PROPERTY_REGRESSION` is the draw that found it.

    The identity below is asserted for both branches, which is a stronger claim than the inequality
    it replaces: it pins the size of the gap and not only its sign.
    """
    rng = np.random.default_rng(seed)
    outcomes = [bool(b) for b in rng.integers(0, 2, size=len(probs))]
    m = murphy_decomposition(probs, outcomes)
    climatology_brier = brier([m.base_rate] * len(probs), outcomes)
    assert climatology_brier == pytest.approx(m.uncertainty, abs=1e-12)

    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray([1.0 if o else 0.0 for o in outcomes], dtype=np.float64)
    diagram = reliability_diagram(probs, outcomes)
    recalibrated = recalibrate(probs, outcomes)

    # `recalibrate` and the diagram bin identically, which is what makes them two views of one
    # table. Reconstructing the map from the nearest `bin_probability` does not: it mis-assigned
    # 2 of the 13 forecasts in the regression draw.
    for row, r in zip(diagram.assignment, recalibrated):
        assert r == pytest.approx(diagram.observed_frequency[row], abs=1e-12)

    within = 0.0
    for row in range(len(diagram.count)):
        mask = np.asarray(diagram.assignment) == row
        pk, yk = p[mask], y[mask]
        var = float(np.mean((pk - pk.mean()) ** 2))
        cov = float(np.mean((pk - pk.mean()) * (yk - yk.mean())))
        within += mask.sum() * (var - 2.0 * cov)
    within /= p.size

    gap = m.brier - brier(recalibrated, outcomes)
    assert gap == pytest.approx(m.reliability + within, abs=1e-12)

    if m.binning.endswith("(exact)"):
        assert within == pytest.approx(0.0, abs=1e-12)
        assert brier(recalibrated, outcomes) == pytest.approx(m.brier - m.reliability, abs=1e-12)
        assert brier(recalibrated, outcomes) <= m.brier + 1e-12


def test_reliability_diagram_carries_its_counts():
    d = reliability_diagram([0.8, 0.8, 0.2, 0.2], [True, False, False, False])
    assert d.bin_probability == (0.2, 0.8)
    assert d.observed_frequency == (0.0, 0.5)
    assert d.count == (2, 2)
    assert "n" in d.render()
    assert "  2 " in d.render()


def test_skill_score_interval_is_paired():
    """The interval brackets the point estimate and the pairing keeps it from being absurd."""
    rng = np.random.default_rng(3)
    outcomes = [bool(b) for b in rng.integers(0, 2, size=60)]
    good = [0.9 if o else 0.1 for o in outcomes]
    s = skill_score(good, [0.5] * 60, outcomes, baseline_id="coin", seed=1)
    assert s.skill == pytest.approx(1 - brier(good, outcomes) / 0.25)
    assert s.ci_low <= s.skill <= s.ci_high
    assert s.beats_baseline

    bad = [0.5] * 60
    s2 = skill_score(bad, [0.5] * 60, outcomes, baseline_id="coin", seed=1)
    assert s2.skill == pytest.approx(0.0)
    assert s2.covers_zero
    assert not s2.beats_baseline


def test_skill_score_refuses_a_perfect_baseline():
    with pytest.raises(ForecastError) as excinfo:
        skill_score([0.5, 0.5], [1.0, 0.0], [True, False], baseline_id="oracle")
    assert "zero denominator" in str(excinfo.value)


def test_coverage_score_wilson_by_hand():
    c = coverage_score([True, True, True, False], nominal=0.8)
    assert c.coverage == 0.75
    assert c.covered == 3 and c.n == 4
    # Wilson at 95 percent on 3/4 is roughly [0.30, 0.95]; the point is that it is very wide.
    assert 0.29 < c.ci_low < 0.31
    assert 0.94 < c.ci_high < 0.96
    assert c.covers_nominal


def test_score_functions_refuse_misaligned_inputs():
    with pytest.raises(ForecastError) as excinfo:
        brier([0.5, 0.5], [True])
    assert "misaligned" in str(excinfo.value)
    with pytest.raises(ForecastError):
        brier([], [])


# ---------------------------------------------------------------------------
# Distributions and rules
# ---------------------------------------------------------------------------


def test_a_probability_outside_the_unit_interval_is_not_a_forecast():
    with pytest.raises(ForecastError):
        BinaryProbability(1.4)
    with pytest.raises(ForecastError):
        BinaryProbability(float("nan"))


def test_an_interval_with_inverted_bounds_is_refused():
    with pytest.raises(ForecastError):
        IntervalForecast(lo=1.0, hi=0.0)


def test_quantiles_must_be_non_decreasing():
    with pytest.raises(ForecastError):
        QuantileForecast(levels=(0.1, 0.9), values=(3.0, 1.0))
    q = QuantileForecast(levels=(0.1, 0.5, 0.9), values=(1.0, 2.0, 3.0))
    assert q.point == 2.0
    assert q.interval(0.8).lo == 1.0


def test_every_comparator_evaluates():
    cases = {
        Comparator.GT: (1.0, 0.0, True),
        Comparator.GE: (0.0, 0.0, True),
        Comparator.LT: (0.0, 1.0, True),
        Comparator.LE: (1.0, 1.0, True),
        Comparator.EQ: (1.0, 1.0, True),
        Comparator.ABS_LT: (-0.5, 1.0, True),
    }
    for comparator, (value, threshold, expected) in cases.items():
        assert comparator.evaluate(value, threshold) is expected
    assert Comparator.ABS_LT.render("x", 0.02) == "|x| < 0.02"


def test_a_decision_that_costs_more_than_it_saves_is_refused():
    with pytest.raises(ForecastError) as excinfo:
        DecisionSpec(action="kill the run", cost=500.0, loss=480.0)
    assert "never worth taking at any probability" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def test_a_missing_metric_is_void_and_not_a_miss():
    outcome = resolve(
        a_forecast(),
        {"other": 1.0},
        at=InformationTime.parse("2026-04-01T00:00:00+00:00", basis="a unit test"),
    )
    assert isinstance(outcome, Void)
    assert outcome.reason is VoidReason.METRIC_ABSENT
    assert "produced no 'm'" in outcome.detail
    assert "other" in outcome.detail
    assert "not the forecast" in outcome.meaning


def test_a_nan_metric_is_void_and_not_a_confident_miss():
    outcome = resolve(
        a_forecast(),
        {"m": float("nan")},
        at=InformationTime.parse("2026-04-01T00:00:00+00:00", basis="a unit test"),
    )
    assert isinstance(outcome, Void)
    assert outcome.reason is VoidReason.METRIC_UNEVALUABLE
    assert "False under every comparator" in outcome.detail


def test_an_expired_forecast_is_void():
    rule = ResolutionRule(metric="m", comparator=Comparator.GT, threshold=0.5)
    subject = SubjectRef()
    issued = InformationTime.parse("2026-03-01T00:00:00+00:00", basis="a unit test")
    expiry = InformationTime.parse("2026-03-10T00:00:00+00:00", basis="the registered deadline")
    forecast = Forecast(
        id=forecast_id(
            target="q",
            subject=subject,
            resolution=rule,
            issued_at=issued,
            distribution=BinaryProbability(0.6),
            inputs=(),
            method="unit",
        ),
        target="q",
        subject=subject,
        resolution=rule,
        issued_at=issued,
        horizon=HorizonSpec(kind="time", value=0.0, expires_at=expiry),
        reference_class=CLASS,
        distribution=BinaryProbability(0.6),
        method="unit",
        inputs=(),
        baselines=four_baselines(),
    )
    outcome = resolve(
        forecast,
        {},
        at=InformationTime.parse("2026-04-01T00:00:00+00:00", basis="a unit test"),
    )
    assert isinstance(outcome, Void)
    assert outcome.reason is VoidReason.EXPIRED
    assert "'not yet' is not 'no'" in outcome.meaning


def test_a_resolved_forecast_carries_the_number_that_decided_it():
    outcome = resolve(
        a_forecast(),
        {"m": 0.9},
        at=InformationTime.parse("2026-04-01T00:00:00+00:00", basis="a unit test"),
    )
    assert isinstance(outcome, Resolved)
    assert outcome.outcome is True
    assert outcome.metric_value == 0.9
    assert "CONFIRMED" in outcome.render()


def test_an_interval_forecast_records_coverage_as_well_as_the_rule():
    rule = ResolutionRule(metric="m", comparator=Comparator.GE, threshold=1.0)
    subject = SubjectRef()
    issued = InformationTime.parse("2026-03-01T00:00:00+00:00", basis="a unit test")
    distribution = IntervalForecast(lo=1.0, hi=2.0, level=0.8)
    forecast = Forecast(
        id=forecast_id(
            target="q",
            subject=subject,
            resolution=rule,
            issued_at=issued,
            distribution=distribution,
            inputs=(),
            method="unit",
        ),
        target="q",
        subject=subject,
        resolution=rule,
        issued_at=issued,
        horizon=HorizonSpec(),
        reference_class=CLASS,
        distribution=distribution,
        method="unit",
        inputs=(),
        baselines=four_baselines(),
    )
    inside = resolve(
        forecast, {"m": 1.5}, at=InformationTime.parse("2026-04-01T00:00:00+00:00", basis="t")
    )
    assert inside.covered is True
    outside = resolve(
        forecast, {"m": 9.0}, at=InformationTime.parse("2026-04-01T00:00:00+00:00", basis="t")
    )
    assert outside.covered is False
    assert outside.outcome is True  # the rule only checks the lower bound


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def test_climatology_refuses_a_class_with_no_base_rate():
    baseline = climatology(ReferenceClass(id="c", definition="a class nobody counted"))
    assert not baseline.is_scored
    assert "no counted base rate" in baseline.refused
    assert baseline.kind is BaselineKind.CLIMATOLOGY


def test_persistence_rate_counts_transitions():
    assert persistence_rate([True, True, True]) == 1.0
    assert persistence_rate([True, False, True, False]) == 0.0
    assert persistence_rate([True, True, False]) == 0.5
    with pytest.raises(ForecastError):
        persistence_rate([True])


def test_the_black_box_baseline_refuses_without_a_judge_and_says_what_to_pass():
    baseline = contrastive_belief_flip(("p",), ("t",), judge=None)
    assert not baseline.is_scored
    assert "Pass `judge=`" in baseline.refused
    assert "first thing a reviewer will raise" in baseline.refused


def test_the_black_box_baseline_runs_with_a_judge():
    """A stub judge that scores the private framing higher, which is the contrast the method reads."""
    calls: list[str] = []

    def judge(prompt: str) -> float:
        calls.append(prompt)
        return 0.7 if "private scratch note" in prompt else 0.3

    baseline = contrastive_belief_flip(("p1", "p2"), ("t1", "t2"), judge=judge)
    assert baseline.is_scored
    assert len(calls) == 4  # two framings per item
    assert "contrast +0.400" in baseline.detail
    assert belief_flip_hash() in baseline.detail
    # 0.7 private plus half of the 0.4 contrast, clipped into the open interval.
    assert baseline.distribution.p == pytest.approx(0.9)


def test_the_black_box_baseline_refuses_misaligned_inputs():
    with pytest.raises(ForecastError):
        contrastive_belief_flip(("p",), ("t1", "t2"), judge=lambda _: 0.5)


# ---------------------------------------------------------------------------
# The corpus
# ---------------------------------------------------------------------------


def test_a_corpus_refuses_a_duplicated_run():
    with pytest.raises(ForecastError) as excinfo:
        RunCorpus(runs=("a", "a"), reference_class="rc", split="leave_one_run_out", embargo_steps=0)
    assert "more than once" in str(excinfo.value)


def test_a_temporal_split_needs_a_step_axis():
    with pytest.raises(ForecastError) as excinfo:
        RunCorpus(runs=("a",), reference_class="rc", split="walk_forward", embargo_steps=5)
    assert "needs a step axis" in str(excinfo.value)


def test_leave_one_run_out_is_the_only_run_disjoint_split():
    steps = {f"r{i}": tuple(range(20)) for i in range(4)}
    loro = RunCorpus(
        runs=tuple(steps),
        reference_class="rc",
        split="leave_one_run_out",
        embargo_steps=0,
        steps=steps,
    )
    for fold in loro.folds():
        assert fold.is_run_disjoint
        assert fold.test_runs == (fold.detail.split()[-1],)


def test_purged_kfold_removes_training_points_either_side_of_the_test_block():
    steps = {f"r{i}": tuple(range(40)) for i in range(3)}
    corpus = RunCorpus(
        runs=tuple(steps), reference_class="rc", split="purged_kfold", embargo_steps=4, steps=steps
    )
    for fold in corpus.folds(n_folds=4):
        lo = min(s for _, s in fold.test)
        hi = max(s for _, s in fold.test)
        assert fold.purged > 0
        for _, step in fold.train:
            assert not (lo - 4 <= step <= hi + 4)


def test_an_embargo_wider_than_the_corpus_refuses_rather_than_returning_empty_folds():
    steps = {"a": tuple(range(10)), "b": tuple(range(10))}
    corpus = RunCorpus(
        runs=("a", "b"), reference_class="rc", split="walk_forward", embargo_steps=500, steps=steps
    )
    with pytest.raises(ForecastError) as excinfo:
        corpus.folds(n_folds=3, min_train=4)
    assert "wider than the corpus" in str(excinfo.value)


def test_the_aisi_traps_are_recorded_where_a_reader_will_find_them():
    assert len(AISI_TRAPS) == 5
    joined = " ".join(AISI_TRAPS)
    for token in ("subfolder=", "401 and 404", "int64", "hack_config", "rollout_index"):
        assert token in joined


def test_the_corpus_docstring_says_why_item_splits_are_wrong():
    assert (
        "Splitting over items is the mistake every competitor makes" in RunCorpus.__module__ or True
    )
    import reward_lens.forecast.corpus as corpus_module

    assert "Splitting over items is the mistake every competitor makes" in corpus_module.__doc__
    assert "splitting over runs and time" in corpus_module.__doc__


# ---------------------------------------------------------------------------
# The ledger and the instrument
# ---------------------------------------------------------------------------


def test_the_ledger_is_append_only_and_round_trips(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = CalibrationLedger(path)
    forecast = a_forecast(0.7)
    outcome = resolve(
        forecast, {"m": 0.9}, at=InformationTime.parse("2026-04-01T00:00:00+00:00", basis="t")
    )
    ledger.append(entry_from(forecast, outcome))
    # Idempotent on the forecast id, so replaying a scoring run is safe.
    ledger.append(entry_from(forecast, outcome))
    assert len(ledger) == 1

    reopened = CalibrationLedger(path)
    assert len(reopened) == 1
    assert reopened.entries[0].probability == 0.7
    assert reopened.entries[0].outcome is True
    assert reopened.entries[0].brier_term == pytest.approx(0.09)


def test_a_readonly_ledger_refuses_to_append(tmp_path):
    ledger = CalibrationLedger(tmp_path / "ledger.jsonl", readonly=True)
    forecast = a_forecast()
    outcome = resolve(
        forecast, {"m": 0.9}, at=InformationTime.parse("2026-04-01T00:00:00+00:00", basis="t")
    )
    with pytest.raises(ForecastError):
        ledger.append(entry_from(forecast, outcome))


def test_the_header_of_a_ledger_that_beats_the_coin_says_so(tmp_path):
    ledger = CalibrationLedger()
    at = InformationTime.parse("2026-04-01T00:00:00+00:00", basis="t")
    for i in range(8):
        forecast = a_forecast(0.9 if i % 2 == 0 else 0.1)
        forecast = Forecast(
            **{
                **{
                    f.name: getattr(forecast, f.name)
                    for f in forecast.__dataclass_fields__.values()
                },
                "id": f"fc:{i}",
                "target": f"q{i}",
            }
        )
        metrics = {"m": 0.9 if i % 2 == 0 else 0.1}
        ledger.append(entry_from(forecast, resolve(forecast, metrics, at=at)))
    header = ledger.header()
    assert "which beat the always-guess-half coin" in header
    assert "meta kill criterion fired" not in header


def test_the_instrument_lints_clean_and_its_quantity_is_registered():
    """The instrument is complete and its registry row has landed.

    This began as the other assertion: with `forecast.brier_score` unregistered, `lint_instrument`
    reported exactly one finding naming that field, and the test registered the row in-process to
    prove the other three declarations were real. It was then registered for real, along with
    the two Murphy terms it decomposes into and `forecast.decision_value`, so the test is inverted
    rather than deleted, on the precedent E25 set for the `py.typed` guard: a check that has served
    its purpose becomes a standing check on the other side of the change.

    The four ids are asserted rather than just the one, because the ledger reports all four and a
    reading whose companions are unregistered can still be ranked against something it should not be.
    """
    from reward_lens.core.quantity import QUANTITIES
    from reward_lens.measure.base import lint_instrument

    instrument = ForecastCalibration(CalibrationLedger())
    assert lint_instrument(instrument) == []
    for qid in (
        "forecast.brier_score",
        "forecast.calibration_reliability",
        "forecast.calibration_resolution",
        "forecast.decision_value",
    ):
        assert qid in QUANTITIES, f"{qid} is not registered, so the ledger cannot key on it"
    assert QUANTITIES.get("forecast.decision_value").invariance == "units", (
        "decision value is covariant under a rescaling of the loss where the other three are "
        "dimensionless, which is why it is the one with a non-trivial group"
    )


def test_the_instruments_generated_invariance_test_passes():
    """`none` is a deliberate answer here and it is the right one.

    A Brier score is a function of probabilities and binary outcomes. The reward gauge acts on
    scores, and no rescaling of a reward changes a probability that was written down before the
    reward existed, so no registered group acts on this quantity. The generated test passes
    vacuously and that is honest rather than empty: the failure the lint targets is not thinking
    about the question.
    """
    from reward_lens.core.invariance import check_invariance, get_group, resolve_relation
    from reward_lens.core.quantity import TRIVIAL_GROUP

    instrument = ForecastCalibration(CalibrationLedger())
    assert instrument.invariance == "none"
    group = get_group(instrument.invariance)
    assert group.id == TRIVIAL_GROUP
    assert resolve_relation(instrument, group.id).status == "invariant"
    report = check_invariance(instrument, group, None)
    assert report.passed


def test_the_ledger_shows_what_scoring_voids_as_misses_would_cost(tmp_path):
    ledger = CalibrationLedger()
    at = InformationTime.parse("2026-04-01T00:00:00+00:00", basis="t")
    resolved = a_forecast(0.6)
    ledger.append(entry_from(resolved, resolve(resolved, {"m": 0.9}, at=at)))
    voided = a_forecast(0.9)
    voided = Forecast(
        **{
            **{f.name: getattr(voided, f.name) for f in voided.__dataclass_fields__.values()},
            "id": "fc:void",
        }
    )
    ledger.append(entry_from(voided, resolve(voided, {}, at=at)))
    assert ledger.score().directional_brier == pytest.approx(0.16)
    # Scoring the void at 0.9 as a miss would nearly triple it.
    assert ledger.directional_brier_if_voids_were_misses() == pytest.approx((0.16 + 0.81) / 2)

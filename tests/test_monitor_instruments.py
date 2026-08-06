"""Unit, property, invariance and refusal tests for the seven monitor instruments.

Four kinds of test, and the fourth is the one that matters most here. A refusal test asserts the
reason **and** the remedy string, because a remedy is a user interface and a refusal whose remedy
says "envelope violated" is a tool that looks broken rather than a tool that looks careful.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from reward_lens.core.envelope import RegimeCondition, RegimeReading
from reward_lens.core.invariance import (
    check_invariance,
    check_unit_refusal,
    parse_group_field,
    resolve_relation,
)
from reward_lens.core.quantity import QUANTITIES, load_quantities
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.types import Access, Component, Phase, Substrate
from reward_lens.measure.base import Context, declared_access, lint_instrument
from reward_lens.monitor import (
    MONITOR,
    AlarmDesign,
    Channel,
    CheckStandardDriftInstrument,
    ConfidenceSequence,
    ConjunctionDetector,
    DetectionDelay,
    EValue,
    OperatingPointFromLoss,
    Session,
    channels_of,
    check_standard_drift,
    design_cusum,
    gaussian_lr_eprocess,
    local_transition_width,
    npv,
    ppv,
    ppv_curve,
    run_cusum,
    standardize,
)
from reward_lens.monitor.cusum import TransitionWindow, _logistic_ramp


@pytest.fixture(autouse=True, scope="module")
def _quantities() -> None:
    load_quantities()


def _channel(values, *, name="c", direction=1) -> Channel:
    values = np.asarray(values, dtype=np.float64)
    return Channel(
        name=name,
        values=values,
        steps=np.arange(values.size),
        direction=direction,
        source="test",
    )


def _instruments() -> list:
    xs = np.array([0.0, 1.0] * 12)
    ramp = _logistic_ramp(40, 60, 15.0, 3.0)
    return [
        ConfidenceSequence(xs),
        EValue(xs, mean_null=0.5),
        AlarmDesign(),
        DetectionDelay(_channel(ramp)),
        ConjunctionDetector([_channel(ramp, name="a"), _channel(ramp * 1.1, name="b")]),
        OperatingPointFromLoss(np.arange(20.0), [0] * 10 + [1] * 10),
        CheckStandardDriftInstrument(
            [Session(str(i), {"p": float(i), "q": 1.0}) for i in range(6)]
        ),
    ]


# ---------------------------------------------------------------------------
# Declarations and lint
# ---------------------------------------------------------------------------


def test_every_instrument_lints_clean() -> None:
    for inst in _instruments():
        assert lint_instrument(inst) == [], inst.name


def test_every_declared_quantity_is_registered_and_its_unit_is_the_catalogue_unit() -> None:
    expected = {
        "monitor.confidence_sequence": "interval",
        "monitor.evalue": "e-value",
        "monitor.arl0": "steps",
        "monitor.detection_delay": "frac window",
        "monitor.conjunction_alarm": "bool",
        "monitor.operating_point": "threshold",
        "monitor.check_standard_drift": "1",
    }
    for cls in MONITOR:
        q = QUANTITIES.get(cls.quantity)
        assert q.unit.as_printed == expected[cls.quantity], cls.__name__


def test_the_access_matrix_is_what_each_instrument_really_needs() -> None:
    """Three read a record and two read nothing. Claiming access you do not need is also a defect.

    An instrument declaring RECORD access it does not use makes the capability report wrong in the
    direction that withholds a reading from somebody who could have had one.
    """
    reads_record = {
        "ConfidenceSequence",
        "EValue",
        "DetectionDelay",
        "ConjunctionDetector",
        "CheckStandardDrift",
    }
    for inst in _instruments():
        matrix = declared_access(inst)
        if inst.name in reads_record:
            assert matrix == {Component.RECORD: Access.RECORD}, inst.name
        else:
            assert matrix == {}, inst.name


def test_no_instrument_answers_a_pre_run_or_deployed_question() -> None:
    for inst in _instruments():
        if inst.name in ("AlarmDesign", "OperatingPointFromLoss"):
            continue
        assert Phase.PRE_RUN not in inst.phases, inst.name
        assert Phase.DEPLOYED not in inst.phases, inst.name


def test_the_envelope_of_the_four_record_readers_requires_a_stationary_grader() -> None:
    """And J5 does not, because J5 is the measurement that establishes it."""
    for inst in _instruments():
        if inst.name in ("ConfidenceSequence", "EValue", "DetectionDelay", "ConjunctionDetector"):
            assert RegimeCondition.STATIONARY_GRADER in inst.envelope.requires, inst.name
            assert RegimeCondition.NO_COMPACTION in inst.envelope.requires, inst.name
        if inst.name == "CheckStandardDrift":
            assert inst.envelope.requires == frozenset()
            assert inst.envelope.unconditional and inst.envelope.justification


def test_the_envelope_refuses_when_the_grader_moved() -> None:
    """The confound a change detector cannot see past, refused with a remedy."""
    ctx = Context(regime_reading=RegimeReading.of(STATIONARY_GRADER=False, NO_COMPACTION=True))
    reading = ConfidenceSequence(np.array([0.0, 1.0] * 12)).estimate(ctx)
    assert isinstance(reading, Refusal)
    assert reading.reason is RefusalReason.ENVELOPE_VIOLATED
    assert "STATIONARY_GRADER" in reading.detail
    assert "restrict the window" in reading.remedy.lower()


def test_a_substrate_and_phase_mismatch_refuse_by_name() -> None:
    ctx = Context(phase=Phase.PRE_RUN, substrate=Substrate.PROGRAM)
    reading = ConfidenceSequence(np.array([0.0, 1.0] * 12)).estimate(ctx)
    assert isinstance(reading, Refusal)
    assert reading.reason is RefusalReason.PHASE_MISMATCH
    assert reading.remedy.strip()


# ---------------------------------------------------------------------------
# The generated invariance test
# ---------------------------------------------------------------------------


def test_every_instrument_names_a_resolvable_invariance_group() -> None:
    for inst in _instruments():
        groups = parse_group_field(inst.invariance)
        assert groups, f"{inst.name} declares {inst.invariance!r}, which names no group"


def test_the_units_group_generates_a_refusal_check_and_not_a_value_check() -> None:
    """Six of the seven declare `units`, whose assertion is a refusal rather than a relation.

    `check_invariance` reports that as a skip with the reason, which is honest: there is no value
    comparison to make. The real check is `check_unit_refusal`, below.
    """
    from reward_lens.core.invariance import InvariancePayload

    payload = InvariancePayload(unit="nats/token")
    for inst in _instruments():
        if inst.invariance != "units":
            continue
        report = check_invariance(
            inst, "units", payload, relation=resolve_relation(inst, "units"), run=lambda _i, _p: 1.0
        )
        assert report.passed
        assert report.skipped, f"{inst.name} claims a units check it cannot have run"


def test_comparing_a_monitor_reading_across_a_unit_boundary_refuses() -> None:
    """The `units` group's real assertion, exercised on this package's own quantities.

    `monitor.detection_delay` is dimensionless per transition window and `monitor.arl0` is a count
    per step. Ranking one against the other is the silent error the whole unit system exists to
    stop, so the comparison must refuse rather than return a number.
    """

    def compare(a, b):
        qa, qb = QUANTITIES.get(a), QUANTITIES.get(b)
        if not qa.unit.compatible_with(qb.unit):
            return Refusal(
                instrument="compare",
                reason=RefusalReason.UNIT_MISMATCH,
                detail=f"{a} is {qa.unit} and {b} is {qb.unit}",
                remedy="convert explicitly with a stated conversion, or compare like with like.",
            )
        return 0.0

    assert check_unit_refusal(compare, "monitor.detection_delay", "monitor.arl0")
    assert check_unit_refusal(compare, "monitor.check_standard_drift", "monitor.arl0")
    # And two quantities in the same unit do compare, so the check above is not vacuous.
    assert compare("monitor.detection_delay", "monitor.detection_delay") == 0.0


def test_the_check_standard_is_invariant_under_a_change_of_representation_basis() -> None:
    """J5 declares `repr.basis`. The check is weak and it is an honest pass, not a vacuous one.

    A drift computed from scalar probe scores cannot move under an orthogonal map on a
    representation, because no representation enters the computation. The generated test says so by
    passing; what would falsify it is a probe whose value is read off an activation direction.
    """
    from reward_lens.core.invariance import InvariancePayload

    sessions = [Session(str(i), {"p": 0.5 + 0.001 * i}) for i in range(6)]
    inst = CheckStandardDriftInstrument(sessions)
    drift = check_standard_drift(sessions)
    assert not isinstance(drift, Refusal)
    payload = InvariancePayload(activations=np.eye(4), readouts=np.eye(4)[:2])
    report = check_invariance(
        inst,
        "repr.basis",
        payload,
        n=16,
        relation=resolve_relation(inst, "repr.basis"),
        run=lambda _i, _p: drift.drift,
    )
    assert report.passed, report.render()


# ---------------------------------------------------------------------------
# Hand-computed values
# ---------------------------------------------------------------------------


def test_standardize_is_a_z_score_against_the_baseline_window() -> None:
    z = standardize([1.0, 3.0, 5.0, 100.0], baseline=3)
    # mean 3, population sd sqrt(8/3) = 1.63299
    assert z[0] == pytest.approx((1 - 3) / math.sqrt(8 / 3))
    assert z[1] == pytest.approx(0.0)


def test_the_cusum_accumulator_is_hand_computable() -> None:
    """`C+ = max(0, C+ + z - k)` with k = 0.5 on z = [1, 1, 1] gives 0.5, 1.0, 1.5."""
    design = design_cusum(1.0, 370.0)
    run = run_cusum(np.array([1.0, 1.0, 1.0]), design, standardized=True)
    assert np.allclose(run.upper, [0.5, 1.0, 1.5])
    assert np.allclose(run.lower, [0.0, 0.0, 0.0])
    assert run.alarm_at is None


def test_the_cusum_fires_exactly_when_the_accumulator_passes_h() -> None:
    design = design_cusum(1.0, 370.0)  # k = 0.5, h = 4.766
    z = np.full(20, 1.0)
    run = run_cusum(z, design, standardized=True)
    # The accumulator grows by 0.5 each step, so it passes 4.766 at index 9 (value 5.0).
    assert run.alarm_at == 9


def test_the_gaussian_lr_eprocess_is_hand_computable() -> None:
    """`exp(shift * S_t - t shift^2/2)` at shift 1, z = [1, 1]: e^0.5 then e^1."""
    e = gaussian_lr_eprocess([1.0, 1.0], 1.0)
    assert e[0] == pytest.approx(math.exp(0.5))
    assert e[1] == pytest.approx(math.exp(1.0))


def test_the_gaussian_lr_eprocess_has_expectation_one_under_the_null() -> None:
    """Which is what makes it an e-value. Simulated, because the identity is the whole licence."""
    rng = np.random.default_rng(31)
    z = rng.standard_normal((40000, 5))
    e = np.array([gaussian_lr_eprocess(row, 1.0)[-1] for row in z])
    assert abs(e.mean() - 1.0) < 0.05, e.mean()


def test_ppv_and_npv_by_hand() -> None:
    assert ppv(1.0, 0.317, 0.025) == pytest.approx(0.025 / (0.025 + 0.317 * 0.975))
    assert npv(0.8, 0.1, 0.5) == pytest.approx((0.9 * 0.5) / (0.9 * 0.5 + 0.2 * 0.5))
    assert math.isnan(ppv(0.0, 0.0, 0.1)), "a detector that never fires has no PPV"


def test_the_ppv_curve_inverts_to_the_base_rate_it_would_need() -> None:
    curve = ppv_curve(0.9, 0.05, target_ppv=0.5)
    pi = curve.prevalence_for_ppv
    assert curve.at(pi) == pytest.approx(0.5, abs=1e-9)


def test_the_logistic_width_recovers_a_planted_one() -> None:
    y = _logistic_ramp(50, 50, 25.0, 4.0)
    window = local_transition_width(y)
    assert not isinstance(window, Refusal)
    assert window.width_steps == pytest.approx(25.0, rel=0.02)
    assert window.midpoint_step == pytest.approx(50.0, abs=0.5)
    assert window.onset_step == pytest.approx(50.0 - 12.5, abs=0.5)
    assert window.source == "local"


def test_the_window_adapts_h4s_fit_without_importing_it() -> None:
    """The integration wiring, checked against H4's real object when it is present.

    Read structurally, so this passes on a tree where `measure.rate.transition` has not landed and
    checks the real thing where it has.
    """
    import types

    duck = types.SimpleNamespace(
        width=18.0, midpoint=60.0, quality=types.SimpleNamespace(r2=0.93), usable=True
    )
    window = TransitionWindow.from_fit(duck)
    assert window.source == "H4"
    assert window.width_steps == 18.0 and window.midpoint_step == 60.0
    assert window.onset_step == 51.0 and window.completion_step == 69.0
    assert window.r_squared == 0.93

    transition = pytest.importorskip(
        "reward_lens.measure.rate.transition", reason="H4 has not landed on this tree"
    )
    t = np.arange(200, dtype=float)
    y = 1.0 / (1.0 + np.exp(-(t - 100.0) * (2 * math.log(9.0) / 25.0)))
    y = y + np.random.default_rng(0).normal(0, 0.02, t.size)
    fit = transition.fit_transition(y)
    if isinstance(fit, Refusal):
        pytest.skip(f"H4 declined to fit the probe series: {fit.reason.name}")
    window = TransitionWindow.from_fit(fit)
    assert window.source == "H4"
    assert window.width_steps == pytest.approx(25.0, rel=0.15)
    assert window.midpoint_step == pytest.approx(100.0, abs=3.0)
    # And the two sign conventions are reconcilable, which is the thing to check at a seam.
    lead = transition.lead_time(80.0, fit)
    assert window.delay_from_onset(80.0) == pytest.approx(-lead.widths_from_onset * fit.width)


def test_the_transition_window_puts_a_delay_in_window_units() -> None:
    w = TransitionWindow(width_steps=20.0, midpoint_step=100.0, source="planted")
    assert w.onset_step == 90.0 and w.completion_step == 110.0
    assert w.delay_from_onset(100.0) == 10.0
    assert w.fraction(10.0) == 0.5
    assert w.lead_to_midpoint(80.0) == 20.0


def test_the_check_standard_drift_is_hand_computable() -> None:
    """One probe, baseline values 1, 2, 3 (mean 2, sample sd 1), current value 5 gives z = 3."""
    sessions = [
        Session("a", {"p": 1.0}),
        Session("b", {"p": 2.0}),
        Session("c", {"p": 3.0}),
        Session("d", {"p": 5.0}),
    ]
    drift = check_standard_drift(sessions, n_baseline=3)
    assert not isinstance(drift, Refusal)
    assert drift.drift == pytest.approx(3.0)
    assert drift.raw_drift == pytest.approx(3.0)
    assert drift.n_baseline == 3


def test_the_probe_set_fingerprint_changes_when_a_probe_is_added() -> None:
    from reward_lens.monitor import probe_set_fingerprint

    assert probe_set_fingerprint(["a", "b"]) == probe_set_fingerprint(["b", "a"])
    assert probe_set_fingerprint(["a", "b"]) != probe_set_fingerprint(["a", "b", "c"])


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(
    st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=8, max_size=60),
    st.sampled_from([0, 1]),
)
def test_a_confidence_sequence_always_contains_the_running_mean(xs: list[float], rung: int) -> None:
    from reward_lens.monitor import confidence_sequence

    trace = confidence_sequence(xs, rung=rung)
    assert np.all(trace.lower <= trace.centre + 1e-12)
    assert np.all(trace.upper >= trace.centre - 1e-12)
    assert np.all(trace.lower >= 0.0) and np.all(trace.upper <= 1.0)


@settings(max_examples=100, deadline=None)
@given(st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=8, max_size=60))
def test_the_anytime_interval_is_never_narrower_than_the_fixed_sample_one(xs: list[float]) -> None:
    from reward_lens.monitor import confidence_sequence

    trace = confidence_sequence(xs, rung=0)
    widths = trace.upper - trace.lower
    fixed = trace.fixed_upper - trace.fixed_lower
    assert np.all(widths >= fixed - 1e-12)


@settings(max_examples=60, deadline=None)
@given(
    st.lists(st.floats(min_value=-5.0, max_value=5.0), min_size=10, max_size=80),
    st.floats(min_value=0.5, max_value=2.0),
)
def test_the_cusum_alarm_is_monotone_in_the_threshold(zs: list[float], shift: float) -> None:
    """A quieter chart never alarms earlier. If it does, the accumulator is wrong."""
    low = design_cusum(shift, 50.0)
    high = design_cusum(shift, 5000.0)
    a = run_cusum(np.asarray(zs), low, standardized=True).alarm_at
    b = run_cusum(np.asarray(zs), high, standardized=True).alarm_at
    if b is not None:
        assert a is not None and a <= b


@settings(max_examples=60, deadline=None)
@given(st.floats(min_value=1e-4, max_value=0.5), st.floats(min_value=1e-4, max_value=0.99))
def test_ppv_rises_with_prevalence_and_falls_with_the_false_positive_rate(
    prevalence: float, fpr: float
) -> None:
    base = ppv(0.9, fpr, prevalence)
    assert ppv(0.9, fpr, min(0.99, prevalence * 1.5)) >= base
    assert ppv(0.9, min(0.999, fpr * 1.5), prevalence) <= base


# ---------------------------------------------------------------------------
# Refusals: the reason and the remedy, not just that something came back
# ---------------------------------------------------------------------------


def test_an_unbounded_stream_refuses_with_unit_mismatch_and_names_the_range() -> None:
    reading = ConfidenceSequence(np.array([0.0, 5.0] * 8)).estimate(Context())
    assert isinstance(reading, Refusal)
    assert reading.reason is RefusalReason.UNIT_MISMATCH
    assert "0, 1" in reading.detail or "[0, 1]" in reading.remedy
    assert "observed range is not a fix" in reading.remedy
    assert reading.statistics["max"] == 5.0


def test_too_few_observations_refuses_with_a_bound_rather_than_a_trivial_interval() -> None:
    reading = ConfidenceSequence(np.array([1.0, 0.0, 1.0])).estimate(Context())
    assert isinstance(reading, Refusal)
    assert reading.reason is RefusalReason.ABOVE_LOD_BELOW_LOQ
    assert "point estimate" in reading.remedy
    assert reading.statistics["n"] == 3


def test_an_evalue_with_no_stated_null_refuses_and_says_why_there_is_no_default() -> None:
    reading = EValue(np.array([0.0, 1.0] * 8)).estimate(Context())
    assert isinstance(reading, Refusal)
    assert reading.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert "mean_null" in reading.remedy
    assert "a default would be this" in reading.remedy


def test_a_conjunction_over_a_named_channel_the_record_lacks_refuses() -> None:
    from reward_lens.core.reading import refuse_incomplete
    from reward_lens.monitor._base import ChannelSet

    present = {
        "a": _channel(np.arange(30.0), name="a"),
        "b": _channel(np.arange(30.0) * 2, name="b"),
    }
    absent = {
        "completion_length": refuse_incomplete(
            "test", field="x", subject="y", remedy="log it and re-run"
        )
    }
    cs = ChannelSet(present=present, absent=absent, n_steps=30)
    reading = ConjunctionDetector(cs, wanted=["a", "b", "completion_length"]).estimate(Context())
    assert isinstance(reading, Refusal)
    assert reading.reason is RefusalReason.RECORD_INCOMPLETE
    assert "completion_length" in reading.detail
    assert "not interchangeable" in reading.remedy


def test_a_conjunction_on_too_short_a_window_refuses_and_explains_the_collapse() -> None:
    """The correctness condition, not a convenience: without a baseline every e-value is the same."""
    channels = [_channel(np.arange(10.0), name=n) for n in ("a", "b")]
    reading = ConjunctionDetector(channels).estimate(Context())
    assert isinstance(reading, Refusal)
    assert reading.reason is RefusalReason.ESS_BELOW_FLOOR
    assert "stop depending on the channel" in reading.remedy


def test_a_delay_on_a_series_with_no_transition_refuses_and_hides_no_lead_time() -> None:
    rng = np.random.default_rng(0)
    reading = DetectionDelay(_channel(rng.standard_normal(120))).estimate(Context())
    assert isinstance(reading, Refusal)
    assert reading.reason is RefusalReason.ENVELOPE_VIOLATED
    assert "carries no transition" in reading.detail
    assert "there is no lead time to report" in reading.remedy
    assert "alarm_index" in reading.statistics


def test_a_chart_that_never_fires_refuses_with_the_null_as_a_result() -> None:
    rng = np.random.default_rng(5)
    # A chart designed for one false alarm per million steps, watching sixty steps of noise with a
    # 0.2-sigma ramp in them. The threshold is 14.6 and the accumulator does not come close, which
    # is the branch: a quiet chart that did not fire is a null result and not a failure.
    y = _logistic_ramp(30, 30, 10.0, 0.2) + rng.standard_normal(60)
    window = TransitionWindow(width_steps=10.0, midpoint_step=30.0, source="planted")
    reading = DetectionDelay(_channel(y), arl0=1e6, baseline_steps=25, window=window).estimate(
        Context()
    )
    assert isinstance(reading, Refusal)
    assert reading.reason is RefusalReason.BELOW_LOD
    assert "accept the null" in reading.remedy
    assert reading.statistics["threshold"] > 0


def test_an_operating_point_on_one_class_refuses_and_points_at_the_curve() -> None:
    reading = OperatingPointFromLoss(np.arange(20.0), [1] * 20).estimate(Context())
    assert isinstance(reading, Refusal)
    assert reading.reason is RefusalReason.LABEL_QUALITY_UNKNOWN
    assert "ppv_curve" in reading.remedy


def test_a_check_standard_with_one_session_refuses() -> None:
    reading = CheckStandardDriftInstrument([Session("a", {"p": 1.0})]).estimate(Context())
    assert isinstance(reading, Refusal)
    assert reading.reason is RefusalReason.RECORD_INCOMPLETE
    assert "one occasion carries none" in reading.remedy


def test_a_check_standard_with_too_short_a_baseline_refuses_to_normalise() -> None:
    sessions = [Session(str(i), {"p": float(i)}) for i in range(3)]
    reading = CheckStandardDriftInstrument(sessions).estimate(Context())
    assert isinstance(reading, Refusal)
    assert reading.reason is RefusalReason.ESS_BELOW_FLOOR
    assert "cannot be told from the probe set's own noise" in reading.remedy


def test_a_check_standard_whose_membership_changed_refuses() -> None:
    sessions = [
        Session("a", {"p": 1.0}),
        Session("b", {"q": 1.0}),
    ]
    drift = check_standard_drift(sessions)
    assert isinstance(drift, Refusal)
    assert drift.reason is RefusalReason.RECORD_INCOMPLETE
    assert "is not a check standard" in drift.remedy


def test_an_alarm_design_below_one_observation_refuses() -> None:
    reading = AlarmDesign(arl0=0.5).estimate(Context())
    assert isinstance(reading, Refusal)
    assert reading.reason is RefusalReason.ENVELOPE_VIOLATED
    assert "between 100 and 10,000" in reading.remedy


def test_every_refusal_in_this_package_carries_an_actionable_remedy() -> None:
    """A remedy is an instruction. A restatement of the failure is not one."""
    banned = ("envelope violated", "access insufficient", "not enough data", "invalid input")
    readings = [
        ConfidenceSequence(np.array([0.0, 5.0] * 8)).estimate(Context()),
        EValue(np.array([0.0, 1.0] * 8)).estimate(Context()),
        DetectionDelay(_channel(np.random.default_rng(1).standard_normal(120))).estimate(Context()),
        OperatingPointFromLoss(np.arange(20.0), [1] * 20).estimate(Context()),
        CheckStandardDriftInstrument([Session("a", {"p": 1.0})]).estimate(Context()),
        AlarmDesign(arl0=0.5).estimate(Context()),
    ]
    for reading in readings:
        assert isinstance(reading, Refusal)
        assert len(reading.remedy) > 60, reading
        assert reading.remedy.strip().lower() not in banned
        # A remedy tells you what to do, so it contains a verb in the imperative somewhere near
        # the front. Checking for that exactly is not worth a parser; checking that it is not a
        # bare restatement of the reason is.
        assert reading.reason.name.lower().replace("_", " ") not in reading.remedy.lower()


# ---------------------------------------------------------------------------
# Channels off a record shape
# ---------------------------------------------------------------------------


def test_a_constant_channel_is_refused_rather_than_silently_watched() -> None:
    """A change detector on a constant series cannot fire, and a quiet chart there means nothing."""

    class _FakeStream:
        def __init__(self, steps):
            self._steps = steps

        def __iter__(self):
            return iter(self._steps)

        def __len__(self):
            return len(self._steps)

    class _Opt:
        def __init__(self, entropy):
            self.entropy = entropy
            self.grad_norm_clipped = None
            self.kl_to_ref = None
            self.kl_to_previous = None
            self.extra: dict[str, float] = {}

    class _Step:
        def __init__(self, index, entropy):
            self.index = index
            self.optimizer = _Opt(entropy)
            self.groups = ()

    class _Run:
        def __init__(self, steps):
            self.steps = _FakeStream(steps)

    run = _Run([_Step(i, 1.0) for i in range(10)])
    channels = channels_of(run, instrument="test", wanted=["entropy"])
    assert "entropy" not in channels.present
    refusal = channels.absent["entropy"]
    assert refusal.reason is RefusalReason.RECORD_INCOMPLETE
    assert "constant at 1" in refusal.remedy
    assert refusal.statistics["constant_at"] == 1.0

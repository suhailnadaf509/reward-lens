"""The Observable runner is the enforcement point for the three gates.

These tests are the executable definition of "the gates are enforced in the kernel, not by
convention" (R5). A measurement that requires a capability the signal does not declare is refused
before any work (R3); a covariant cross-signal comparison without a frame is refused (gate 2); an
ad hoc number with no scorecard is EXPLORATORY and the same number under a scorecard provider is
CALIBRATED (gate 1); a number produced under a frozen study is REGISTERED (gate 3). The signal
here is a minimal stand-in exposing only ``caps`` and ``meta.fingerprint``, which is all the
runner and Context touch, so the test isolates the gate logic from any model.
"""

from __future__ import annotations

import pytest

from reward_lens.core.errors import CapabilityError, GaugeError
from reward_lens.core.gates import CalibrationRef
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.types import Capability, GaugeStatus, TrustLevel
from reward_lens.measure import base as mb


class _Meta:
    fingerprint = "mfp:test"


class _FakeSignal:
    caps = Capability.SCORES | Capability.ACTIVATIONS
    meta = _Meta()


@pytest.fixture(autouse=True)
def _reset_provider():
    mb.set_calibration_provider(lambda name, subj, regime: None)
    yield
    mb.set_calibration_provider(lambda name, subj, regime: None)


def test_capability_gate_refuses_before_work():
    class NeedsGrad(mb.BaseObservable):
        name = "NeedsGrad"
        capabilities = Capability.GRADIENTS

        def measure(self, ctx):
            return ctx.emit(1.0)

    with pytest.raises(CapabilityError):
        mb.run(NeedsGrad(), mb.Context(signal=_FakeSignal()))


def test_estimate_turns_the_capability_gate_into_a_refusal_with_a_remedy():
    """The same gate through the `estimate` entry point, which returns rather than raises.

    `run` is the 2.0.1 `Observable` path, typed `-> Evidence`, and raising there is right for a
    caller who made a programming error. `estimate` is typed `-> Reading`, and a refusal is a value
    with a remedy and never an exception. A missing capability is the commonest thing a capability
    report exists to warn about in advance, so it is anticipated by definition.
    """

    class NeedsGrad(mb.BaseObservable):
        name = "NeedsGrad"
        capabilities = Capability.GRADIENTS

        def measure(self, ctx):
            return ctx.emit(1.0)

    reading = NeedsGrad().estimate(mb.Context(signal=_FakeSignal()))

    assert isinstance(reading, Refusal)
    assert reading.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert reading.instrument == "NeedsGrad"
    assert "GRADIENTS" in reading.statistics["missing"]
    # The remedy is an instruction, not a restatement of the failure.
    assert "Supply a signal offering GRADIENTS" in reading.remedy
    assert "lower rung" in reading.remedy


def test_an_instrument_with_no_signal_at_all_refuses_and_says_what_to_declare():
    class NeedsGrad(mb.BaseObservable):
        name = "NeedsGrad"
        capabilities = Capability.GRADIENTS

        def measure(self, ctx):
            return ctx.emit(1.0)

    reading = NeedsGrad().estimate(mb.Context())

    assert isinstance(reading, Refusal)
    assert reading.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert "declare Capability.NONE instead" in reading.remedy
    assert "a program has no activations" in reading.remedy


def test_gauge_gate_refuses_covariant_comparison_without_frame():
    class Angle(mb.BaseObservable):
        name = "Angle"
        gauge_status = GaugeStatus.COVARIANT

        def measure(self, ctx):
            return ctx.emit(0.9)

    with pytest.raises(GaugeError):
        mb.run(Angle(), mb.Context(signal=_FakeSignal(), is_comparison=True, frame=None))


def test_calibration_gate_default_is_exploratory():
    class Mean(mb.BaseObservable):
        name = "Mean"

        def measure(self, ctx):
            return ctx.emit(0.5)

    ev = mb.run(Mean(), mb.Context(signal=_FakeSignal()))
    assert ev.trust is TrustLevel.EXPLORATORY
    assert ev.calibration is None
    assert ev.subject.signals == ("mfp:test",)


def test_calibration_provider_upgrades_trust():
    class Mean(mb.BaseObservable):
        name = "Mean"

        def measure(self, ctx):
            return ctx.emit(0.5)

    mb.set_calibration_provider(
        lambda name, subj, regime: CalibrationRef("ev:sc", "planted") if name == "Mean" else None
    )
    ev = mb.run(Mean(), mb.Context(signal=_FakeSignal()))
    assert ev.trust is TrustLevel.CALIBRATED
    assert ev.is_calibrated


def test_registration_gate_makes_evidence_registered():
    class Mean(mb.BaseObservable):
        name = "Mean"

        def measure(self, ctx):
            return ctx.emit(0.5)

    ev = mb.run(Mean(), mb.Context(signal=_FakeSignal(), study="study:demo@v1#abc"))
    assert ev.trust is TrustLevel.REGISTERED
    assert ev.provenance.study == "study:demo@v1#abc"

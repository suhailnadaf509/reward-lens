"""Running instruments against a converted `Run`, with one guarantee to keep.

The guarantee: *every record-only instrument returns Evidence or a Refusal against it, none an
exception.* Three things stand between an instrument and that guarantee, and only the first is the
instrument's own doing.

**A record is not a signal, and the capability gate raises rather than refusing.** `measure.base.run`
checks the instrument's declared `capabilities` against the signal's and raises `CapabilityError`
either way: with no signal, because there is nothing to check against, and with a signal that
declares less than the instrument needs. So `BaseObservable.estimate`, whose own docstring says
"Evidence or Refusal. Never a bare float, never a silent degradation", raises on the capability
dimension. That is the one place in the instrument contract where the refusal architecture is not
honoured, and it is closed here by checking the capability before calling `estimate` and returning
`Refusal(ACCESS_INSUFFICIENT)`. Anticipating a declared condition is not the same as catching a
broad exception.

**What the record holds is not what the recorder could reach.** `Run.access` is what was reachable
at capture time, and `BaseObservable.preflight` reads whatever is in `Context.access` as what the
analyst can reach now. Those are different matrices and the campaign is the case that separates
them: it ran forward passes and captured activations, and nobody holding the converted store can do
either. `reader_access` builds the second one.

**A capability on a signal and a capability in a record mean different things.** A signal with
`ACTIVATIONS` can produce them on demand; a record either already holds them or never will.
`capabilities_in_record` reports the second, which is strictly weaker, and it is measured from the
record rather than declared.

`sweep` classifies four outcomes and the fourth must stay empty: `Evidence`, a `note` Evidence,
a `Refusal`, and an escaped exception. The `note` class exists because twelve of the shipped
instruments return `ctx.emit({"note": "... none injected"})` when their input is absent, which
satisfies the letter of the clause and not its intent. Counting them separately keeps the
acceptance number honest.

**And the measurement that says why the capability report has to be honest.** Handing every
instrument a signal that declares `Capability(~0)` and an access matrix of everything, so nothing
refuses at the gate, makes sixteen of the thirty-eight raise `AttributeError` or `TypeError` inside
`measure`: they reach for `signal.capture()` or `signal.readouts()` on an object that has neither.
Refusing on a capability the record does not hold is not conservatism, it is the only thing between
this sweep and sixteen tracebacks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from reward_lens.core.envelope import ConditionReading, RegimeCondition, RegimeReading
from reward_lens.core.evidence import Evidence
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.types import (
    Access,
    AccessMatrix,
    Capability,
    Component,
    ModelFP,
    Phase,
    Substrate,
)
from reward_lens.measure.base import (
    Context,
    declared_access,
    declared_capabilities,
)
from reward_lens.record.schema import Run
from reward_lens.record.tensors import AbsentRef, StoredRef

#: What anyone holding a converted record can reach: the log, and nothing they can call. This is
#: `PROFILE_AUDITOR` rebuilt from the record rather than assumed, so a record that carries no task
#: reference does not claim task access.
READER_COMPONENTS = (Component.TASK, Component.GRADER, Component.RECORD)


def reader_access(run: Run) -> AccessMatrix:
    """What a holder of this record can reach, as against what its recorder could.

    `RECORD` on every component the run names, and nothing else. Not derived from `Run.access`,
    which says what the campaign could do: a reader who inherits that matrix is told they can run
    the grader, and they cannot.
    """
    return {
        component: Access.RECORD
        for component in READER_COMPONENTS
        if run.component(component) is not None
    }


def capabilities_in_record(run: Run, *, steps: int | None = None) -> tuple[Capability, int, int]:
    """What this record already holds, expressed in the signal vocabulary.

    Returns the capabilities, how many steps were scanned and how many the run has. The default
    scans every step, and it has to: on the converted campaign the eight ProcessBench banks sit at
    step indices 597 to 606, so a scan bounded at the front finds no `STEP_SCORES` and produces a
    refusal that is wrong about the record. A full scan of the thousand banks reads 266 MB of
    sidecars and takes about 25 seconds, which is the honest price of the answer.

    ``steps`` bounds the scan, and then the result is a **lower bound** over the window rather than
    a statement about the record. `sweep` carries the window into every refusal it builds so a
    bounded answer says how bounded it is.

    `LINEAR_READOUT` is read off `ComponentRef.extra["readout_vectors"]`, which is where the
    converter puts the campaign's recorded reward directions because the record has no field for
    them.
    """
    caps = Capability.NONE
    grader = run.component(Component.GRADER)
    if grader is not None and grader.extra.get("readout_vectors"):
        caps |= Capability.LINEAR_READOUT
    total = len(run.steps)
    seen = 0
    for step in run.steps:
        if steps is not None and seen >= steps:
            break
        seen += 1
        for group in step.groups:
            for traj in group.trajectories:
                if traj.scores is not None:
                    caps |= Capability.SCORES
                for turn in traj.turns:
                    if turn.step_score is not None:
                        caps |= Capability.STEP_SCORES
                    if turn.spans:
                        caps |= Capability.SPAN_TYPES
                if traj.capture is not None and any(
                    isinstance(t, StoredRef) for t in traj.capture.tensors.values()
                ):
                    caps |= Capability.ACTIVATIONS
    return caps, seen, total


def absent_capture_reasons(run: Run, *, steps: int = 1) -> dict[str, int]:
    """Why the captured tensors this record references are not here, counted by reason.

    A capture manifest with no bytes behind it is the honest state of the campaign's activations,
    and this is what makes it visible rather than implied by `ACTIVATIONS` being absent above.
    """
    counts: dict[str, int] = {}
    seen = 0
    for step in run.steps:
        seen += 1
        for group in step.groups:
            for traj in group.trajectories:
                if traj.capture is None:
                    continue
                for ref in traj.capture.tensors.values():
                    if isinstance(ref, AbsentRef):
                        counts[ref.reason.name] = counts.get(ref.reason.name, 0) + 1
        if seen >= steps:
            break
    return counts


def regime_over(run: Run, *, limit: int | None = None) -> tuple[RegimeReading, int, int]:
    """Fold the per-step regime readings into one, conservatively.

    Returns the folded reading, how many steps were folded, and how many the run has. A condition
    that fails anywhere in the folded window fails; one that is indeterminate anywhere and fails
    nowhere is indeterminate; one absent from every step stays absent, because
    `EnvelopeSpec.classify` distinguishes "nobody measured it" from "somebody measured it and could
    not tell" and folding them together would lose the distinction the whole envelope machinery is
    built on.

    ``limit`` caps the fold, and when it bites the detail on every folded condition says so. A
    verdict folded over eight of a thousand banks is a verdict about eight banks, and a reader has
    to be able to see that in the reading rather than in a docstring.
    """
    total = len(run.steps)
    folded: dict[RegimeCondition, ConditionReading] = {}
    seen = 0
    for step in run.steps:
        if limit is not None and seen >= limit:
            break
        seen += 1
        for condition, reading in step.regime_measured.conditions.items():
            current = folded.get(condition)
            if current is None:
                folded[condition] = reading
                continue
            if current.holds is False:
                continue
            if reading.holds is False or reading.holds is None:
                folded[condition] = reading
    if seen < total:
        folded = {
            condition: ConditionReading(
                condition=reading.condition,
                holds=reading.holds,
                statistic=reading.statistic,
                threshold=reading.threshold,
                provenance=reading.provenance,
                detail=f"{reading.detail} [folded over {seen} of {total} steps]",
            )
            for condition, reading in folded.items()
        }
    return RegimeReading(conditions=folded), seen, total


# ---------------------------------------------------------------------------
# The signal a record is
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _RecordMeta:
    """The `SignalMeta` surface `Context.subject` reads, and nothing more."""

    fingerprint: ModelFP
    adapter: str = "record"


@dataclass(frozen=True)
class RecordSignal:
    """The recorded grader, as much of a `RewardSignal` as a record can be.

    Deliberately not a `RewardSignal`: it implements `meta` and `caps` and none of the seven
    methods, because every one of them calls the model and a record cannot. It exists so that
    `Context.subject` can name what was measured and so the capability check compares against what
    the record holds instead of against nothing. Anything that tries to `score` through it gets an
    `AttributeError`, which is the correct outcome: a record is not callable and pretending
    otherwise is how a recomputed number gets mistaken for a recorded one.
    """

    meta: _RecordMeta
    caps: Capability
    intervention_fingerprints: tuple[str, ...] = ()

    @classmethod
    def of(cls, run: Run, *, caps: Capability | None = None) -> "RecordSignal":
        grader = run.component(Component.GRADER)
        fingerprint = ModelFP(
            str(grader.model_fp) if grader is not None and grader.model_fp else f"record:{run.id}"
        )
        return cls(
            meta=_RecordMeta(fingerprint=fingerprint),
            caps=capabilities_in_record(run)[0] if caps is None else caps,
        )


def substrate_of(run: Run) -> Substrate | None:
    grader = run.component(Component.GRADER)
    return None if grader is None else grader.substrate


def context_for(
    run: Run,
    *,
    limit: int | None = None,
    phase: Phase = Phase.POST_RUN,
    caps: Capability | None = None,
) -> tuple[Context, RegimeReading, int, int]:
    """A `Context` describing what this record is, for an instrument to be preflighted against.

    `Phase.POST_RUN` is not a default anyone should override lightly: the record exists and the run
    is over, which is the definition of the phase, and it is what makes an in-run instrument refuse
    with `PHASE_MISMATCH` rather than compute something meaningless.

    ``caps`` is threaded through rather than recomputed so that the capability the harness gates on
    and the capability `measure.base.run` re-checks against `ctx.signal.caps` are the same value. If
    they could differ, an instrument the harness admitted would raise inside the runner, which is
    the failure the harness exists to prevent.
    """
    reading, folded, total = regime_over(run, limit=limit)
    ctx = Context(
        signal=RecordSignal.of(run, caps=caps),
        view=None,
        readout="reward",
        access=reader_access(run),
        substrate=substrate_of(run),
        phase=phase,
        regime_reading=reading,
        lod=None,
    )
    return ctx, reading, folded, total


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InstrumentOutcome:
    """What one instrument did when pointed at the record.

    ``kind`` is one of ``evidence``, ``note``, ``refusal``, ``exception``. The last one is the
    failure the guarantee is about and it carries the exception so a test failure names the
    instrument and the traceback rather than a count.
    """

    instrument: str
    kind: str
    reading: Any = None
    error: BaseException | None = None

    @property
    def refusal_reason(self) -> RefusalReason | None:
        return self.reading.reason if isinstance(self.reading, Refusal) else None

    def render(self) -> str:
        if self.kind == "refusal":
            return f"{self.instrument:24s} REFUSAL  {self.reading.reason.name}"
        if self.kind == "exception":
            return f"{self.instrument:24s} RAISED   {type(self.error).__name__}: {self.error}"
        if self.kind == "note":
            note = self.reading.value.get("note", "")
            return f"{self.instrument:24s} NOTE     {note}"
        return f"{self.instrument:24s} EVIDENCE {self.reading.observable}"


@dataclass
class SweepReport:
    """Every instrument's outcome, and the counts the guarantee is stated in."""

    outcomes: tuple[InstrumentOutcome, ...] = ()
    record_only: tuple[str, ...] = ()
    capabilities: Capability = Capability.NONE
    access: AccessMatrix = field(default_factory=dict)
    regime_steps: tuple[int, int] = (0, 0)
    capability_steps: tuple[int, int] = (0, 0)

    def of_kind(self, kind: str) -> tuple[InstrumentOutcome, ...]:
        return tuple(o for o in self.outcomes if o.kind == kind)

    @property
    def exceptions(self) -> tuple[InstrumentOutcome, ...]:
        return self.of_kind("exception")

    @property
    def reasons(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for outcome in self.of_kind("refusal"):
            name = outcome.reading.reason.name
            counts[name] = counts.get(name, 0) + 1
        return dict(sorted(counts.items()))

    def for_instrument(self, name: str) -> InstrumentOutcome | None:
        for outcome in self.outcomes:
            if outcome.instrument == name:
                return outcome
        return None

    def render(self) -> str:
        folded, total = self.regime_steps
        scanned, steps = self.capability_steps
        lines = [
            f"{len(self.outcomes)} instruments against the record "
            f"(regime folded over {folded} of {total} steps)",
            f"  record holds: {self.capabilities!s} (scanned {scanned} of {steps} steps)",
            "  reader access: "
            + (
                ", ".join(
                    f"{c.name}:{a.name}"
                    for c, a in sorted(self.access.items(), key=lambda kv: kv[0].name)
                )
                or "nothing"
            ),
            f"  evidence {len(self.of_kind('evidence'))}, note {len(self.of_kind('note'))}, "
            f"refusal {len(self.of_kind('refusal'))}, exception {len(self.exceptions)}",
            f"  refusal reasons: {self.reasons}",
        ]
        lines.extend("  " + o.render() for o in self.outcomes)
        return "\n".join(lines)


def _is_note(reading: Any) -> bool:
    """An Evidence whose value is only a note is a non-answer wearing a measurement.

    Twelve of the shipped instruments return one when their injected input is absent, measured by
    running all thirty-eight with every capability and every access granted. That case is a
    `Refusal` with a remedy; classifying it here keeps the count from counting
    a "none injected" string as a reading.
    """
    return (
        isinstance(reading, Evidence)
        and isinstance(reading.value, Mapping)
        and set(reading.value) == {"note"}
    )


def is_record_only(instrument: Any, access: AccessMatrix) -> bool:
    """Whether ``access`` satisfies everything this instrument declares it needs.

    An instrument that declares no access matrix is included, because an empty requirement is
    satisfied by any access and excluding it would be reading the blank as a declaration. Section
    4.2 makes an undeclared field a lint finding rather than an exemption.
    """
    from reward_lens.core.types import satisfies

    return satisfies(access, declared_access(instrument))


def run_instrument(
    instrument: Any,
    ctx: Context,
    *,
    caps: Capability,
    window: tuple[int, int] = (0, 0),
) -> InstrumentOutcome:
    """One instrument against one context, with the capability gate turned into a refusal.

    The capability comparison happens here and not inside `estimate` because `measure.base.run`
    raises on it. `missing_from` is the same call the runner makes, so this refuses in exactly the
    cases the runner would have raised in and in no others.

    ``window`` is how many of the run's steps the capability scan looked at. It goes in the
    refusal's statistics because a capability read off part of a record is a lower bound, and a
    refusal built on a lower bound has to say so or it is asserting more than it measured.
    """
    name = getattr(instrument, "name", type(instrument).__name__)
    needed = declared_capabilities(instrument)
    missing = needed.missing_from(caps)
    if missing and missing != Capability.NONE:
        scanned, total = window
        bounded = (
            ""
            if scanned >= total
            else (
                f" The record was scanned over {scanned} of its {total} steps, so what it holds is a "
                f"lower bound and a wider scan could change this."
            )
        )
        return InstrumentOutcome(
            instrument=name,
            kind="refusal",
            reading=Refusal(
                instrument=name,
                reason=RefusalReason.RECORD_INCOMPLETE,
                detail=(
                    f"this instrument declares it needs {needed!s} from the signal, and the record "
                    f"holds {caps!s}; {missing!s} is not in it. A record is not a callable signal: "
                    f"what it did not capture cannot be produced from it." + bounded
                ),
                remedy=(
                    f"point this instrument at the live grader, which can produce {missing!s} on "
                    f"demand, or re-record with a capture spec that stores it. If the instrument "
                    f"reads only numbers another arm already measured, its capability declaration "
                    f"is wider than its code and narrowing it is the fix."
                ),
                statistics={
                    "needs": str(needed),
                    "record_holds": str(caps),
                    "missing": str(missing),
                    "steps_scanned": scanned,
                    "steps_total": total,
                },
            ),
        )
    try:
        reading = instrument.estimate(ctx)
    except BaseException as exc:  # noqa: BLE001 - recorded as a failure, never converted
        return InstrumentOutcome(instrument=name, kind="exception", error=exc)
    if isinstance(reading, Refusal):
        return InstrumentOutcome(instrument=name, kind="refusal", reading=reading)
    if _is_note(reading):
        return InstrumentOutcome(instrument=name, kind="note", reading=reading)
    return InstrumentOutcome(instrument=name, kind="evidence", reading=reading)


def sweep(
    run: Run,
    instruments: Iterable[Any],
    *,
    limit: int | None = None,
    caps: Capability | None = None,
    caps_steps: int | None = None,
) -> SweepReport:
    """Run every instrument against the record and classify what came back.

    Exceptions are recorded rather than converted. Turning an escaped exception into a refusal here
    would hide exactly the failure this sweep exists to detect, and the acceptance test asserts the
    exception list is empty rather than asserting that nothing was raised anywhere.

    ``caps`` skips the capability scan when the caller already ran one, which matters because the
    full scan of the converted campaign decodes every bank.
    """
    instruments = tuple(instruments)
    if caps is None:
        caps, scanned, steps = capabilities_in_record(run, steps=caps_steps)
    else:
        scanned, steps = (caps_steps or len(run.steps)), len(run.steps)
    ctx, _, folded, total = context_for(run, limit=limit, caps=caps)
    access = ctx.access or {}
    outcomes = tuple(
        run_instrument(i, ctx, caps=caps, window=(scanned, steps)) for i in instruments
    )
    record_only = tuple(
        getattr(i, "name", type(i).__name__) for i in instruments if is_record_only(i, access)
    )
    return SweepReport(
        outcomes=outcomes,
        record_only=record_only,
        capabilities=caps,
        access=access,
        regime_steps=(folded, total),
        capability_steps=(scanned, steps),
    )


def _observable_classes() -> Iterable[tuple[str, type]]:
    """Every class in `measure` that inherits `BaseObservable` and is defined in its own module."""
    import importlib
    import inspect
    import pkgutil

    import reward_lens.measure as measure_pkg
    from reward_lens.measure.base import BaseObservable

    for module_info in pkgutil.walk_packages(measure_pkg.__path__, measure_pkg.__name__ + "."):
        module = importlib.import_module(module_info.name)
        for name, obj in vars(module).items():
            if not inspect.isclass(obj) or not issubclass(obj, BaseObservable):
                continue
            if obj is BaseObservable or obj.__module__ != module.__name__:
                continue
            yield f"{obj.__module__}.{name}", obj


def _declares_a_name(cls: type) -> bool:
    """Whether this class named itself, which is how an instrument is told from a base class.

    `BaseObservable.name` is the placeholder ``"observable"`` and the instrument contract requires
    `name` on every instrument, so a class that never set one has not declared itself to be one.
    That is the only structural signal available: `ControlInstrument` is an abstract base whose
    `estimate` is fully implemented and whose `compute` raises `NotImplementedError`, so it
    satisfies the `runtime_checkable` `Instrument` protocol and a discovery sweep picks it up as a
    real instrument unless something like this excludes it.
    """
    from reward_lens.measure.base import BaseObservable

    return getattr(cls, "name", BaseObservable.name) != BaseObservable.name


def shipped_instruments() -> tuple[Any, ...]:
    """Every named `BaseObservable` subclass in `measure`, constructed with no arguments.

    Discovered by import rather than listed, so an instrument added to the battery next week is
    swept without anyone remembering to add it here. Constructed bare on purpose: the acceptance
    clause is about what happens when a record is all you have, and an instrument handed its inputs
    is not being asked that question. `unnamed_bases` and `uninstantiable` name everything this
    skips, so the pass count cannot be inflated by a quiet exclusion.
    """
    out: list[Any] = []
    for _, cls in _observable_classes():
        if not _declares_a_name(cls):
            continue
        try:
            out.append(cls())
        except TypeError:
            continue
    return tuple(out)


def uninstantiable() -> tuple[str, ...]:
    """Named instruments that cannot be constructed without arguments, named rather than skipped."""
    out: list[str] = []
    for key, cls in _observable_classes():
        if not _declares_a_name(cls):
            continue
        try:
            cls()
        except TypeError:
            out.append(key)
    return tuple(sorted(out))


def unnamed_bases() -> tuple[str, ...]:
    """`BaseObservable` subclasses that never set a `name`, so they are bases and not instruments."""
    return tuple(sorted(key for key, cls in _observable_classes() if not _declares_a_name(cls)))


def access_declaration_findings(instruments: Sequence[Any]) -> tuple[str, ...]:
    """Instruments that carry an access matrix under a name `declared_access` does not read.

    The instrument contract and `declared_access` both say `requires`. Four of the control
    instruments declare `access` instead, so their access matrix is never checked by any preflight
    and the refusal it would produce never fires. Reported rather than fixed: those files belong to
    another package.
    """
    out: list[str] = []
    for instrument in instruments:
        name = getattr(instrument, "name", type(instrument).__name__)
        if declared_access(instrument):
            continue
        for alias in ("access", "access_matrix", "requires_access"):
            value = getattr(instrument, alias, None)
            if isinstance(value, Mapping) and value:
                out.append(
                    f"{name} declares its access matrix as `{alias}`; the instrument contract "
                    f"and `declared_access` read `requires`, so preflight never checks it"
                )
                break
    return tuple(out)


__all__ = [
    "READER_COMPONENTS",
    "InstrumentOutcome",
    "RecordSignal",
    "SweepReport",
    "absent_capture_reasons",
    "access_declaration_findings",
    "capabilities_in_record",
    "context_for",
    "is_record_only",
    "reader_access",
    "regime_over",
    "run_instrument",
    "shipped_instruments",
    "substrate_of",
    "sweep",
    "uninstantiable",
    "unnamed_bases",
]

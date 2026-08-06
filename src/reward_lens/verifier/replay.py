"""D10 Determinism and replay fidelity: whether the record can be re-graded to the score it holds.

Says, with your numbers in place of the specification's: "Replaying the recorded trajectory
reproduces the score on 87% of tasks. The other 13% are not auditable."

This is the smallest instrument in the series and the one the rest of it stands on. If replaying a
recorded task through the grader does not return the score the record says it got, then every
post-hoc claim about that task is a claim about a number nobody can reconstruct: the coverage
attribution, the mutation kill, the metamorphic violation and the false positive are all computed
against a grader that is not the grader that produced the record. Three instruments in this package
already declare `STATIONARY_GRADER` as a precondition and name `env.replay_fidelity` as the thing
that measures it, so this closes a loop that was open by design until now.

**Why it is a fraction of tasks and not a spread of scores.** The related and different question is
A7's: run the same policy on the same task twenty times and report how far the scores range, which
is `env.flakiness` in percentage points and is a property of the environment's variance. D10 asks a
binary question per task, "did it come back the same", and reports the fraction that did. The two
are worth keeping apart because they have different remedies: a spread says the environment is
noisy and you need more replicates, and a fidelity below one says particular tasks are not
auditable and names them.

**How this fits `RecomputeRef`.** `record/tensors.py` already carries the discipline for a value
that is not stored but can be recomputed: a recipe naming the exact engine and revision, an
`expected_numerics_floor` that the caller may not quietly widen, and a seven-member `AbsenceReason`
vocabulary for saying why a value is not here. A recomputed *score* is the same object as a
recomputed *tensor* and it gets the same treatment, so a task that will not replay comes back as an
`AbsentRef` carrying one of those reasons rather than as a zero or a `None`. The mapping is stated
once, in `TaskReplay.absence`:

- the grader raised on the recorded inputs, so the recipe cannot be honoured: `RECOMPUTE_UNAVAILABLE`
- the replayed score differs from the recorded one by more than the floor: `NUMERICS_FLOOR_EXCEEDED`
- the record carries no score for this task, so there is nothing to reproduce: `NOT_CAPTURED`
- no grader was supplied to replay with: `RECOMPUTE_UNSUPPORTED`

Kill condition, from the catalogue: **if replay fidelity is 100% everywhere.** That is a real
possibility for a pure-function grader on a fixed corpus and it is why the instrument is worth
running on graders that shell out, hit a network, read a clock or import a library whose version
floats. On a grader that does none of those things a fidelity of 1.0 is the expected answer and
this instrument has told you something cheap and true rather than something interesting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from reward_lens.core.envelope import ConditionReading, EnvelopeSpec, RegimeCondition
from reward_lens.core.evidence import Uncertainty, register_payload
from reward_lens.core.invariance import Relation
from reward_lens.core.quantity import (
    BiasStatement,
    CostModel,
    EstimatorEntry,
    register_estimator,
)
from reward_lens.core.reading import Reading, Refusal, RefusalReason
from reward_lens.core.types import (
    Access,
    AccessMatrix,
    Capability,
    Component,
    GaugeStatus,
    Phase,
    Substrate,
)
from reward_lens.measure.base import BaseObservable, Context, run
from reward_lens.record.tensors import AbsenceReason, AbsentRef
from reward_lens.verifier import Rollout, RolloutCorpus, VerifierUnderTest, ensure_quantities
from reward_lens.verifier.metamorphic import resolve_grader

#: What counts as reproducing a score. The same discipline `RecomputeRef.expected_numerics_floor`
#: applies to a recomputed tensor: it is not a tolerance a caller may widen quietly to make a
#: report look better. A grader whose score moves by more than this between two runs on identical
#: inputs is not the same instrument twice.
#:
#: The default is tight rather than generous because a programmatic grader's score is usually a
#: small rational or a 0/1, and a genuine float wobble at 1e-9 is a different thing from a grader
#: that returns 1.0 one time and 0.0 the next.
DEFAULT_SCORE_FLOOR = 1e-9

#: The fidelity at or above which `STATIONARY_GRADER` is treated as holding. Deliberately 1.0:
#: the condition says "grader weights and rubric weights unchanged across the window", and a task
#: that does not replay is direct evidence that the grader is not returning the same answer to the
#: same question. Anything below 1.0 fails, which will fire often, and firing often is the correct
#: behaviour for a precondition three other instruments in this package depend on.
#:
#: This threshold appears in an envelope and is therefore not this module's to fix alone. It is a
#: parameter of `ReplayReport.condition_reading` and the default is recorded here so that a change
#: is an amendment rather than a drift.
STATIONARY_GRADER_FLOOR = 1.0


@register_payload
@dataclass(frozen=True)
class TaskReplay:
    """One recorded task, replayed. The unit of the fidelity fraction.

    ``absence`` is the bridge to the record. A task that did not reproduce is not a failure of this
    instrument, it is a value the record cannot supply, and `record/tensors.py` already has the
    vocabulary for that. Reusing it means a replay failure and a missing tensor arrive at a
    downstream consumer in the same shape rather than in two shapes that have to be special-cased.

    It is held as the reason's *name* rather than as the enum member, and that is a workaround
    rather than a preference: `register_payload`'s codec walks a dataclass's fields and can encode
    primitives, sequences, mappings, arrays and other registered payloads, with no branch for an
    enum. A payload carrying an `AbsenceReason` field raises `cannot encode value of type
    AbsenceReason` the first time it is emitted. `absence_reason` gives the member back.
    """

    id: str
    recorded: float | None
    replayed: float | None
    deviation: float | None
    reproduced: bool
    absence: str = ""
    error: str = ""
    repeats: int = 1
    replays: tuple[float, ...] = ()
    deterministic: bool = True

    @property
    def absence_reason(self) -> AbsenceReason | None:
        """The record's `AbsenceReason` member, or None when the task replayed."""
        return AbsenceReason[self.absence] if self.absence else None

    @property
    def auditable(self) -> bool:
        """Whether any post-hoc claim about this task rests on a reconstructible number."""
        return self.reproduced and self.deterministic

    def as_absent_ref(self) -> AbsentRef | None:
        """The record's own way of saying this value is not here, or None when it is.

        Statistics are floats because `AbsentRef.of` types them that way, so the recorded score,
        the replayed one and the gap between them travel with the absence and a reader does not
        have to go back to the report to see how badly it missed.
        """
        reason = self.absence_reason
        if reason is None:
            return None
        stats: dict[str, float] = {}
        if self.deviation is not None and math.isfinite(self.deviation):
            stats["deviation"] = float(self.deviation)
        if self.recorded is not None:
            stats["recorded"] = float(self.recorded)
        if self.replayed is not None:
            stats["replayed"] = float(self.replayed)
        return AbsentRef.of(reason, detail=self._detail(), **stats)

    def _detail(self) -> str:
        if self.absence == AbsenceReason.RECOMPUTE_UNAVAILABLE.name:
            return f"replaying task {self.id} raised {self.error}"
        if self.absence == AbsenceReason.NUMERICS_FLOOR_EXCEEDED.name:
            return (
                f"task {self.id} replayed to {self.replayed!r} against a recorded {self.recorded!r}"
            )
        if self.absence == AbsenceReason.NOT_CAPTURED.name:
            return f"task {self.id} carries no recorded score, so there is nothing to reproduce"
        return f"task {self.id} was not replayed"

    def render(self) -> str:
        if not self.absence and self.reproduced:
            flag = "" if self.deterministic else "  [NON-DETERMINISTIC across repeats]"
            return f"{self.id}: reproduced {self.recorded!r}{flag}"
        return f"{self.id}: {self.absence or 'NOT REPLAYED'} - {self._detail()}"


@register_payload
@dataclass(frozen=True)
class ReplayReport:
    """The value of `env.replay_fidelity`: what fraction of the record can be re-graded.

    ``baseline_assumed_fidelity`` is 1.0 and it is the mandatory baseline in the same shape D8's
    is. "The record replays" is the assumption every post-hoc analysis makes without stating it,
    it predicts a fidelity of exactly one, and this instrument's whole content is the size of the
    gap between that prediction and the measurement.
    """

    grader: str
    fingerprint: str
    rung: int
    n_tasks: int
    n_attempted: int
    n_reproduced: int
    n_mismatched: int
    n_unreplayable: int
    n_no_recorded_score: int
    score_floor: float
    repeats: int
    n_nondeterministic: int
    tasks: tuple[TaskReplay, ...] = ()
    baseline_assumed_fidelity: float = 1.0
    notes: tuple[str, ...] = ()

    @property
    def replay_fidelity(self) -> float:
        """The headline: reproduced over attempted.

        The denominator is tasks with a recorded score, not all tasks. A record that never stored
        a score for a task is not a replay failure, it is a record with a gap, and folding the two
        together turns a capture-policy question into a determinism question.
        """
        return float("nan") if self.n_attempted == 0 else self.n_reproduced / self.n_attempted

    @property
    def headline(self) -> float:
        return self.replay_fidelity

    @property
    def deterministic_fraction(self) -> float:
        """Tasks that gave the same answer on every repeat. Only meaningful above one repeat.

        This is not `env.flakiness`. A7 reports the *spread* of scores across identical runs in
        percentage points and owns that quantity; this is a per-task binary, and it is here because
        a task that disagrees with itself cannot have a replay fidelity that means anything.
        """
        if self.repeats < 2 or self.n_attempted == 0:
            return float("nan")
        return (self.n_attempted - self.n_nondeterministic) / self.n_attempted

    @property
    def unauditable(self) -> tuple[TaskReplay, ...]:
        """The tasks no post-hoc analysis can say anything about. The actionable half."""
        return tuple(t for t in self.tasks if not t.auditable)

    def absent_refs(self) -> tuple[AbsentRef, ...]:
        """Every failed replay in the record's own absence vocabulary, for a downstream consumer."""
        return tuple(ref for t in self.tasks if (ref := t.as_absent_ref()) is not None)

    def condition_reading(self, floor: float = STATIONARY_GRADER_FLOOR) -> ConditionReading:
        """`STATIONARY_GRADER`, measured. What three other instruments in this package consult.

        `metamorphic.py`, `sensitivity.py` and `fuzz.py` each declare
        `measured_by={STATIONARY_GRADER: "env.replay_fidelity"}` and until now nothing produced
        that quantity, so the condition came back unmeasured and every envelope that required it
        declined to admit. This is the other end of that wire.

        `holds` is None rather than False when nothing could be attempted, because "we could not
        check" and "we checked and it fails" call for different responses and the envelope's own
        `classify` splits them.
        """
        fidelity = self.replay_fidelity
        return ConditionReading(
            condition=RegimeCondition.STATIONARY_GRADER,
            holds=None if math.isnan(fidelity) else bool(fidelity >= floor),
            statistic=fidelity,
            threshold=floor,
            detail=(
                f"{self.n_reproduced} of {self.n_attempted} recorded scores reproduced at a floor "
                f"of {self.score_floor:g}"
                if self.n_attempted
                else "no task in this corpus carried a recorded score"
            ),
        )

    def render(self) -> str:
        fidelity = self.replay_fidelity
        pct = 100.0 * (1.0 - fidelity) if not math.isnan(fidelity) else float("nan")
        lines = [
            f"Replaying the recorded trajectory reproduces the score on {fidelity:.0%} of tasks "
            f"({self.n_reproduced} of {self.n_attempted}). The other {pct:.0f}% are not auditable.",
            f"    {self.n_mismatched} replayed to a different score, "
            f"{self.n_unreplayable} raised, "
            f"{self.n_no_recorded_score} carried no recorded score to compare against",
            f"    score floor {self.score_floor:g}; baseline (the assumption that a record "
            f"replays) predicts {self.baseline_assumed_fidelity:.0%}",
        ]
        if self.repeats > 1:
            lines.append(
                f"    {self.n_nondeterministic} of {self.n_attempted} tasks disagreed with "
                f"themselves across {self.repeats} repeats "
                f"(deterministic fraction {self.deterministic_fraction:.3f}). The score *spread* "
                f"is A7's env.flakiness, not this."
            )
        for t in self.unauditable[:10]:
            lines.append(f"      {t.render()}")
        if len(self.unauditable) > 10:
            lines.append(f"      ... and {len(self.unauditable) - 10} more")
        lines += [f"    note: {n}" for n in self.notes]
        return "\n".join(lines)


def replay_corpus(
    grader: Callable[..., float] | VerifierUnderTest,
    corpus: RolloutCorpus | Sequence[Rollout],
    *,
    score_floor: float = DEFAULT_SCORE_FLOOR,
    repeats: int = 1,
    rung: int = 0,
) -> ReplayReport | Refusal:
    """Re-grade every recorded rollout and compare against the score the record holds.

    A rollout whose call raises is recorded and counted, never skipped: the exception is the
    grader's behaviour on that input and it is precisely the case where the record cannot be
    audited. Nothing here converts it into a refusal, because one unreplayable task out of a
    thousand is a measurement of 0.999 rather than a failure to measure.

    ``repeats`` above one turns on the determinism half. Each task is graded that many times and a
    task whose own answers disagree is marked non-deterministic and does not count as reproduced,
    because a score that is only sometimes right is not a score the record can be audited against.
    """
    rows = list(corpus)
    if not rows:
        return Refusal(
            instrument="ReplayFidelity",
            reason=RefusalReason.ACCESS_INSUFFICIENT,
            detail="the corpus is empty, so there is nothing to replay",
            remedy=(
                "supply a corpus of recorded rollouts, each carrying the score the record says "
                "the grader produced. D10 compares a re-grade against a recorded score; with no "
                "record there is nothing to compare against and a fidelity of 1.0 on zero tasks "
                "would be the most misleading number this library could return."
            ),
            statistics={"n": 0},
        )
    if repeats < 1:
        raise ValueError(f"repeats must be at least 1, got {repeats}")

    fn, subject, name = resolve_grader(grader)
    scored = [r for r in rows if r.score is not None]
    if not scored:
        return Refusal(
            instrument="ReplayFidelity",
            reason=RefusalReason.ACCESS_INSUFFICIENT,
            detail=f"none of the {len(rows)} rollouts carries a recorded score",
            remedy=(
                "set `Rollout.score` from the record. Replay fidelity is a comparison against "
                "what the record says happened, and a corpus with no recorded scores supports "
                "the re-grade but not the comparison."
            ),
            statistics={"n": len(rows), "with_score": 0},
        )

    tasks: list[TaskReplay] = []
    reproduced = mismatched = unreplayable = nondeterministic = 0

    for rollout in rows:
        if rollout.score is None:
            tasks.append(
                TaskReplay(
                    id=rollout.id,
                    recorded=None,
                    replayed=None,
                    deviation=None,
                    reproduced=False,
                    absence=AbsenceReason.NOT_CAPTURED.name,
                    repeats=repeats,
                )
            )
            continue

        values: list[float] = []
        error = ""
        for _ in range(repeats):
            try:
                values.append(float(fn(**dict(rollout.inputs))))
            except Exception as exc:  # noqa: BLE001 - the grader raising IS the measurement here
                error = f"{type(exc).__name__}: {exc}"
                break

        if error:
            unreplayable += 1
            tasks.append(
                TaskReplay(
                    id=rollout.id,
                    recorded=float(rollout.score),
                    replayed=None,
                    deviation=None,
                    reproduced=False,
                    absence=AbsenceReason.RECOMPUTE_UNAVAILABLE.name,
                    error=error,
                    repeats=repeats,
                    replays=tuple(values),
                )
            )
            continue

        stable = max(values) - min(values) <= score_floor
        if not stable:
            nondeterministic += 1
        deviation = abs(values[0] - float(rollout.score))
        hit = deviation <= score_floor and stable
        if hit:
            reproduced += 1
        elif stable:
            mismatched += 1
        tasks.append(
            TaskReplay(
                id=rollout.id,
                recorded=float(rollout.score),
                replayed=values[0],
                deviation=deviation,
                reproduced=hit,
                absence="" if hit else AbsenceReason.NUMERICS_FLOOR_EXCEEDED.name,
                repeats=repeats,
                replays=tuple(values),
                deterministic=stable,
            )
        )

    notes: list[str] = []
    if repeats == 1:
        notes.append(
            "one replay per task, so the determinism half did not run: a task that reproduces "
            "once may still be non-deterministic. Pass repeats>=2 to separate a grader that "
            "reproduces from one that happened to."
        )

    return ReplayReport(
        grader=name,
        fingerprint=getattr(subject.meta, "fingerprint", ""),
        rung=rung,
        n_tasks=len(rows),
        n_attempted=len(scored),
        n_reproduced=reproduced,
        n_mismatched=mismatched,
        n_unreplayable=unreplayable,
        n_no_recorded_score=len(rows) - len(scored),
        score_floor=score_floor,
        repeats=repeats,
        n_nondeterministic=nondeterministic,
        tasks=tuple(tasks),
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# The instrument
# ---------------------------------------------------------------------------

#: `TASK:QUERY + a record`, which is what the catalogue's access line says. The grader has to be
#: callable again, and the record has to hold the score it produced the first time. Neither alone
#: supports the comparison.
ACCESS_QUERY_AND_RECORD: AccessMatrix = {
    Component.TASK: Access.QUERY,
    Component.GRADER: Access.QUERY,
    Component.RECORD: Access.RECORD,
}

_D10_SUBSTRATES = frozenset(
    {Substrate.PROGRAM, Substrate.PROCEDURAL, Substrate.COMPOSITE, Substrate.NEURAL_GEN}
)
_D10_PHASES = frozenset({Phase.POST_RUN, Phase.IN_RUN, Phase.DEPLOYED})

#: D10 is the instrument the envelope machinery consults, so it cannot itself require a measured
#: regime without the check running in a circle: `STATIONARY_GRADER` is measured *by this*, and an
#: envelope requiring it here would ask this instrument to consult its own output.
D10_ENVELOPE = EnvelopeSpec(
    unconditional=True,
    justification=(
        "a census over a record: it re-runs a grader on inputs the record holds and compares "
        "against the score the record holds, so no property of the training process can make the "
        "comparison wrong. It is also the measurement of STATIONARY_GRADER itself, so requiring "
        "that condition here would be circular."
    ),
)


class ReplayFidelity(BaseObservable):
    """D10 `env.replay_fidelity`: what fraction of the record can be re-graded to its own score.

    Kill condition, from the catalogue: **if replay fidelity is 100% everywhere.** A pure-function
    grader on a fixed corpus will return 1.0 and that is the expected answer rather than a defect;
    the instrument earns its place on graders that shell out, read a clock, hit a network or
    depend on a library version that is not pinned. If it comes back 1.0 on a harness that does any
    of those things, check `repeats` first: at one replay per task the determinism half did not
    run, and a grader that agrees with the record once has not been shown to agree with itself.
    """

    name = "ReplayFidelity"
    version = "1.0"
    quantity = "env.replay_fidelity"
    capabilities = Capability.SCORES
    requires = ACCESS_QUERY_AND_RECORD
    substrates = _D10_SUBSTRATES
    phases = _D10_PHASES
    envelope = D10_ENVELOPE
    invariance = "none"
    invariance_relation = Relation("invariant")
    baselines = ("the assumption that a record replays, which predicts a fidelity of 1.0",)
    rung = 0
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = None
    deviations = (
        "a reproduced score is not a reproduced computation. Two different code paths that "
        "arrive at the same number count as a reproduction here, which is the right call for "
        "auditing a score and the wrong one for auditing a grader.",
        "the comparison is against the score the corpus carries, which is the record's claim "
        "rather than an independent fact. A record that stored the wrong score consistently "
        "reports a fidelity of zero, and a record that stored a re-grade rather than the original "
        "reports one.",
    )

    def __init__(
        self,
        grader: Callable[..., float] | VerifierUnderTest | None = None,
        corpus: RolloutCorpus | Sequence[Rollout] | None = None,
        *,
        score_floor: float = DEFAULT_SCORE_FLOOR,
        repeats: int = 1,
    ) -> None:
        ensure_quantities()
        self.grader = grader
        self.corpus = corpus
        self.score_floor = score_floor
        self.repeats = repeats
        self.rung = 0 if repeats < 2 else 1

    @property
    def subject(self) -> Any:
        if self.grader is None:
            raise ValueError(f"{self.name} was constructed without a grader")
        _, subject, _ = resolve_grader(self.grader)
        return subject

    def estimate(self, ctx: Context | None = None) -> Reading:
        if ctx is None:
            ctx = Context(signal=self.subject, view=self.corpus, readout="score")
        pre = self.preflight(ctx)
        if not pre.ok and pre.refusal is not None:
            return pre.refusal
        return run(self, ctx)

    def measure(self, ctx: Context) -> Any:
        grader = self.grader
        if grader is None:
            grader = getattr(ctx.signal, "verifier", None) or getattr(ctx.signal, "fn", None)
        corpus = self.corpus if self.corpus is not None else ctx.view
        if grader is None or corpus is None:
            missing = [n for n, v in (("grader", grader), ("corpus", corpus)) if v is None]
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail=f"no {' and no '.join(missing)} was supplied",
                remedy=(
                    "pass `ReplayFidelity(grader, corpus)`, or put the corpus on `ctx.view` and "
                    "a grader-backed subject on `ctx.signal`. D10 needs both ends: something to "
                    "call again, and a record of what it returned the first time."
                ),
                statistics={"missing": missing},
            )
        result = replay_corpus(
            grader,
            corpus,
            score_floor=self.score_floor,
            repeats=self.repeats,
            rung=self.rung,
        )
        if isinstance(result, Refusal):
            return result
        return ctx.emit(
            result,
            uncertainty=Uncertainty(
                n=result.n_attempted,
                method="census over the record; no interval, every task was attempted",
            ),
            subject_extra={
                "score_floor": f"{result.score_floor:g}",
                "baseline_assumed_fidelity": f"{result.baseline_assumed_fidelity:g}",
            },
        )


def replay_fidelity(
    grader: Callable[..., float] | VerifierUnderTest,
    corpus: RolloutCorpus | Sequence[Rollout],
    **kwargs: Any,
) -> Reading:
    """Run D10 and return the Reading. The one-call form, for a card renderer."""
    return ReplayFidelity(grader, corpus, **kwargs).estimate()


def stationary_grader_reading(
    grader: Callable[..., float] | VerifierUnderTest,
    corpus: RolloutCorpus | Sequence[Rollout],
    *,
    floor: float = STATIONARY_GRADER_FLOOR,
    **kwargs: Any,
) -> ConditionReading:
    """Measure `STATIONARY_GRADER` for an envelope, straight from a grader and a record.

    The convenience that makes the wire usable. D3, D4 and D5 all require the condition and name
    `env.replay_fidelity` as its measurement, and the shortest path from "I have a grader and a
    corpus" to "my envelope admits" should be one call rather than four.

    A refusal from the underlying measurement becomes `holds=None`, not `holds=False`. Failing to
    measure a condition is not the same as measuring it and finding it violated, and the envelope's
    `classify` reports the two differently on purpose.
    """
    reading = replay_corpus(grader, corpus, **kwargs)
    if isinstance(reading, Refusal):
        return ConditionReading(
            condition=RegimeCondition.STATIONARY_GRADER,
            holds=None,
            statistic=float("nan"),
            threshold=floor,
            detail=f"replay fidelity could not be measured: {reading.detail}",
        )
    return reading.condition_reading(floor)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def _register() -> None:
    """One estimator, which is what `spec/QUANTITIES.yaml` declares for `env.replay_fidelity`.

    The catalogue's ladder for D10 is OPEN and the registry says `rungs: 1`, so a single entry is
    the faithful reading. The ``repeats`` parameter is not a second rung: it turns on a different
    check on the same quantity rather than a better estimator of it.
    """
    ensure_quantities()
    register_estimator(
        EstimatorEntry(
            quantity="env.replay_fidelity",
            impl="env.replay_fidelity.regrade",
            requires=ACCESS_QUERY_AND_RECORD,
            envelope=D10_ENVELOPE,
            rung=0,
            bias=BiasStatement(
                direction="upward",
                why=(
                    "reproducing the score is weaker than reproducing the computation: two "
                    "different paths to the same number count as a reproduction. A grader whose "
                    "internals have changed but whose outputs happen to agree on this corpus "
                    "reports a fidelity of 1.0, so the measured fidelity is a ceiling on how much "
                    "of the record is genuinely auditable."
                ),
            ),
            cost=CostModel(note="one grader call per recorded task, times `repeats`; no GPU"),
            substrates=_D10_SUBSTRATES,
            phases=_D10_PHASES,
            run=None,
        )
    )


_register()


__all__ = [
    "ACCESS_QUERY_AND_RECORD",
    "DEFAULT_SCORE_FLOOR",
    "D10_ENVELOPE",
    "STATIONARY_GRADER_FLOOR",
    "ReplayFidelity",
    "ReplayReport",
    "TaskReplay",
    "replay_corpus",
    "replay_fidelity",
    "stationary_grader_reading",
]

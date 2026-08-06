"""Study specifications: the unit of confirmatory work.

A Study is data (a spec) plus a thin analysis function, never a subsystem. The spec states the
hypotheses, each with a registered prediction whose sign and effect predate the run; the subjects;
the analysis plan; and the kill criteria as schema fields, not prose, so the scoreboard can render
them and a reviewer can check them. Freezing the spec (``freeze``) hashes it and records the git
sha, after which Evidence produced under it is REGISTERED, and any edit creates a new visible
version.

Everything here is a plain, serializable dataclass so the spec can be hashed and stored. The
analysis is named by a dotted path, not held as a callable, so the frozen content is stable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from reward_lens.core.provenance import Cost
from reward_lens.studies.void import StudyOutcome, Void

Comparator = Literal[">", "<", ">=", "<=", "==", "!=", "abs>", "abs<"]


@dataclass(frozen=True)
class Prediction:
    """A registered, checkable prediction.

    ``metric`` names the quantity the analysis will compute; ``comparator`` and ``threshold`` state
    the predicted relationship (for example ``metric="spearman_chi_vs_drift", comparator=">",
    threshold=0.3``). ``effect`` and ``ci_excludes`` optionally register a point effect size and a
    value the CI should exclude. This is what makes a claim uneditable after seeing the data: the
    prediction is hashed into the frozen study.
    """

    metric: str
    comparator: Comparator
    threshold: float
    effect: float | None = None
    ci_excludes: float | None = None
    rationale: str = ""

    def check(self, value: float) -> bool:
        t = self.threshold
        ops = {
            ">": value > t,
            "<": value < t,
            ">=": value >= t,
            "<=": value <= t,
            "==": value == t,
            "!=": value != t,
            "abs>": abs(value) > t,
            "abs<": abs(value) < t,
        }
        return bool(ops[self.comparator])


@dataclass(frozen=True)
class Hypothesis:
    """A hypothesis with its registered prediction and the scoreboard row it addresses."""

    id: str
    statement: str
    prediction: Prediction
    scoreboard_row: str | None = None  # e.g. "T9"


@dataclass(frozen=True)
class KillCriterion:
    """A schema-fielded kill criterion.

    If ``metric`` stands in ``comparator`` relation to ``threshold`` after the run, the criterion
    fires and the study produces a first-class negative-result report rather than a hidden failure.
    The description states, in one sentence, what a fired criterion means scientifically.
    """

    id: str
    metric: str
    comparator: Comparator
    threshold: float
    description: str = ""

    def fired(self, value: float) -> bool:
        return Prediction(self.metric, self.comparator, self.threshold).check(value)


@dataclass(frozen=True)
class SubjectQuery:
    """A declarative description of the study's subjects (signals, organisms, datasets).

    Held as ids/specs so the frozen study names exactly what it ran on. The runner resolves these
    to concrete objects at run time; the engine test injects them directly.
    """

    signals: tuple[str, ...] = ()
    organisms: tuple[str, ...] = ()
    datasets: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StudySpec:
    """The full specification of a confirmatory study."""

    id: str
    title: str
    science: str  # e.g. "S03-thermo"
    hypotheses: tuple[Hypothesis, ...]
    analysis: str  # dotted path to analysis(run) -> StudyResult
    subjects: SubjectQuery = field(default_factory=SubjectQuery)
    kill_criteria: tuple[KillCriterion, ...] = ()
    oracle_budget: Cost | None = None
    version: int = 1
    notes: str = ""

    def __canonical__(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "science": self.science,
            "hypotheses": [
                {
                    "id": h.id,
                    "statement": h.statement,
                    "prediction": {
                        "metric": h.prediction.metric,
                        "comparator": h.prediction.comparator,
                        "threshold": h.prediction.threshold,
                        "effect": h.prediction.effect,
                        "ci_excludes": h.prediction.ci_excludes,
                    },
                    "scoreboard_row": h.scoreboard_row,
                }
                for h in self.hypotheses
            ],
            "analysis": self.analysis,
            "subjects": {
                "signals": list(self.subjects.signals),
                "organisms": list(self.subjects.organisms),
                "datasets": list(self.subjects.datasets),
                "extra": self.subjects.extra,
            },
            "kill_criteria": [
                {
                    "id": k.id,
                    "metric": k.metric,
                    "comparator": k.comparator,
                    "threshold": k.threshold,
                }
                for k in self.kill_criteria
            ],
            "version": self.version,
        }


@dataclass
class StudyResult:
    """The outcome of a study run.

    ``outcomes`` maps each hypothesis id to "confirmed" / "refuted" / "void"; ``metrics`` holds the
    computed values the predictions and kill criteria were checked against; ``evidence`` lists the
    Evidence ids the study produced (its adjudicating evidence); ``killed`` is True if any kill
    criterion fired. A study that fired a kill criterion is not a failure; it is a publishable
    negative result, and this object carries it as first-class data.

    Three fields exist because a missing metric must not be able to masquerade as a reading.
    ``voids`` maps a hypothesis or kill-criterion id to the named, remediable reason it could not
    be adjudicated. ``kill_outcomes`` records "fired" / "passed" / "void" per criterion, so a
    criterion that could not be evaluated is distinguishable from one that was evaluated and
    passed; the old runner could not tell those apart and that was the most damaging thing about
    it. ``outcome`` is the study-level verdict: a study with any void is ``VOID``, and a void study
    is a work item rather than a result.
    """

    outcomes: dict[str, str]
    metrics: dict[str, float]
    evidence: list[str] = field(default_factory=list)
    killed: bool = False
    killed_by: list[str] = field(default_factory=list)
    summary: str = ""
    voids: dict[str, Void] = field(default_factory=dict)
    kill_outcomes: dict[str, str] = field(default_factory=dict)
    outcome: StudyOutcome = StudyOutcome.RESULT


Outcome = Literal["confirmed", "refuted", "void"]


__all__ = [
    "Comparator",
    "Prediction",
    "Hypothesis",
    "KillCriterion",
    "SubjectQuery",
    "StudySpec",
    "StudyResult",
    "StudyOutcome",
    "Outcome",
    "Void",
]

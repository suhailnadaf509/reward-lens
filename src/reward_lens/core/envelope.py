"""The validity envelope: what an instrument needs to be true, not merely available.

Access, phase and substrate all fail loudly. Regime fails quietly, and that is the whole reason
this module exists. It is the difference between an instrument being *unavailable* and an
instrument being *silently wrong*.

The worked case. `F1`, the selection term `η·Cov_group(A, f)`, needs a record and a
featuriser and nothing else, so it will happily compute on any run ever recorded. But it is a
first-order expansion, so it means nothing if the step is large; it assumes the group has spread,
so it means nothing on all-fail groups; it assumes the advantage transform is the one you think it
is, so it means nothing if the estimator z-scores; and it assumes the trajectory has one
generating policy, so it means nothing under partial rollouts. Four ways to get a confident wrong
number, and access can see none of them.

An envelope makes all four checkable and makes the check appear in the reading.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Literal, Mapping

QuantityID = str
EstimatorID = str
EvidenceID = str


class RegimeCondition(enum.Enum):
    """The conditions an estimator can depend on, each measurable from a record.

    Twelve were there from the start. `DESIGN_CROSSED` is a thirteenth and it is a deliberate
    addition rather than a drift: A2's crossed-design qualifier had been carried as a hard
    precondition returning `ENVELOPE_VIOLATED` with the gap named in `deviations`, which works and
    is not what the type system is for.
    """

    #: Ad = tau_relax * |d log lambda / dt| below threshold.
    QUASI_STATIC = enum.auto()
    #: The step is small enough that the O(eta^2) term is negligible. Measured by Lambda.
    LINEAR_RESPONSE = enum.auto()
    #: K > 1 and std(r) > 0 for a stated fraction of groups.
    GROUP_NONDEGENERATE = enum.auto()
    #: Staleness below a bound, and segment provenance is singular.
    NEAR_POLICY = enum.auto()
    #: Grader weights and rubric weights unchanged across the window.
    STATIONARY_GRADER = enum.auto()
    #: The task distribution is not responding to the score.
    EXOGENOUS_CURRICULUM = enum.auto()
    #: No prefix rewrite inside the measurement window.
    NO_COMPACTION = enum.auto()
    #: The effect exceeds the limit of detection.
    ABOVE_LOD = enum.auto()
    #: Importance weights have not degenerated.
    ESS_ADEQUATE = enum.auto()
    #: The moment generating function exists; the Hill estimate is below a stated bound.
    LIGHT_TAILED = enum.auto()
    #: Curl mass below a stated bound.
    SCALAR_REPRESENTABLE = enum.auto()
    #: The loss-mask policy is unchanged across the window.
    MASK_STABLE = enum.auto()
    #: Every subject was scored by every rater at every occasion, so the expected-mean-square
    #: inversion the G-study rests on is the one that applies. The thirteenth condition, added
    #: because it is the cleanest instance in the library of the class an envelope exists for: an
    #: unbalanced design does not raise and nothing is missing, it just silently returns wrong
    #: variance components from an inversion that assumes a crossed design.
    DESIGN_CROSSED = enum.auto()


OnViolation = Literal["refuse", "bound", "downgrade"]


class EnvelopeLintError(Exception):
    """An envelope that cannot be enforced. Raised at construction, so it fails at import."""


@dataclass(frozen=True)
class EnvelopeSpec:
    """What this instrument needs to be TRUE.

    Two lint rules live in `__post_init__` rather than in a linter, because an envelope that
    cannot be enforced should not be constructible at all:

    An empty ``requires`` fails unless the instrument passes ``unconditional=True``, which is the
    code form of an explicit ``# envelope: unconditional`` justification and forces the author to
    write down that they thought about it.

    Every condition in ``requires`` must appear in ``measured_by``, so no instrument can declare a
    precondition nobody can check. A declared-but-unmeasurable precondition is worse than no
    precondition, because it reads as rigour and enforces nothing.

    And the third rule, which is easy to get wrong: ``on_violation``
    of ``"bound"`` without a ``bound_estimator`` is not a policy, it is a promise with nothing
    behind it. The type makes it impossible.
    """

    requires: frozenset[RegimeCondition] = frozenset()
    measured_by: Mapping[RegimeCondition, QuantityID] = field(default_factory=dict)
    on_violation: OnViolation = "refuse"
    bound_estimator: EstimatorID | None = None
    unconditional: bool = False
    justification: str = ""

    def __post_init__(self) -> None:
        if not self.requires and not self.unconditional:
            raise EnvelopeLintError(
                "an instrument with no regime preconditions must say so explicitly, by passing "
                "unconditional=True with a justification. Almost nothing is unconditional, and an "
                "empty envelope is far more often an author who has not looked than one who has."
            )
        if self.unconditional and not self.justification:
            raise EnvelopeLintError(
                "unconditional=True needs a justification saying why this estimator holds in every "
                "regime. One sentence."
            )
        # A condition mapped to an empty id is not measured. It appears in `measured_by`, which was
        # the whole of this check before, and it names nothing, so it passed the check and measured
        # nothing. That is the same defect as an unregistered id one step earlier, and the empty
        # spelling is the easier one to write by accident: a key with no value, a field defaulted to
        # "", a mapping built from a record whose measurer is still OPEN.
        named_conditions = {c for c, qid in self.measured_by.items() if qid}
        unmeasurable = self.requires - named_conditions
        if unmeasurable:
            names = ", ".join(sorted(c.name for c in unmeasurable))
            blank = sorted(
                c.name for c in (self.requires & set(self.measured_by)) - named_conditions
            )
            extra = (
                f" {', '.join(blank)} {'is' if len(blank) == 1 else 'are'} present in `measured_by` "
                f"with no quantity id, which is not a measurer."
                if blank
                else ""
            )
            raise EnvelopeLintError(
                f"envelope requires {names} but declares no way to measure {'it' if len(unmeasurable) == 1 else 'them'}. "
                f"Every condition in `requires` must appear in `measured_by` against a quantity id, "
                f"or the instrument is claiming a precondition nobody can check.{extra}"
            )
        # Deferred, and guarded on there being something to look up. `core.quantity` imports this
        # module and this module builds `UNCONDITIONAL` at import, so an unguarded import here
        # re-enters `core.quantity` while it is still initialising. Every envelope constructed
        # inside that window is unconditional and carries no `measured_by`, so asking the question
        # only when there is a name to resolve breaks the cycle without weakening the check.
        named = [qid for qid in self.measured_by.values() if qid]
        if named:
            from reward_lens.core.quantity import QUANTITIES

            unregistered = sorted(qid for qid in named if qid not in QUANTITIES)
        else:
            unregistered = []
        if unregistered:
            raise EnvelopeLintError(
                f"envelope names {', '.join(unregistered)} in `measured_by`, and "
                f"spec/QUANTITIES.yaml carries no such row. Appearing in `measured_by` was the "
                f"whole check that a precondition is measurable, so an id that resolves to nothing "
                f"passes that check and measures nothing. Register the quantity or name the one "
                f"that already exists."
            )
        if self.on_violation == "bound" and not self.bound_estimator:
            raise EnvelopeLintError(
                "on_violation='bound' promises a weaker estimator that survives outside the "
                "envelope, so it has to name one."
            )

    def admits(self, reading: "RegimeReading | None") -> bool:
        """Whether this envelope's conditions hold in a measured regime.

        A condition that could not be determined does **not** admit. Unknown is not a pass: the
        entire failure mode this module addresses is a check that did not happen reading as a
        check that succeeded.
        """
        if reading is None:
            return not self.requires
        return all(reading.holds(c) is True for c in self.requires)

    def classify(
        self, reading: "RegimeReading | None"
    ) -> tuple[list["ConditionReading"], list["ConditionReading"], list["RegimeCondition"]]:
        """Split the required conditions into (held, failed, never measured).

        `admits` deliberately folds the last two together, because as a gate it must: unknown is
        not a pass. But a *report* that folds them buries the real failures. One capability report
        over a synthetic run produced eighteen refusals of which sixteen were "nobody has measured
        `LINEAR_RESPONSE` yet", and the two that mattered were lost in the list.

        The split that makes both correct is per condition rather than per reading. A condition
        **absent from the reading** was never measured, and the honest report is that the check did
        not run. A condition **present with `holds=None`** was measured and came back
        indeterminate, which is a failure: somebody looked and could not tell.
        """
        held: list[ConditionReading] = []
        failed: list[ConditionReading] = []
        unmeasured: list[RegimeCondition] = []
        for c in sorted(self.requires, key=lambda c: c.name):
            cr = None if reading is None else reading.conditions.get(c)
            if cr is None:
                unmeasured.append(c)
            elif cr.holds is True:
                held.append(cr)
            else:
                failed.append(cr)
        return held, failed, unmeasured

    def violations(self, reading: "RegimeReading | None") -> list["ConditionReading"]:
        """The conditions that failed or could not be determined, for the refusal detail."""
        if reading is None:
            return [
                ConditionReading(
                    condition=c, holds=None, statistic=float("nan"), threshold=float("nan")
                )
                for c in sorted(self.requires, key=lambda c: c.name)
            ]
        out = []
        for c in sorted(self.requires, key=lambda c: c.name):
            cr = reading.conditions.get(c)
            if cr is None:
                out.append(
                    ConditionReading(
                        condition=c, holds=None, statistic=float("nan"), threshold=float("nan")
                    )
                )
            elif cr.holds is not True:
                out.append(cr)
        return out


UNCONDITIONAL = EnvelopeSpec(
    unconditional=True,
    justification=(
        "a census over a record: it counts what is there and asserts nothing about the process "
        "that produced it, so no regime can make the count wrong."
    ),
)


@dataclass(frozen=True)
class ConditionReading:
    """One measured condition, with the numbers that decided it."""

    condition: RegimeCondition
    #: True, False, or None for could-not-be-determined. Three states, and the third is real:
    #: it renders as `unknown` in the capability report rather than as a failure, because "we
    #: could not check" and "we checked and it fails" call for different responses.
    holds: bool | None
    statistic: float
    threshold: float
    provenance: EvidenceID | None = None
    detail: str = ""

    def render(self) -> str:
        state = {True: "ok", False: "FAIL", None: "unknown"}[self.holds]
        if self.holds is None and self.detail:
            return f"{self.condition.name:<22} {state:<8} {self.detail}"
        return (
            f"{self.condition.name:<22} {state:<8} "
            f"{self.statistic:.4g} (threshold {self.threshold:.4g})"
            + (f"  {self.detail}" if self.detail else "")
        )


@dataclass(frozen=True)
class RegimeReading:
    """The envelope, measured. Cheap, cached, consulted by every preflight."""

    conditions: Mapping[RegimeCondition, ConditionReading] = field(default_factory=dict)

    def holds(self, condition: RegimeCondition) -> bool | None:
        cr = self.conditions.get(condition)
        return None if cr is None else cr.holds

    def render(self) -> str:
        return "\n".join(
            self.conditions[c].render() for c in sorted(self.conditions, key=lambda c: c.name)
        )

    @classmethod
    def of(cls, **kwargs: bool | None) -> "RegimeReading":
        """Build a reading from condition names, for tests and for synthetic records.

        ``RegimeReading.of(GROUP_NONDEGENERATE=True, NEAR_POLICY=None)``. The statistic and
        threshold are NaN, which is honest: this constructor records a verdict somebody supplied
        rather than a measurement, and the rendered reading shows that.
        """
        conds = {}
        for name, holds in kwargs.items():
            c = RegimeCondition[name]
            conds[c] = ConditionReading(
                condition=c,
                holds=holds,
                statistic=float("nan"),
                threshold=float("nan"),
                detail="supplied, not measured",
            )
        return cls(conditions=conds)


__all__ = [
    "UNCONDITIONAL",
    "ConditionReading",
    "EnvelopeLintError",
    "EnvelopeSpec",
    "OnViolation",
    "RegimeCondition",
    "RegimeReading",
]

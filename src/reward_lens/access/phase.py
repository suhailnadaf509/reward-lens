"""When the question is being asked, resolved from what exists rather than from what is reachable.

Phase is the small dimension and it is in the type system for one reason: it is not derivable from
access, and every attempt to derive it produces a wrong answer for somebody. An auditor holding a
complete record is `POST_RUN` with `RECORD` access. A lab replaying its own record is also
`POST_RUN`, with `CONTROL` access, and it can stand up arms. Those two have nothing in common on
the access axis and the same answer on this one, so a resolver that reads the access matrix would
have to invent the difference.

This function therefore takes no `AccessMatrix`. It reads the four things that actually decide it:
whether a record exists, whether it is still being written, whether something in this process is
attached to a live loop, and whether all that is on hand is a served artifact. The signature is the
enforcement, which is why it is worth stating in the docstring.

What phase decides is whether a reading can still change a decision. An `IN_RUN` reading of a run
that is already finished is not wrong, it is late.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from reward_lens.access.matrix import RecordFacts, record_facts
from reward_lens.core.types import Phase


@dataclass(frozen=True)
class PhaseResolution:
    """The phase, and the observation that settled it.

    ``enforced`` is False when nobody supplied a phase and the report is carrying a placeholder so
    the field has a type. It renders as UNRESOLVED rather than as whatever the placeholder happens
    to be, because a printed `POST_RUN` that nobody established is the sort of thing a reader takes
    at face value.
    """

    phase: Phase
    note: str
    declared: bool = False
    enforced: bool = True

    def render(self) -> str:
        return self.phase.name if self.enforced else "UNRESOLVED"


def _declared_phase(value: Any) -> Phase | None:
    if isinstance(value, Phase):
        return value
    if isinstance(value, str):
        try:
            return Phase[value.strip().upper()]
        except KeyError:
            return None
    return None


def resolve_phase(
    *,
    record: Any = None,
    live: bool = False,
    artifact_only: bool = False,
    declared: Phase | str | None = None,
) -> PhaseResolution:
    """Resolve the phase. Deliberately takes no access matrix; see the module docstring.

    ``live`` is the caller saying a training loop is attached in this process, which is the tap's
    situation and the only thing that makes a question an in-run question. ``artifact_only`` is the
    deployed case, where the weights are serving and neither the record nor the grader is on hand.

    An open record beats ``live=False``. A record that says it is still being written is a run in
    progress whether or not this process is the one writing it, and reading it as finished would
    let a monitoring instrument report a final number on a partial run.
    """
    explicit = _declared_phase(declared)
    if explicit is not None:
        return PhaseResolution(explicit, "declared by the caller", declared=True)

    if artifact_only:
        return PhaseResolution(
            Phase.DEPLOYED, "an artifact is serving and no record or grader was supplied"
        )

    facts: RecordFacts = record_facts(record)
    if facts.complete is False:
        where = f" at step {facts.n_steps:,}" if facts.n_steps is not None else ""
        return PhaseResolution(Phase.IN_RUN, f"the record is still being written{where}")
    if live:
        return PhaseResolution(Phase.IN_RUN, "a training loop is attached in this process")
    if facts.is_present:
        where = f" of {facts.n_steps:,} steps" if facts.n_steps is not None else ""
        return PhaseResolution(Phase.POST_RUN, f"a closed record{where} was supplied")
    return PhaseResolution(
        Phase.PRE_RUN,
        "no record supplied, so nothing has been optimised that this could be after",
    )


__all__ = ["PhaseResolution", "resolve_phase"]

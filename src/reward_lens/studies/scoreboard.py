"""The theorem scoreboard.

The scoreboard registers theorem rows T1 through T14 (and any new ones) with a status in {open,
confirmed, refuted, mixed} linked to the adjudicating Evidence ids. T1 through T8 are standing
theorems the program instantiates inside real reward models; T9 through T14 are candidate laws this
program originates. The difference is a rendered property of the row, not a separate mechanism.

The design is emphatic on one point: refutations render as prominently as confirmations. A
scoreboard that hid its refutations would be a marketing document, not a scientific instrument, so
"refuted" is a first-class status with the same visual weight as "confirmed". The scoreboard
persists to a JSON file so it composes across studies and can be exported to the site.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from reward_lens.studies.spec import StudyResult

RowKind = Literal["standing_theorem", "candidate_law"]
RowStatus = Literal["open", "confirmed", "refuted", "mixed"]


@dataclass
class ScoreboardRow:
    """One theorem row."""

    id: str
    title: str
    kind: RowKind
    status: RowStatus = "open"
    adjudicating_evidence: list[str] = field(default_factory=list)
    studies: list[str] = field(default_factory=list)
    science: str = ""


# The standard rows. T1-T8 are standing theorems to instantiate; T9-T14 are the candidate laws
# this program originates. Titles are compressed from the design's scoreboard notes.
DEFAULT_ROWS: tuple[ScoreboardRow, ...] = (
    ScoreboardRow(
        "T1", "Constructive unhackable-subspace finder", "standing_theorem", science="S4"
    ),
    ScoreboardRow("T2", "Distortion equilibrium", "standing_theorem", science="S8/S12"),
    ScoreboardRow(
        "T3", "RLHF speed proportional to teacher variance", "standing_theorem", science="S12/S3"
    ),
    ScoreboardRow("T4", "Proxy-true reward angle", "standing_theorem", science="S2/S12"),
    ScoreboardRow("T5", "Heavy tail defeats KL control", "standing_theorem", science="S3/S4"),
    ScoreboardRow("T6", "Identifiability up to shift and scale", "standing_theorem", science="S2"),
    ScoreboardRow("T7", "No single scalar for a population", "standing_theorem", science="S11"),
    ScoreboardRow(
        "T8", "Scalar head cannot express intransitivity", "standing_theorem", science="S2"
    ),
    ScoreboardRow(
        "T9", "Fluctuation-dissipation for reward hacking", "candidate_law", science="S3"
    ),
    ScoreboardRow(
        "T10", "Belief factorization and gauge=channel-kernel", "candidate_law", science="S8/S2"
    ),
    ScoreboardRow(
        "T11", "Evaluator-model divergence precedes hacking", "candidate_law", science="S13/AT"
    ),
    ScoreboardRow(
        "T12", "Coherence/Welch law and Hodge obstruction", "candidate_law", science="S5/S6"
    ),
    ScoreboardRow("T13", "Value convergence excess", "candidate_law", science="AT"),
    ScoreboardRow("T14", "Honesty unraveling law", "candidate_law", science="S15"),
)


class Scoreboard:
    """A persisted registry of theorem rows and their status."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else None
        self.rows: dict[str, ScoreboardRow] = {
            r.id: ScoreboardRow(**asdict(r)) for r in DEFAULT_ROWS
        }
        if self.path and self.path.exists():
            self._load()

    def _load(self) -> None:
        assert self.path is not None
        data = json.loads(self.path.read_text())
        for rid, row in data.items():
            self.rows[rid] = ScoreboardRow(**row)

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({rid: asdict(r) for rid, r in self.rows.items()}, indent=2))

    def register_row(self, row: ScoreboardRow) -> None:
        self.rows[row.id] = row

    def update_from_result(self, study_id: str, spec_hypotheses, result: StudyResult) -> None:
        """Fold a study's outcomes into the rows its hypotheses address.

        A hypothesis maps to a scoreboard row via ``scoreboard_row``. Confirmed and refuted outcomes
        update the row's status; a row that receives both confirming and refuting evidence across
        hypotheses or studies becomes "mixed". The adjudicating Evidence ids are recorded so the row
        can cite exactly what settled it.
        """
        for h in spec_hypotheses:
            if not h.scoreboard_row:
                continue
            row = self.rows.get(h.scoreboard_row)
            if row is None:
                continue
            outcome = result.outcomes.get(h.id, "void")
            if study_id not in row.studies:
                row.studies.append(study_id)
            row.adjudicating_evidence.extend(
                e for e in result.evidence if e not in row.adjudicating_evidence
            )
            row.status = _merge_status(row.status, outcome)
        self.save()

    def render_markdown(self) -> str:
        lines = [
            "| Row | Title | Kind | Status | Science | Adjudicating evidence |",
            "|---|---|---|---|---|---|",
        ]
        for rid in sorted(self.rows, key=_row_sort_key):
            r = self.rows[rid]
            kind = "standing" if r.kind == "standing_theorem" else "candidate law"
            ev = ", ".join(r.adjudicating_evidence[:3]) + (
                "..." if len(r.adjudicating_evidence) > 3 else ""
            )
            status = r.status.upper() if r.status in ("confirmed", "refuted") else r.status
            lines.append(f"| {r.id} | {r.title} | {kind} | {status} | {r.science} | {ev} |")
        return "\n".join(lines)


def _merge_status(current: RowStatus, outcome: str) -> RowStatus:
    # A void does not move a row. It was not read, so it is not evidence in either direction.
    if outcome == "void":
        return current
    mapped: RowStatus = "confirmed" if outcome == "confirmed" else "refuted"
    if current == "open":
        return mapped
    if current == mapped:
        return current
    return "mixed"


def _row_sort_key(rid: str) -> tuple[int, str]:
    try:
        return (int(rid[1:]), rid)
    except ValueError:
        return (999, rid)


__all__ = ["ScoreboardRow", "Scoreboard", "DEFAULT_ROWS", "RowKind", "RowStatus"]

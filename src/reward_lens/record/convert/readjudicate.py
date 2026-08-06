"""Re-adjudicating the campaign's twenty-seven cards through the void-aware runner.

A missing metric produces `VOID` with a named reason, and this runs that against the real thing:
the campaign's own frozen specs, its own recorded metrics, and the runner as it now stands. Nothing
is recomputed. The metrics are read back out of the ``campaign.adjudication.*`` rows exactly as
they were written, so the only thing that changes between the campaign's verdicts and these is the
adjudication code.

**Why the frozen study is rebuilt rather than re-frozen.** `run_study` freezes a bare `StudySpec`
before running it, which would stamp today's date and the current git sha onto a study that was
frozen on 2026-07-18 against commit ``f93f4b5``. The whole content of gate 3 is that the freeze
predates the evidence, so a re-adjudication that re-freezes has destroyed the property it is
supposed to be checking. `frozen_study` rebuilds the `FrozenStudy` from the campaign's own
``specs/frozen/*.json`` and `verify_spec_hash` checks that the rebuilt spec hashes to the recorded
``spec_hash``, which is what makes "the same spec" a measurement rather than a claim.

**Why the store is opened read-only.** `run_study` takes a store and appends to it whenever the
analysis calls `measure` or `record`. The analysis here does neither, and passing the campaign's
own store opened read-only turns that from a promise into a guarantee: an append would raise.

**What the eight blocked cards were blocked by.** Six of them name a missing intermediate
observable with the roster key and the bank it was wanted for; two failed at subject resolution
with a `PermissionError`. `metric_arcs_from_reason` turns both into an arc id, so the void that
comes out names the metric, the hypothesis that registered it, and the arc that was supposed to
produce it. That is the difference between "no value for ``campaign.bias.battery``" and a work
item.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from reward_lens.core.store import EvidenceStore
from reward_lens.core.types import StudyID, content_hash
from reward_lens.record.convert.store import ADJUDICATION_PREFIX, CampaignRow, CampaignStore
from reward_lens.studies.freeze import FrozenStudy
from reward_lens.studies.runner import run_study
from reward_lens.studies.spec import (
    Hypothesis,
    KillCriterion,
    Prediction,
    StudyResult,
    StudySpec,
    SubjectQuery,
)
from reward_lens.studies.void import StudyOutcome, Void

#: The campaign's own phrasing for a metric whose upstream observable never landed. Six of the
#: eight blocked cards carry it; the pattern is matched rather than the whole string compared,
#: because the observable, the roster and the bank differ per card and they are the parts that
#: make the arc a work item.
MISSING_INTERMEDIATE = re.compile(
    r"no intermediate '(?P<observable>[^']+)'"
    r"(?: for roster_key='(?P<roster>[^']*)')?"
    r"(?: slice='(?P<slice>[^']*)')?"
)

#: The other two blocked cards. `resolve_subjects` is the campaign's own step that turns a study's
#: declared subjects into loaded objects, and it hit a filesystem permission error on the machine
#: the close-out ran on.
SUBJECT_FAILURE = re.compile(r"resolve_subjects failed: (?P<error>.+)$")


def _spec_from_canonical(obj: Mapping[str, Any]) -> StudySpec:
    """Rebuild a `StudySpec` from `StudySpec.__canonical__`.

    The inverse is not in `studies.spec`, which is why it is here. It has to be exact, because the
    spec hash is computed over the canonical form and a field reconstructed with the wrong default
    changes the hash and so changes the study id.
    """
    subjects = obj.get("subjects", {}) or {}
    return StudySpec(
        id=obj["id"],
        title=obj["title"],
        science=obj["science"],
        hypotheses=tuple(
            Hypothesis(
                id=h["id"],
                statement=h["statement"],
                prediction=Prediction(
                    metric=h["prediction"]["metric"],
                    comparator=h["prediction"]["comparator"],
                    threshold=h["prediction"]["threshold"],
                    effect=h["prediction"].get("effect"),
                    ci_excludes=h["prediction"].get("ci_excludes"),
                ),
                scoreboard_row=h.get("scoreboard_row"),
            )
            for h in obj.get("hypotheses", ())
        ),
        analysis=obj["analysis"],
        subjects=SubjectQuery(
            signals=tuple(subjects.get("signals", ()) or ()),
            organisms=tuple(subjects.get("organisms", ()) or ()),
            datasets=tuple(subjects.get("datasets", ()) or ()),
            extra=dict(subjects.get("extra", {}) or {}),
        ),
        kill_criteria=tuple(
            KillCriterion(
                id=k["id"],
                metric=k["metric"],
                comparator=k["comparator"],
                threshold=k["threshold"],
            )
            for k in obj.get("kill_criteria", ())
        ),
        version=obj.get("version", 1),
    )


def frozen_study(document: Mapping[str, Any]) -> FrozenStudy:
    """Rebuild the campaign's `FrozenStudy` from one ``specs/frozen/*.json`` document."""
    spec = _spec_from_canonical(document["spec"])
    return FrozenStudy(
        spec=spec,
        study_id=StudyID(document["study_id"]),
        git_sha=document["git_sha"],
        frozen_at=document["frozen_at"],
        spec_hash=document["spec_hash"],
    )


def verify_spec_hash(frozen: FrozenStudy) -> bool:
    """Whether the rebuilt spec hashes to the hash the campaign recorded.

    False means the reconstruction differs from what was frozen somewhere, and every verdict
    computed from it would be a verdict about a different study.
    """
    return content_hash(frozen.spec.__canonical__(), "spec") == frozen.spec_hash


def load_frozen_specs(directory: str | Path) -> dict[str, FrozenStudy]:
    """Every frozen study in a directory, keyed by spec id.

    Documents without a ``spec`` block are skipped: the campaign's freeze directory also holds a
    manifest and three analysis-input files, and those are not studies.
    """
    out: dict[str, FrozenStudy] = {}
    for path in sorted(Path(directory).glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or "spec" not in document:
            continue
        if "hypotheses" not in (document.get("spec") or {}):
            continue
        frozen = frozen_study(document)
        out[frozen.spec.id] = frozen
    return out


def metric_arcs_from_reason(reason: str, *, spec_id: str, metrics: Iterable[str]) -> dict[str, str]:
    """Map every metric a blocked card registered to the arc that was supposed to produce it.

    The campaign recorded one reason per card rather than one per metric, because the analysis
    stopped at the first missing input. So every absent metric on a blocked card gets the same arc,
    which is honest: they were all blocked by the same thing, and a per-metric arc would be a
    precision the record does not have.
    """
    match = MISSING_INTERMEDIATE.search(reason or "")
    if match is not None:
        observable = match.group("observable")
        roster = match.group("roster") or ""
        bank = match.group("slice") or ""
        suffix = f"@{roster}/{bank}" if roster or bank else ""
        arc = f"arc:{observable}{suffix}"
        return {m: arc for m in metrics}
    match = SUBJECT_FAILURE.search(reason or "")
    if match is not None:
        return {m: f"arc:{spec_id}/resolve_subjects" for m in metrics}
    return {}


def _recorded(row: CampaignRow, store: CampaignStore) -> Mapping[str, Any]:
    value = store.value(row)
    return value if isinstance(value, dict) else {}


@dataclass(frozen=True)
class CardReadjudication:
    """One card, as the campaign recorded it and as the void-aware runner reads it now."""

    card: str
    spec_id: str
    frozen: FrozenStudy
    spec_hash_verified: bool
    recorded_status: str
    recorded_reason: str
    recorded_outcomes: Mapping[str, str]
    metrics: Mapping[str, float]
    result: StudyResult

    @property
    def outcome(self) -> StudyOutcome:
        return self.result.outcome

    @property
    def hypothesis_voids(self) -> tuple[str, ...]:
        return tuple(h for h, o in self.result.outcomes.items() if o == "void")

    @property
    def kill_voids(self) -> tuple[str, ...]:
        return tuple(k for k, o in self.result.kill_outcomes.items() if o == "void")

    @property
    def voids(self) -> Mapping[str, Void]:
        return self.result.voids

    @property
    def changed(self) -> bool:
        """Whether any hypothesis outcome differs from what the campaign recorded."""
        return any(self.recorded_outcomes.get(h) != o for h, o in self.result.outcomes.items())


@dataclass
class ReadjudicationReport:
    """The counts the re-adjudication is stated in, plus everything behind them."""

    cards: tuple[CardReadjudication, ...] = ()
    missing_specs: tuple[str, ...] = ()

    @property
    def inconclusive_cards(self) -> int:
        """Card-level rows still carrying the word. The target is zero."""
        return sum(1 for c in self.cards if str(getattr(c.outcome, "value", "")) == "inconclusive")

    @property
    def inconclusive_hypotheses(self) -> int:
        """Hypothesis-level rows still carrying the word. The target is zero."""
        return sum(1 for c in self.cards for o in c.result.outcomes.values() if o == "inconclusive")

    @property
    def void_cards(self) -> tuple[str, ...]:
        return tuple(c.card for c in self.cards if c.outcome is StudyOutcome.VOID)

    @property
    def void_hypotheses(self) -> tuple[tuple[str, str], ...]:
        return tuple((c.card, h) for c in self.cards for h in c.hypothesis_voids)

    @property
    def void_kills(self) -> tuple[tuple[str, str], ...]:
        return tuple((c.card, k) for c in self.cards for k in c.kill_voids)

    @property
    def recorded_inconclusive_cards(self) -> tuple[str, ...]:
        return tuple(c.card for c in self.cards if c.recorded_status == "inconclusive")

    @property
    def recorded_inconclusive_hypotheses(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (c.card, h)
            for c in self.cards
            for h, o in c.recorded_outcomes.items()
            if o == "inconclusive"
        )

    @property
    def unverified_specs(self) -> tuple[str, ...]:
        return tuple(c.card for c in self.cards if not c.spec_hash_verified)

    def unnamed_voids(self) -> tuple[str, ...]:
        """Voids whose detail does not name the metric that was absent. Should be empty.

        The reason string is the whole value of the distinction. A `VOID` with no named arc is
        `inconclusive` spelled differently, so this is the check that the reason survived.
        """
        out: list[str] = []
        for card in self.cards:
            for key, void in card.voids.items():
                metric = _metric_for(card, key)
                if metric is None or metric not in void.detail:
                    out.append(f"{card.card}/{key}")
        return tuple(out)

    def arcless_voids(self) -> tuple[str, ...]:
        """Voids that name no arc. Every void here should name one; see `metric_arcs_from_reason`."""
        return tuple(f"{c.card}/{k}" for c in self.cards for k, v in c.voids.items() if not v.arc)

    def render(self) -> str:
        lines = [
            f"re-adjudicated {len(self.cards)} cards from their frozen specs",
            f"  card-level inconclusive: {self.inconclusive_cards} "
            f"(campaign recorded {len(self.recorded_inconclusive_cards)})",
            f"  hypothesis-level inconclusive: {self.inconclusive_hypotheses} "
            f"(campaign recorded {len(self.recorded_inconclusive_hypotheses)})",
            f"  card-level VOID: {len(self.void_cards)}  {sorted(self.void_cards)}",
            f"  hypothesis-level VOID: {len(self.void_hypotheses)}",
            f"  kill-criterion VOID: {len(self.void_kills)}",
            f"  spec hashes verified: {len(self.cards) - len(self.unverified_specs)}"
            f"/{len(self.cards)}",
        ]
        for card, hypothesis in self.void_hypotheses:
            void = next(c for c in self.cards if c.card == card).voids[hypothesis]
            lines.append(f"    {card}/{hypothesis}: {void.arc or 'no arc'}")
        return "\n".join(lines)


def _metric_for(card: CardReadjudication, key: str) -> str | None:
    for h in card.frozen.spec.hypotheses:
        if h.id == key:
            return h.prediction.metric
    for k in card.frozen.spec.kill_criteria:
        if k.id == key:
            return k.metric
    return None


def readjudicate(
    store_path: str | Path,
    specs_dir: str | Path,
    *,
    sidecar_dirs: Sequence[str | Path] = (),
    only: Sequence[str] = (),
) -> ReadjudicationReport:
    """Re-run every campaign card's adjudication against its own frozen spec and metrics.

    ``only`` restricts to named cards, which is what a fast test uses. The store is opened
    read-only and handed to `run_study`, so nothing this function does can write to the archive.
    """
    campaign = CampaignStore(store_path, sidecar_dirs=sidecar_dirs)
    campaign.assert_no_blind_payloads()
    specs = load_frozen_specs(specs_dir)
    readonly = EvidenceStore(Path(store_path), readonly=True)

    cards: list[CardReadjudication] = []
    missing: list[str] = []
    wanted = set(only)
    for row in campaign.by_prefix(ADJUDICATION_PREFIX):
        card = row.card or row.observable.rsplit(".", 1)[-1]
        if wanted and card not in wanted:
            continue
        spec_id = str(row.extra.get("spec", ""))
        frozen = specs.get(spec_id)
        if frozen is None:
            missing.append(card)
            continue
        recorded = _recorded(row, campaign)
        metrics = {str(k): v for k, v in (recorded.get("metrics") or {}).items()}
        recorded_outcomes = {str(k): str(v) for k, v in (recorded.get("outcomes") or {}).items()}
        reason = str(recorded.get("reason") or "")
        registered = [h.prediction.metric for h in frozen.spec.hypotheses]
        registered += [k.metric for k in frozen.spec.kill_criteria]
        arcs = metric_arcs_from_reason(
            reason,
            spec_id=spec_id,
            metrics=[m for m in registered if metrics.get(m) is None],
        )

        def analysis(_run: Any, _metrics: Mapping[str, float] = metrics) -> StudyResult:
            """Hand the runner the metrics the campaign computed, and nothing else.

            The runner adjudicates against the frozen predictions rather than against anything the
            analysis claims, which is the split that stops a science moving a threshold after
            seeing the data. Here it is also what makes the comparison clean: the metrics are
            byte-identical to the campaign's and only the adjudication has changed.
            """
            return StudyResult(outcomes={}, metrics=dict(_metrics))

        _, result = run_study(
            frozen,
            subjects={},
            store=readonly,
            analysis_fn=analysis,
            metric_arcs=arcs,
        )
        cards.append(
            CardReadjudication(
                card=card,
                spec_id=spec_id,
                frozen=frozen,
                spec_hash_verified=verify_spec_hash(frozen),
                recorded_status=str(recorded.get("status", "")),
                recorded_reason=reason,
                recorded_outcomes=recorded_outcomes,
                metrics=metrics,
                result=result,
            )
        )
    cards.sort(key=lambda c: c.card)
    return ReadjudicationReport(cards=tuple(cards), missing_specs=tuple(sorted(missing)))


__all__ = [
    "MISSING_INTERMEDIATE",
    "SUBJECT_FAILURE",
    "CardReadjudication",
    "ReadjudicationReport",
    "frozen_study",
    "load_frozen_specs",
    "metric_arcs_from_reason",
    "readjudicate",
    "verify_spec_hash",
]

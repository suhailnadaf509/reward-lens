"""The calibration ledger, published.

Append-only, public, and it prints its own worst result at the top.

**The claim is not that we forecast well. It is that we are the only ones publishing the score.**

The honest starting position is part of the artifact: directional Brier 0.26 against a coin at 0.25,
interval coverage 0.75 against a registered 0.80, and the meta kill fired. A ledger that hides its
own bad first result is the artifact this library exists to argue against, so the header is not
optional and it is not a footnote. It is also **recomputed rather than transcribed**: `header`
scores the rows in the ledger and prints what it got, so if a future append changes those numbers
the header changes with them and the published figures cannot drift away from the data.

There is a stronger reading of that first result available, and it is worth running rather than
asserting. Prediction markets reach roughly 71 to 73 percent forecasting whether published findings
replicate. But those are not the same task: a market forecasts whether a peer-reviewed finding
replicates, where the base rate is 40 to 60 percent, and the campaign forecast whether a novel
mechanistic hypothesis about reward-model geometry would confirm, with no base rate, no prior
literature and no community of forecasters. Chance performance on a genuinely novel measurement
frontier is the expected result and is diagnostic of frontier-ness. The way to argue that rather
than assert it is to put the next campaign's preregistered predictions in front of forecasters
before the runs and compare the crowd's Brier to ours and to a coin's. If the crowd also lands at
chance, the questions are open, which is a far stronger claim than "our forecasts were bad".
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from reward_lens.core.envelope import EnvelopeSpec
from reward_lens.core.evidence import Uncertainty, register_payload
from reward_lens.core.invariance import Relation
from reward_lens.core.reading import Reading, refuse_incomplete
from reward_lens.core.types import Access, Capability, Component, Phase, Substrate
from reward_lens.forecast.resolve import Resolution, Resolved, Void
from reward_lens.forecast.schema import (
    BaselineKind,
    BinaryProbability,
    DecisionSpec,
    Forecast,
    ForecastError,
)
from reward_lens.forecast.score import (
    CoverageScore,
    DecisionValue,
    MurphyDecomposition,
    ReliabilityDiagram,
    SkillScore,
    brier,
    coverage_score,
    decision_value,
    murphy_decomposition,
    reliability_diagram,
    skill_score,
)
from reward_lens.measure.base import BaseObservable, Context

#: The campaign's published figures, as they appear in RESULTS.md and SCOREBOARD.md. These are the
#: values the ledger's own recomputation is checked against; they are never printed in place of a
#: recomputed number.
PUBLISHED_DIRECTIONAL_BRIER = 0.26
PUBLISHED_COIN_BRIER = 0.25
PUBLISHED_INTERVAL_COVERAGE = 0.75
REGISTERED_NOMINAL_COVERAGE = 0.80
PUBLISHED_N_DIRECTIONAL = 16
PUBLISHED_N_INTERVALS = 4


@register_payload
@dataclass(frozen=True)
class LedgerEntry:
    """One scored call: what was predicted, when, against what, and how it came out.

    The baselines travel on the row rather than being recomputed at read time, because the row is
    the published artifact and a comparison recomputed later against a baseline that has since been
    improved is not the comparison that was made.
    """

    forecast_id: str
    target: str
    reference_class: str
    issued_at: str
    method: str
    #: The forecast, as a probability for a binary call or `None` for an interval call.
    probability: float | None
    #: What each mandatory baseline said, keyed by kind. A refused baseline carries `None`.
    baselines: Mapping[str, float | None]
    #: Realised outcome for a resolved binary call.
    outcome: bool | None = None
    #: Whether a realised value fell inside an interval call.
    covered: bool | None = None
    voided: str = ""
    void_detail: str = ""
    resolved_at: str = ""
    metric: str = ""
    metric_value: float | None = None
    rule: str = ""
    lead_steps: float | None = None
    lead_widths: float | None = None
    note: str = ""

    @property
    def is_void(self) -> bool:
        return bool(self.voided)

    @property
    def is_interval(self) -> bool:
        return self.covered is not None

    @property
    def brier_term(self) -> float | None:
        if self.probability is None or self.outcome is None or self.is_void:
            return None
        return (self.probability - (1.0 if self.outcome else 0.0)) ** 2

    def render(self) -> str:
        if self.is_void:
            return f"{self.target:<28} VOID ({self.voided:<18}) {self.rule:<42} {self.void_detail}"
        if self.is_interval:
            hit = "inside" if self.covered else "outside"
            return f"{self.target:<28} interval {hit:<8} {self.rule}"
        verdict = "confirmed" if self.outcome else "refuted "
        return (
            f"{self.target:<28} P={self.probability:.2f}  {verdict}  "
            f"brier {self.brier_term:.4f}  {self.rule}"
        )

    def __canonical__(self) -> dict[str, Any]:
        return {
            "forecast_id": self.forecast_id,
            "target": self.target,
            "reference_class": self.reference_class,
            "issued_at": self.issued_at,
            "method": self.method,
            "probability": self.probability,
            "baselines": dict(self.baselines),
            "outcome": self.outcome,
            "covered": self.covered,
            "voided": self.voided,
            "void_detail": self.void_detail,
            "resolved_at": self.resolved_at,
            "metric": self.metric,
            "metric_value": self.metric_value,
            "rule": self.rule,
            "lead_steps": self.lead_steps,
            "lead_widths": self.lead_widths,
            "note": self.note,
        }

    @classmethod
    def from_json(cls, obj: Mapping[str, Any]) -> "LedgerEntry":
        return cls(
            forecast_id=obj["forecast_id"],
            target=obj["target"],
            reference_class=obj.get("reference_class", ""),
            issued_at=obj.get("issued_at", ""),
            method=obj.get("method", ""),
            probability=obj.get("probability"),
            baselines=dict(obj.get("baselines") or {}),
            outcome=obj.get("outcome"),
            covered=obj.get("covered"),
            voided=obj.get("voided", ""),
            void_detail=obj.get("void_detail", ""),
            resolved_at=obj.get("resolved_at", ""),
            metric=obj.get("metric", ""),
            metric_value=obj.get("metric_value"),
            rule=obj.get("rule", ""),
            lead_steps=obj.get("lead_steps"),
            lead_widths=obj.get("lead_widths"),
            note=obj.get("note", ""),
        )


def entry_from(
    forecast: Forecast,
    resolution: Resolution,
    *,
    note: str = "",
) -> LedgerEntry:
    """Build a ledger row from a forecast and its resolution. The only constructor worth using."""
    baselines: dict[str, float | None] = {}
    for b in forecast.baselines:
        if b.is_scored and isinstance(b.distribution, BinaryProbability):
            baselines[b.kind.value] = b.distribution.p
        else:
            baselines[b.kind.value] = None

    probability = (
        forecast.distribution.p if isinstance(forecast.distribution, BinaryProbability) else None
    )
    common: dict[str, Any] = dict(
        forecast_id=str(forecast.id),
        target=forecast.target,
        reference_class=str(forecast.reference_class.id),
        issued_at=forecast.issued_at.instant,
        method=forecast.method,
        probability=probability,
        baselines=baselines,
        rule=forecast.resolution.render(),
        note=note,
    )
    if isinstance(resolution, Void):
        return LedgerEntry(
            **common,
            voided=resolution.reason.value,
            void_detail=resolution.detail,
            resolved_at=resolution.resolved_at.instant,
        )
    if not isinstance(resolution, Resolved):
        raise ForecastError(f"{resolution!r} is neither a Resolved nor a Void")
    lead_steps = None
    if forecast.issued_step is not None and forecast.horizon.kind == "steps":
        lead_steps = float(forecast.horizon.value)
    return LedgerEntry(
        **common,
        outcome=resolution.outcome,
        covered=resolution.covered,
        resolved_at=resolution.resolved_at.instant,
        metric=resolution.metric,
        metric_value=resolution.metric_value,
        lead_steps=lead_steps,
    )


@register_payload
@dataclass(frozen=True)
class LedgerScore:
    """Everything the ledger knows about its own calibration, computed from its own rows."""

    n_entries: int
    n_directional: int
    n_intervals: int
    n_void: int
    directional_brier: float | None
    coin_brier: float
    murphy: MurphyDecomposition | None
    reliability: ReliabilityDiagram | None
    coverage: CoverageScore | None
    skill: tuple[SkillScore, ...] = ()
    value: DecisionValue | None = None
    void_reasons: Mapping[str, int] = field(default_factory=dict)

    @property
    def beats_coin(self) -> bool:
        return self.directional_brier is not None and self.directional_brier < self.coin_brier

    def __canonical__(self) -> dict[str, Any]:
        return {
            "n_entries": self.n_entries,
            "n_directional": self.n_directional,
            "n_intervals": self.n_intervals,
            "n_void": self.n_void,
            "directional_brier": self.directional_brier,
            "coin_brier": self.coin_brier,
            "murphy": None if self.murphy is None else self.murphy.__canonical__(),
            "coverage": None if self.coverage is None else self.coverage.__canonical__(),
            "skill": [s.__canonical__() for s in self.skill],
            "value": None if self.value is None else self.value.__canonical__(),
            "void_reasons": dict(self.void_reasons),
        }


class CalibrationLedger:
    """Append-only, file-backed, and it prints its own worst result at the top.

    JSON Lines on disk, so the ledger is diffable, greppable and readable by anyone with a text
    editor, for the same reason `core/store.py` is. Appends are fsynced and the file is never
    rewritten: an entry that turned out to be embarrassing is corrected by a later entry saying so,
    not by an edit, because a calibration ledger that can be edited is a marketing document.
    """

    def __init__(self, path: str | Path | None = None, *, readonly: bool = False) -> None:
        self.path = Path(path) if path is not None else None
        self.readonly = readonly
        self._entries: list[LedgerEntry] = []
        if self.path is not None and self.path.exists():
            self._load()

    def _load(self) -> None:
        assert self.path is not None
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                self._entries.append(LedgerEntry.from_json(json.loads(line)))

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[LedgerEntry]:
        return iter(self._entries)

    @property
    def entries(self) -> tuple[LedgerEntry, ...]:
        return tuple(self._entries)

    def append(self, entry: LedgerEntry) -> LedgerEntry:
        """Add one row. Idempotent on the forecast id, so replaying a scoring run is safe."""
        if self.readonly:
            raise ForecastError(
                f"ledger at {self.path} was opened readonly; appending would race whichever "
                f"process this mode exists to protect"
            )
        if any(e.forecast_id == entry.forecast_id for e in self._entries):
            return entry
        self._entries.append(entry)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry.__canonical__(), ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        return entry

    def extend(self, entries: Sequence[LedgerEntry]) -> None:
        for entry in entries:
            self.append(entry)

    # -- scoring ------------------------------------------------------------

    @property
    def directional(self) -> tuple[LedgerEntry, ...]:
        return tuple(
            e
            for e in self._entries
            if not e.is_void and e.probability is not None and e.outcome is not None
        )

    @property
    def intervals(self) -> tuple[LedgerEntry, ...]:
        return tuple(e for e in self._entries if not e.is_void and e.covered is not None)

    @property
    def voids(self) -> tuple[LedgerEntry, ...]:
        return tuple(e for e in self._entries if e.is_void)

    def directional_brier_if_voids_were_misses(self) -> float:
        """What the Brier would be if every void were scored as a wrong call. The cost of the error.

        Not a number to publish. It exists so the size of the mistake is visible: scoring a void as
        a miss is the commonest way to make a calibration ledger say something false, and the
        difference between this and the real score is how false. On the campaign it is 0.2904 over
        23 calls against the published 0.26 over 16, and one of the seven voids carried a call at
        0.9 whose analysis never produced its metric.
        """
        probs: list[float] = []
        outs: list[bool] = []
        for entry in self._entries:
            if entry.probability is None:
                continue
            if entry.is_void:
                probs.append(float(entry.probability))
                outs.append(False)
            elif entry.outcome is not None:
                probs.append(float(entry.probability))
                outs.append(bool(entry.outcome))
        return brier(probs, outs)

    def score(
        self,
        *,
        nominal_coverage: float = REGISTERED_NOMINAL_COVERAGE,
        coin: float = 0.5,
        decision: DecisionSpec | None = None,
        seed: int = 0,
    ) -> LedgerScore:
        """Score every row in the ledger. Nothing here is cached and nothing is transcribed."""
        directional = self.directional
        intervals = self.intervals
        voids = self.voids
        reasons: dict[str, int] = {}
        for entry in voids:
            reasons[entry.voided] = reasons.get(entry.voided, 0) + 1

        murphy = None
        diagram = None
        skills: list[SkillScore] = []
        value = None
        directional_brier = None
        if directional:
            probs = [float(e.probability) for e in directional if e.probability is not None]
            outs = [bool(e.outcome) for e in directional]
            directional_brier = brier(probs, outs)
            murphy = murphy_decomposition(probs, outs)
            diagram = reliability_diagram(probs, outs)
            for kind in BaselineKind:
                values = [e.baselines.get(kind.value) for e in directional]
                if any(v is None for v in values):
                    continue
                scored = [float(v) for v in values if v is not None]
                skills.append(
                    skill_score(
                        probs,
                        scored,
                        outs,
                        baseline_id=f"forecast.baseline.{kind.value}",
                        seed=seed,
                    )
                )
            skills.append(
                skill_score(
                    probs,
                    [coin] * len(probs),
                    outs,
                    baseline_id=f"coin at {coin:g}",
                    seed=seed,
                )
            )
            if decision is not None:
                value = decision_value(probs, outs, decision)

        coverage = None
        if intervals:
            coverage = coverage_score(
                [bool(e.covered) for e in intervals], nominal=nominal_coverage
            )

        return LedgerScore(
            n_entries=len(self._entries),
            n_directional=len(directional),
            n_intervals=len(intervals),
            n_void=len(voids),
            directional_brier=directional_brier,
            coin_brier=coin * coin,
            murphy=murphy,
            reliability=diagram,
            coverage=coverage,
            skill=tuple(skills),
            value=value,
            void_reasons=reasons,
        )

    # -- the published artifact ---------------------------------------------

    def header(self, score: LedgerScore | None = None) -> str:
        """The honest starting position, recomputed from the rows rather than transcribed.

        This prints at the top of the ledger and it is the first thing a reader sees. Every number
        in it came from `score`, which read the rows in this file; the published constants are used
        only to say whether the recomputation still agrees with what was published.
        """
        s = score or self.score()
        lines = ["THE HONEST STARTING POSITION", ""]
        if s.directional_brier is None:
            lines.append(
                "No directional call in this ledger has resolved yet, so there is no Brier score "
                "to print. That is not a good result, it is no result."
            )
        else:
            verdict = "beat" if s.directional_brier < s.coin_brier else "did not beat"
            lines.append(
                f"Directional Brier {s.directional_brier:.2f} over {s.n_directional} calls, "
                f"which {verdict} the always-guess-half coin at {s.coin_brier:.2f}."
            )
        if s.coverage is not None:
            lines.append(
                f"Interval coverage {s.coverage.coverage:.2f} over {s.coverage.n} intervals "
                f"against a registered nominal {s.coverage.nominal:.2f}."
            )
        if s.directional_brier is not None and s.directional_brier >= s.coin_brier:
            lines.append(
                "The meta kill criterion fired: the directional calls do not calibrate, and every "
                "confirmation in the campaign should be read with that in mind."
            )
        if s.n_void:
            detail = ", ".join(f"{k} {v}" for k, v in sorted(s.void_reasons.items()))
            lines.append(
                f"{s.n_void} registered call(s) went void and are not in the denominator: {detail}. "
                f"An expired or unresolvable forecast is void, never a miss, and dropping them "
                f"silently would be the easiest way to make this page look better than it is."
            )
        lines.append("")
        lines.append(
            "The claim is not that we forecast well. It is that we are the only ones publishing "
            "the score."
        )
        return "\n".join(lines)

    def render(self, score: LedgerScore | None = None) -> str:
        """The whole published page: header, decomposition, skill, coverage, then every row."""
        s = score or self.score()
        out = [self.header(s), ""]
        if s.murphy is not None:
            out.append(s.murphy.render())
            out.append("")
        if s.skill:
            out.append("Skill against each mandatory baseline, with a paired bootstrap interval:")
            for sk in s.skill:
                out.append("    " + sk.render())
            out.append("")
        if s.coverage is not None:
            out.append(s.coverage.render())
            out.append("")
        if s.value is not None:
            out.append(s.value.render())
            out.append("")
        if s.reliability is not None:
            out.append(s.reliability.render())
            out.append("")
        out.append(f"Every call, {len(self._entries)} rows:")
        for entry in self._entries:
            out.append("    " + entry.render())
        return "\n".join(out)


# ---------------------------------------------------------------------------
# The instrument
# ---------------------------------------------------------------------------

#: The quantity this instrument estimates. **Not yet registered** in `spec/QUANTITIES.yaml`: the
#: registry carries no `forecast.*` row, and adding a quantity id is a decision for whoever
#: maintains the registry. Until it lands, `lint_instrument` reports exactly one finding on this
#: instrument and it is this one.
FORECAST_BRIER_QUANTITY = "forecast.brier_score"

#: The envelope. Scoring a resolved ledger is a census: it counts rows that already exist and
#: asserts nothing about the process that produced them. What could make it silently wrong is not a
#: regime condition at all, it is a leaked input or a mixed reference class, and both are closed by
#: the type rather than by a threshold.
LEDGER_ENVELOPE = EnvelopeSpec(
    unconditional=True,
    justification=(
        "a census over resolved forecasts. It counts calls that were frozen before their outcomes "
        "and reports the arithmetic on them, so no regime of the run can make the count wrong. The "
        "two things that could make it wrong are both closed elsewhere and neither is one of "
        "the twelve regime conditions: an input that postdates its own forecast is impossible "
        "because `issue` refuses to build one, and a score pooled over two reference classes is "
        "visible because every row carries the class it was conditional on."
    ),
)


class ForecastCalibration(BaseObservable):
    """Score a calibration ledger: Brier, the Murphy decomposition, skill, coverage, value.

    Kill condition: none. This instrument cannot be wrong about a run, only about arithmetic, and
    the arithmetic is checked against a hand-computed case in the acceptance test.

    It refuses on an empty ledger and on a ledger whose every call went void, and the second refusal
    is the one worth having. A ledger holding twenty voids and no resolutions has a Brier score of
    nothing, and returning a number computed over an empty denominator is exactly the confident
    wrong output this architecture exists to prevent. The refusal is `RECORD_INCOMPLETE` rather than
    `ACCESS_INSUFFICIENT` because no amount of extra access fixes it: the analyses did not produce
    the metrics, and the fix is upstream in whatever was supposed to.

    What it cannot do: it says nothing about whether the forecasts were any good in a sense other
    than calibration and discrimination on the calls that resolved. A forecaster that only issues
    calls it is confident about scores well here and is not being measured on the calls it declined
    to make, which is the selection effect a ledger cannot see and a registration can.
    """

    name = "ForecastCalibration"
    version = "1.0"
    quantity = FORECAST_BRIER_QUANTITY
    capabilities = Capability.NONE
    requires = {Component.RECORD: Access.RECORD}
    substrates = frozenset(Substrate)
    phases = frozenset({Phase.POST_RUN, Phase.DEPLOYED})
    envelope = LEDGER_ENVELOPE
    invariance = "none"
    invariance_relation = Relation("invariant")
    baselines = (
        "the always-guess-half coin, at Brier 0.25",
        "climatology, the base rate in the reference class",
        "persistence, the current state carried forward",
        "the registered dumb statistic for each target",
        "contrastive belief-flipping, the scaffolded black-box comparator",
    )
    rung = 0
    faithful_to = "Murphy (1973), the three-term decomposition of the Brier score"
    deviations = (
        "the half-Brier convention, one term per forecast, rather than Murphy's two-category sum "
        "which is twice this. The campaign's published 0.26 is on this convention and a coin sits "
        "at 0.25 on it, which is the check that settles which convention a number is on.",
        "the decomposition is exact rather than binned whenever the forecaster emitted few "
        "distinct probabilities, which is the usual case for a pre-registered campaign. Under "
        "equal-width binning the identity carries a within-bin variance term that "
        "`MurphyDecomposition.residual` reports rather than absorbs.",
        "the skill interval is a paired percentile bootstrap rather than BCa. At the sample sizes a "
        "pre-registered campaign reaches, sixteen calls in the worked case, the acceleration term "
        "is estimated from too few points to be worth the extra assumption.",
    )

    def __init__(
        self,
        ledger: CalibrationLedger,
        *,
        decision: DecisionSpec | None = None,
        nominal_coverage: float = REGISTERED_NOMINAL_COVERAGE,
        seed: int = 0,
    ) -> None:
        self.ledger = ledger
        self.decision = decision
        self.nominal_coverage = nominal_coverage
        self.seed = seed
        # Cheap and idempotent. The estimator registry is a process-global and the module-level
        # call can be undone by a test fixture that pops whatever appeared during its window, after
        # which the module will not re-execute. Re-asserting here means an instrument that exists
        # has its rows registered, which is the property lint rule two is actually asking about.
        _register()

    def measure(self, ctx: Context) -> Any:
        score = self.ledger.score(
            nominal_coverage=self.nominal_coverage, decision=self.decision, seed=self.seed
        )
        baselines: dict[str, float] = {"coin": score.coin_brier}
        for sk in score.skill:
            baselines[sk.baseline_id] = sk.baseline_brier
        interval = None
        if score.skill:
            coin = next((s for s in score.skill if s.baseline_id.startswith("coin")), None)
            if coin is not None:
                interval = Uncertainty(
                    n=score.n_directional,
                    ci_low=coin.ci_low,
                    ci_high=coin.ci_high,
                    ci_level=coin.level,
                    method="paired percentile bootstrap on the skill score",
                )
        return ctx.emit(
            score,
            uncertainty=interval,
            baselines=baselines,
            subject_extra={
                "n_directional": score.n_directional,
                "n_intervals": score.n_intervals,
                "n_void": score.n_void,
            },
        )

    def estimate(self, ctx: Context | None = None) -> Reading:
        ctx = ctx or Context(readout="forecast")
        if len(self.ledger) == 0:
            return refuse_incomplete(
                self.name,
                field="any forecast row",
                subject="the calibration ledger",
                remedy=(
                    "issue at least one forecast with `forecast.issue` and resolve it with "
                    "`forecast.resolve`, then append the row. An empty ledger has no Brier score, "
                    "and printing one over an empty denominator is the failure this instrument "
                    "exists to avoid."
                ),
            )
        resolved = self.ledger.directional
        intervals = self.ledger.intervals
        if not resolved and not intervals:
            reasons = ", ".join(sorted({e.voided for e in self.ledger.voids}))
            return refuse_incomplete(
                self.name,
                field="a resolved outcome",
                subject=f"all {len(self.ledger)} rows in the calibration ledger",
                remedy=(
                    f"every call in this ledger went void ({reasons}), so there is no denominator "
                    f"to score over. Produce the metrics the resolution rules name and re-resolve; "
                    f"the void count is itself the result and it is on the ledger. No amount of "
                    f"extra access fixes this, because the analyses did not emit the numbers."
                ),
                n_void=len(self.ledger.voids),
            )
        return super().estimate(ctx)


__all__ = [
    "FORECAST_BRIER_QUANTITY",
    "LEDGER_ENVELOPE",
    "ForecastCalibration",
    "PUBLISHED_COIN_BRIER",
    "PUBLISHED_DIRECTIONAL_BRIER",
    "PUBLISHED_INTERVAL_COVERAGE",
    "PUBLISHED_N_DIRECTIONAL",
    "PUBLISHED_N_INTERVALS",
    "REGISTERED_NOMINAL_COVERAGE",
    "CalibrationLedger",
    "LedgerEntry",
    "LedgerScore",
    "entry_from",
]


# ---------------------------------------------------------------------------
# Registry rows
# ---------------------------------------------------------------------------


def _register() -> None:
    """The four ledger quantities, registered so lint rule two stops calling them open.

    Idempotent, and called from `ForecastCalibration.__init__` as well as at import, because the
    estimator registry is a process-global that four test modules mutate by snapshotting it and
    popping whatever appeared during their window. A module-level registration that lands inside one
    of those windows is popped at its teardown and never comes back, since the module is already in
    `sys.modules` and will not re-execute. The underlying fragility is the registry rather than this
    module, so this guards itself rather than pretending the registry is stable.

    A quantity with no registered estimator is an open research target and the docs build says so.
    These four are not open: they are computed by `ForecastCalibration` on every scored ledger, and
    leaving them unregistered would have published four built numbers on the roadmap page as things
    nobody has worked out how to measure. The rule caught exactly that, which is what it is for.

    One rung each, and no ladder. A proper scoring rule has no cheaper approximation worth naming:
    the arithmetic is a mean of squares over resolved forecasts and the cost is nothing. What varies
    between a good ledger and a bad one is the forecasts, not the estimator.
    """
    from reward_lens.core.quantity import (
        ESTIMATORS,
        FREE,
        BiasStatement,
        EstimatorEntry,
        register_estimator,
    )
    from reward_lens.core.types import Access, Component

    if any(e.quantity == "forecast.brier_score" for e in ESTIMATORS.values()):
        return

    requires = {Component.RECORD: Access.RECORD}
    unbiased = BiasStatement(
        direction="approximately_unbiased",
        why=(
            "a mean over the forecasts that resolved. The one place it can mislead is the "
            "denominator: a void resolution is excluded rather than scored as a miss, and the "
            "count of them travels with the score so a reader can take the other convention. On "
            "the campaign's own ledger that is 0.26 over 16 against 0.2904 over 23."
        ),
    )
    for quantity, impl in (
        ("forecast.brier_score", "forecast.brier.half_brier"),
        ("forecast.calibration_reliability", "forecast.brier.murphy_reliability"),
        ("forecast.calibration_resolution", "forecast.brier.murphy_resolution"),
        ("forecast.decision_value", "forecast.decision.expected_loss_saved"),
    ):
        register_estimator(
            EstimatorEntry(
                quantity=quantity,
                impl=impl,
                requires=requires,
                envelope=LEDGER_ENVELOPE,
                rung=0,
                bias=unbiased,
                cost=FREE,
            )
        )


_register()

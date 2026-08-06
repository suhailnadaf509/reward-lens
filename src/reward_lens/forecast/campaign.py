"""Re-scoring the published campaign through this ledger.

The campaign froze 27 cards and 53 hypotheses on 2026-07-18 and ran them on 2026-07-19. Its
published meta-ledger reports a directional Brier of 0.26 over 16 calls against a coin at 0.25, an
interval coverage of 0.75 over 4 intervals against a registered nominal of 0.80, and a meta kill
that fired. This module rebuilds those calls as `Forecast` objects, resolves them through
`resolve()` against the metrics the adjudication rows recorded, and scores them through
`CalibrationLedger`.

Nothing is copied. Every verdict is **recomputed** from the frozen comparator and threshold in the
card's own spec file applied to the recorded metric, so the reproduction is a check on the frozen
rules and the recorded numbers rather than a re-print of the published outcome column. The
`verify_against_recorded` helper then compares the recomputed verdicts with the ones the store
holds, and disagreements are returned rather than raised.

Two things this found, neither of them fixed here.

**Every measurement in the campaign postdates the freeze.** All 1,363 rows in
`campaign-results/runs/campaign/evidence.jsonl` were created on 2026-07-19 and the freeze is
2026-07-18T23:46:57.951556+00:00, so zero rows predate it. That is what a pre-registration is
supposed to look like and it is now checkable in one line rather than asserted in a README.

**Twenty-two of the 23 directional calls recorded no inputs.** Their probabilities were priors from
reading the literature, which is legitimate, and it means the information barrier has nothing to
certify for them: `issue` refuses an empty input set, so these are constructed directly and every
ledger row says so in its note. The one call with recorded inputs is the LADDER interval, whose
three small-rung index tables are named in `ladder_intervals.json`, and that one goes through the
barrier for real. See `ladder_interval_forecast`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from reward_lens.core.store import EvidenceStore
from reward_lens.core.types import EvidenceID, SubjectRef
from reward_lens.forecast.barrier import issue
from reward_lens.forecast.baselines import (
    climatology,
    contrastive_belief_flip,
    dumb_statistic,
    persistence,
)
from reward_lens.forecast.ledger import CalibrationLedger, entry_from
from reward_lens.forecast.resolve import Resolution, resolve
from reward_lens.forecast.schema import (
    BinaryProbability,
    Comparator,
    Forecast,
    ForecastError,
    HorizonSpec,
    InformationTime,
    IntervalForecast,
    ReferenceClass,
    ReferenceClassID,
    ResolutionRule,
    forecast_id,
)

#: The main freeze, from `specs/frozen/manifest.json`. Every card spec carries the same instant.
MAIN_FREEZE = "2026-07-18T23:46:57.951556+00:00"

#: The ladder intervals were written later, at runbook 5b, and before the 8B arc. The campaign says
#: so in its own `interval_rule` string and the barrier now enforces it.
LADDER_INTERVAL_FREEZE = "2026-07-19T14:38:14.080678+00:00"

ADJUDICATION_PREFIX = "campaign.adjudication."
RESULT_PREFIX = "campaign.result."


def campaign_reference_class() -> ReferenceClass:
    """The population the campaign's directional calls were conditional on, and its missing base rate.

    `n` and `base_rate` are both `None`, and that is the finding rather than a gap in this code. The
    campaign forecast whether a novel mechanistic hypothesis about reward-model geometry would
    confirm; there was no prior literature, no community of forecasters and no counted base rate for
    that class, so climatology refuses and the refusal is on every ledger row. Prediction markets
    reach 71 to 73 percent on whether a *published, peer-reviewed* finding replicates, where the base
    rate is 40 to 60 percent. That is a different reference class and comparing the two numbers
    directly is the mistake this field is about to make.
    """
    return ReferenceClass(
        id=ReferenceClassID("campaign.novel-mechanistic-hypothesis"),
        definition=(
            "a first-contact mechanistic hypothesis about reward-model geometry, pre-registered "
            "with a mechanically evaluable threshold, run once on a fixed bank of open reward "
            "models with no pilot and no prior literature on the specific quantity"
        ),
        n=None,
        base_rate=None,
    )


# ---------------------------------------------------------------------------
# Reading the freeze and the store
# ---------------------------------------------------------------------------


def _decode(obj: Any) -> Any:
    """Undo the campaign store's `__map__`/`__seq__`/`__type__` tagging.

    The store predates this library's codec registration for the campaign's own payload module, so
    `EvidenceStore.get` cannot decode these rows without importing a package that is not installed.
    Reading the tagging directly is what `record/convert/store.py` does for the same reason.
    """
    if isinstance(obj, dict):
        if "__map__" in obj:
            return {k: _decode(v) for k, v in obj["__map__"].items()}
        if "__seq__" in obj:
            return [_decode(v) for v in obj["__seq__"]]
        if "__type__" in obj:
            return {k: _decode(v) for k, v in obj.get("fields", {}).items()}
        return {k: _decode(v) for k, v in obj.items()}
    return obj


@dataclass(frozen=True)
class CampaignFreeze:
    """What was registered, before anything ran."""

    directional: tuple[Mapping[str, Any], ...]
    intervals: tuple[Mapping[str, Any], ...]
    specs: Mapping[str, Mapping[str, Any]]
    ladder_intervals: Mapping[str, Any]
    frozen_at: str

    def rule_for(self, card: str, hypothesis: str) -> ResolutionRule | None:
        """The frozen comparator and threshold for one hypothesis, read from the card's own spec."""
        spec = self.specs.get(card)
        if spec is None:
            return None
        for hyp in spec.get("spec", {}).get("hypotheses", []):
            if hyp.get("id") != hypothesis:
                continue
            prediction = hyp.get("prediction", {})
            return ResolutionRule(
                metric=prediction["metric"],
                comparator=Comparator(prediction["comparator"]),
                threshold=float(prediction["threshold"]),
                definition=hyp.get("statement", ""),
            )
        return None


def load_freeze(specs_dir: str | Path) -> CampaignFreeze:
    """Read the frozen pre-registration: the probabilities, the card specs, the ladder intervals."""
    root = Path(specs_dir)
    probabilities = json.loads((root / "probabilities.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    ladder = json.loads((root / "ladder_intervals.json").read_text(encoding="utf-8"))
    specs: dict[str, Mapping[str, Any]] = {}
    for path in sorted(root.glob("campaign-*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        study = doc.get("study_id", "")
        spec_id = doc.get("spec", {}).get("id", "")
        card = _card_of(spec_id) or _card_of(study)
        if card:
            specs[card] = doc
    return CampaignFreeze(
        directional=tuple(probabilities.get("directional", ())),
        intervals=tuple(probabilities.get("intervals", ())),
        specs=specs,
        ladder_intervals=ladder,
        frozen_at=manifest.get("created_at", MAIN_FREEZE),
    )


def _card_of(spec_id: str) -> str:
    """The card name a spec id belongs to, matched against the store's own card labels."""
    return _SPEC_TO_CARD.get(spec_id, "")


#: The campaign's spec ids and card labels agree except in two places, which is why this is a table
#: rather than an upper-case of the suffix. `campaign-decomp` is card `T3-DECOMP` and
#: `campaign-field-flat` is card `T3-FIELD`.
_SPEC_TO_CARD: dict[str, str] = {
    "campaign-adj-avp": "ADJ-AVP",
    "campaign-atlas-vce": "ATLAS-VCE",
    "campaign-cal-transfer": "CAL-TRANSFER",
    "campaign-capacity-welch": "CAPACITY-WELCH",
    "campaign-chi-drift": "CHI-DRIFT",
    "campaign-conf-partial": "CONF-PARTIAL",
    "campaign-decomp": "T3-DECOMP",
    "campaign-emb-lora": "EMB-LORA",
    "campaign-err-rb2": "ERR-RB2",
    "campaign-eval-aware": "EVAL-AWARE",
    "campaign-fact-kui": "FACT-KUI",
    "campaign-field-flat": "T3-FIELD",
    "campaign-forecast-cat": "FORECAST-CAT",
    "campaign-forensic-receipt": "FORENSIC-RECEIPT",
    "campaign-gauge-e19": "GAUGE-E19",
    "campaign-gauge-xfam": "GAUGE-XFAM",
    "campaign-hack-fore": "HACK-FORE",
    "campaign-hump": "HUMP",
    "campaign-judge-vbc": "JUDGE-VBC",
    "campaign-ladder": "LADDER",
    "campaign-meta-ledger": "META-LEDGER",
    "campaign-ppe-bon": "PPE-BON",
    "campaign-style-rmb": "STYLE-RMB",
    "campaign-surgery": "SURGERY",
    "campaign-topo-hodge": "TOPO-HODGE",
    "campaign-values-contest": "VALUES-CONTEST",
    "campaign-verif-prm": "VERIF-PRM",
}


@dataclass(frozen=True)
class CardAdjudication:
    """One card's recorded close-out, decoded."""

    card: str
    status: str
    outcomes: Mapping[str, str]
    metrics: Mapping[str, Any]
    killed: bool
    killed_by: tuple[str, ...]
    created_at: str


def load_adjudications(store_path: str | Path) -> dict[str, CardAdjudication]:
    """Read every `campaign.adjudication.*` row, decoded, keyed by card.

    Opened readonly: this is a published artifact and nothing here should ever touch it.
    """
    store = EvidenceStore(Path(store_path), readonly=True)
    out: dict[str, CardAdjudication] = {}
    for envelope in store._index.values():  # noqa: SLF001 - no public raw-envelope iterator
        observable = envelope.get("observable", "")
        if not observable.startswith(ADJUDICATION_PREFIX):
            continue
        value = _decode(envelope.get("value"))
        card = str(value.get("card") or envelope["subject"].get("extra", {}).get("card", ""))
        out[card] = CardAdjudication(
            card=card,
            status=str(value.get("status", "")),
            outcomes=dict(value.get("outcomes") or {}),
            metrics=dict(value.get("metrics") or {}),
            killed=bool(value.get("killed", False)),
            killed_by=tuple(value.get("killed_by") or ()),
            created_at=envelope.get("created_at", ""),
        )
    return out


def load_ladder_result(store_path: str | Path) -> Mapping[str, Any]:
    """The LADDER result payload, which carries the measured 8B value for each ladder quantity."""
    store = EvidenceStore(Path(store_path), readonly=True)
    for envelope in store._index.values():  # noqa: SLF001
        if envelope.get("observable") == f"{RESULT_PREFIX}LADDER":
            meta = _decode(envelope.get("value")).get("meta", {})
            return dict(meta.get("per_quantity", {}))
    return {}


# ---------------------------------------------------------------------------
# Rebuilding the forecasts
# ---------------------------------------------------------------------------


def _prior_baselines(reference_class: ReferenceClass) -> tuple:
    """The four mandatory comparators for a call made from no measurement at all.

    Every one refuses, and every refusal is true of the campaign as it was actually run. That is the
    point of putting them on the row: the campaign shipped 23 directional calls against zero
    comparators, and until this ledger existed there was nowhere for that fact to appear.
    """
    return (
        climatology(reference_class),
        persistence(
            None,
            detail="no prior state: each hypothesis was first contact with its own quantity",
        ),
        dumb_statistic(
            None,
            name="none_registered",
            refused=(
                "the campaign registered no dumb statistic for these hypotheses. Most are "
                "correlations and AUC gaps over fixed banks rather than transcript-level calls, so "
                "the transcript comparators in `stats.baselines` do not apply; the comparator that "
                "does is a permutation or label-shuffle null on the same statistic, and 8 of the 23 "
                "directional cards carry one in their frozen power note while 15 do not. Register "
                "one per hypothesis at freeze time."
            ),
        ),
        contrastive_belief_flip((), (), judge=None),
    )


def campaign_forecasts(freeze: CampaignFreeze) -> tuple[Forecast, ...]:
    """Rebuild the campaign's 23 directional calls as `Forecast` objects.

    Constructed directly rather than through `issue`, because these calls recorded no measured
    inputs and `issue` refuses an empty input set. That refusal is correct and the workaround is not
    a loophole: the barrier certifies that inputs predate the issue, and a call with no inputs has
    nothing to certify. Every row this produces carries a note saying so, and the one campaign
    forecast that *does* have recorded inputs goes through the barrier in
    `ladder_interval_forecast`.
    """
    reference = campaign_reference_class()
    issued = InformationTime.parse(
        freeze.frozen_at,
        basis="the campaign's frozen spec manifest, before any run",
    )
    out: list[Forecast] = []
    for entry in freeze.directional:
        card = str(entry["card"])
        hypothesis = str(entry["hypothesis"])
        rule = freeze.rule_for(card, hypothesis)
        if rule is None:
            raise ForecastError(
                f"{card}:{hypothesis} is registered in probabilities.json and has no matching "
                f"hypothesis in the card's frozen spec, so its resolution rule cannot be read. The "
                f"pre-registration disagrees with itself and that is a finding, not something to "
                f"work around."
            )
        subject = SubjectRef(extra={"card": card, "hypothesis": hypothesis})
        distribution = BinaryProbability(float(entry["prob"]))
        out.append(
            Forecast(
                id=forecast_id(
                    target=f"campaign.{card}.{hypothesis}",
                    subject=subject,
                    resolution=rule,
                    issued_at=issued,
                    distribution=distribution,
                    inputs=(),
                    method="pre-registered prior",
                ),
                target=f"campaign.{card}.{hypothesis}",
                subject=subject,
                resolution=rule,
                issued_at=issued,
                horizon=HorizonSpec(kind="time", value=0.0),
                reference_class=reference,
                distribution=distribution,
                method="pre-registered prior",
                inputs=(),
                baselines=_prior_baselines(reference),
                meta_plan=str(entry.get("rationale", "")) or None,
            )
        )
    return tuple(out)


def campaign_interval_forecasts(freeze: CampaignFreeze) -> tuple[Forecast, ...]:
    """The interval calls: two from `probabilities.json` and three from the ladder freeze.

    The two in `probabilities.json` are one-sided equivalence bounds written as a comparator and a
    threshold, and the campaign counted them as intervals. They are represented here as the interval
    `[0, threshold]` on the absolute metric, which is what the bound says and is what reproduces the
    published coverage. Recorded so that anyone reading the number knows it mixes two shapes.
    """
    reference = campaign_reference_class()
    prior_issued = InformationTime.parse(
        freeze.frozen_at, basis="the campaign's frozen spec manifest, before any run"
    )
    ladder_issued = InformationTime.parse(
        freeze.ladder_intervals.get("frozen_at", LADDER_INTERVAL_FREEZE),
        basis="ladder_intervals.json, written at runbook 5b and before the 8B arc",
    )
    out: list[Forecast] = []

    for entry in freeze.intervals:
        card = str(entry["card"])
        hypothesis = str(entry["hypothesis"])
        threshold = float(entry["threshold"])
        rule = ResolutionRule(
            metric=str(entry["metric"]),
            comparator=Comparator(str(entry["comparator"])),
            threshold=threshold,
            definition=str(entry.get("rationale", "")),
        )
        subject = SubjectRef(extra={"card": card, "hypothesis": hypothesis})
        distribution = IntervalForecast(lo=0.0, hi=threshold, level=0.8)
        out.append(
            _bare_forecast(
                target=f"campaign.{card}.{hypothesis}",
                subject=subject,
                rule=rule,
                issued=prior_issued,
                distribution=distribution,
                reference=reference,
                method="pre-registered one-sided equivalence bound, counted as an interval",
            )
        )

    for quantity, band in sorted(freeze.ladder_intervals.get("intervals", {}).items()):
        subject = SubjectRef(extra={"card": "LADDER", "quantity": quantity})
        rule = ResolutionRule(
            metric=f"{quantity}_measured_8b",
            comparator=Comparator.GE,
            threshold=float(band["lo"]),
            definition=(
                f"the measured 8B {quantity} falls inside the 80 percent prediction interval "
                f"extrapolated from the 0.6B, 1.7B and 4B rungs"
            ),
        )
        distribution = IntervalForecast(
            lo=float(band["lo"]),
            hi=float(band["hi"]),
            level=0.8,
            point=float(band["point"]),
        )
        out.append(
            _bare_forecast(
                target=f"campaign.LADDER.{quantity}",
                subject=subject,
                rule=rule,
                issued=ladder_issued,
                distribution=distribution,
                reference=reference,
                method="OLS on log(params_b), 80 percent prediction interval at 7.6B",
            )
        )
    return tuple(out)


def _bare_forecast(
    *,
    target: str,
    subject: SubjectRef,
    rule: ResolutionRule,
    issued: InformationTime,
    distribution: Any,
    reference: ReferenceClass,
    method: str,
) -> Forecast:
    return Forecast(
        id=forecast_id(
            target=target,
            subject=subject,
            resolution=rule,
            issued_at=issued,
            distribution=distribution,
            inputs=(),
            method=method,
        ),
        target=target,
        subject=subject,
        resolution=rule,
        issued_at=issued,
        horizon=HorizonSpec(kind="time", value=0.0),
        reference_class=reference,
        distribution=distribution,
        method=method,
        inputs=(),
        baselines=_prior_baselines(reference),
    )


# ---------------------------------------------------------------------------
# The barrier, on the one campaign forecast that has recorded inputs
# ---------------------------------------------------------------------------


def ladder_rung_inputs(freeze: CampaignFreeze) -> tuple[str, ...]:
    """The three small-rung evidence ids the ladder intervals were fitted on."""
    return tuple(str(e) for e in freeze.ladder_intervals.get("source_evidence", ()))


def held_out_rung_id(store_path: str | Path) -> str:
    """The 8B index-table row the ladder deliberately held out, found by its subject.

    Looked up rather than hard-coded, so the leakage test fails loudly if the store changes shape
    instead of quietly testing a string that no longer names anything.
    """
    store = EvidenceStore(Path(store_path), readonly=True)
    for envelope in store._index.values():  # noqa: SLF001
        if envelope.get("observable") != "campaign.index.table":
            continue
        extra = envelope.get("subject", {}).get("extra", {})
        if extra.get("roster_key") == "skywork-v2-qwen3-8b" and extra.get("slice") == "rb2-full":
            return str(envelope["id"])
    raise ForecastError(
        "no `campaign.index.table` row for skywork-v2-qwen3-8b on rb2-full is in this store, so "
        "the held-out rung the ladder forecast excluded cannot be named."
    )


def ladder_interval_forecast(
    store: EvidenceStore,
    freeze: CampaignFreeze,
    *,
    quantity: str = "crystallization",
    extra_inputs: Sequence[str] = (),
) -> Forecast:
    """Issue the LADDER interval through the real barrier, against the real campaign store.

    This is the campaign's one forecast with recorded inputs, and the timing is genuinely tight: the
    three small rungs were measured at 13:29, 13:31 and 13:38 on 2026-07-19, the intervals were
    frozen at 14:38, and the held-out 8B rung was measured at 15:34, 56 minutes after the freeze.
    Passing the 8B row in ``extra_inputs`` is therefore a real leak on real data and the barrier
    raises on it naming both timestamps.
    """
    band = freeze.ladder_intervals["intervals"][quantity]
    subject = SubjectRef(extra={"card": "LADDER", "quantity": quantity})
    rule = ResolutionRule(
        metric=f"{quantity}_measured_8b",
        comparator=Comparator.GE,
        threshold=float(band["lo"]),
        definition=(
            f"the measured 8B {quantity} falls inside the 80 percent prediction interval "
            f"extrapolated from the 0.6B, 1.7B and 4B rungs"
        ),
    )
    reference = campaign_reference_class()
    return issue(
        target=f"campaign.LADDER.{quantity}",
        subject=subject,
        resolution=rule,
        distribution=IntervalForecast(
            lo=float(band["lo"]),
            hi=float(band["hi"]),
            level=0.8,
            point=float(band["point"]),
        ),
        inputs=tuple(EvidenceID(i) for i in (*ladder_rung_inputs(freeze), *extra_inputs)),
        at=InformationTime.parse(
            freeze.ladder_intervals.get("frozen_at", LADDER_INTERVAL_FREEZE),
            basis="ladder_intervals.json, written at runbook 5b and before the 8B arc",
        ),
        store=store,
        reference_class=reference,
        horizon=HorizonSpec(kind="time", value=0.0),
        method="OLS on log(params_b), 80 percent prediction interval at 7.6B",
        baselines=_prior_baselines(reference),
    )


# ---------------------------------------------------------------------------
# Resolving and scoring
# ---------------------------------------------------------------------------


def resolve_campaign(
    forecasts: Sequence[Forecast],
    adjudications: Mapping[str, CardAdjudication],
    ladder_measured: Mapping[str, Any] | None = None,
) -> list[tuple[Forecast, Resolution]]:
    """Resolve every rebuilt forecast against the metrics the campaign recorded.

    The verdict is recomputed from the frozen comparator and threshold, never read off the recorded
    outcome column. A card the analysis declared inconclusive voids with that reason, which is what
    keeps the denominator at 16 rather than 23.
    """
    ladder_measured = ladder_measured or {}
    out: list[tuple[Forecast, Resolution]] = []
    for forecast in forecasts:
        card = str(forecast.subject.extra.get("card", ""))
        hypothesis = str(forecast.subject.extra.get("hypothesis", ""))
        quantity = str(forecast.subject.extra.get("quantity", ""))
        adjudication = adjudications.get(card)
        if adjudication is None:
            metrics: Mapping[str, Any] = {}
            inconclusive = ""
            at = InformationTime.parse(MAIN_FREEZE, basis="no adjudication row for this card")
        else:
            metrics = dict(adjudication.metrics)
            at = InformationTime.parse(
                adjudication.created_at, basis=f"the campaign.adjudication.{card} row was written"
            )
            recorded = adjudication.outcomes.get(hypothesis, "")
            inconclusive = (
                f"the {card} analysis reported {hypothesis} inconclusive under its own registered "
                f"criteria"
                if recorded == "inconclusive" or adjudication.status == "inconclusive"
                else ""
            )
        if quantity:
            measured = ladder_measured.get(quantity, {}).get("measured_8b")
            metrics = dict(metrics)
            if measured is not None:
                metrics[f"{quantity}_measured_8b"] = float(measured)
            inconclusive = ""
        out.append((forecast, resolve(forecast, metrics, at=at, inconclusive=inconclusive)))
    return out


def verify_against_recorded(
    resolutions: Sequence[tuple[Forecast, Resolution]],
    adjudications: Mapping[str, CardAdjudication],
) -> tuple[str, ...]:
    """Compare every recomputed verdict with the one the store recorded. Returns disagreements.

    Returned rather than raised, because a disagreement is a finding about the campaign or about
    this reconstruction and both want reading before either is called a bug.
    """
    findings: list[str] = []
    for forecast, resolution in resolutions:
        card = str(forecast.subject.extra.get("card", ""))
        hypothesis = str(forecast.subject.extra.get("hypothesis", ""))
        if not hypothesis:
            continue
        adjudication = adjudications.get(card)
        if adjudication is None:
            continue
        recorded = adjudication.outcomes.get(hypothesis)
        if recorded is None:
            continue
        if getattr(resolution, "is_void", False):
            if recorded != "inconclusive":
                findings.append(
                    f"{card}:{hypothesis} recomputed as void ({resolution.reason.value}) and the "
                    f"store records {recorded!r}"
                )
            continue
        recomputed = "confirmed" if resolution.outcome else "refuted"
        if recomputed != recorded:
            findings.append(
                f"{card}:{hypothesis} recomputed as {recomputed} from {resolution.metric}="
                f"{resolution.metric_value:.6g} against the frozen rule {resolution.rule}, and "
                f"the store records {recorded!r}"
            )
    return tuple(findings)


def rescore_campaign(
    store_path: str | Path,
    specs_dir: str | Path,
    *,
    ledger_path: str | Path | None = None,
) -> tuple[CalibrationLedger, CampaignFreeze, tuple[str, ...]]:
    """The whole route: freeze, rebuild, resolve, score, ledger. Returns the ledger and any findings.

    The published figures it should reproduce are a directional Brier of 0.26 over 16 calls, a coin
    at 0.25, an interval coverage of 0.75 over 4, and the meta kill fired.
    """
    freeze = load_freeze(specs_dir)
    adjudications = load_adjudications(store_path)
    ladder_measured = load_ladder_result(store_path)

    directional = campaign_forecasts(freeze)
    intervals = campaign_interval_forecasts(freeze)
    resolutions = resolve_campaign(
        list(directional) + list(intervals), adjudications, ladder_measured
    )
    findings = verify_against_recorded(resolutions, adjudications)

    ledger = CalibrationLedger(ledger_path)
    for forecast, resolution in resolutions:
        note = (
            "registered as a prior; no measured inputs were recorded, so the information barrier "
            "has nothing to certify for this call"
            if not forecast.inputs
            else ""
        )
        ledger.append(entry_from(forecast, resolution, note=note))
    return ledger, freeze, findings


__all__ = [
    "ADJUDICATION_PREFIX",
    "LADDER_INTERVAL_FREEZE",
    "MAIN_FREEZE",
    "RESULT_PREFIX",
    "CampaignFreeze",
    "CardAdjudication",
    "campaign_forecasts",
    "campaign_interval_forecasts",
    "campaign_reference_class",
    "held_out_rung_id",
    "ladder_interval_forecast",
    "ladder_rung_inputs",
    "load_adjudications",
    "load_freeze",
    "load_ladder_result",
    "rescore_campaign",
    "resolve_campaign",
    "verify_against_recorded",
]

"""C3 `instrument.recovery_auc`: the head-to-head nobody publishes.

Against one planted key, every localisation method in the library gets a row, and the table prints
in measured order. The catalogue's kill condition is "nothing; a bad number here is the point",
which is the only entry in the catalogue that says so, and it is why this instrument has no path
that suppresses a row.

**The reference gate.** C3's envelope line in the source reads "`REFERENCE_UNCERTIFIED` must
not fire", which names a `RefusalReason` rather than a `RegimeCondition`, and the catalogue carries
the unconditional justification instead. So the reference requirement is not
an envelope condition here, it is a refusal: a recovery number is a calibration against a planted
key, and a key with no measured uncertainty of its own cannot calibrate anything. `u_homogeneity is
None` means nobody checked whether two plants with different seeds give the same answer, and the
Model Organism Lottery says they do not.

The refusal is **bounded**. The table is computed and travels as the refusal's `partial`, because
the losses are the deliverable and withholding them to punish an uncertified reference would be the
instrument working against its own purpose. What the refusal withholds is the *certification*: the
number is not Evidence, its trust is not ratified, and the reason says which of the three
uncertainty terms was never measured.

**Scope limit, three lines in.** A recovery AUC measures whether a method ranks the planted
candidates above the unplanted ones. It does not measure whether the method found the *mechanism*,
and a method can rank a key perfectly by reading a marker the plant happened to correlate with. That
is why the panel below includes a zero-parameter behavioural correlation and a string match: if
those rank the key as well as the white-box methods do, the white-box row has not demonstrated
anything about internals, whatever its AUC.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from reward_lens.core.budget import IncrementalValidity
from reward_lens.core.envelope import EnvelopeSpec
from reward_lens.core.evidence import Uncertainty
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.reading import Reading, Refusal, RefusalReason, bounded_refusal
from reward_lens.core.reference import ReferenceMaterial, uncertified_refusal
from reward_lens.core.types import Capability
from reward_lens.measure.base import Context
from reward_lens.measure.selection._common import (
    ACCESS_ORGANISM_MUTATE,
    SelectionInstrument,
    emit_white_box,
)
from reward_lens.measure.selection.table import (
    RecoveryRow,
    RecoveryTable,
    recovery_auc,
    score_row,
)
from reward_lens.policy.selection import MethodClass

#: A recovery table describes methods scored against a planted key that is already in hand. The
#: twelve regime conditions are all properties of a training run and none of them can make a rank
#: statistic over candidates wrong. The precondition that does bite is that the reference is
#: certified, and that is a refusal rather than a regime condition.
RECOVERY_ENVELOPE = EnvelopeSpec(
    unconditional=True,
    justification=(
        "a rank statistic of methods against a planted key already in hand. It asserts nothing "
        "about any training process, so no regime of one can make it wrong. The condition that does "
        "bite, that the reference material carries an uncertainty of its own, is a "
        "REFERENCE_UNCERTIFIED refusal rather than a RegimeCondition: the source's Env "
        "column names a RefusalReason there."
    ),
)

#: The catalogue's four baselines for C3, verbatim.
RECOVERY_BASELINES: tuple[str, ...] = (
    "string match",
    "length",
    "scaffolded black-box prompting",
    "a coherent irrelevant semantic direction",
)


class InstrumentRecoveryTable(SelectionInstrument):
    """C3. Every localisation method against one planted key, losses included.

    White-box: the panel reads activations, so an `IncrementalValidity` record is mandatory on the
    reading and this instrument supplies one. **The bar is decorrelation plus
    signal, not superiority**, and this is the instrument in the library whose whole subject is
    comparing methods, so its incremental record is built from the panel itself: the own score is
    the best claimable white-box row, the baseline is the best row that read no internals, and the
    ensemble is the two combined per candidate. A white-box method that loses the table and is
    uncorrelated with the black-box row is a better result than one that wins and is redundant, and
    only those four numbers can say which happened.

    What it cannot do. It ranks candidates; it does not find mechanisms. And every row is scored on
    one organism family at one dose, so a method that wins here has won on this matrix: the
    transfer to a real 8B reward model is a `Transfer` term nobody in this package has measured.
    """

    name = "InstrumentRecoveryTable"
    version = "1.0"
    quantity = "instrument.recovery_auc"
    capabilities = Capability.ACTIVATIONS
    requires = ACCESS_ORGANISM_MUTATE
    envelope = RECOVERY_ENVELOPE
    invariance = "repr.basis"
    #: Every method in the panel is either a rank statistic over candidates or an inner product
    #: between two directions that rotate together, so a shared orthogonal change of representation
    #: basis leaves every row exactly where it was. A test asserts it rather than the declaration
    #: standing alone.
    invariance_relation = INVARIANT
    baselines = RECOVERY_BASELINES
    rung = 1
    faithful_to = "C3, an instrument-comparison experiment with a ground truth"
    deviations = (
        "the recovery statistic is the rank of planted candidates against unplanted ones, which "
        "scores localisation and not mechanism. A method that ranks the key by reading a marker "
        "the plant correlates with scores the same as one that found the computation",
        "the interval resamples candidates stratified by planted status, so it carries the "
        "uncertainty from having few candidates and not the uncertainty from having one organism. "
        "The second is larger and it is a Transfer term this instrument does not measure",
        "rows cited from a stored campaign result are not re-measured here, and their intervals are "
        "whatever the stored result carried. `RecoveryRow.source` names every such row",
    )

    def __init__(
        self,
        panel: Mapping[str, tuple[MethodClass, Sequence[float]]] | None = None,
        planted: Sequence[bool] | None = None,
        *,
        certificate: Any = None,
        cited_rows: Sequence[RecoveryRow] = (),
        candidate_ids: Sequence[str] = (),
        organism: str = "",
        ours: str = "",
        n_parameters: Mapping[str, int] | None = None,
        n_boot: int = 1000,
        seed: int = 0,
        level: float = 0.95,
        require_certified: bool = True,
    ) -> None:
        self.panel = dict(panel or {})
        self.planted = np.asarray(planted if planted is not None else [], dtype=bool)
        self.certificate = certificate
        self.cited_rows = tuple(cited_rows)
        self.candidate_ids = tuple(candidate_ids)
        self.organism = organism
        self.ours = ours
        self.n_parameters = dict(n_parameters or {})
        self.n_boot = int(n_boot)
        self.seed = int(seed)
        self.level = float(level)
        self.require_certified = bool(require_certified)

    # -- the table ---------------------------------------------------------

    def table(self) -> RecoveryTable | Refusal:
        """Score every panel method against the planted key and assemble the table."""
        if not self.panel and not self.cited_rows:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.RECORD_INCOMPLETE,
                detail="no method was supplied to score, so there is no table",
                remedy=(
                    "pass `panel={method_id: (MethodClass, per_candidate_scores)}` with one score "
                    "per candidate, or `cited_rows=[...]` for rows read from a stored result. "
                    "`measure.selection.panel.run_panel` builds a panel from a trained organism."
                ),
            )
        rows: list[RecoveryRow] = list(self.cited_rows)
        n_candidates = int(self.planted.size)
        for method_id, (method_class, scores) in sorted(self.panel.items()):
            s = np.asarray(scores, dtype=np.float64).ravel()
            if s.size != n_candidates:
                return Refusal(
                    instrument=self.name,
                    reason=RefusalReason.UNIT_MISMATCH,
                    detail=(
                        f"method {method_id!r} scored {s.size} candidates and the planted key names "
                        f"{n_candidates}. Every row of a recovery table has to be scored against "
                        f"the same candidate set or the AUCs are not comparable"
                    ),
                    remedy=(
                        "score every method on identical candidates in identical order. A method "
                        "that cannot score a candidate should return a constant for it and declare "
                        "that in its detail, which costs it rank rather than silently changing the "
                        "denominator."
                    ),
                    statistics={"method": method_id, "scored": int(s.size), "key": n_candidates},
                )
            rows.append(
                score_row(
                    method_id,
                    method_class,
                    s,
                    self.planted,
                    n_parameters=int(self.n_parameters.get(method_id, 0)),
                    n_boot=self.n_boot,
                    seed=self.seed,
                    level=self.level,
                )
            )
        if int(np.count_nonzero(self.planted)) == 0 and self.panel:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.RECORD_INCOMPLETE,
                detail=(
                    f"the answer key marks none of the {n_candidates} candidates as planted, so "
                    f"every rank statistic is undefined"
                ),
                remedy=(
                    "supply the planted mask from the organism's `AnswerKey`. If the plant is "
                    "graded rather than binary, threshold it and record the threshold: a recovery "
                    "AUC needs a two-class key."
                ),
                statistics={"n_candidates": n_candidates},
            )
        return RecoveryTable(
            rows=rows,
            organism=self.organism,
            reference_id=getattr(self.certificate, "reference_id", "") or "",
            ours=self.ours,
            n_candidates=n_candidates,
        )

    # -- the incremental record -------------------------------------------

    def incremental(self, table: RecoveryTable) -> IncrementalValidity | None:
        """What the best white-box row adds over the best row that read no internals.

        Built from the panel rather than from a separate run, because the panel already *is* the
        comparison: every method scored the same candidates, so the per-candidate scores are paired
        by construction and the error vectors are directly comparable.

        The errors are per candidate: a method errs on a candidate when its score puts that
        candidate on the wrong side of the midpoint between the planted and unplanted score means.
        The ensemble averages the two methods' standardised scores, which is the combining rule M9
        calls `standardised_margin`, and it is the right one here because the panel's members are on
        wildly different scales: a cosine and a covariance are both scores and they are not
        commensurable.
        """
        inside = [
            r
            for r in table.ranked()
            if r.method_class.is_white_box and r.may_carry_a_claim and r.is_measured_here
        ]
        outside = [
            r for r in table.ranked() if not r.method_class.is_white_box and r.is_measured_here
        ]
        if not inside or not outside:
            return None
        own, base = inside[0], outside[0]
        own_scores = np.asarray(self.panel[own.method_id][1], dtype=np.float64).ravel()
        base_scores = np.asarray(self.panel[base.method_id][1], dtype=np.float64).ravel()
        ensemble = _standardise(own_scores) + _standardise(base_scores)
        ens_auc, _, _ = recovery_auc(ensemble, self.planted)
        return IncrementalValidity(
            own_score=float(own.auc),
            baseline_score=float(base.auc),
            baseline_id=base.method_id,
            error_correlation=_error_correlation(own_scores, base_scores, self.planted),
            ensemble_score=float(ens_auc),
        )

    # -- the reading -------------------------------------------------------

    def estimate(self, ctx: Context | None = None) -> Reading:
        ctx = ctx or Context(readout="score")
        pre = self.preflight(ctx)
        if not pre.ok and pre.refusal is not None:
            return pre.refusal
        ctx._observable = self
        try:
            return self._read(ctx)
        finally:
            ctx._observable = None

    def measure(self, ctx: Context) -> Any:
        return self._read(ctx)

    def _read(self, ctx: Context) -> Any:
        table = self.table()
        if isinstance(table, Refusal):
            return table
        record = self.incremental(table)
        material: ReferenceMaterial | None = None
        if self.certificate is not None:
            material = self.certificate.material()

        baselines = {
            r.method_id: float(r.auc)
            for r in table.ranked()
            if r.method_class
            in (MethodClass.BLACK_BOX, MethodClass.DUMB_BASELINE, MethodClass.CONTROL)
        }
        if record is None:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.NO_MATCHED_CONTROL,
                detail=(
                    "the panel has no measured white-box row, or no measured row that read nothing "
                    "internal, so there is no increment to record and one is mandatory on a "
                    "white-box reading"
                ),
                remedy=(
                    "add at least one method that reads internals and at least one that does not. "
                    "The black-box behavioural correlation and the string match are both cheap and "
                    "both belong in every run of this table: the comparison is the instrument."
                ),
                statistics={"n_rows": table.n_methods, "n_measured": table.n_measured_here},
            )

        if self.require_certified and (material is None or not material.is_certified):
            bound = emit_white_box(
                ctx,
                table,
                incremental=record,
                baselines=baselines,
                reference=material,
                uncertainty=Uncertainty(
                    n=table.n_candidates,
                    method=(
                        f"stratified bootstrap over candidates, {self.n_boot:,} resamples, "
                        f"{self.level:.0%} percentile"
                    ),
                ),
                subject_extra={"organism": self.organism, "bound": "table_without_certification"},
            )
            refusal = (
                uncertified_refusal(self.name, material)
                if material is not None
                else Refusal(
                    instrument=self.name,
                    reason=RefusalReason.REFERENCE_UNCERTIFIED,
                    detail=(
                        "no reference certificate was supplied, so the planted key carries no "
                        "uncertainty of its own and cannot calibrate a recovery number"
                    ),
                    remedy=(
                        "certify the organism family with `measure.labels.reference.certify`, or "
                        "build one with `organisms.dose.certified_micro_reference`, which trains "
                        "plants at several doses and seeds and re-measures after further training "
                        "so all three ISO Guide 35 terms exist."
                    ),
                )
            )
            return bounded_refusal(
                self.name,
                RefusalReason.REFERENCE_UNCERTIFIED,
                detail=refusal.detail,
                remedy=refusal.remedy,
                bound=bound,
                **{
                    **refusal.statistics,
                    "n_methods": table.n_methods,
                    "our_rank": table.our_rank(),
                    "n_losers": len(table.losers()),
                },
            )

        return emit_white_box(
            ctx,
            table,
            incremental=record,
            baselines=baselines,
            reference=material,
            uncertainty=Uncertainty(
                n=table.n_candidates,
                method=(
                    f"stratified bootstrap over candidates, {self.n_boot:,} resamples, "
                    f"{self.level:.0%} percentile"
                ),
            ),
            subject_extra={"organism": self.organism, "reference": table.reference_id},
        )


def _standardise(x: np.ndarray) -> np.ndarray:
    sd = float(np.std(x))
    return (x - float(np.mean(x))) / sd if sd > 0 else x - float(np.mean(x))


def _error_correlation(a: np.ndarray, b: np.ndarray, planted: np.ndarray) -> float:
    """Correlation between two methods' per-candidate errors, NaN when one never errs.

    NaN rather than zero, following M9's `phi`. A method that ranks every candidate correctly has no
    errors to correlate, and reporting zero for it would assert that the two fail independently
    about a pair where one of them does not fail.
    """
    y = np.asarray(planted).astype(bool)
    if y.all() or (~y).any() is False:
        return float("nan")

    def errs(s: np.ndarray) -> np.ndarray:
        if not y.any() or y.all():
            return np.zeros_like(s)
        mid = 0.5 * (float(np.mean(s[y])) + float(np.mean(s[~y])))
        high_is_planted = float(np.mean(s[y])) >= float(np.mean(s[~y]))
        called = (s >= mid) if high_is_planted else (s <= mid)
        return (called != y).astype(np.float64)

    ea, eb = errs(np.asarray(a, dtype=np.float64)), errs(np.asarray(b, dtype=np.float64))
    if ea.size < 2 or float(np.std(ea)) == 0.0 or float(np.std(eb)) == 0.0:
        return float("nan")
    return float(np.corrcoef(ea, eb)[0, 1])


def campaign_rows(store_dir: str, *, sidecar_dirs: Sequence[str] = ()) -> list[RecoveryRow]:
    """The recovery rows the campaign already published, read out of its evidence store.

    `campaign.result.ADJ-AVP` is a row of exactly this table and it is the row where the library
    loses: attribution recovers the planted key at 1.0 and patching at 0.434 over 56 components of
    which 3 are planted, and the card's kill criterion fired. Reading it here rather than restating
    it means the table's cited rows carry the evidence id they came from.

    These rows are **not re-measured**: they were taken on a GPU against an 8B model and this
    process has neither. `RecoveryRow.source` names the stored observable for every one.
    """
    from reward_lens.record.convert.store import CampaignStore

    store = CampaignStore(store_dir, sidecar_dirs=sidecar_dirs)
    store.assert_no_blind_payloads()
    out: list[RecoveryRow] = []
    for row in store.by_observable("campaign.result.ADJ-AVP"):
        value = store.value(row)
        fields = value if isinstance(value, dict) else getattr(value, "__dict__", {})
        metrics = _table_metrics(fields)
        meta = fields.get("meta") or {}
        n_planted = int(meta.get("n_planted", 0) or 0)
        n_components = int(meta.get("n_components", 0) or 0)
        evidence_id = row.get("id") if isinstance(row, dict) else getattr(row, "id", None)
        for method_id, key, method_class in (
            ("campaign.attribution", "attribution_recovery_auc", MethodClass.SUPERVISED_DIFFMEAN),
            ("campaign.patching", "patching_recovery_auc", MethodClass.SUPERVISED_DIFFMEAN),
        ):
            if key not in metrics:
                continue
            out.append(
                RecoveryRow(
                    method_id=method_id,
                    method_class=method_class,
                    auc=float(metrics[key]),
                    n_planted=n_planted,
                    n_unplanted=max(n_components - n_planted, 0),
                    ci_low=float(metrics.get("recovery_gap_ci_low", float("nan")))
                    if key == "patching_recovery_auc"
                    else float("nan"),
                    ci_high=float("nan"),
                    source="campaign.result.ADJ-AVP",
                    evidence_id=str(evidence_id) if evidence_id else None,
                    detail=(
                        f"stored campaign result on an 8B reward model, {n_components} components, "
                        f"{n_planted} planted; not re-measured here"
                    ),
                )
            )
    return out


def _table_metrics(fields: Mapping[str, Any]) -> dict[str, float]:
    """`{metric: value}` from the campaign's two-column `TablePayload` shape."""
    out: dict[str, float] = {}
    for entry in fields.get("rows") or ():
        pair = list(entry)
        if len(pair) == 2:
            try:
                out[str(pair[0])] = float(pair[1])
            except (TypeError, ValueError):
                continue
    return out


__all__ = [
    "RECOVERY_BASELINES",
    "RECOVERY_ENVELOPE",
    "InstrumentRecoveryTable",
    "campaign_rows",
]

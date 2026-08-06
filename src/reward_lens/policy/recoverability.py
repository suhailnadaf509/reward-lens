"""`PolicyReadoutProbe`: what reading the policy's activations buys over reading its text.

This is the library's first white-box reading in the strict sense: a reading produced by an
instrument that opens the network, carrying an `IncrementalValidity` record whose `error_correlation`
was measured rather than inferred. Every other white-box instrument in the battery emits
`incremental=None`, which is why lint rule four ("a white-box instrument whose reading has
`incremental is None` fails lint") has never been enforceable: there was nothing for it to pass or
fail on.

The measurement is deliberately the simplest white-box thing there is, because the point is not the
probe. Fit a ridge probe on the policy's residual stream at one site, cross-fitted over stratified
folds so no item is scored by a probe that saw it, and classify a binary outcome recorded alongside
the items. Then run M3's six dumb baselines on the same items and the same labels, and hand both to
M9, which computes what the probe adds to the ensemble of the six with a paired interval and the
correlation between the probe's errors and the best baseline's.

**The bar is decorrelation plus signal, not superiority.** A probe ten points worse than a length
baseline and uncorrelated with it is worth more than one two points better and redundant, because
the ensemble of the first pair beats either member and the ensemble of the second does not. That is
the sentence this instrument exists to be able to return. On a run whose grader is a length
function, the honest expected outcome is that the probe is redundant, and reporting that is the
instrument working.

**What it cannot do.** The probe is fitted on activations computed from *these* weights, and the
completions were produced by whatever checkpoint the record was written from. If those differ, the
probe is reading a counterfactual: what this policy would have represented had it produced text it
did not produce. That is the `NEAR_POLICY` envelope condition, measured by `run.max_staleness`, and
it is the condition most likely to be violated in practice because the natural thing to do is fit a
probe on the final checkpoint against the whole run. The instrument also says nothing about whether
the policy *uses* the direction it recovers: linear decodability is not mediation, and C7's
dissociation is the instrument for that question.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from reward_lens.core.envelope import EnvelopeSpec, RegimeCondition
from reward_lens.core.evidence import Uncertainty, register_payload
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.reading import Refusal, RefusalReason, refuse_incomplete
from reward_lens.core.types import (
    Access,
    AccessMatrix,
    Capability,
    Component,
    GaugeStatus,
    Phase,
    Site,
    Substrate,
)
from reward_lens.measure.base import BaseObservable, Context
from reward_lens.measure.meta.incremental import Detector, IncrementalValidityReading
from reward_lens.measure.rate.regime import MEASURED_BY
from reward_lens.policy.base import PositionSpec
from reward_lens.runtime.backend import CaptureSpec
from reward_lens.stats.baselines import (
    ALL_SIX,
    BaselineBank,
    DetectionTask,
    auroc,
    is_scored,
    run_bank,
    stratified_folds,
)

#: Both conditions this instrument depends on, and the quantity that measures each. `NEAR_POLICY`
#: because the probe reads this checkpoint's activations for text an older checkpoint produced, and
#: `ABOVE_LOD` because an increment smaller than the substrate's disagreement with itself is not
#: attributable to the probe.
PROBE_ENVELOPE = EnvelopeSpec(
    requires=frozenset({RegimeCondition.NEAR_POLICY, RegimeCondition.ABOVE_LOD}),
    measured_by={
        RegimeCondition.NEAR_POLICY: MEASURED_BY[RegimeCondition.NEAR_POLICY],
        RegimeCondition.ABOVE_LOD: MEASURED_BY[RegimeCondition.ABOVE_LOD],
    },
    on_violation="refuse",
)


@register_payload
@dataclass
class Recoverability:
    """What a probe on the policy's activations recovered, and what that added over the text.

    Kept as a payload rather than a bare dict because the `IncrementalValidity` record on it is the
    reason the reading exists, and a dict lets a caller drop it without anything noticing.
    """

    site: str
    n_items: int
    d_model: int
    n_folds: int
    ridge_alpha: float
    own_accuracy: float
    own_auroc: float
    best_baseline_id: str
    best_baseline_accuracy: float
    error_correlation: float
    ensemble_accuracy: float
    ensemble_gain: float
    increment: float
    ci_low: float
    ci_high: float
    ci_level: float
    is_redundant: bool
    adds_nothing: bool
    #: The baselines that actually entered the ensemble. A scored baseline whose per-item scores
    #: are constant carries no margin and is dropped here, so this is shorter than `baseline_auroc`
    #: whenever a comparator ran and separated nothing, and the difference is worth being able to
    #: see: an increment over two baselines is a weaker claim than an increment over six.
    baseline_ids_used: list[str] = field(default_factory=list)
    baseline_auroc: dict[str, float] = field(default_factory=dict)
    baseline_refusals: dict[str, str] = field(default_factory=dict)
    baselines: dict[str, float] = field(default_factory=dict)
    says: str = ""


def _ridge_scores(
    features: np.ndarray, labels: np.ndarray, *, n_folds: int, seed: int, alpha: float
) -> np.ndarray:
    """Cross-fitted ridge decision values, one per item, each from a probe that did not see it.

    The penalty is `alpha` times the mean eigenvalue of the training Gram matrix, which is a
    scale-free rule fixed in advance rather than a hyperparameter tuned on the evaluation set.
    Tuning it there is the free parameter that turns a probe into a fitted comparator, and the whole
    argument of the baseline bank is that fitted parameters have to be counted.

    Ridge with an isotropic penalty is orthogonally equivariant: rotating the features rotates the
    fitted weights the same way and leaves every prediction where it was. That is why this
    instrument declares `repr.basis` INVARIANT and means it.
    """
    x = np.asarray(features, dtype=np.float64)
    y = np.where(np.asarray(labels).astype(int) > 0, 1.0, -1.0)
    out = np.zeros(x.shape[0], dtype=np.float64)
    for fold in stratified_folds(np.asarray(labels).astype(int), n_folds, seed=seed):
        if fold.size == 0:
            continue
        train = np.ones(x.shape[0], dtype=bool)
        train[fold] = False
        xt, yt = x[train], y[train]
        if xt.shape[0] < 2 or np.unique(yt).size < 2:
            continue
        mean = xt.mean(axis=0)
        xc = xt - mean
        gram = xc.T @ xc
        penalty = alpha * float(np.trace(gram)) / max(xc.shape[1], 1)
        weights = np.linalg.solve(gram + penalty * np.eye(xc.shape[1]), xc.T @ (yt - yt.mean()))
        out[fold] = (x[fold] - mean) @ weights + yt.mean()
    return out


def _midpoint(scores: np.ndarray, labels: np.ndarray) -> float:
    """The threshold `accuracy_at_midpoint` uses, so own and baseline scores are comparable."""
    y = np.asarray(labels).astype(int)
    return 0.5 * (float(scores[y == 1].mean()) + float(scores[y == 0].mean()))


class PolicyReadoutProbe(BaseObservable):
    """How much of a recorded outcome is linearly present in the policy's residual stream.

    White-box: it needs `Access.FORWARD` on the policy and reads activations, so an
    `IncrementalValidity` record is mandatory on its reading and this instrument supplies one
    through `Context.emit(incremental=...)`.

    What it cannot do, beyond the module docstring's two. The probe is linear, so a quantity the
    policy represents nonlinearly reads as absent, and "not linearly decodable" is routinely
    reported as "not represented". The cross-fit removes the optimism of scoring an item with a
    probe that saw it and does not remove the optimism of choosing the site after looking: a site
    chosen because it scored best on this item set is a fitted parameter this payload does not
    count, so the site belongs in the claim.
    """

    name = "PolicyReadoutProbe"
    version = "1.0"
    capabilities = Capability.ACTIVATIONS
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "incremental validity"
    deviations = (
        "the ridge penalty is fixed at alpha times the mean eigenvalue of the training Gram "
        "matrix rather than selected, so the probe takes one hyperparameter chosen in advance "
        "and none chosen on the data",
        "M9's increment is computed on the same items the ensemble is combined on, so it is "
        "optimistic in the way any in-sample ensemble gain is; that bias is stated on M9's own "
        "estimator entry and is inherited unchanged here",
    )

    # -- the instrument declarations ---------------------------------------
    quantity = "policy.readout_recoverability"
    requires: AccessMatrix = {
        Component.POLICY: Access.FORWARD,
        Component.RECORD: Access.RECORD,
    }
    #: A policy is a generative network. It has no reward head, so `NEURAL_SCALAR` would be a claim
    #: about a `w_r` that does not exist, and the other four substrates have no activations at all.
    substrates = frozenset({Substrate.NEURAL_GEN})
    phases = frozenset({Phase.PRE_RUN, Phase.POST_RUN})
    envelope = PROBE_ENVELOPE
    #: The probe is ridge with an isotropic penalty, which is orthogonally equivariant, so a shared
    #: change of representation basis leaves every out-of-fold prediction exactly where it was.
    invariance = "repr.basis"
    invariance_relation = INVARIANT
    baselines = ALL_SIX
    rung = 0

    def __init__(
        self,
        *,
        labels: Sequence[int] | np.ndarray,
        texts: Sequence[str] = (),
        site: Site | None = None,
        markers: tuple[str, ...] = (),
        seed_labels: tuple[Any, ...] = (),
        n_folds: int = 5,
        ridge_alpha: float = 1.0,
        seed: int = 0,
        n_resamples: int = 2000,
    ) -> None:
        self.labels = np.asarray(labels).astype(int).ravel()
        self.texts = tuple(texts)
        self.site = site
        self.markers = tuple(markers)
        self.seed_labels = tuple(seed_labels)
        self.n_folds = int(n_folds)
        self.ridge_alpha = float(ridge_alpha)
        self.seed = int(seed)
        self.n_resamples = int(n_resamples)

    # -- the measurement ----------------------------------------------------

    def measure(self, ctx: Context) -> Any:
        subject = ctx.signal
        items = list(ctx.view)
        y = self.labels

        if y.size != len(items):
            return refuse_incomplete(
                self.name,
                field="one label per item",
                subject=f"{len(items)} items and {y.size} labels",
                remedy=(
                    "pass a label array aligned item-for-item with the view. A probe scored against "
                    "misaligned labels reports a floor, and nothing downstream can tell that from a "
                    "genuine null."
                ),
                n_items=len(items),
                n_labels=int(y.size),
            )
        if np.unique(y).size < 2:
            return refuse_incomplete(
                self.name,
                field="a label set containing both classes",
                subject=f"{len(items)} items, all labelled {int(y[0]) if y.size else 'nothing'}",
                remedy=(
                    "supply items of both classes. Every discrimination statistic is undefined on "
                    "one class, so widen the window until both appear, or lower the threshold that "
                    "produced the labels and record which threshold you used."
                ),
                n_items=len(items),
                n_positive=int((y == 1).sum()),
            )

        site = self.site or Site(int(subject.meta.n_layers) - 1, "resid_post")
        spec = CaptureSpec(sites=(site,), position=PositionSpec("final"), dtype="float32")
        capture = next(iter(subject.capture(items, spec)))
        features = capture.tensors[site].detach().to("cpu").numpy().astype(np.float64)

        own_scores = _ridge_scores(
            features,
            y,
            n_folds=self.n_folds,
            seed=self.seed,
            alpha=self.ridge_alpha,
        )
        if float(np.std(own_scores)) == 0.0:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.BELOW_LOD,
                detail=(
                    f"every cross-fitted probe score is identical ({float(own_scores[0]):.4g}), so "
                    f"the probe separated nothing at all on {len(items)} items at {site}"
                ),
                remedy=(
                    "read a site with more variance, or supply more items: a fold with one class "
                    "in its training half fits no direction and contributes a constant. "
                    f"n={len(items)} over {self.n_folds} folds leaves "
                    f"{len(items) // max(self.n_folds, 1)} items per fold."
                ),
                statistics={"n_items": len(items), "site": str(site)},
            )

        task = DetectionTask(
            labels=y,
            texts=self.texts,
            markers=self.markers,
            seed_labels=self.seed_labels,
            name=f"{self.name}:{site}",
        )
        bank = run_bank(task)

        own = Detector.from_scores(
            "policy.readout_probe",
            own_scores,
            y,
            threshold=_midpoint(own_scores, y),
            note=f"ridge probe on {site}, {self.n_folds}-fold cross-fitted",
        )
        baseline_detectors = [
            Detector.from_scores(
                bid,
                reading.scores,
                y,
                threshold=_midpoint(np.asarray(reading.scores, dtype=np.float64), y),
                note=reading.detail,
            )
            for bid, reading in sorted(bank.readings.items())
            if is_scored(reading) and float(np.std(reading.scores)) > 0.0
        ]
        if not baseline_detectors:
            return refuse_incomplete(
                self.name,
                field="at least one black-box baseline that could run",
                subject=self._bank_summary(bank),
                remedy=(
                    "supply the inputs the bank needs on the same items: `texts` for the string, "
                    "length, TF-IDF and n-gram baselines, `series` for the gradient-norm peak, and "
                    "a `judge` callable for the scaffolded prompt. Incremental validity is defined "
                    "against a set already run, and with an empty set this reading would be the "
                    "probe's own accuracy wearing a different name."
                ),
                **{k: v for k, v in self._bank_counts(bank).items()},
            )

        increment = IncrementalValidityReading(
            own=own,
            baselines_run=baseline_detectors,
            n_resamples=self.n_resamples,
            seed=self.seed,
        ).compute()
        if isinstance(increment, Refusal):
            return increment

        record = increment.record
        payload = Recoverability(
            site=str(site),
            n_items=len(items),
            d_model=int(features.shape[1]),
            n_folds=self.n_folds,
            ridge_alpha=self.ridge_alpha,
            own_accuracy=own.score,
            own_auroc=float(auroc(own_scores, y)),
            best_baseline_id=increment.best_baseline_id,
            best_baseline_accuracy=increment.best_baseline_score,
            error_correlation=increment.error_correlation,
            ensemble_accuracy=increment.ensemble_with_own,
            ensemble_gain=record.ensemble_gain,
            increment=increment.increment,
            ci_low=increment.ci_low,
            ci_high=increment.ci_high,
            ci_level=increment.ci_level,
            is_redundant=record.is_redundant,
            adds_nothing=increment.adds_nothing,
            baseline_ids_used=list(increment.baseline_ids),
            baseline_auroc={
                bid: float(reading.auroc)
                for bid, reading in sorted(bank.readings.items())
                if is_scored(reading)
            },
            baseline_refusals={
                bid: reading.detail
                for bid, reading in sorted(bank.readings.items())
                if not is_scored(reading)
            },
            baselines=dict(bank.as_mapping()),
            says=increment.says(),
        )
        return ctx.emit(
            payload,
            uncertainty=Uncertainty(
                ci_low=increment.ci_low,
                ci_high=increment.ci_high,
                ci_level=increment.ci_level,
                n=len(items),
                method=(
                    f"accuracy at the class-mean midpoint, cross-fitted over {self.n_folds} "
                    f"stratified folds; the interval is on the M9 increment, from "
                    f"{self.n_resamples:,} paired resamples over items"
                ),
            ),
            baselines=dict(bank.as_mapping()),
            incremental=record,
        )

    @staticmethod
    def _bank_summary(bank: BaselineBank) -> str:
        parts = [
            f"{bid} refused ({getattr(reading, 'reason', 'unknown')})"
            for bid, reading in sorted(bank.readings.items())
            if not is_scored(reading)
        ]
        return "; ".join(parts) or "no baseline produced a usable score"

    @staticmethod
    def _bank_counts(bank: BaselineBank) -> dict[str, int]:
        scored = sum(1 for r in bank.readings.values() if is_scored(r))
        return {"n_baselines_scored": scored, "n_baselines_refused": len(bank.readings) - scored}


__all__ = [
    "PROBE_ENVELOPE",
    "PolicyReadoutProbe",
    "Recoverability",
]

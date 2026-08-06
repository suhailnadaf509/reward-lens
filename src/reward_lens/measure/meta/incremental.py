"""M9, incremental validity: what this instrument adds to the ones already run.

The bar is **decorrelation plus signal, not superiority.** An instrument ten points worse than the
best method already in the bank and uncorrelated with it is worth more than one two points better
and redundant, because the ensemble of the first pair beats either member and the ensemble of the
second does not. A lone score cannot express that, which is why almost every published comparison
of a new detector against an old one is answering a question nobody asked.

Two numbers carry the whole reading and only one of them is common.

**The increment.** The score of the ensemble of the set plus this instrument, minus the score of the
ensemble of the set alone, on one fixed item set with the pairing preserved. That is the
psychometric definition of incremental validity carried over intact: what the predictor adds to a
model that already contains the others. It is reported with an interval from a paired resample over
items, because the whole point is being able to say that an increment is indistinguishable from
zero, and a point estimate cannot say that.

**The error correlation.** Between the two instruments' per-item *errors*, not between their scores.
Two instruments that agree about which items are hard have correlated errors even when their scores
differ a lot, and that is the case where an ensemble gains nothing. This is the statistic nobody
reports. The nearest published work makes the white-box-minus-black-box gap its benchmark and infers
complementarity from ensemble gain, which is inference where a measurement was available.

**The increment is a property of the combining rule as well as of the instrument, and that is
measured rather than hidden.** The default rule averages per-item margins; a rule that averages
standardised scores and then decides is a different estimator of the same construct and gives a
different number on the same data. `combiner` names which one produced the reading, and a claim that
an instrument adds nothing should be checked under both before it is made.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from reward_lens.core.budget import IncrementalValidity
from reward_lens.core.envelope import EnvelopeSpec, RegimeCondition
from reward_lens.core.evidence import Uncertainty, register_payload
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.quantity import (
    BaselineID,
    BiasStatement,
    CostModel,
    EstimatorEntry,
    register_estimator,
)
from reward_lens.core.reading import Refusal, RefusalReason, refuse_incomplete
from reward_lens.core.types import (
    Access,
    Capability,
    Component,
    GaugeStatus,
    Phase,
    Substrate,
)
from reward_lens.measure.meta._base import MetaInstrument
from reward_lens.measure.rate.regime import MEASURED_BY

#: Every input is a finished reading on a shared item set, so the record is all that is touched.
INCREMENTAL_ACCESS: dict[Component, Access] = {Component.RECORD: Access.RECORD}

#: The two numbers an increment has to be read against. The first is the null the interval is tested
#: against; the second is the number a paper reports instead, which is the instrument's own score,
#: and which says nothing at all about what it adds.
INCREMENTAL_BASELINES: tuple[BaselineID, ...] = (
    "baseline.no_increment",
    "baseline.own_score_alone",
)

#: An increment smaller than the substrate's disagreement with itself is not attributable to the
#: instrument. That is `ABOVE_LOD`, and M1 is what measures it.
INCREMENTAL_ENVELOPE = EnvelopeSpec(
    requires=frozenset({RegimeCondition.ABOVE_LOD}),
    measured_by={RegimeCondition.ABOVE_LOD: MEASURED_BY[RegimeCondition.ABOVE_LOD]},
    on_violation="refuse",
)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Detector:
    """One instrument's per-item verdict on a shared item set, as a signed margin.

    ``margin`` is positive when the instrument got the item right and its magnitude is how
    confidently. A ranking instrument's margin is the chosen option's score minus the best rival's;
    a binary detector's is the score minus its threshold, signed by the label. Both reduce to the
    same object, which is what lets one combining rule serve every kind of instrument.

    The sign convention is the whole interface and it is checked nowhere, because it cannot be: an
    instrument whose margins are all negative is either always wrong or has its sign flipped, and no
    amount of inspection distinguishes those. `score` being near zero is the symptom.
    """

    id: str
    margin: np.ndarray
    note: str = ""

    def __post_init__(self) -> None:
        m = np.asarray(self.margin, dtype=np.float64).ravel()
        object.__setattr__(self, "margin", m)
        if not self.id.strip():
            raise ValueError(
                "a detector needs an id; an unnamed one cannot appear in a baseline set"
            )

    @classmethod
    def from_scores(
        cls,
        id: str,
        scores: Sequence[float] | np.ndarray,
        labels: Sequence[float] | np.ndarray,
        *,
        threshold: float = 0.0,
        note: str = "",
    ) -> "Detector":
        """A binary detector's margin: distance from the threshold, signed by the label."""
        s = np.asarray(scores, dtype=np.float64).ravel()
        y = np.asarray(labels, dtype=np.float64).ravel()
        if s.size != y.size:
            raise ValueError(f"{s.size} scores and {y.size} labels are not the same item set")
        return cls(id=id, margin=(s - threshold) * np.where(y > 0, 1.0, -1.0), note=note)

    @property
    def n(self) -> int:
        return int(self.margin.size)

    @property
    def correct(self) -> np.ndarray:
        return self.margin > 0

    @property
    def errors(self) -> np.ndarray:
        """1.0 where this instrument got the item wrong. What the error correlation is taken over."""
        return (~self.correct).astype(np.float64)

    @property
    def score(self) -> float:
        return float(self.correct.mean()) if self.n else float("nan")


Combiner = Callable[[Sequence[Detector]], np.ndarray]


def mean_margin(detectors: Sequence[Detector]) -> np.ndarray:
    """The default rule: average the members' margins and take the sign.

    Every member contributes on its own scale, which is deliberate: a member whose margins are large
    because its scores are large gets more weight, and standardising them first is a different rule
    that the caller can pass and the reading will name.
    """
    return np.mean(np.stack([d.margin for d in detectors], axis=0), axis=0)


def standardised_margin(detectors: Sequence[Detector]) -> np.ndarray:
    """One member, one vote, after putting every member's margins on a common scale.

    Worth having beside `mean_margin` because the two disagree, and the disagreement is a real
    sensitivity of the increment rather than a detail: a panel whose members' score scales differ by
    two orders of magnitude is effectively a single-member ensemble under `mean_margin`.
    """
    rows = []
    for d in detectors:
        sd = float(np.std(d.margin, ddof=1)) if d.n > 1 else 0.0
        rows.append(d.margin / sd if sd > 0 else d.margin)
    return np.mean(np.stack(rows, axis=0), axis=0)


def phi(a: np.ndarray, b: np.ndarray) -> float:
    """Correlation between two error vectors, or NaN when one of them has no variance.

    NaN and not zero. A detector that is right on every item has no errors to correlate, and
    reporting an error correlation of zero for it would say "these two fail independently" about a
    pair where one of them does not fail.
    """
    if a.size < 2 or float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------------------
# The reading
# ---------------------------------------------------------------------------


@register_payload
@dataclass
class Increment:
    """What one instrument adds to a set already run, with the interval and the error correlation."""

    own_id: str
    own_score: float
    baseline_ids: tuple[str, ...]
    best_baseline_id: str
    best_baseline_score: float
    ensemble_baseline: float
    ensemble_with_own: float
    increment: float
    ci_low: float
    ci_high: float
    ci_level: float
    error_correlation: float
    error_correlation_with_ensemble: float
    n: int
    n_resamples: int
    combiner: str
    baselines: Mapping[str, float] = field(default_factory=dict)

    @property
    def adds_nothing(self) -> bool:
        """Whether the increment's interval covers zero. A finding about the instrument."""
        return self.ci_low <= 0.0 <= self.ci_high

    @property
    def subtracts(self) -> bool:
        """Whether the whole interval is below zero: adding it makes the set worse.

        A separate property from `adds_nothing` because the three cases are three different
        findings, and the one that gets collapsed is this one. An interval excluding zero on the
        wrong side is not a win with a sign, it is evidence that the instrument is contributing
        errors the set does not already make.
        """
        return self.ci_high < 0.0

    @property
    def record(self) -> IncrementalValidity:
        """The kernel's record, so this reading composes with anything that already reads one."""
        return IncrementalValidity(
            own_score=self.own_score,
            baseline_score=self.best_baseline_score,
            baseline_id=self.best_baseline_id,
            error_correlation=self.error_correlation,
            ensemble_score=self.ensemble_with_own,
        )

    def says(self) -> str:
        if self.adds_nothing:
            verdict = (
                "an interval covering zero, so on this item set and under this combining rule it "
                "adds nothing measurable to the set already run"
            )
        elif self.subtracts:
            verdict = (
                "an interval entirely below zero, so on this item set and under this combining "
                "rule it makes the set worse rather than adding to it"
            )
        else:
            verdict = (
                "an interval entirely above zero, so it adds something the set already run does "
                "not have"
            )
        correlation = (
            "undefined (one of the two error vectors has no variance)"
            if math.isnan(self.error_correlation)
            else f"{self.error_correlation:+.2f}"
        )
        return (
            f"{self.own_id} scores {self.own_score:.3f} on its own against {self.best_baseline_score:.3f} "
            f"for the best of the {len(self.baseline_ids)} instruments already run. Added to their "
            f"ensemble it moves {self.ensemble_baseline:.3f} to {self.ensemble_with_own:.3f}, an "
            f"increment of {self.increment:+.4f} [{self.ci_low:+.4f}, {self.ci_high:+.4f}]: "
            f"{verdict}. Its errors correlate {correlation} with the best baseline's."
        )

    def render(self) -> str:
        return "\n".join(
            [
                self.says(),
                f"    combining rule: {self.combiner}",
                f"    {self.record.render()}",
                f"    n = {self.n:,} items, {self.n_resamples:,} paired resamples",
            ]
        )


# ---------------------------------------------------------------------------
# The instrument
# ---------------------------------------------------------------------------


class IncrementalValidityReading(MetaInstrument):
    """M9. Whether an instrument adds anything over the ones already run, measured.

    Required on every white-box reading. The reading it produces is deliberately capable of
    saying no: an increment whose interval covers zero is the intended output for a redundant
    instrument, not a failure of the measurement.
    """

    name = "IncrementalValidity"
    version = "1.0"
    quantity = "instrument.incremental_validity"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    requires = INCREMENTAL_ACCESS
    substrates = frozenset(
        {
            Substrate.NEURAL_SCALAR,
            Substrate.NEURAL_GEN,
            Substrate.PROGRAM,
            Substrate.PROCEDURAL,
            Substrate.HUMAN,
            Substrate.COMPOSITE,
        }
    )
    phases = frozenset({Phase.PRE_RUN, Phase.POST_RUN, Phase.DEPLOYED})
    envelope = INCREMENTAL_ENVELOPE
    invariance = "units"
    invariance_relation = INVARIANT
    baselines = INCREMENTAL_BASELINES
    rung = 0
    faithful_to = "M9"
    deviations = (
        "the registry gives `instrument.incremental_validity` the unit `4-tuple` and this reading "
        "carries six numbers plus an interval. The four are on the reading as the kernel's own "
        "`IncrementalValidity` record, reachable as `Increment.record`, so anything consuming the "
        "registered shape gets it unchanged",
        "`Evidence.incremental` is the field that is mandatory on a white-box reading and "
        "`Context.emit` takes no argument that sets it, so the record travels in the value payload "
        "instead. Noted rather than worked around, because the fix is one keyword on `emit` and it "
        "is not this package's to make",
    )

    def __init__(
        self,
        own: Detector | None = None,
        baselines_run: Sequence[Detector] = (),
        *,
        combiner: Combiner = mean_margin,
        combiner_name: str = "",
        n_resamples: int = 2000,
        ci: float = 0.95,
        seed: int = 0,
    ) -> None:
        self.own = own
        self.baselines_run = tuple(baselines_run)
        self.combiner = combiner
        self.combiner_name = combiner_name or getattr(combiner, "__name__", "combiner")
        self.n_resamples = int(n_resamples)
        self.ci = float(ci)
        self.seed = int(seed)

    def compute(self) -> Any:
        own = self.own
        if own is None or own.n == 0:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail="no instrument was supplied to measure the increment of",
                remedy=(
                    "pass the instrument's per-item margins as `own=Detector(id=..., margin=...)`, "
                    "where the margin is positive on items it got right. For a binary detector, "
                    "`Detector.from_scores(id, scores, labels)` builds it."
                ),
                statistics={"n": 0},
            )
        if not self.baselines_run:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail=(
                    "no baseline instruments were supplied. Incremental validity is defined over a "
                    "set already run, and with an empty set the reading would be the instrument's "
                    "own score wearing a different name"
                ),
                remedy=(
                    "pass the instruments already run on the same items as "
                    "`baselines_run=[Detector(...), ...]`. If nothing has been run, M3's dumb "
                    "baseline bank is the set to run first: a claim with no baseline is not a claim, "
                    "and an increment over nothing is not an increment."
                ),
                statistics={"n_baselines": 0},
            )
        sizes = {d.id: d.n for d in (own, *self.baselines_run)}
        if len(set(sizes.values())) > 1:
            return refuse_incomplete(
                self.name,
                field="a common item set",
                subject=f"the {len(sizes)} instruments ({sizes})",
                remedy=(
                    "re-run every instrument on identical items and pass the margins aligned "
                    "item-for-item. The increment is a paired quantity: computed across different "
                    "item sets it contains a sampling difference that nothing downstream can "
                    "separate from the instrument's contribution."
                ),
                sizes=sizes,
            )

        base = list(self.baselines_run)
        ens_base = self.combiner(base) > 0
        ens_all = self.combiner([*base, own]) > 0
        increment = float(ens_all.mean() - ens_base.mean())

        scores = np.array([d.score for d in base], dtype=np.float64)
        best = int(np.argmax(scores))
        best_detector = base[best]
        ens_base_errors = (~ens_base).astype(np.float64)

        rng = np.random.default_rng(self.seed)
        n = own.n
        draws = np.empty(self.n_resamples, dtype=np.float64)
        all_f = ens_all.astype(np.float64)
        base_f = ens_base.astype(np.float64)
        for r in range(self.n_resamples):
            idx = rng.integers(0, n, n)
            draws[r] = float(all_f[idx].mean() - base_f[idx].mean())
        alpha = (1.0 - self.ci) / 2.0
        lo, hi = (float(x) for x in np.quantile(draws, [alpha, 1.0 - alpha]))

        return Increment(
            own_id=own.id,
            own_score=own.score,
            baseline_ids=tuple(d.id for d in base),
            best_baseline_id=best_detector.id,
            best_baseline_score=float(scores[best]),
            ensemble_baseline=float(ens_base.mean()),
            ensemble_with_own=float(ens_all.mean()),
            increment=increment,
            ci_low=lo,
            ci_high=hi,
            ci_level=self.ci,
            error_correlation=phi(own.errors, best_detector.errors),
            error_correlation_with_ensemble=phi(own.errors, ens_base_errors),
            n=n,
            n_resamples=self.n_resamples,
            combiner=self.combiner_name,
            baselines={
                "baseline.no_increment": 0.0,
                "baseline.own_score_alone": own.score,
            },
        )

    def uncertainty(self, computed: Increment) -> Uncertainty | None:
        return Uncertainty(
            ci_low=computed.ci_low,
            ci_high=computed.ci_high,
            ci_level=computed.ci_level,
            n=computed.n,
            method=(
                f"paired percentile bootstrap over items, {computed.n_resamples:,} resamples, "
                f"combining rule {computed.combiner}"
            ),
        )

    def payload(self, computed: Increment) -> dict[str, Any]:
        return {
            "own_id": computed.own_id,
            "own_score": computed.own_score,
            "baseline_ids": list(computed.baseline_ids),
            "best_baseline_id": computed.best_baseline_id,
            "best_baseline_score": computed.best_baseline_score,
            "ensemble_baseline": computed.ensemble_baseline,
            "ensemble_with_own": computed.ensemble_with_own,
            "increment": computed.increment,
            "ci_low": computed.ci_low,
            "ci_high": computed.ci_high,
            "ci_level": computed.ci_level,
            "error_correlation": computed.error_correlation,
            "error_correlation_with_ensemble": computed.error_correlation_with_ensemble,
            "ensemble_gain": computed.record.ensemble_gain,
            "is_redundant": computed.record.is_redundant,
            "adds_nothing": computed.adds_nothing,
            "subtracts": computed.subtracts,
            "n": computed.n,
            "n_resamples": computed.n_resamples,
            "combiner": computed.combiner,
            "incremental_validity": computed.record.__canonical__(),
            "baselines": dict(computed.baselines),
            "says": computed.says(),
        }


def register_ladder() -> list[str]:
    """Register M9's rungs for its two quantities. Not called at import, by design."""
    entries = [
        EstimatorEntry(
            quantity="instrument.incremental_validity",
            impl="m9.paired_ensemble_increment",
            requires=INCREMENTAL_ACCESS,
            envelope=INCREMENTAL_ENVELOPE,
            rung=0,
            bias=BiasStatement(
                direction="unknown",
                why=(
                    "the increment is measured on the same items the ensemble is combined on, so it "
                    "is optimistic in the way any in-sample ensemble gain is. A held-out increment "
                    "is the higher rung and it needs a split the caller has to declare"
                ),
            ),
            cost=CostModel(note="arithmetic on readings already taken"),
            run=None,
        ),
        EstimatorEntry(
            quantity="instrument.error_correlation",
            impl="m9.error_phi",
            requires=INCREMENTAL_ACCESS,
            envelope=INCREMENTAL_ENVELOPE,
            rung=0,
            bias=BiasStatement(
                direction="approximately_unbiased",
                why=(
                    "a correlation between two observed error vectors on one item set. It is "
                    "attenuated by however noisy each instrument's own scoring is, in the "
                    "direction Spearman described in 1904, and A3 is the instrument that corrects "
                    "for that when the reliabilities are known"
                ),
            ),
            cost=CostModel(note="arithmetic on readings already taken"),
            run=None,
        ),
    ]
    for e in entries:
        register_estimator(e)
    return [e.impl for e in entries]


__all__ = [
    "INCREMENTAL_ACCESS",
    "INCREMENTAL_BASELINES",
    "INCREMENTAL_ENVELOPE",
    "Combiner",
    "Detector",
    "Increment",
    "IncrementalValidityReading",
    "mean_margin",
    "phi",
    "register_ladder",
    "standardised_margin",
]

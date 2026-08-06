"""Shared machinery for series L: the binomial interval, the stratified rank statistic, and emit.

Three things every instrument in this package needs and none of them belongs in five copies.

`wilson_interval` is the interval a proportion estimated from an audit gets. Not the normal
approximation: at the sample sizes a hand audit can afford, and at the error rates label audits
actually find, the Wald interval crosses zero and reports a negative lower bound on a count. A
label set with 0 wrong out of 20 audited has a 95% upper bound of 0.161, and printing "0% error"
for that sample is the single most common way a label-quality claim is overstated.

`stratified_auc` is the rank statistic conditioned on a stratum. An AUC pooled over every candidate
implicitly scores the detector against a uniform prior over the strata; conditioning on the stratum
removes whatever the stratum alone predicts, which is what L5 needs and what nothing in `stats/roc`
does.

`emit_with_reference` exists because `Context.emit` has no way to pass a reference material through
to `make_evidence`, and the reference is what caps the trust. The kernel's `make_evidence` already
takes `reference=` and already applies `compute_trust`'s cap; `Context.emit` simply does not forward
it, and the trust level is part of the Evidence content id, so patching it afterwards would leave
the id disagreeing with the content. Until `emit` forwards the reference, this helper does what
`emit` does, gate for gate, plus the two fields `emit` drops.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from reward_lens.core.evidence import Evidence, Uncertainty, make_evidence, register_payload
from reward_lens.core.provenance import Cost, Provenance, capture_provenance
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.types import Access, AccessMatrix, Component, GaugeStatus, Phase, Substrate
from reward_lens.measure.base import BaseObservable, Context, lookup_calibration

# The average-rank helper `stats.roc` itself imports for the Mann-Whitney identity. Reached by the
# same private name it uses, rather than reimplemented here, so the two cannot drift on ties.
from reward_lens.stats.effects import _rankdata

#: 1 - alpha/2 quantiles of the standard normal, for the two confidence levels an audit uses.
_Z: dict[float, float] = {
    0.90: 1.6448536269514722,
    0.95: 1.959963984540054,
    0.99: 2.5758293035489004,
}


def z_for(level: float) -> float:
    """The two-sided normal quantile for a confidence level, refusing to guess an unlisted one.

    Three levels, tabulated rather than computed, because the alternative is either a dependency on
    `scipy.stats` for one number or an inverse-erf approximation whose error nobody checks. A caller
    who needs a fourth level says so and it gets tabulated.
    """
    try:
        return _Z[round(float(level), 6)]
    except KeyError:
        raise ValueError(
            f"no normal quantile tabulated for level {level}; this package carries "
            f"{sorted(_Z)} and refuses to approximate a fourth rather than ship an inverse-erf "
            f"whose error nobody has checked."
        ) from None


@register_payload
@dataclass(frozen=True)
class Proportion:
    """A count over a sample, with the Wilson interval on the proportion it estimates.

    ``point`` is `k/n` and it is the number people quote. ``low`` and ``high`` are the score
    interval, which is what makes the quote honest: at n = 200 a measured 4% carries a 95% interval
    roughly two points wide either side, and a leaderboard gap narrower than that is not a gap.

    An empty sample is not an error here. `n = 0` gives a point of NaN and the full [0, 1] interval,
    because "nobody has audited this" is a real state of knowledge and the correct interval for it
    is the whole line.
    """

    k: int
    n: int
    point: float
    low: float
    high: float
    level: float = 0.95
    method: str = "wilson score"

    @property
    def is_measured(self) -> bool:
        return self.n > 0

    @property
    def width(self) -> float:
        return self.high - self.low

    def render(self) -> str:
        if not self.is_measured:
            return "not measured (n = 0); the interval is the whole line"
        return (
            f"{self.point:.4g} [{self.low:.4g}, {self.high:.4g}] "
            f"({self.k}/{self.n}, {self.level:.0%} {self.method})"
        )

    def __canonical__(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "n": self.n,
            "point": self.point,
            "low": self.low,
            "high": self.high,
            "level": self.level,
            "method": self.method,
        }


def wilson_interval(k: int, n: int, *, level: float = 0.95) -> Proportion:
    """The Wilson score interval for `k` successes in `n` trials.

        centre = (p + z^2/2n) / (1 + z^2/n)
        half   = z/(1 + z^2/n) * sqrt(p(1-p)/n + z^2/4n^2)

    Wilson rather than Wald for the reason that decides every label audit: an audit that finds no
    errors has `p = 0`, and Wald gives the interval `[0, 0]`, which reads as a proof that the labels
    are perfect and is produced by twenty items. Wilson gives `[0, 0.161]` at n = 20, which is the
    honest statement, and it is what makes the score ceiling a bound rather than a boast.

    Clamped to `[0, p] x [p, 1]`, which is not cosmetic. The Wilson interval provably contains the
    observed proportion, because it is the set of `p` satisfying `|p_hat - p| <= z*sqrt(p(1-p)/n)`
    and `p = p_hat` makes the left side zero. Floating-point cancellation breaks that at the ends:
    at `k = 0, n = 7` the computed lower endpoint is `2.8e-17`, which is above the point estimate of
    zero, and an interval that excludes its own point estimate is the kind of thing that survives
    review and then confuses somebody a year later. Clamping restores the guarantee exactly.
    """
    if n < 0:
        raise ValueError(f"n cannot be negative; got {n}")
    if not 0 <= k <= max(n, 0):
        raise ValueError(f"k must lie in [0, n]; got k={k}, n={n}")
    if n == 0:
        return Proportion(k=0, n=0, point=float("nan"), low=0.0, high=1.0, level=level)
    z = z_for(level)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return Proportion(
        k=int(k),
        n=int(n),
        point=p,
        low=min(max(0.0, centre - half), p),
        high=max(min(1.0, centre + half), p),
        level=level,
    )


def stratified_auc(
    scores: np.ndarray, labels: np.ndarray, strata: np.ndarray
) -> tuple[float, int, int]:
    """`P(score of a positive > score of a negative | both in the same stratum)`, ties at one half.

    The pooled AUC asks whether a positive outranks a *randomly drawn* negative, which is the same
    as scoring the detector against a uniform prior over whatever the strata index. This asks
    whether it outranks a negative *from the same stratum*, so anything the stratum alone predicts
    is removed by construction rather than adjusted for afterwards.

    Weighted by pairs rather than by stratum, which is the choice that matters. A stratum holding
    one positive and two negatives contributes two comparisons and a stratum holding forty
    positives and six hundred negatives contributes twenty-four thousand, and averaging the
    per-stratum AUCs instead would give the first stratum the same voice as the second.

    Returns the statistic, the number of strata that contributed at least one pair, and the total
    pair count. A stratum with no positive or no negative contributes nothing and is not an error:
    it is a stratum where the comparison is undefined, and counting it as 0.5 would be inventing
    observations.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel().astype(int)
    strata = np.asarray(strata).ravel()
    if not (scores.size == labels.size == strata.size):
        raise ValueError(
            f"scores, labels and strata must be the same length; got {scores.size}, "
            f"{labels.size}, {strata.size}. A detector aligned to the wrong items scores noise "
            f"and reports it as a floor."
        )
    wins = 0.0
    pairs = 0.0
    used = 0
    for st in np.unique(strata):
        m = strata == st
        y = labels[m] == 1
        n_pos = int(np.count_nonzero(y))
        n_neg = int(y.size) - n_pos
        if n_pos == 0 or n_neg == 0:
            continue
        # The rank identity rather than a pairwise difference matrix: exact under ties, and O(m log
        # m) instead of O(m^2), which is what makes a thousand bootstrap draws over twenty thousand
        # candidates finish in seconds rather than hours.
        ranks = _rankdata(scores[m])
        wins += float(ranks[y].sum()) - n_pos * (n_pos + 1.0) / 2.0
        pairs += float(n_pos * n_neg)
        used += 1
    if pairs == 0.0:
        return float("nan"), used, 0
    return wins / pairs, used, int(pairs)


def cluster_bootstrap_ci(
    statistic: Any,
    clusters: Sequence[Any],
    *,
    n_boot: int = 1000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """A percentile interval resampling whole clusters, because the items inside one are not free.

    Every series-L statistic is computed over items nested inside something: steps inside a
    trajectory, audited labels inside an annotator's batch, transcripts inside a task. Resampling
    the items would treat nineteen thousand steps as nineteen thousand independent observations
    when there are two thousand trajectories, and would give an interval roughly three times too
    narrow. ``clusters`` is the list of cluster payloads; ``statistic`` maps a resampled list of
    them to a number.
    """
    if not clusters:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(clusters)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        value = statistic([clusters[i] for i in idx])
        if np.isfinite(value):
            draws.append(float(value))
    if not draws:
        return float("nan"), float("nan")
    alpha = (1.0 - level) / 2.0
    return (
        float(np.quantile(draws, alpha)),
        float(np.quantile(draws, 1.0 - alpha)),
    )


# ---------------------------------------------------------------------------
# Refusals this series owns
# ---------------------------------------------------------------------------


def label_quality_refusal(instrument: str, *, what: str, remedy: str, **statistics: Any) -> Refusal:
    """`LABEL_QUALITY_UNKNOWN`: scoring against labels with no measured error rate.

    The remedy is written as an instruction because a refusal without one is a tool that looks
    broken rather than a tool that looks careful. ``what`` names the specific thing that was not
    measured, so the same reason can carry "no audit sample" and "no procedure-violation column"
    without the reader having to guess which.
    """
    return Refusal(
        instrument=instrument,
        reason=RefusalReason.LABEL_QUALITY_UNKNOWN,
        detail=(
            f"{what}. Scoring an instrument against an answer key of unmeasured quality measures "
            f"the answer key: the reported number is the sum of what the instrument got right and "
            f"what the labels got wrong, and nothing in the number separates them."
        ),
        remedy=remedy,
        statistics=dict(statistics),
    )


# ---------------------------------------------------------------------------
# Emit
# ---------------------------------------------------------------------------


def emit_with_reference(
    ctx: Context,
    value: Any,
    *,
    reference: Any = None,
    quantity: str = "",
    uncertainty: Uncertainty | None = None,
    gauge: GaugeStatus | None = None,
    baselines: Mapping[str, float] | None = None,
    parents: tuple[str, ...] = (),
    cost: Cost | None = None,
    subject_extra: dict[str, Any] | None = None,
) -> Evidence:
    """`Context.emit`, plus the reference that caps the trust and the quantity the instrument declares.

    Gate for gate identical to `Context.emit`: gate 1 looks up a calibration reference for this
    instrument and subject, gate 3 reads the frozen study off the context, and the trust level falls
    out of `make_evidence` rather than being set here. What it adds is the two arguments
    `make_evidence` already accepts and `emit` does not forward.

    The reference matters because `compute_trust` caps at `CALIBRATED` when the reference is
    uncertified, and that cap is the one rule this whole series exists to enforce. Routing around
    `emit` is not a preference: `trust` is inside the Evidence content id, so capping it after the
    fact would produce a row whose id disagrees with its own trust level.

    The quantity matters for a duller reason. `Evidence.quantity` documents an empty value as "the
    instrument has not been retrofitted", `lint_instrument` reports it as a gap, and `emit` drops
    the declaration on the floor, so a fully retrofitted instrument still writes an empty field.
    """
    obs = ctx._observable
    name = getattr(obs, "name", "anonymous")
    version = getattr(obs, "version", "0")
    gauge_status = gauge or getattr(obs, "gauge_status", GaugeStatus.INVARIANT)
    subject = ctx.subject(subject_extra)
    calibration = lookup_calibration(name, subject, ctx.regime)
    prov = capture_provenance(parents=parents, study=ctx.study, cost=cost)
    if cost is not None:
        prov = Provenance(
            git_sha=prov.git_sha,
            config_hash=prov.config_hash,
            seeds=prov.seeds,
            cost=cost,
            oracle_calls=prov.oracle_calls,
            parents=tuple(parents),
            study=ctx.study,
            extra=prov.extra,
        )
    return make_evidence(
        observable=name,
        observable_version=version,
        subject=subject,
        value=value,
        uncertainty=uncertainty,
        gauge=gauge_status,
        calibration=calibration,
        provenance=prov,
        registered=ctx.study is not None,
        quantity=quantity or getattr(obs, "quantity", ""),
        reference=reference,
        baselines=baselines,
    )


# ---------------------------------------------------------------------------
# The shared instrument base
# ---------------------------------------------------------------------------

#: Reading a labelled corpus and its answer key. `Access.RECORD` on `GOLD` is the labels; the
#: instruments that also read the items declare `Component.RECORD` alongside.
ACCESS_LABELLED_CORPUS: AccessMatrix = {
    Component.GOLD: Access.RECORD,
    Component.RECORD: Access.RECORD,
}

#: Planting an organism at a controlled dose. `MUTATE` on `GOLD` is the whole of L1's rung 1: you
#: cannot measure between-seed homogeneity without being able to make a second plant.
ACCESS_PLANT: AccessMatrix = {Component.GOLD: Access.MUTATE}

#: Every substrate, and the declaration is deliberate rather than lazy. A label's error rate is a
#: property of the answer key, not of the grader being scored against it, so restricting these
#: instruments by substrate would exclude cases where they apply exactly as well. The one axis that
#: does restrict them is access, and they declare that.
ALL_SUBSTRATES: frozenset[Substrate] = frozenset(Substrate)

#: Every phase. An answer key can be audited before a run, during one, after one, or in deployment,
#: and the audit is the same audit.
ALL_PHASES: frozenset[Phase] = frozenset(Phase)


class LabelsInstrument(BaseObservable):
    """The shared declarations for series L, so five instruments do not restate them.

    Two of the twelve section-4.2 declarations are the same for every instrument here and the rest
    are not, so this carries exactly those two and leaves the rest to the subclass. Anything with a
    plausible default is left undeclared on purpose: a plausible default is indistinguishable from a
    decision, and `lint_instrument` is what turns an undeclared field into a finding.
    """

    substrates = ALL_SUBSTRATES
    phases = ALL_PHASES
    gauge_status = GaugeStatus.INVARIANT


__all__ = [
    "ACCESS_LABELLED_CORPUS",
    "ACCESS_PLANT",
    "ALL_PHASES",
    "ALL_SUBSTRATES",
    "LabelsInstrument",
    "Proportion",
    "cluster_bootstrap_ci",
    "emit_with_reference",
    "label_quality_refusal",
    "stratified_auc",
    "wilson_interval",
    "z_for",
]

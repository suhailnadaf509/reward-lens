"""Scoring: a proper rule, the Murphy decomposition, skill with an interval, and decision value.

Four things, in increasing order of how rarely anybody computes them.

A **proper scoring rule** is the floor. Brier and log are both here; Brier is the default because
it is bounded and because a single confident miss under the log rule is an infinite score, which
turns a ledger into a report about its worst row.

The **Murphy decomposition** splits the Brier score into reliability, resolution and uncertainty
with `BS = REL - RES + UNC`, and it is the difference between "our forecasts were bad" and "our
forecasts were badly calibrated but did discriminate" or "they were well calibrated and said nothing
the base rate did not". Those are three different problems with three different fixes and one number
cannot tell them apart.

A **skill score with an interval**, because a skill score is a ratio of two noisy numbers and a
point estimate of one is nearly meaningless at the sample sizes anyone actually has. Sixteen
forecasts is a small n and the interval says so.

And **decision value**, which nobody in this field computes and which is the number that answers "so
what". It is the classic cost-loss value score from operational meteorology: given what acting costs
and what the event costs if you did not act, what fraction of the achievable saving over climatology
did this forecaster deliver. A forecaster with an unimpressive Brier can have real decision value if
its errors sit away from the decision threshold, and a forecaster with a good Brier can have none.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from reward_lens.core.evidence import register_payload
from reward_lens.forecast.schema import DecisionSpec, ForecastError

# ---------------------------------------------------------------------------
# Proper scoring rules
# ---------------------------------------------------------------------------


def brier(probabilities: Sequence[float], outcomes: Sequence[bool]) -> float:
    """The mean squared error of a probability forecast. Proper, bounded, and the default.

    The half-Brier convention (one term per forecast, `(p - y)^2`) rather than the original
    two-category sum, which is twice this. Both are in the literature and they differ by a factor of
    two; the campaign's published 0.26 is on this convention and a coin sits at 0.25 on it, which is
    the check that settles which one a number is on.
    """
    p, y = _paired(probabilities, outcomes)
    return float(np.mean((p - y) ** 2))


def brier_terms(probabilities: Sequence[float], outcomes: Sequence[bool]) -> np.ndarray:
    """The per-forecast squared errors, kept because the interval bootstraps over them."""
    p, y = _paired(probabilities, outcomes)
    return np.asarray((p - y) ** 2, dtype=np.float64)


def log_score(
    probabilities: Sequence[float], outcomes: Sequence[bool], *, floor: float = 1e-6
) -> float:
    """The mean negative log likelihood. Proper, unbounded, and reported with its floor stated.

    ``floor`` clips the probability away from 0 and 1. Without it a single confident miss is
    infinite and the mean is a report about that one row; with it the worst achievable single score
    is `-log(floor)`, which is 13.8 at the default and is still enough to dominate a small ledger.
    The floor is a parameter and not a constant because it changes the number, and a scoring
    convention that changes the number should be visible at the call site.
    """
    p, y = _paired(probabilities, outcomes)
    p = np.clip(p, floor, 1.0 - floor)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _paired(
    probabilities: Sequence[float], outcomes: Sequence[bool]
) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(probabilities, dtype=np.float64).ravel()
    y = np.asarray([1.0 if bool(o) else 0.0 for o in outcomes], dtype=np.float64).ravel()
    if p.size != y.size:
        raise ForecastError(
            f"{p.size} probabilities against {y.size} outcomes. A score computed on a misaligned "
            f"pair is a number about nothing, and it is the commonest way a re-scoring goes wrong."
        )
    if p.size == 0:
        raise ForecastError(
            "no resolved forecasts to score. An empty ledger has no Brier score; report the count "
            "of voids instead, which is the finding."
        )
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        bad = int(np.sum(~np.isfinite(p) | (p < 0) | (p > 1)))
        raise ForecastError(f"{bad} of {p.size} forecast values are not probabilities")
    return p, y


# ---------------------------------------------------------------------------
# The Murphy decomposition
# ---------------------------------------------------------------------------


@register_payload
@dataclass(frozen=True)
class MurphyDecomposition:
    """`BS = REL - RES + UNC`, and the three terms are three different problems.

    **Reliability** is how far the observed frequency in each probability bin sits from the
    probability the bin claimed. Zero is perfect calibration. It is the term a recalibration fixes,
    and fixing it needs no new information at all.

    **Resolution** is how far those bin frequencies sit from the overall base rate. Bigger is
    better. It is the term that says the forecaster distinguished cases, and no recalibration can
    manufacture it: it is the only one of the three that requires actually knowing something.

    **Uncertainty** is `ō(1 - ō)`, the base rate's own variance. It is a property of the target and
    not of the forecaster, and it is the reason two Brier scores from two different targets are not
    comparable.

    ``binning`` records how the bins were formed, because the decomposition is exact only when
    forecasts are binned by their distinct values, and approximate (with a within-bin variance term
    silently folded into reliability) when they are binned by width.
    """

    brier: float
    reliability: float
    resolution: float
    uncertainty: float
    n: int
    n_bins: int
    base_rate: float
    binning: str

    @property
    def residual(self) -> float:
        """`BS - (REL - RES + UNC)`. Exactly zero under distinct-value binning; a check, not a term."""
        return self.brier - (self.reliability - self.resolution + self.uncertainty)

    def render(self) -> str:
        return (
            f"Brier {self.brier:.4f} = reliability {self.reliability:.4f} "
            f"- resolution {self.resolution:.4f} + uncertainty {self.uncertainty:.4f}   "
            f"(n={self.n}, {self.n_bins} bins by {self.binning}, base rate {self.base_rate:.4f})"
        )

    def __canonical__(self) -> dict[str, Any]:
        return {
            "brier": self.brier,
            "reliability": self.reliability,
            "resolution": self.resolution,
            "uncertainty": self.uncertainty,
            "n": self.n,
            "n_bins": self.n_bins,
            "base_rate": self.base_rate,
            "binning": self.binning,
        }


def _bin(p: np.ndarray, bins: int | None) -> tuple[np.ndarray, int, bool]:
    """Assign each forecast to a bin. Returns `(assignments, n_bins, exact)`.

    One function rather than a copy in each of the three places that bin, because the diagram, the
    decomposition and the recalibration are three views of one table and two of them silently
    disagreeing is the failure that makes a decomposition unreadable.

    ``exact`` is True when the bins are the distinct forecast values, which is the condition under
    which the Murphy identity closes to machine precision and under which recalibration provably
    cannot raise the Brier score. Neither holds under equal-width bins, and the difference is not a
    rounding matter: see `recalibrate`.
    """
    distinct = np.unique(p)
    if bins is None and distinct.size <= max(10, p.size // 2):
        return np.searchsorted(distinct, p), int(distinct.size), True
    n_bins = bins or 10
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    return np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1), int(n_bins), False


def recalibrate(
    probabilities: Sequence[float], outcomes: Sequence[bool], *, bins: int | None = None
) -> np.ndarray:
    """Replace each forecast by the observed frequency in its own bin. The reliability fix.

    This is what "reliability is the term a recalibration fixes" means operationally, and having it
    in the module rather than in a docstring makes the claim checkable. Under distinct-value binning
    it is exact and it buys exactly the reliability term:

        `BS(recalibrated) = BS - REL`

    to machine precision, which is the finite-sample statement of properness.

    **Under equal-width binning it can raise the Brier score**, and the guarantee does not merely
    weaken, it fails. The general identity is

        `BS - BS(recalibrated) = REL + (1/n) sum_k n_k [var_k(p) - 2 cov_k(p, y)]`

    where the sum runs over bins and `var_k`, `cov_k` are within-bin. Distinct-value bins have no
    within-bin spread, so both vanish and the difference is `REL >= 0`. An equal-width bin holding
    several distinct forecasts that genuinely track their outcomes has `cov_k(p, y) > 0`, and
    flattening them to the bin mean throws that away. Measured over 200,000 draws from a perfectly
    calibrated generator at 10 equal-width bins, recalibration raised the Brier score in 18 of them,
    worst case 0.16484 to 0.17424 on eleven forecasts.

    So the honest reading of a reliability term computed on equal-width bins is that part of it is
    within-bin spread rather than miscalibration, which is what `MurphyDecomposition.binning`
    labelling itself "(approximate)" is warning about.
    """
    p, y = _paired(probabilities, outcomes)
    assignments, n_bins, _ = _bin(p, bins)
    out = np.empty_like(p)
    for k in range(n_bins):
        mask = assignments == k
        if np.any(mask):
            out[mask] = float(np.mean(y[mask]))
    return out


def murphy_decomposition(
    probabilities: Sequence[float],
    outcomes: Sequence[bool],
    *,
    bins: int | None = None,
) -> MurphyDecomposition:
    """Decompose the Brier score into reliability, resolution and uncertainty.

    Bins by **distinct forecast value** when there are few of them, which makes the decomposition
    exact rather than approximate. The campaign issued sixteen calls taking five distinct
    probabilities, so distinct-value binning gives five bins with no within-bin spread and the
    identity closes to machine precision. Equal-width binning is used when a forecaster emits
    continuous probabilities, and then the identity carries a small residual that `residual` reports
    rather than hides.
    """
    p, y = _paired(probabilities, outcomes)
    n = p.size
    base = float(np.mean(y))
    uncertainty = base * (1.0 - base)

    assignments, n_bins, exact = _bin(p, bins)
    binning = (
        "distinct forecast value (exact)" if exact else f"{n_bins} equal-width bins (approximate)"
    )

    reliability = 0.0
    resolution = 0.0
    used = 0
    for k in range(n_bins):
        mask = assignments == k
        n_k = int(np.sum(mask))
        if n_k == 0:
            continue
        used += 1
        p_k = float(np.mean(p[mask]))
        o_k = float(np.mean(y[mask]))
        reliability += n_k * (p_k - o_k) ** 2
        resolution += n_k * (o_k - base) ** 2
    reliability /= n
    resolution /= n

    return MurphyDecomposition(
        brier=float(np.mean((p - y) ** 2)),
        reliability=reliability,
        resolution=resolution,
        uncertainty=uncertainty,
        n=int(n),
        n_bins=used,
        base_rate=base,
        binning=binning,
    )


# ---------------------------------------------------------------------------
# Skill, with an interval
# ---------------------------------------------------------------------------


@register_payload
@dataclass(frozen=True)
class SkillScore:
    """Skill against one baseline, with a paired bootstrap interval on the skill itself.

    `skill = 1 - BS_forecast / BS_baseline`. Positive is better than the baseline, zero is the
    baseline, negative is worse.

    The interval is a **paired** percentile bootstrap over forecasts: each resample draws the same
    indices for both the forecaster and the baseline, so the resampling respects the fact that the
    two scored the same events. Resampling them independently would widen the interval by the
    between-event variance the pairing already removes, which at n = 16 is most of it.

    ``covers_zero`` is the sentence a reader wants: an interval that includes zero has not
    established skill, whatever the point estimate says.
    """

    baseline_id: str
    skill: float
    ci_low: float
    ci_high: float
    level: float
    n: int
    forecast_brier: float
    baseline_brier: float
    method: str = "paired percentile bootstrap"

    @property
    def covers_zero(self) -> bool:
        return self.ci_low <= 0.0 <= self.ci_high

    @property
    def beats_baseline(self) -> bool:
        """Point estimate positive **and** the interval clear of zero."""
        return self.skill > 0.0 and not self.covers_zero

    def render(self) -> str:
        verdict = "beats" if self.beats_baseline else "does not beat"
        return (
            f"{self.baseline_id:<40} skill {self.skill:+.4f} "
            f"[{self.ci_low:+.4f}, {self.ci_high:+.4f}] at {self.level:.0%}  "
            f"({self.forecast_brier:.4f} against {self.baseline_brier:.4f}, n={self.n}): "
            f"{verdict} it"
        )

    def __canonical__(self) -> dict[str, Any]:
        return {
            "baseline_id": self.baseline_id,
            "skill": self.skill,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "level": self.level,
            "n": self.n,
            "forecast_brier": self.forecast_brier,
            "baseline_brier": self.baseline_brier,
            "method": self.method,
        }


def skill_score(
    probabilities: Sequence[float],
    baseline_probabilities: Sequence[float],
    outcomes: Sequence[bool],
    *,
    baseline_id: str,
    level: float = 0.9,
    n_boot: int = 4000,
    seed: int = 0,
) -> SkillScore:
    """Skill against one baseline, with a paired percentile bootstrap interval."""
    f_terms = brier_terms(probabilities, outcomes)
    b_terms = brier_terms(baseline_probabilities, outcomes)
    if f_terms.size != b_terms.size:
        raise ForecastError(
            f"{f_terms.size} forecasts against {b_terms.size} baseline forecasts; a skill score "
            f"needs the baseline to have called every event the forecaster called."
        )
    n = int(f_terms.size)
    bs_f = float(np.mean(f_terms))
    bs_b = float(np.mean(b_terms))
    if bs_b <= 0.0:
        raise ForecastError(
            f"baseline {baseline_id!r} has a Brier score of {bs_b}, so it was perfect and the "
            f"skill ratio has a zero denominator. A baseline that never errs on the sample is "
            f"either a leak or a target with no variation; report the two Brier scores directly."
        )
    point = 1.0 - bs_f / bs_b

    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=np.float64)
    kept = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        denom = float(np.mean(b_terms[idx]))
        if denom <= 0.0:
            continue
        draws[kept] = 1.0 - float(np.mean(f_terms[idx])) / denom
        kept += 1
    if kept < max(100, n_boot // 10):
        raise ForecastError(
            f"only {kept} of {n_boot} bootstrap resamples produced a finite skill score, because "
            f"the baseline scored zero on most of them. The interval would be an artifact of the "
            f"resamples that happened to work; report the two Brier scores instead."
        )
    tail = (1.0 - level) / 2.0
    lo, hi = np.quantile(draws[:kept], [tail, 1.0 - tail])
    return SkillScore(
        baseline_id=baseline_id,
        skill=point,
        ci_low=float(lo),
        ci_high=float(hi),
        level=level,
        n=n,
        forecast_brier=bs_f,
        baseline_brier=bs_b,
    )


# ---------------------------------------------------------------------------
# The reliability diagram
# ---------------------------------------------------------------------------


@register_payload
@dataclass(frozen=True)
class ReliabilityDiagram:
    """Observed frequency against forecast probability, per bin, with the counts.

    The counts are not decoration. A reliability diagram plotted without them is the most misleading
    chart in forecasting: a bin holding one forecast plots as a point exactly as far from the
    diagonal as a bin holding four hundred, and the eye reads the two the same way. `render` prints
    the count on every row for that reason.
    """

    bin_probability: tuple[float, ...]
    observed_frequency: tuple[float, ...]
    count: tuple[int, ...]
    base_rate: float
    binning: str
    #: Which row of this table each forecast landed in, in input order. Carried because the nearest
    #: row by `|bin_probability - p|` is **not** the row a forecast is in: under equal-width bins the
    #: rows are labelled by their within-bin mean, and a forecast near a bin edge can sit closer to
    #: its neighbour's mean than to its own. Reconstructing the assignment that way mis-assigned 2
    #: of 13 forecasts on the draw that broke `test_property_brier_is_proper`.
    assignment: tuple[int, ...] = ()

    @property
    def n(self) -> int:
        return int(sum(self.count))

    def render(self) -> str:
        lines = [
            f"reliability ({self.binning}, n={self.n}, base rate {self.base_rate:.3f})",
            f"    {'forecast':>10}  {'observed':>10}  {'n':>5}   deviation",
        ]
        for p, o, c in zip(self.bin_probability, self.observed_frequency, self.count):
            bar = "#" * min(40, c)
            lines.append(f"    {p:>10.3f}  {o:>10.3f}  {c:>5}   {o - p:+.3f}  {bar}")
        return "\n".join(lines)

    def __canonical__(self) -> dict[str, Any]:
        return {
            "bin_probability": list(self.bin_probability),
            "observed_frequency": list(self.observed_frequency),
            "count": list(self.count),
            "base_rate": self.base_rate,
            "binning": self.binning,
        }


def reliability_diagram(
    probabilities: Sequence[float], outcomes: Sequence[bool], *, bins: int | None = None
) -> ReliabilityDiagram:
    """Bin the forecasts and report the observed frequency in each, with the counts.

    Binned by distinct value under the same rule as the Murphy decomposition, so the diagram and
    the decomposition are two views of one table rather than two computations that can disagree.
    """
    p, y = _paired(probabilities, outcomes)
    assignments, n_bins, exact = _bin(p, bins)
    groups: list[tuple[float, np.ndarray]] = []
    row_of: dict[int, int] = {}
    for k in range(n_bins):
        mask = assignments == k
        if not np.any(mask):
            continue
        row_of[k] = len(groups)
        groups.append((float(np.mean(p[mask])), y[mask]))
    binning = "distinct forecast value" if exact else f"{n_bins} equal-width bins"
    return ReliabilityDiagram(
        bin_probability=tuple(g[0] for g in groups),
        observed_frequency=tuple(float(np.mean(g[1])) for g in groups),
        count=tuple(int(g[1].size) for g in groups),
        base_rate=float(np.mean(y)),
        binning=binning,
        assignment=tuple(row_of[int(k)] for k in assignments),
    )


# ---------------------------------------------------------------------------
# Decision value
# ---------------------------------------------------------------------------


@register_payload
@dataclass(frozen=True)
class DecisionValue:
    """Expected loss saved under the forecast's own DecisionSpec. The number that answers "so what".

    The cost-loss value score, from operational meteorology and unchanged here:

        `V = (E_climatology - E_forecast) / (E_climatology - E_perfect)`

    where each `E` is the expected loss per event of a decision rule that acts when the probability
    exceeds `cost / loss`. `V = 1` is a perfect forecast, `V = 0` is no better than always doing
    whatever climatology says is best, and `V < 0` means acting on this forecaster loses money
    relative to ignoring it.

    Two things it exposes that a Brier score does not. A forecaster whose errors all sit far from
    the decision threshold has value even with an unremarkable Brier, because the errors never
    change an action. And value is zero by construction when climatology already dominates: if the
    base rate is below the cost-loss ratio, never acting is optimal and no forecast that never
    crosses the threshold can improve on it, which is a fact about the decision rather than about
    the forecaster and is why `climatology_action` travels with the reading.
    """

    value: float
    expected_loss: float
    climatology_loss: float
    perfect_loss: float
    threshold: float
    hits: int
    misses: int
    false_alarms: int
    correct_rejections: int
    n: int
    unit: str
    action: str
    climatology_action: str

    @property
    def loss_saved(self) -> float:
        """Expected loss saved per event against climatology, in the decision's own unit."""
        return self.climatology_loss - self.expected_loss

    @property
    def loss_saved_total(self) -> float:
        return self.loss_saved * self.n

    def render(self) -> str:
        return (
            f"decision value {self.value:+.4f} on '{self.action}' (act above P={self.threshold:.4g}): "
            f"{self.expected_loss:.3f} {self.unit} per event against climatology's "
            f"{self.climatology_loss:.3f} and a perfect forecast's {self.perfect_loss:.3f}; "
            f"{self.loss_saved:+.3f} {self.unit} saved per event, {self.loss_saved_total:+.2f} over "
            f"n={self.n}. Contingency: {self.hits} hit, {self.misses} miss, "
            f"{self.false_alarms} false alarm, {self.correct_rejections} correct rejection. "
            f"Climatology would {self.climatology_action}."
        )

    def __canonical__(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "expected_loss": self.expected_loss,
            "climatology_loss": self.climatology_loss,
            "perfect_loss": self.perfect_loss,
            "threshold": self.threshold,
            "hits": self.hits,
            "misses": self.misses,
            "false_alarms": self.false_alarms,
            "correct_rejections": self.correct_rejections,
            "n": self.n,
            "unit": self.unit,
            "action": self.action,
        }


def decision_value(
    probabilities: Sequence[float],
    outcomes: Sequence[bool],
    spec: DecisionSpec,
) -> DecisionValue:
    """Expected loss saved under `spec`, as the cost-loss value score."""
    p, y = _paired(probabilities, outcomes)
    n = int(p.size)
    threshold = spec.threshold
    acted = p > threshold
    event = y > 0.5

    hits = int(np.sum(acted & event))
    false_alarms = int(np.sum(acted & ~event))
    misses = int(np.sum(~acted & event))
    correct_rejections = int(np.sum(~acted & ~event))

    base = float(np.mean(y))
    e_forecast = ((hits + false_alarms) * spec.cost + misses * spec.loss) / n
    e_perfect = base * spec.cost
    always_act = spec.cost
    never_act = base * spec.loss
    e_clim = min(always_act, never_act)
    clim_action = "always act" if always_act <= never_act else "never act"

    denominator = e_clim - e_perfect
    if denominator <= 0.0:
        # Climatology is already optimal: the base rate sits exactly at the cost-loss ratio, or
        # there is no variation to exploit. Value is zero and that is a fact about the decision.
        value = 0.0
    else:
        value = (e_clim - e_forecast) / denominator

    return DecisionValue(
        value=float(value),
        expected_loss=float(e_forecast),
        climatology_loss=float(e_clim),
        perfect_loss=float(e_perfect),
        threshold=float(threshold),
        hits=hits,
        misses=misses,
        false_alarms=false_alarms,
        correct_rejections=correct_rejections,
        n=n,
        unit=spec.unit,
        action=spec.action,
        climatology_action=clim_action,
    )


# ---------------------------------------------------------------------------
# Interval coverage
# ---------------------------------------------------------------------------


@register_payload
@dataclass(frozen=True)
class CoverageScore:
    """Realised coverage of a set of interval forecasts against their nominal level.

    Reported with a Wilson interval, because coverage over four intervals is a binomial proportion
    with n = 4 and the naive standard error is meaningless there. The campaign's own note applies:
    at this count coverage is a companion diagnostic, not a kill metric, and the interval is how
    that is said in numbers rather than in a caveat.
    """

    covered: int
    n: int
    nominal: float
    ci_low: float
    ci_high: float

    @property
    def coverage(self) -> float:
        return self.covered / self.n if self.n else float("nan")

    @property
    def covers_nominal(self) -> bool:
        return self.ci_low <= self.nominal <= self.ci_high

    def render(self) -> str:
        return (
            f"interval coverage {self.coverage:.4f} ({self.covered}/{self.n}) against a nominal "
            f"{self.nominal:.2f}; Wilson 95% [{self.ci_low:.3f}, {self.ci_high:.3f}], which "
            f"{'includes' if self.covers_nominal else 'excludes'} the nominal rate"
        )

    def __canonical__(self) -> dict[str, Any]:
        return {
            "covered": self.covered,
            "n": self.n,
            "nominal": self.nominal,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
        }


def coverage_score(hits: Sequence[bool], *, nominal: float, level: float = 0.95) -> CoverageScore:
    """Realised coverage with a Wilson score interval."""
    values = [bool(h) for h in hits]
    n = len(values)
    if n == 0:
        raise ForecastError(
            "no interval forecasts to score. Report the count of voids; an empty coverage is not "
            "a coverage of zero."
        )
    k = sum(values)
    z = {0.9: 1.6449, 0.95: 1.9600, 0.99: 2.5758}.get(level, 1.9600)
    phat = k / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return CoverageScore(
        covered=k,
        n=n,
        nominal=nominal,
        ci_low=max(0.0, centre - half),
        ci_high=min(1.0, centre + half),
    )


# ---------------------------------------------------------------------------
# Lead time, and the width this package does not fit
# ---------------------------------------------------------------------------


def transition_fit(
    outcome: Sequence[float],
    steps: Sequence[float] | None = None,
    *,
    series: str = "outcome",
    seed: int = 0,
) -> Any:
    """Instrument H4's transition fit, called through. This package does not fit widths.

    Returns H4's `TransitionFit`, or the `Refusal` it produces when the series carries no
    identifiable transition. Both come back unchanged: translating a refusal into something else
    here would lose the remedy, and the remedy is the product.

    This is a delegation and not a wrapper with opinions. `run.transition_width` has one owner and
    a second fit would be a second answer to one question, which is how two numbers for one
    quantity end up in one report. An earlier draft of this package carried its own logistic
    fallback, written before H4 landed; on a planted logistic of width 21.9722 steps the fallback
    returned 23.11 clean and 62.80 at noise sd 0.03, and H4 returned 21.9722 and 21.955. It was
    deleted rather than kept behind a flag.
    """
    from reward_lens.measure.rate.transition import fit_transition

    return fit_transition(outcome, steps, series=series, seed=seed)


def forecast_lead_time(
    alarm_step: float,
    outcome: Sequence[float],
    steps: Sequence[float] | None = None,
    *,
    series: str = "outcome",
    seed: int = 0,
) -> Any:
    """How far ahead a call came, in fractions of the fitted transition width. One call, one fit.

    Returns H4's `LeadTime`, which carries the fraction of a width, the same lead measured from the
    start of the rise, the absolute step count labelled as not comparable across runs, and the
    sampling resolution of the series so a lead finer than one logging interval is visibly rounded
    rather than quietly reported.

    Returns a `Refusal` when the series has no fitted transition to divide by. That is the honest
    answer for a run with no transition in it, and it is the case the toy GRPO fixtures in this
    repository are in: a 0.6M-parameter model optimising against a length grader for 200 steps is a
    real optimisation trace and not a behavioural transition, so a lead time on it exists only in
    absolute steps and this refuses rather than manufacturing a width.
    """
    from reward_lens.measure.rate.transition import lead_time as _lead_time

    fit = transition_fit(outcome, steps, series=series, seed=seed)
    if not hasattr(fit, "usable"):
        return fit
    return _lead_time(alarm_step, fit)


__all__ = [
    "CoverageScore",
    "DecisionValue",
    "MurphyDecomposition",
    "ReliabilityDiagram",
    "SkillScore",
    "brier",
    "brier_terms",
    "coverage_score",
    "decision_value",
    "log_score",
    "forecast_lead_time",
    "transition_fit",
    "murphy_decomposition",
    "recalibrate",
    "reliability_diagram",
    "skill_score",
]

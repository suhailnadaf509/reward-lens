"""E-values and confidence sequences: peek freely (J1).

Ville (1939): for a nonnegative martingale started at 1, ``P(exists t : M_t >= a) <= 1/a``. So
``p_t = 1 / max_{s<=t} M_s`` is an anytime-valid p-value and inverting a family of such tests gives
a time-uniform interval. Nothing is corrected for the number of looks afterwards, because there was
never a fixed number of looks to correct for. That is the whole licence, and it is the difference
between a monitor you may read at every step and a monitor you may read once.

**Three rungs, and each one is a real cost.**

- Rung 0, alpha-spending Hoeffding. Split the budget as ``delta_n = 6 delta / (pi^2 n^2)``, which
  sums to ``delta`` because ``sum 1/n^2 = pi^2/6``, and apply Hoeffding at each ``n`` with its own
  slice. Simple, needs no state, and pays ``2 log n`` inside the square root forever.
- Rung 1, polynomial stitching (Howard, Ramdas, McAuliffe and Sekhon 2021). Cover geometrically
  spaced epochs and union-bound over epochs instead of over steps, which pays ``log log n``. 1.41
  times narrower than rung 0 at 100 observations and 1.70 times at 10,000, measured here.
- Rung 2, betting with the ONS schedule. Bet on the mean, adapt the stake to what the data have
  shown, and invert the capital process. Narrowest of the three on a stream whose variance is well
  under the Hoeffding worst case, which a hack rate near 0.1 is, and it costs a search over the
  candidate mean at each look.

**The mandatory baseline, and it is the whole argument for the package.** A fixed-sample interval is
valid at one sample size declared in advance. `peeking_cost` runs the experiment a practitioner
actually runs, recomputing the interval at every step and reading it at every step, and counts how
often it excluded the truth. On 4,000 Bernoulli(0.1) streams of 200 steps at a nominal 5%, the
fixed-sample Wilson interval is wrong on 37.5% of runs, 7.5 times the level it advertises, and the
rung-0 anytime-valid interval on 0.0%. The price is width: 0.042 against 0.192 at step 200. Both
numbers are produced by code in this module and neither is quoted from anywhere.

**Rungs 0 and 2 are the vendored `cif.py`, unmodified.** `certified-interventional-fidelity` is MIT
and pure Python, and it ships exactly the alpha-spending radius and the ONS betting sequence this
rung ladder asks for. Rung 1 is not in it, so rung 1 is `stats.sequential.stitched_radius`. See
`monitor/_vendor/cif.py` for the provenance and the licence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from reward_lens.core.envelope import EnvelopeSpec
from reward_lens.core.evidence import Uncertainty
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.types import Capability, GaugeStatus
from reward_lens.measure.base import Context
from reward_lens.monitor._base import MONITOR_ENVELOPE, RECORD_ACCESS, MonitorInstrument
from reward_lens.monitor._vendor import cif
from reward_lens.monitor.operating_point import PPVCurve, ppv_curve
from reward_lens.stats.sequential import (
    PeekingCost,
    fixed_sample_radius,
    peeking_miscoverage,
    running_evalue,
    stitched_radius,
    ville_pvalue,
    ville_threshold,
)

Rung = Literal[0, 1, 2]

#: What each rung is, for the reading and for the capability report.
RUNG_NAMES: dict[int, str] = {
    0: "Hoeffding by alpha-spending 6d/(pi^2 n^2)",
    1: "polynomial stitching (Howard et al. 2021)",
    2: "betting with the ONS schedule",
}

#: Rungs 0 and 2 come from the vendored file rather than being written twice.
RUNG_SOURCE: dict[int, str] = {
    0: "monitor._vendor.cif.cs_radius",
    1: "stats.sequential.stitched_radius",
    2: "monitor._vendor.cif.BettingCSTracker",
}


# ---------------------------------------------------------------------------
# Confidence sequences
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfidenceTrace:
    """A time-uniform interval, at every step, with the fixed-sample one beside it.

    ``lower`` and ``upper`` hold at all ``n`` simultaneously with probability at least
    ``1 - delta``. ``fixed_lower`` and ``fixed_upper`` hold at any single ``n`` chosen in advance,
    and at no other, which is what `peeking` measures the cost of.
    """

    n: np.ndarray
    centre: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    fixed_lower: np.ndarray
    fixed_upper: np.ndarray
    delta: float
    rung: int
    lo_bound: float = 0.0
    hi_bound: float = 1.0

    @property
    def final(self) -> tuple[float, float]:
        return float(self.lower[-1]), float(self.upper[-1])

    @property
    def final_fixed(self) -> tuple[float, float]:
        return float(self.fixed_lower[-1]), float(self.fixed_upper[-1])

    @property
    def width_ratio(self) -> float:
        """How many times wider the anytime-valid interval is at the end. The price of peeking."""
        a = float(self.upper[-1] - self.lower[-1])
        f = float(self.fixed_upper[-1] - self.fixed_lower[-1])
        return a / f if f > 0 else float("nan")

    def excludes(self, value: float) -> bool:
        """Whether the interval has ever excluded ``value``, at any step so far."""
        return bool(np.any((self.lower > value) | (self.upper < value)))

    def first_exclusion(self, value: float) -> int | None:
        """The first step at which the interval excluded ``value``, or None."""
        bad = np.where((self.lower > value) | (self.upper < value))[0]
        return int(self.n[bad[0]]) if bad.size else None

    def render(self) -> str:
        lo, hi = self.final
        flo, fhi = self.final_fixed
        return (
            f"anytime-valid {1 - self.delta:.0%} interval at n={int(self.n[-1])}: "
            f"[{lo:.4f}, {hi:.4f}]  (rung {self.rung}, {RUNG_NAMES[self.rung]})\n"
            f"    fixed-sample interval at the same n: [{flo:.4f}, {fhi:.4f}], "
            f"{self.width_ratio:.2f}x narrower and valid at that n only"
        )


def _radius_fn(rung: Rung, delta: float, sigma: float):
    if rung == 0:
        return lambda n: cif.cs_radius(n, delta)
    if rung == 1:
        return lambda n: stitched_radius(n, delta, sigma=sigma)
    raise ValueError(f"rung {rung} has no closed-form radius; rung 2 inverts a capital process")


def confidence_sequence(
    observations: Sequence[float],
    *,
    delta: float = 0.05,
    rung: Rung = 0,
    sigma: float = 0.5,
    every: int = 1,
) -> ConfidenceTrace:
    """The time-uniform interval on the mean of a [0, 1]-bounded stream, at every step.

    ``every`` subsamples the reported trace for rung 2 only, where each point costs a fresh binary
    search over the candidate mean and the full trace is quadratic. It does not change the interval
    at the points it reports; a confidence sequence is valid at every stopping time whether or not
    anybody looked, so looking at every tenth step is a subset of the guarantee rather than a
    weakening of it.
    """
    x = np.asarray(observations, dtype=np.float64).ravel()
    if x.size == 0:
        raise ValueError("a confidence sequence needs at least one observation")
    if np.any((x < -1e-12) | (x > 1 + 1e-12)):
        raise ValueError(
            "observations must lie in [0, 1]. Every bound here is a boundedness bound and there is "
            "no honest way to apply it to an unbounded stream; rescale by a known range and say so, "
            "or use a sub-Gaussian sequence with a stated variance proxy."
        )
    x = np.clip(x, 0.0, 1.0)
    means = np.cumsum(x) / np.arange(1, x.size + 1, dtype=np.float64)
    idx = np.arange(1, x.size + 1)
    keep = idx if (rung != 2 or every <= 1) else idx[(idx % every == 0) | (idx == x.size)]

    if rung in (0, 1):
        radius = _radius_fn(rung, delta, sigma)
        lo = np.array([means[n - 1] - radius(int(n)) for n in keep])
        hi = np.array([means[n - 1] + radius(int(n)) for n in keep])
    else:
        tracker = cif.BettingCSTracker(delta=delta)
        lo_list: list[float] = []
        hi_list: list[float] = []
        wanted = set(int(n) for n in keep)
        for i, xi in enumerate(x, 1):
            tracker.update(float(xi))
            if i in wanted:
                a, b = tracker.cs_interval
                lo_list.append(a)
                hi_list.append(b)
        lo = np.array(lo_list)
        hi = np.array(hi_list)

    fixed = np.array([fixed_sample_radius(int(n), delta) for n in keep])
    centre = means[keep - 1]
    return ConfidenceTrace(
        n=keep,
        centre=centre,
        lower=np.clip(lo, 0.0, 1.0),
        upper=np.clip(hi, 0.0, 1.0),
        fixed_lower=np.clip(centre - fixed, 0.0, 1.0),
        fixed_upper=np.clip(centre + fixed, 0.0, 1.0),
        delta=float(delta),
        rung=int(rung),
    )


def peeking_cost(
    *,
    delta: float = 0.05,
    rung: Rung = 0,
    truth: float = 0.1,
    n_steps: int = 200,
    n_runs: int = 2000,
    seed: int = 0,
    sigma: float = 0.5,
) -> PeekingCost:
    """How often the fixed-sample interval would have been wrong, under exactly these looks.

    The mandatory J1 baseline, run rather than asserted. Rung 2 is not offered here: the betting
    interval has no closed-form radius, so simulating 2,000 streams of it costs minutes rather than
    milliseconds, and the point of the comparison is made by either analytic rung.
    """
    if rung == 2:
        raise ValueError(
            "rung 2 has no closed-form radius, so the peeking simulation would re-invert a capital "
            "process 400,000 times. Run the comparison at rung 0 or 1; the fixed-sample side is "
            "identical and it is the side the comparison is about."
        )
    return peeking_miscoverage(
        _radius_fn(rung, delta, sigma),
        truth=truth,
        delta=delta,
        n_steps=n_steps,
        n_runs=n_runs,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# E-processes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EProcessTrace:
    """A test martingale, its running maximum, and the anytime-valid p-value that follows.

    ``alarm_at`` is the first step where the capital crossed ``1/alpha``. There is no multiplicity
    correction attached to it and none is needed: Ville bounds the probability that it ever crosses,
    over an unbounded horizon, at ``alpha``.
    """

    capital: np.ndarray
    running_max: np.ndarray
    mean_null: float
    alpha: float
    alarm_at: int | None
    n: int

    @property
    def evalue(self) -> float:
        """The final e-value. Evidence against the null, in units of a fair bet."""
        return float(self.capital[-1]) if self.capital.size else 1.0

    @property
    def pvalue(self) -> float:
        """``1 / sup_s M_s``, valid at every stopping time."""
        return ville_pvalue(float(self.running_max[-1]) if self.running_max.size else 1.0)

    @property
    def threshold(self) -> float:
        return ville_threshold(self.alpha)

    def render(self) -> str:
        fired = (
            f"crossed 1/alpha = {self.threshold:.4g} at step {self.alarm_at}"
            if self.alarm_at is not None
            else f"never crossed 1/alpha = {self.threshold:.4g}"
        )
        return (
            f"e-process against mean <= {self.mean_null:.4g} over {self.n} observations: "
            f"e = {self.evalue:.4g}, anytime-valid p = {self.pvalue:.4g}; {fired}"
        )


def eprocess(
    observations: Sequence[float],
    mean_null: float,
    *,
    alpha: float = 0.05,
    adaptive: bool = True,
    lam: float = 0.5,
) -> EProcessTrace:
    """The test martingale for "this bounded stream's mean is at most ``mean_null``".

    One-sided, because a monitor watches for a rise and a two-sided version would halve the power
    for a direction nobody is watching. The bet is the ONS schedule by default, which adapts to an
    effect whose size was not known in advance; ``adaptive=False`` fixes the stake at ``lam`` and is
    what the hand-computed unit test checks against.
    """
    x = np.asarray(observations, dtype=np.float64).ravel()
    if np.any((x < -1e-12) | (x > 1 + 1e-12)):
        raise ValueError("observations must lie in [0, 1] for the betting construction to be valid")
    capital = running_evalue(np.clip(x, 0.0, 1.0), float(mean_null), lam=lam, adaptive=adaptive)
    running_max = np.maximum.accumulate(capital) if capital.size else capital
    thresh = ville_threshold(alpha)
    crossed = np.where(capital >= thresh)[0]
    return EProcessTrace(
        capital=capital,
        running_max=running_max,
        mean_null=float(mean_null),
        alpha=float(alpha),
        alarm_at=int(crossed[0]) if crossed.size else None,
        n=int(x.size),
    )


# ---------------------------------------------------------------------------
# A record adapter that produces a genuinely bounded stream
# ---------------------------------------------------------------------------


def abstention_stream(run, *, window: tuple[int, int] | None = None) -> np.ndarray:
    """Per-trajectory grader abstention, in run order. A real Bernoulli stream off a real record.

    A `ScoreTree` leaf carries `abstained`, so "did the grader decline to answer on this rollout" is
    a per-rollout indicator that a record always has when the tap was fitted. It is the honest
    bounded quantity a GRPO record actually carries: an entropy is not in [0, 1], a reward is not in
    [0, 1], and rescaling either by its observed range would put the data inside a bound the data
    chose.

    Trajectories whose score tree carries no `abstained` field at all are skipped rather than
    counted as non-abstentions, because a record that does not distinguish an abstention from a zero
    is exactly the record this indicator would be measuring wrongly.
    """
    steps = list(run.steps) if window is None else list(run.steps.slice(*window))
    out: list[float] = []
    for step in steps:
        for traj in step.trajectories:
            tree = traj.scores
            flag = getattr(tree, "abstained", None)
            if flag is None:
                continue
            out.append(1.0 if flag else 0.0)
    return np.asarray(out, dtype=np.float64)


# ---------------------------------------------------------------------------
# The two instruments
# ---------------------------------------------------------------------------

J1_BASELINES: tuple[str, ...] = (
    "baseline.fixed_sample_interval",
    "baseline.fixed_sample_under_peeking",
)

#: J1 reads a stream of bounded observations in the order they arrived, and the two things that can
#: silently break that are a grader that moved underneath it and a prefix rewrite that reordered it.
#: Both are in `MONITOR_ENVELOPE`.
J1_ENVELOPE: EnvelopeSpec = MONITOR_ENVELOPE


class _J1Base(MonitorInstrument):
    """What the two J1 instruments share: the stream, the rung, and the refusals."""

    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    requires = RECORD_ACCESS
    envelope = J1_ENVELOPE
    invariance = "units"
    invariance_relation = INVARIANT
    baselines = J1_BASELINES

    #: Below this many observations, the alpha-spending radius exceeds the whole [0, 1] support and
    #: the interval is the trivial one. Returning [0, 1] and calling it a measurement is the shape
    #: of the error this library exists to prevent, so it is a refusal with a bound attached.
    MIN_OBSERVATIONS: int = 8

    def __init__(
        self,
        observations: Sequence[float],
        *,
        delta: float = 0.05,
        rung: Rung | None = None,
        mean_null: float | None = None,
        source: str = "supplied",
        every: int = 1,
    ) -> None:
        self.observations = np.asarray(observations, dtype=np.float64).ravel()
        self.delta = float(delta)
        # `rung` is a protocol field the capability report reads off the class and it also selects
        # the estimator. `None` means "the one this instrument declares", so the two cannot disagree
        # by an argument default.
        self.rung = int(type(self).rung if rung is None else rung)
        self.mean_null = mean_null
        self.source = source
        self.every = int(every)

    def _guard(self) -> Refusal | None:
        x = self.observations
        if x.size == 0:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.VOID,
                detail="the stream is empty, so there is no sequence to be uniform over.",
                remedy=(
                    "Supply at least one observation. If the record carries no per-rollout "
                    "indicator at all, that is a `RECORD_INCOMPLETE` upstream and the fix is to log "
                    "the field, not to widen the window."
                ),
            )
        if np.any((x < -1e-12) | (x > 1 + 1e-12)):
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.UNIT_MISMATCH,
                detail=(
                    f"the stream ranges over [{x.min():.4g}, {x.max():.4g}] and every bound in this "
                    f"instrument is a boundedness bound on [0, 1]."
                ),
                remedy=(
                    "Rescale by a range you can state a priori and record the range on the reading, "
                    "or ask for the same quantity on a genuinely bounded channel: an abstention "
                    "indicator, a pass rate, a per-rollout hack flag. Rescaling by the observed "
                    "range is not a fix, because the bound would then be chosen by the data."
                ),
                statistics={"min": float(x.min()), "max": float(x.max()), "n": int(x.size)},
            )
        if x.size < self.MIN_OBSERVATIONS:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ABOVE_LOD_BELOW_LOQ,
                detail=(
                    f"{x.size} observations. The rung-{self.rung} radius at n={x.size} is "
                    f"{cif.cs_radius(x.size, self.delta):.3f}, which covers the whole [0, 1] "
                    f"support, so the interval is the trivial one and carries no information."
                ),
                remedy=(
                    f"Collect at least {self.MIN_OBSERVATIONS} observations, or read the point "
                    f"estimate {float(np.mean(x)):.4g} with no interval and treat it as such. The "
                    f"sequence stays valid at every n; it is simply uninformative at this one."
                ),
                statistics={"n": int(x.size), "point_estimate": float(np.mean(x))},
            )
        return None


class ConfidenceSequence(_J1Base):
    """J1. The anytime-valid interval on a bounded rate, readable at every step.

    "The interval on the abstention rate is [0.06, 0.21] at rollout 96 and it has never been
    invalid, despite our having looked at every one." The last clause is the product.

    What it cannot do: it says nothing about an unbounded channel. Every bound here is Hoeffding's
    or a betting bound on [0, 1], and an entropy or a KL has no such support, so those channels get
    a CUSUM (J2) and not an interval. Rescaling an unbounded series by its own observed range would
    put the bound inside the data, and the instrument refuses instead.
    """

    name = "ConfidenceSequence"
    version = "1.0"
    quantity = "monitor.confidence_sequence"
    faithful_to = (
        "Ville (1939); Howard et al. (2021) at rung 1; Waudby-Smith and Ramdas betting at rung 2"
    )
    deviations = (
        "rung 1 uses the sub-Gaussian stitched boundary with the Hoeffding variance proxy 1/4 "
        "rather than an empirical-Bernstein variance process, so it is loose on a rate near zero, "
        "which is exactly where a hack rate sits. Rung 2 recovers that and costs a search.",
    )
    rung = 0

    def compute(self, ctx: Context) -> ConfidenceTrace | Refusal:
        bad = self._guard()
        if bad is not None:
            return bad
        return confidence_sequence(
            self.observations, delta=self.delta, rung=self.rung, every=self.every
        )

    def payload(self, computed: ConfidenceTrace) -> dict:
        lo, hi = computed.final
        flo, fhi = computed.final_fixed
        return {
            "interval": [lo, hi],
            "fixed_sample_interval": [flo, fhi],
            "width_ratio": computed.width_ratio,
            "point_estimate": float(computed.centre[-1]),
            "n": int(computed.n[-1]),
            "delta": computed.delta,
            "rung": computed.rung,
            "rung_name": RUNG_NAMES[computed.rung],
            "rung_source": RUNG_SOURCE[computed.rung],
            "source": self.source,
            "lower": [float(v) for v in computed.lower],
            "upper": [float(v) for v in computed.upper],
            "steps": [int(v) for v in computed.n],
            "baselines": self.baseline_map(computed),
            "rendered": computed.render(),
        }

    def baseline_map(self, computed: ConfidenceTrace) -> dict[str, float]:
        flo, fhi = computed.final_fixed
        return {"baseline.fixed_sample_interval": float(fhi - flo)}

    def uncertainty(self, computed: ConfidenceTrace) -> Uncertainty:
        lo, hi = computed.final
        return Uncertainty(
            ci_low=lo,
            ci_high=hi,
            ci_level=1.0 - computed.delta,
            n=int(computed.n[-1]),
            method=f"anytime-valid, {RUNG_NAMES[computed.rung]}",
        )


class EValue(_J1Base):
    """J1. Evidence against a stated null rate, as an e-value, readable at every step.

    An e-value is interpretable as the payoff of a bet that would be fair under the null, so 20
    means "a bet at 20:1 against the null would have broken even", and it is the number to report
    when the question is how much evidence there is rather than where the interval sits. Its
    reciprocal running maximum is a p-value valid at every stopping time.

    ``mean_null`` has to be stated. There is no default null rate for a hack rate and inventing one
    would make the e-value a function of this instrument's opinion rather than of the data.
    """

    name = "EValue"
    version = "1.0"
    quantity = "monitor.evalue"
    faithful_to = "Ville (1939); Shafer and Vovk's betting interpretation"
    deviations = (
        "the stake follows an ONS schedule with the clip at 0.5, which is `cif.py`'s choice and not "
        "a tuned one. A different schedule changes the power and never the validity.",
    )
    rung = 2

    def compute(self, ctx: Context) -> EProcessTrace | Refusal:
        bad = self._guard()
        if bad is not None:
            return bad
        if self.mean_null is None:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail="no null rate was stated, and an e-value is evidence against a stated null.",
                remedy=(
                    "Pass `mean_null=` with the rate you are testing against: the rate from a "
                    "pre-change window, the rate a previous run settled at, or a rate you are "
                    "willing to accept. There is no default and a default would be this "
                    "instrument's opinion wearing your number."
                ),
            )
        return eprocess(self.observations, self.mean_null, alpha=self.delta)

    def payload(self, computed: EProcessTrace) -> dict:
        curve: PPVCurve | None = None
        if computed.alarm_at is not None:
            # An alarm at level alpha has a false-alarm probability of at most alpha over the whole
            # horizon, which is the fpr the PPV curve needs. Sensitivity is not measurable without
            # labels, so the curve is drawn at the best case and says so: even at perfect
            # sensitivity, this is what an alarm is worth.
            curve = ppv_curve(1.0, computed.alpha)
        return {
            "evalue": computed.evalue,
            "pvalue_anytime_valid": computed.pvalue,
            "mean_null": computed.mean_null,
            "alpha": computed.alpha,
            "threshold": computed.threshold,
            "alarm_at": computed.alarm_at,
            "n": computed.n,
            "source": self.source,
            "capital": [float(v) for v in computed.capital],
            "ppv_curve": None
            if curve is None
            else {
                "sensitivity_assumed": curve.sensitivity,
                "fpr": curve.fpr,
                "prevalences": list(curve.prevalences),
                "ppv": list(curve.ppv),
                "note": (
                    "drawn at perfect sensitivity, which is the best case. The realised PPV cannot "
                    "exceed these numbers and needs labels to pin down."
                ),
            },
            "baselines": self.baseline_map(computed),
            "rendered": computed.render(),
        }

    def baseline_map(self, computed: EProcessTrace) -> dict[str, float]:
        # The fixed-sample comparator for an e-value is the one-look test at the same level, whose
        # rejection threshold is a p-value of alpha. Reported as the level itself so the two
        # numbers on the reading are commensurable.
        return {"baseline.fixed_sample_interval": float(computed.alpha)}

    def uncertainty(self, computed: EProcessTrace) -> Uncertainty:
        return Uncertainty(n=computed.n, method="anytime-valid, Ville")


__all__ = [
    "J1_BASELINES",
    "J1_ENVELOPE",
    "RUNG_NAMES",
    "RUNG_SOURCE",
    "ConfidenceSequence",
    "ConfidenceTrace",
    "EProcessTrace",
    "EValue",
    "Rung",
    "abstention_stream",
    "confidence_sequence",
    "eprocess",
    "peeking_cost",
]

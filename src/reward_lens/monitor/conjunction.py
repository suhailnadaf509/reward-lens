"""The conjunction detector, where the conjunction is the detector (J3).

Measured, in the ALFWorld collapse study: entropy alone false-alarms in three distinct ways, while
the conjunction of entropy monotone decline, prediction saturation and episode-length pinning
precedes collapse by 15 to 30 steps. The architectural consequence is the whole point of this
module. A monitoring layer is not a bank of independent thresholds with a shared dashboard. It
**composes predicates over multiple channels with a joint false-alarm rate**, and the composition
has to be legal.

**How the composition is made legal.** E-values do not multiply legally under arbitrary dependence.
Vovk and Wang (2021) show that the merging function valid under arbitrary dependence is the
*arithmetic mean*, and that it is essentially the only admissible symmetric one; the product is
valid across *independent* e-values, and over *time within one channel*, where each factor is
conditionally an e-value given the past. Entropy, reward spread and completion length on the same
training run are about as dependent as three series can be, so multiplying their e-values would
inflate the evidence by a factor that is not bounded in general.

What survives that correction is better than what it replaces, because a conjunction does not need a
merging rule at all. The conjunction fires when every channel has crossed, so under the global null

    P(all K channels fire) <= min_k P(channel k fires) <= min_k alpha_k

with **no dependence assumption whatever**. That bound is exact, it is free, and it is what
`ConjunctionAlarm.joint_far_bound` reports. The independence-optimistic product is reported beside
it, labelled, because the gap between the two is what a reader is entitled to see rather than to be
told.

**e-BH is still needed, for the ledger rather than for the alarm.** The question "which channels
moved" is a multiple-testing problem over a per-feature ledger, and e-BH (Wang and Ramdas 2022)
controls the false discovery rate there under arbitrary dependence, which plain BH on p-values does
not. `reward_lens.stats.multiplicity` already holds the p-value procedures and this does not write a
second BH; `stats.sequential.ebh` is the e-value one, which is a different procedure and not a
duplicate.

**The kill condition is live, and on most designs it fires.** If a single channel matches the
conjunction's operating characteristic, the conjunction is decoration. `measure_conjunction` settles
it by measurement rather than by assertion, comparing at a **matched** false-alarm rate rather than
at a common threshold, because at a common threshold the conjunction wins by construction: its
statistic is ``min_k e_k``, which Ville bounds at the same level as any single ``e_k``.

Measured, at a 5% false-alarm rate over 100 pre-change steps, a 0.6-sigma shift on all three
channels, noise equicorrelation 0.5, 1,500 runs:

| in-control scale | no artefacts | artefacts on 60% of runs |
|---|---|---|
| estimated from the monitored window | 22.4% against 41.8% | 8.3% against 29.5% |
| robust (MAD) on the monitored window | 19.3% against 44.3% | 13.5% against 32.1% |
| known, from a clean reference | 22.3% against 38.3% | **23.8% against 20.1%** |

conjunction first, best single channel second. **One cell out of six goes the conjunction's way**,
and reading the table across is the whole result. Against Gaussian noise the conjunction can only
lose: every channel is already well controlled and requiring all of them to agree is strictly
stricter. It starts to win only when the false alarms are *channel-specific structural artefacts*,
which is what "entropy alone false-alarms in three distinct ways" actually describes, **and** when
the in-control scale comes from somewhere other than the window being monitored. Estimating the
scale from a contaminated window is what destroys the advantage: the artefact inflates that
channel's own denominator, which shrinks its post-change signal, and the conjunction takes the
minimum so it inherits the worst of them.

The engineering consequence is concrete and it ties J3 to J5. **A conjunction detector earns its
place only with a clean reference scale**, so `ConjunctionDetector` takes a per-channel
``reference`` of in-control ``(mean, sd)``, and a frozen probe set measured on a quiet run is where
that reference comes from. Without one, the honest recommendation on the evidence here is to watch
the best single channel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from reward_lens.core.evidence import Uncertainty
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.types import Capability, GaugeStatus
from reward_lens.measure.base import Context
from reward_lens.monitor._base import (
    MONITOR_ENVELOPE,
    RECORD_ACCESS,
    Channel,
    ChannelSet,
    MonitorInstrument,
)
from reward_lens.monitor.cusum import standardize
from reward_lens.monitor.operating_point import ppv_curve
from reward_lens.stats.sequential import ebh, merge_e, ville_threshold

# ---------------------------------------------------------------------------
# One channel's e-process
# ---------------------------------------------------------------------------


def gaussian_lr_eprocess(z_series: Sequence[float], shift: float = 1.0) -> np.ndarray:
    """The likelihood-ratio test martingale for a mean shift on a standardized channel.

    ``M_t = exp(shift * S_t - t shift^2 / 2)`` where ``S_t`` is the running sum. Under the null that
    the standardized series is standard normal, ``M`` is a nonnegative martingale with ``E[M_0] = 1``
    exactly, so Ville applies and ``M_t >= 1/alpha`` is an anytime-valid rejection at level
    ``alpha``. It is the same object the CUSUM is built from, without the reset, and the difference
    between the two is the difference between a lifetime false-alarm probability and a false-alarm
    rate per step.

    Returned in the natural scale rather than the log, because an e-value is read as a betting
    payoff and a reader who wants 40 should not have to exponentiate 3.7. It overflows above about
    ``exp(709)``, which is evidence so overwhelming that the distinction between it and infinity has
    no operational content.
    """
    z = np.nan_to_num(np.asarray(z_series, dtype=np.float64).ravel(), nan=0.0)
    t = np.arange(1, z.size + 1, dtype=np.float64)
    log_m = shift * np.cumsum(z) - 0.5 * t * shift * shift
    return np.exp(np.clip(log_m, -700.0, 700.0))


@dataclass(frozen=True)
class ChannelAlarm:
    """One channel's e-process, its own alarm, and its own false-alarm budget."""

    name: str
    evalue: np.ndarray
    alpha: float
    alarm_at: int | None
    direction: int

    @property
    def threshold(self) -> float:
        return ville_threshold(self.alpha)

    @property
    def final_evalue(self) -> float:
        return float(self.evalue[-1]) if self.evalue.size else 1.0

    @property
    def fired(self) -> bool:
        return self.alarm_at is not None

    def above(self) -> np.ndarray:
        """The indicator of "currently above threshold", which is what a conjunction ands."""
        return self.evalue >= self.threshold

    def render(self) -> str:
        where = f"step index {self.alarm_at}" if self.fired else "never"
        return (
            f"{self.name:<22} e = {self.final_evalue:>12.4g}  alarm {where:<16} "
            f"(alpha {self.alpha:.3g}, threshold {self.threshold:.4g})"
        )


def channel_alarm(
    channel: Channel,
    *,
    alpha: float = 0.01,
    shift: float = 1.0,
    baseline_steps: int | None = None,
    reference: tuple[float, float] | None = None,
) -> ChannelAlarm:
    """Build one channel's e-process on its oriented, standardized series.

    ``channel.oriented`` signs the series so that larger always means worse, using the direction the
    channel *declares* rather than one fitted here. Fitting the direction on the same series the
    alarm is read from is how a detector learns to fire on whichever way the data happened to go.

    ``reference`` is an in-control ``(mean, sd)`` measured somewhere else, on a run known to be
    quiet. Supplying one is not a refinement: `measure_conjunction` shows it is the difference
    between a conjunction that earns its place and one that does not. See that function.
    """
    if reference is not None:
        mu, sd = reference
        if sd <= 0:
            raise ValueError(f"a reference standard deviation must be positive; got {sd}")
        z = (np.asarray(channel.oriented, dtype=np.float64) - mu) / sd
    else:
        z = standardize(channel.oriented, baseline_steps)
    e = gaussian_lr_eprocess(z, shift)
    hit = np.where(e >= ville_threshold(alpha))[0]
    return ChannelAlarm(
        name=channel.name,
        evalue=e,
        alpha=float(alpha),
        alarm_at=int(hit[0]) if hit.size else None,
        direction=channel.direction,
    )


# ---------------------------------------------------------------------------
# The conjunction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConjunctionAlarm:
    """The conjunction over channels, its joint false-alarm bound, and each channel alone.

    ``fired_at`` is the first index at which **every** channel's e-process is above its own
    threshold at the same time. ``all_crossed_at`` is the weaker "each has crossed by now" version,
    reported because the two answer different questions: the first is a state of the system and the
    second is a history of it.
    """

    channels: tuple[ChannelAlarm, ...]
    fired_at: int | None
    all_crossed_at: int | None
    n_steps: int
    ledger: Mapping[str, float]
    ebh_rejected: tuple[str, ...]
    ebh_threshold: float
    fdr_alpha: float

    @property
    def k(self) -> int:
        return len(self.channels)

    @property
    def joint_far_bound(self) -> float:
        """``min_k alpha_k``. Valid under arbitrary dependence, with no assumption at all.

        A conjunction is an intersection, so its probability cannot exceed the smallest of its
        parts'. That is the guarantee, and it is the number to quote.
        """
        return min((c.alpha for c in self.channels), default=1.0)

    @property
    def joint_far_if_independent(self) -> float:
        """``prod_k alpha_k``. What the rate would be if the channels were independent.

        They are not. Entropy, reward spread and completion length on one training run move
        together, and this number is reported only so the distance between the guaranteed bound and
        the optimistic one is visible rather than assumed away.
        """
        out = 1.0
        for c in self.channels:
            out *= c.alpha
        return out

    @property
    def merged_evalue(self) -> float:
        """The final merged e-value, by the rule that is valid under arbitrary dependence."""
        return merge_e([c.final_evalue for c in self.channels], dependence="arbitrary").value

    def single_channel_alarms(self) -> dict[str, int | None]:
        return {c.name: c.alarm_at for c in self.channels}

    def render(self) -> str:
        head = [
            f"conjunction over {self.k} channels, {self.n_steps} steps.",
            f"    joint false-alarm bound {self.joint_far_bound:.4g} under arbitrary dependence "
            f"(the guarantee); {self.joint_far_if_independent:.4g} if the channels were "
            f"independent, which they are not.",
            f"    merged e-value {self.merged_evalue:.4g} by arithmetic mean, which is the rule "
            f"that is valid here.",
        ]
        if self.fired_at is not None:
            head.append(f"    all channels above threshold together at index {self.fired_at}.")
        elif self.all_crossed_at is not None:
            head.append(
                f"    every channel crossed at some point, all by index {self.all_crossed_at}, "
                f"but never all at once."
            )
        else:
            head.append("    the conjunction never fired.")
        head.append(
            f"    e-BH at FDR {self.fdr_alpha:.3g} rejects "
            f"{list(self.ebh_rejected) or 'nothing'} at e >= {self.ebh_threshold:.4g}."
        )
        return "\n".join(head + ["    " + c.render() for c in self.channels])


def conjunction_alarm(
    channels: Sequence[Channel],
    *,
    alpha: float = 0.01,
    shift: float = 1.0,
    baseline_steps: int | None = None,
    fdr_alpha: float = 0.05,
    reference: Mapping[str, tuple[float, float]] | None = None,
) -> ConjunctionAlarm:
    """Run every channel's e-process and and them together, with a ledger under e-BH.

    ``alpha`` is per channel. The conjunction's own false-alarm probability is at most the smallest
    of them, so tightening one channel tightens the conjunction and the other channels are then only
    contributing power, which is the right way round: a conjunction should not be able to make its
    own guarantee worse by adding a channel.
    """
    alarms = tuple(
        channel_alarm(
            c,
            alpha=alpha,
            shift=shift,
            baseline_steps=baseline_steps,
            reference=(reference or {}).get(c.name),
        )
        for c in channels
    )
    if not alarms:
        return ConjunctionAlarm((), None, None, 0, {}, (), float("inf"), fdr_alpha)
    n = min(a.evalue.size for a in alarms)
    above = np.vstack([a.above()[:n] for a in alarms])
    all_now = np.all(above, axis=0)
    fired = np.where(all_now)[0]
    crossed = np.maximum.accumulate(above, axis=1)
    all_ever = np.all(crossed, axis=0)
    ever = np.where(all_ever)[0]

    ledger = {a.name: a.final_evalue for a in alarms}
    result = ebh([ledger[a.name] for a in alarms], fdr_alpha)
    rejected = tuple(a.name for a, r in zip(alarms, result.rejected) if r)
    return ConjunctionAlarm(
        channels=alarms,
        fired_at=int(fired[0]) if fired.size else None,
        all_crossed_at=int(ever[0]) if ever.size else None,
        n_steps=int(n),
        ledger=ledger,
        ebh_rejected=rejected,
        ebh_threshold=result.threshold,
        fdr_alpha=float(fdr_alpha),
    )


# ---------------------------------------------------------------------------
# Measuring the kill condition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OperatingRow:
    """One detector's detection rate and delay at a matched false-alarm rate."""

    name: str
    target_far: float
    achieved_far: float
    threshold: float
    detection: float
    median_delay: float

    def render(self) -> str:
        delay = "never" if math.isnan(self.median_delay) else f"{self.median_delay:5.1f} steps"
        return (
            f"{self.name:<20} FAR {self.achieved_far:6.2%}  detected {self.detection:6.1%}  "
            f"median delay {delay}   (threshold {self.threshold:.4g})"
        )


@dataclass(frozen=True)
class ConjunctionComparison:
    """The conjunction against each channel alone, **at a matched false-alarm rate**.

    Matching matters and comparing at a common threshold does not. The conjunction statistic is
    ``min_k e_k(t)``, which is bounded by Ville at the same level as any single ``e_k``, so at a
    common threshold the conjunction is guaranteed to have the lower false-alarm rate and would win
    that comparison by construction. The question worth asking is what it detects **when both are
    tuned to the same false-alarm rate**, and that is what this measures: sweep the threshold on
    each statistic, pick the threshold that hits the target rate on the pre-change window, and read
    the detection rate and delay off there.

    J3's kill condition is that a single channel matches the conjunction's operating characteristic.
    This is the measurement that would settle it.
    """

    n_runs: int
    n_pre: int
    n_post: int
    shift: float
    rho: float
    channels_shifted: int
    k: int
    target_far: float
    conjunction: OperatingRow
    singles: tuple[OperatingRow, ...]
    artefact_rate: float = 0.0
    artefact_size: float = 0.0
    scale: str = "estimated"

    @property
    def best_single(self) -> OperatingRow:
        return max(self.singles, key=lambda r: r.detection)

    @property
    def kill_condition_met(self) -> bool:
        """True when the best single channel detects at least as often at the same false-alarm rate.

        Tolerance of one percentage point on the detection rate, so a Monte Carlo wobble does not
        decide it.
        """
        return self.best_single.detection >= self.conjunction.detection - 0.01

    def render(self) -> str:
        lines = [
            f"conjunction against single channels at a matched {self.target_far:.1%} false-alarm "
            f"rate over {self.n_pre} pre-change steps.",
            f"    {self.n_runs} runs, {self.k} channels of which {self.channels_shifted} carry a "
            f"{self.shift:.2g}-sigma shift, noise equicorrelation {self.rho:.2g}, "
            + (
                f"channel-specific artefacts on {self.artefact_rate:.0%} of runs at "
                f"{self.artefact_size:.2g} sigma."
                if self.artefact_rate > 0
                else "no channel-specific artefacts."
            )
            + f" In-control scale: {self.scale}.",
            "    " + self.conjunction.render(),
        ]
        for row in self.singles:
            lines.append("    " + row.render())
        verdict = (
            f"the best single channel ({self.best_single.name}) matches or beats the conjunction "
            f"at the same false-alarm rate. J3's kill condition is MET on this design"
            if self.kill_condition_met
            else f"the conjunction detects {self.conjunction.detection - self.best_single.detection:+.1%} "
            f"more than the best single channel at the same false-alarm rate. The kill condition "
            f"is not met here"
        )
        lines.append(f"    {verdict}.")
        return "\n".join(lines)


def _operating_row(name: str, statistic: np.ndarray, n_pre: int, target_far: float) -> OperatingRow:
    """Sweep the threshold on a running statistic and read off the matched operating point.

    ``statistic`` is ``(n_runs, n_steps)``. The false-alarm rate is the fraction of runs whose
    statistic first crosses inside the pre-change window; the detection rate is the fraction whose
    first crossing is at or after it. The threshold is chosen as the smallest one whose realised
    false-alarm rate is at or below the target, which is the conservative direction: a detector
    tuned slightly quieter than the target cannot win the comparison by spending more budget.
    """
    # A false alarm happens if and only if the pre-change running maximum crosses, so the realised
    # false-alarm rate at a threshold is the fraction of runs whose pre-change maximum is at or
    # above it. Sorting the maxima gives the whole curve without re-running any chart.
    pre_max = np.max(statistic[:, :n_pre], axis=1)
    order = np.sort(pre_max)
    idx = int(np.floor((1.0 - target_far) * order.size))
    threshold = float(order[min(idx, order.size - 1)])
    if threshold <= 0 or not math.isfinite(threshold):
        threshold = float(np.max(pre_max))
    achieved = float(np.mean(pre_max >= threshold))
    crossed = statistic >= threshold
    first = np.where(crossed.any(axis=1), crossed.argmax(axis=1), -1)
    detected = first >= n_pre
    delays = (first[detected] - n_pre).astype(np.float64)
    return OperatingRow(
        name=name,
        target_far=float(target_far),
        achieved_far=achieved,
        threshold=threshold,
        detection=float(np.mean(detected)),
        median_delay=float(np.median(delays)) if delays.size else float("nan"),
    )


def measure_conjunction(
    *,
    k: int = 3,
    channels_shifted: int = 3,
    shift: float = 0.6,
    rho: float = 0.5,
    n_pre: int = 100,
    n_post: int = 100,
    n_runs: int = 1000,
    target_far: float = 0.05,
    artefact_rate: float = 0.0,
    artefact_size: float = 2.0,
    artefact_length: int = 12,
    scale: str = "estimated",
    seed: int = 0,
) -> ConjunctionComparison:
    """Plant a shift on some channels of a correlated multivariate stream and compare, matched.

    ``rho`` is the equicorrelation of the channels' noise, and it is the parameter the whole
    argument turns on: at ``rho = 0`` the channels are independent and a product merge would be
    legal, at ``rho`` near 1 they are one channel wearing three names. The default of 0.5 is between
    them and is not tuned to make the conjunction look good.

    **``artefact_rate`` is the parameter that decides the kill condition, and it is the one the
    ALFWorld result is actually about.** Under Gaussian noise alone a conjunction can only lose
    power, because every channel is already well controlled and requiring all of them to agree is
    strictly stricter. The measured claim is different: entropy false-alarms *in three distinct
    ways*, meaning channel-specific structural artefacts rather than noise. A transient in one
    channel forces that channel's threshold up to keep its false-alarm rate at target, which costs
    it detection, and leaves the conjunction untouched because the other channels did not move.
    Setting ``artefact_rate`` above zero plants exactly that: on that fraction of runs one channel
    gets a bump of ``artefact_size`` sigmas lasting ``artefact_length`` steps at a random pre-change
    position.

    ``channels_shifted`` below ``k`` is the case the conjunction is *bad* at, and it is offered so
    the comparison can be run there too: a conjunction requiring all three cannot fire when only two
    move, which is a real cost of the design and not a defect in the measurement.
    """
    rng = np.random.default_rng(seed)
    n = n_pre + n_post
    cov = np.full((k, k), rho, dtype=np.float64)
    np.fill_diagonal(cov, 1.0)
    chol = np.linalg.cholesky(cov)
    names = [f"channel{i}" for i in range(k)]
    mean = np.zeros((k, n))
    mean[:channels_shifted, n_pre:] = shift

    stat_single = np.empty((k, n_runs, n), dtype=np.float64)
    for r in range(n_runs):
        raw = chol @ rng.standard_normal((k, n)) + mean
        if artefact_rate > 0 and rng.random() < artefact_rate:
            victim = int(rng.integers(k))
            start = int(rng.integers(0, max(1, n_pre - artefact_length)))
            raw[victim, start : start + artefact_length] += artefact_size
        pre = raw[:, :n_pre]
        if scale == "known":
            z = raw
        elif scale == "robust":
            med = np.median(pre, axis=1, keepdims=True)
            mad = 1.4826 * np.median(np.abs(pre - med), axis=1, keepdims=True)
            z = (raw - med) / np.where(mad > 0, mad, 1.0)
        elif scale == "estimated":
            mu = pre.mean(axis=1, keepdims=True)
            sd = pre.std(axis=1, keepdims=True)
            z = (raw - mu) / np.where(sd > 0, sd, 1.0)
        else:
            raise ValueError(f"scale must be 'estimated', 'robust' or 'known'; got {scale!r}")
        for i in range(k):
            stat_single[i, r] = gaussian_lr_eprocess(z[i], shift)
    stat_conj = np.min(stat_single, axis=0)

    return ConjunctionComparison(
        n_runs=n_runs,
        n_pre=n_pre,
        n_post=n_post,
        shift=shift,
        rho=rho,
        channels_shifted=channels_shifted,
        k=k,
        target_far=float(target_far),
        conjunction=_operating_row("conjunction", stat_conj, n_pre, target_far),
        singles=tuple(
            _operating_row(name, stat_single[i], n_pre, target_far) for i, name in enumerate(names)
        ),
        artefact_rate=float(artefact_rate),
        artefact_size=float(artefact_size),
        scale=str(scale),
    )


# ---------------------------------------------------------------------------
# The instrument
# ---------------------------------------------------------------------------

J3_BASELINES: tuple[str, ...] = (
    "baseline.single_channel_alarm",
    "baseline.best_single_channel",
)

#: Below this many channels a conjunction is not a conjunction. Two is the minimum that composes at
#: all, and the instrument says so rather than reporting a one-channel "conjunction" whose joint
#: false-alarm bound is just that channel's own.
MIN_CHANNELS: int = 2

#: The shortest in-control window the standardization is allowed to use, and the shortest window
#: that may then be monitored.
#:
#: **This is not a convenience default, it is a correctness condition.** The e-process is a test
#: against a stated in-control distribution, and standardizing against the whole series estimates
#: that distribution from the data the test is about. When it does, the sum of the standardized
#: series is exactly zero at the last step, so every channel's e-value ends at exactly
#: ``exp(-n shift^2 / 2)`` regardless of what the channel did: three different channels return the
#: same number, and the number is a function of the window length alone. Measured on the shipped
#: 12-step fixture before the floor was added, where all three channels returned 0.002479.
MIN_BASELINE_STEPS: int = 8
MIN_MONITOR_STEPS: int = 4


class ConjunctionDetector(MonitorInstrument):
    """J3. The conjunction over channels, with a joint false-alarm rate that is a real bound.

    "The conjunction of entropy decline, reward-spread saturation and length pinning fired at step
    62. Entropy alone fired three times, twice before anything happened."

    What it cannot do, and it is the design's real cost: a conjunction requiring every channel
    cannot fire when one channel is missing from the record or constant on it. On the shipped GRPO
    fixture two of the three channels the ALFWorld result names are unavailable, `completion_length`
    because every completion hits the 12-token cap so the series is constant, and prediction
    saturation because nothing in the record measures it. The instrument refuses by name rather than
    quietly running a two-channel conjunction and calling it the three-channel result.

    **Pass a ``reference`` if you have one.** Without a clean in-control ``(mean, sd)`` per channel,
    the scale is estimated from the window being monitored, and the module docstring's table shows
    that this is what costs a conjunction its advantage over the best single channel. A frozen probe
    set measured on a quiet run (J5) is where a reference comes from.
    """

    name = "ConjunctionDetector"
    version = "1.0"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = (
        "Ville (1939) per channel; Vovk and Wang (2021) merging; Wang and Ramdas (2022) e-BH"
    )
    deviations = (
        "multiplying the channels' e-values would compose them as if they were independent, and "
        "they are not, so the conjunction is bounded instead by the minimum of the per-channel "
        "levels, which needs no dependence assumption, and the merged e-value uses the arithmetic "
        "mean.",
        "each channel's e-process is the Gaussian likelihood-ratio martingale on the standardized "
        "series, so the per-channel level is exact only to the extent that the standardized channel "
        "is standard normal in control. An autocorrelated channel spends its budget faster than the "
        "stated alpha.",
        "the in-control mean and standard deviation are estimated from a baseline window rather "
        "than known, unless a `reference` is supplied, so the stated alpha is exact only in the "
        "limit of a long baseline. At the 8-step floor the level is optimistic, and the reading "
        "records which of the two produced the scale.",
    )

    quantity = "monitor.conjunction_alarm"
    requires = RECORD_ACCESS
    envelope = MONITOR_ENVELOPE
    invariance = "units"
    invariance_relation = INVARIANT
    baselines = J3_BASELINES
    rung = 0

    def __init__(
        self,
        channels: ChannelSet | Sequence[Channel],
        *,
        wanted: Sequence[str] | None = None,
        alpha: float = 0.01,
        shift: float = 1.0,
        baseline_steps: int | None = None,
        fdr_alpha: float = 0.05,
        reference: Mapping[str, tuple[float, float]] | None = None,
    ) -> None:
        self.channel_set = channels if isinstance(channels, ChannelSet) else None
        if isinstance(channels, ChannelSet):
            names = list(wanted) if wanted is not None else channels.names()
            self.channels = [channels[n] for n in names if n in channels]
            self.missing = [n for n in names if n not in channels]
        else:
            self.channels = list(channels)
            self.missing = []
        self.wanted = list(wanted) if wanted is not None else None
        self.alpha = float(alpha)
        self.shift = float(shift)
        self.baseline_steps = baseline_steps
        self.fdr_alpha = float(fdr_alpha)
        self.reference = dict(reference) if reference else None

    def _baseline(self, n: int) -> int:
        """The in-control window: a third of the series, never fewer than the floor."""
        if self.baseline_steps is not None:
            return int(self.baseline_steps)
        return max(MIN_BASELINE_STEPS, n // 3)

    def compute(self, ctx: Context) -> ConjunctionAlarm | Refusal:
        # Two ways to be short of channels, and they are the same refusal. The caller named a
        # channel the record does not carry, or the caller named nothing and the record carries
        # fewer than two. A conjunction run over fewer channels than it names is a different
        # detector with a different false-alarm rate, and running it silently is how the ALFWorld
        # result gets cited for a claim it does not support.
        named_and_missing = bool(self.wanted) and bool(self.missing)
        if named_and_missing or len(self.channels) < MIN_CHANNELS:
            missing = ", ".join(self.missing) or "none named"
            reasons = ""
            if self.channel_set is not None:
                reasons = " ".join(
                    f"`{n}`: {self.channel_set.absent[n].detail}."
                    for n in self.missing
                    if n in self.channel_set.absent
                )
            asked = (
                f"you asked for {len(self.wanted)} channels and {len(self.channels)} are usable"
                if self.wanted
                else f"{len(self.channels)} usable channel(s), and a conjunction needs at least "
                f"{MIN_CHANNELS}"
            )
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.RECORD_INCOMPLETE,
                detail=f"{asked}. Unavailable: {missing}. {reasons}".strip(),
                remedy=(
                    "Log the missing channels on each step and re-run, or name only the channels "
                    "this record carries and read the result as the detector it actually is. The "
                    "joint false-alarm bound of a two-channel conjunction is not the bound of a "
                    "three-channel one, so the two are not interchangeable even when they agree."
                ),
                statistics={
                    "n_channels": len(self.channels),
                    "have": [c.name for c in self.channels],
                    "missing": self.missing,
                },
            )
        n = min(c.n for c in self.channels)
        baseline = self._baseline(n)
        if n - baseline < MIN_MONITOR_STEPS or baseline < MIN_BASELINE_STEPS:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ESS_BELOW_FLOOR,
                detail=(
                    f"{n} steps, which leaves {max(0, n - baseline)} steps to monitor after a "
                    f"{baseline}-step in-control window. The floor is {MIN_BASELINE_STEPS} baseline "
                    f"steps and {MIN_MONITOR_STEPS} monitored steps."
                ),
                remedy=(
                    f"Monitor a window of at least "
                    f"{MIN_BASELINE_STEPS + MIN_MONITOR_STEPS} steps, or supply "
                    f"`baseline_steps=` from a separate in-control run so the whole of this window "
                    f"can be monitored. Standardizing against the same steps the test is about "
                    f"makes every channel's e-value collapse to exp(-n shift^2 / 2) and stop "
                    f"depending on the channel at all, so a shorter window does not give a weaker "
                    f"answer, it gives a number that is not about the data."
                ),
                statistics={"n_steps": n, "baseline_steps": baseline},
            )
        return conjunction_alarm(
            self.channels,
            alpha=self.alpha,
            shift=self.shift,
            baseline_steps=baseline,
            fdr_alpha=self.fdr_alpha,
            reference=self.reference,
        )

    def payload(self, computed: ConjunctionAlarm) -> dict:
        curve = ppv_curve(1.0, computed.joint_far_bound)
        return {
            "fired": computed.fired_at is not None,
            "fired_at": computed.fired_at,
            "all_crossed_at": computed.all_crossed_at,
            "n_channels": computed.k,
            "baseline_steps": None
            if self.reference
            else self._baseline(min(c.n for c in self.channels)),
            "scale_source": "clean reference" if self.reference else "the monitored window",
            "channels": [c.name for c in computed.channels],
            "missing_channels": self.missing,
            "joint_far_bound": computed.joint_far_bound,
            "joint_far_if_independent": computed.joint_far_if_independent,
            "merged_evalue": computed.merged_evalue,
            "merge_rule": "arithmetic mean, valid under arbitrary dependence",
            "per_channel_alarm": computed.single_channel_alarms(),
            "per_channel_evalue": dict(computed.ledger),
            "ebh_rejected": list(computed.ebh_rejected),
            "ebh_threshold": computed.ebh_threshold,
            "fdr_alpha": computed.fdr_alpha,
            "ppv_curve": {
                "sensitivity_assumed": 1.0,
                "fpr": computed.joint_far_bound,
                "prevalences": list(curve.prevalences),
                "ppv": list(curve.ppv),
                "note": (
                    "drawn at the guaranteed joint false-alarm bound and perfect sensitivity. The "
                    "realised PPV cannot exceed these."
                ),
            },
            "baselines": self.baseline_map(computed),
            "rendered": computed.render(),
        }

    def baseline_map(self, computed: ConjunctionAlarm) -> Mapping[str, float]:
        fired = [c.alarm_at for c in computed.channels if c.alarm_at is not None]
        return {
            # How many channels alone would have raised an alarm. On a quiet run this is the count
            # of alarms a bank of independent thresholds would have produced and the conjunction
            # did not.
            "baseline.single_channel_alarm": float(len(fired)),
            "baseline.best_single_channel": float(min(fired)) if fired else float("nan"),
        }

    def uncertainty(self, computed: ConjunctionAlarm) -> Uncertainty:
        return Uncertainty(
            n=computed.n_steps,
            method="anytime-valid per channel; joint bound by intersection, no dependence assumed",
        )


__all__ = [
    "J3_BASELINES",
    "MIN_CHANNELS",
    "ChannelAlarm",
    "ConjunctionAlarm",
    "ConjunctionComparison",
    "ConjunctionDetector",
    "channel_alarm",
    "conjunction_alarm",
    "gaussian_lr_eprocess",
    "measure_conjunction",
]

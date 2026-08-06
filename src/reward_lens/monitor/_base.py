"""What the five monitoring instruments share: the declarations, the runner seam, the channels.

Three things live here because putting them in each instrument would mean five chances to get them
inconsistent.

**The declarations.** All five apply to every substrate, because a monitor watches a series of
numbers and does not care what produced them. All five answer an in-run or post-run question, and
none answers a pre-run one: there is no series before the run. Four of the five require the run's
grader to have held still across the window, which is the one confound a change detector cannot see
past. The fifth is the instrument that measures whether it did.

**The seam.** `Observable.measure` returns `Evidence` by contract and `Instrument.estimate` returns
`Reading`. Every instrument here can decide to refuse after looking at the series, so the dispatch
sits here once, in the same shape `measure.decision._base.DecisionInstrument` uses. It is not
imported from there: one instrument family reaching into another's underscore module is a
dependency that is invisible at the point where it breaks.

**The channels.** A monitor watches several series at once, and getting a series off a `Run` is
where the honest refusals are. `channels_of` returns what a run carries and, for each channel it
cannot build, the reason and the remedy. On the shipped 200-step GRPO record that is four channels
present out of nine: three are missing because TRL never wrote them, one is present and constant
because every completion hits the token cap, and one is present and moves only within the noise of
its own float32 representation. Those are three different failures with three different fixes, so
they are three different refusals rather than one absent key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from reward_lens.core.envelope import EnvelopeSpec, RegimeCondition
from reward_lens.core.evidence import Evidence, Uncertainty
from reward_lens.core.gates import require_frame_for_comparison
from reward_lens.core.reading import Reading, Refusal, RefusalReason, refuse_incomplete
from reward_lens.core.types import Access, Component, Phase, Substrate
from reward_lens.measure.base import BaseObservable, Context, run
from reward_lens.measure.rate.regime import MEASURED_BY

# ---------------------------------------------------------------------------
# The shared declarations
# ---------------------------------------------------------------------------

#: Every substrate. A monitor consumes a scalar per step and is indifferent to whether a network, a
#: unit test or a panel of humans produced it.
ALL_SUBSTRATES: frozenset[Substrate] = frozenset(
    {
        Substrate.NEURAL_SCALAR,
        Substrate.NEURAL_GEN,
        Substrate.PROGRAM,
        Substrate.PROCEDURAL,
        Substrate.HUMAN,
        Substrate.COMPOSITE,
    }
)

#: `IN_RUN` because that is what a monitor is for, and `POST_RUN` because an auditor replaying a
#: finished record asks the same question of the same series and gets the same answer. `PRE_RUN` is
#: excluded because there is no series yet, and `DEPLOYED` because a served artifact emits no
#: training steps.
MONITOR_PHASES: frozenset[Phase] = frozenset({Phase.IN_RUN, Phase.POST_RUN})

#: Reading a record is all any of these needs. Nothing here calls a grader, loads a checkpoint or
#: differentiates anything, which is why the whole layer runs at auditor access on somebody else's
#: run.
RECORD_ACCESS: dict[Component, Access] = {Component.RECORD: Access.RECORD}

#: A loss statement and a prevalence are the caller's, not the record's, so J4 needs no access at
#: all. Declared as an empty matrix rather than a record requirement, because claiming a
#: requirement the instrument does not have would make the capability report wrong in the direction
#: that withholds a reading.
NO_ACCESS: dict[Component, Access] = {}

#: The two conditions a change detector cannot see past, and the reason each one matters.
#:
#: `STATIONARY_GRADER`: if the grader's weights or rubric moved inside the window, a detected shift
#: is the grader moving and the alarm cannot tell which. The instrument is not being asked whether
#: the policy changed; it is being asked whether the *measured series* changed, and under a moving
#: ruler those are different questions with the same answer.
#:
#: `NO_COMPACTION`: a prefix rewrite inside the window silently reorders the sequence a sequential
#: test consumes, and every anytime-valid guarantee in this package is a statement about a sequence
#: in the order it arrived.
MONITOR_ENVELOPE = EnvelopeSpec(
    requires=frozenset({RegimeCondition.STATIONARY_GRADER, RegimeCondition.NO_COMPACTION}),
    measured_by={
        RegimeCondition.STATIONARY_GRADER: MEASURED_BY[RegimeCondition.STATIONARY_GRADER],
        RegimeCondition.NO_COMPACTION: MEASURED_BY[RegimeCondition.NO_COMPACTION],
    },
    on_violation="refuse",
)


# ---------------------------------------------------------------------------
# The seam
# ---------------------------------------------------------------------------


class MonitorInstrument(BaseObservable):
    """Preflight, compute once, refuse or emit. One dispatch for the family.

    ``compute`` takes the `Context` because four of the five read a series off a record and the
    fifth reads a stated loss, and forcing the record-free one to accept a context it ignores is
    cheaper than two seams.
    """

    _computed: Any = None

    substrates = ALL_SUBSTRATES
    phases = MONITOR_PHASES

    def compute(self, ctx: Context) -> Any:  # pragma: no cover - abstract
        """The instrument's arithmetic. Returns a payload object or a `Refusal`."""
        raise NotImplementedError

    def payload(self, computed: Any) -> dict[str, Any]:  # pragma: no cover - abstract
        raise NotImplementedError

    def uncertainty(self, computed: Any) -> Uncertainty | None:
        return None

    def baseline_map(self, computed: Any) -> Mapping[str, float] | None:
        """The declared baselines, as numbers, for `Evidence.baselines`."""
        return None

    def gated_emit(self, ctx: Context, computed: Any) -> Evidence:
        """Hand a computed payload to the runner, or apply the runner's gates by hand.

        `run` resolves `ctx.signal.caps` and a monitor reading a record has no signal to resolve.
        The no-signal branch does what `run` would do minus the check that has nothing to check
        against, including setting `ctx._observable`, which is what `Context.emit` reads the name,
        version and quantity off. Gate 2 applies in both branches.
        """
        self._computed = computed
        try:
            if ctx.signal is not None:
                return run(self, ctx)
            if ctx.is_comparison:
                require_frame_for_comparison(self.gauge_status, ctx.frame)
            ctx._observable = self
            try:
                return self.measure(ctx)
            finally:
                ctx._observable = None
        finally:
            self._computed = None

    def estimate(self, ctx: Context) -> Reading:
        pre = self.preflight(ctx)
        if not pre.ok and pre.refusal is not None:
            return pre.refusal
        out = self.compute(ctx)
        if isinstance(out, Refusal):
            return out
        return self.gated_emit(ctx, out)

    def measure(self, ctx: Context) -> Evidence:
        out = self._computed if self._computed is not None else self.compute(ctx)
        if isinstance(out, Refusal):
            raise ValueError(
                f"{self.name}.measure was called on a measurement that declines to produce "
                f"Evidence: {out.reason.name}. Call `estimate`, which returns the refusal as a "
                f"value carrying its remedy."
            )
        body = self.payload(out)
        return ctx.emit(
            body,
            uncertainty=self.uncertainty(out),
            baselines=self.baseline_map(out),
        )


# ---------------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Channel:
    """One per-step scalar series, with where it came from and which way is worse.

    ``direction`` is +1 when a rise is the worrying move and -1 when a fall is. It is a declaration
    rather than something fitted, because fitting the direction on the same series the alarm is
    read from is how a detector learns to fire on whichever way the data happened to go.
    """

    name: str
    values: np.ndarray
    steps: np.ndarray
    direction: int
    source: str
    detail: str = ""

    @property
    def n(self) -> int:
        return int(self.values.size)

    @property
    def oriented(self) -> np.ndarray:
        """The series signed so that larger always means worse. What a detector consumes."""
        return self.values * float(self.direction)

    def render(self) -> str:
        arrow = "rise" if self.direction > 0 else "fall"
        return (
            f"{self.name:<26} n={self.n:<5} {arrow} is the worrying move   "
            f"[{self.source}]{('  ' + self.detail) if self.detail else ''}"
        )


@dataclass(frozen=True)
class ChannelSet:
    """What a run carries, and for everything it does not, why not.

    The absent half is kept rather than dropped because "three channels were watched" and "three
    channels were watched and four more were unavailable" are different states, and a conjunction
    detector's whole claim is about how many channels agreed.
    """

    present: dict[str, Channel] = field(default_factory=dict)
    absent: dict[str, Refusal] = field(default_factory=dict)
    n_steps: int = 0

    def __getitem__(self, name: str) -> Channel:
        return self.present[name]

    def __contains__(self, name: str) -> bool:
        return name in self.present

    def names(self) -> list[str]:
        return sorted(self.present)

    def render(self) -> str:
        lines = [f"{len(self.present)} channels over {self.n_steps} steps"]
        for name in sorted(self.present):
            lines.append("    " + self.present[name].render())
        for name in sorted(self.absent):
            lines.append(f"    {name:<26} unavailable: {self.absent[name].detail}")
        return "\n".join(lines)


#: The resolution floor a logged channel has to clear before a change detector may be run on it.
#: A range of 32 float32 units in the last place is about five bits of real variation, and eight
#: distinct values is the smallest set a standardization can be built from without the z-scores
#: being a relabelling of the quantisation levels. Both floors are cleared by five to seven orders
#: of magnitude on every channel of the shipped record except one, and failed by the one.
MIN_ULPS: float = 32.0
MIN_DISTINCT_VALUES: int = 8

#: What each channel is, where it is read from, and which direction is the worrying one. The three
#: the ALFWorld collapse study names are the first three: entropy declining, the policy saturating,
#: and episode length pinning to a bound. The rest are here because a real GRPO record carries them
#: and a conjunction over channels a record does not have is not a detector.
CHANNEL_SPECS: tuple[tuple[str, str, int, str], ...] = (
    ("entropy", "optimizer.entropy", -1, "policy entropy; monotone decline precedes collapse"),
    (
        "reward_std",
        "optimizer.extra['reward_std']",
        -1,
        "spread of the reward within a batch; saturation is the prediction collapsing to one answer",
    ),
    (
        "completion_length",
        "optimizer.extra['completions/mean_length']",
        +1,
        "mean completion length; pinning to the cap is the third channel of the conjunction",
    ),
    ("grad_norm", "optimizer.grad_norm_clipped", +1, "logged gradient norm; the baseline's series"),
    ("reward", "optimizer.extra['reward']", +1, "mean reward per step"),
    ("kl_to_ref", "optimizer.kl_to_ref", +1, "KL to the reference policy"),
    ("kl_to_previous", "optimizer.kl_to_previous", +1, "KL to the previous policy"),
    (
        "group_mean",
        "mean over groups of group_stats.mean",
        +1,
        "the per-step mean score, aggregated over groups",
    ),
    (
        "degenerate_fraction",
        "fraction of groups with group_stats.degenerate",
        +1,
        "how much of the batch produced no gradient signal at all",
    ),
)


def _step_value(step: Any, source: str) -> float | None:
    """Pull one channel's value off one step, by the source string in `CHANNEL_SPECS`."""
    opt = step.optimizer
    if source == "optimizer.entropy":
        return opt.entropy
    if source == "optimizer.grad_norm_clipped":
        return opt.grad_norm_clipped
    if source == "optimizer.kl_to_ref":
        return opt.kl_to_ref
    if source == "optimizer.kl_to_previous":
        return opt.kl_to_previous
    if source.startswith("optimizer.extra["):
        key = source[len("optimizer.extra['") : -2]
        value = opt.extra.get(key)
        return None if value is None else float(value)
    if source == "mean over groups of group_stats.mean":
        means = [g.group_stats.mean for g in step.groups if g.group_stats.mean is not None]
        return float(np.mean(means)) if means else None
    if source == "fraction of groups with group_stats.degenerate":
        if not step.groups:
            return None
        return float(np.mean([1.0 if g.group_stats.degenerate else 0.0 for g in step.groups]))
    raise KeyError(f"no reader for channel source {source!r}")


def channels_of(
    run: Any,
    *,
    window: tuple[int, int] | None = None,
    instrument: str = "monitor",
    wanted: Sequence[str] | None = None,
) -> ChannelSet:
    """Every per-step channel this run actually carries, and a refusal for each one it does not.

    Three conditions, each a refusal rather than a silent drop, and the second and third both fire
    on the shipped 200-step GRPO record.

    **Enough finite values.** Fewer than three and there is no series. TRL leaves `kl_to_ref`,
    `kl_to_previous` and the group-degeneracy fraction unwritten on that record, and that is a
    `RECORD_INCOMPLETE` whose fix is upstream.

    **Not constant.** `completions/mean_length` is logged on every step and every value is 12.0,
    because `max_completion_length` is 12 and every completion hits it. A change detector on a
    constant series cannot fire, and reporting "no alarm" from it would be reporting the absence of
    variance as the absence of a problem.

    **Above the recording's own resolution**, which is the one that is easy to miss. On that same
    record, `optimizer.entropy` takes **six distinct values across 200 steps** and its whole range
    spans **six float32 units in the last place** at its magnitude of 11.93. It is not constant, so
    the previous check passes it, and standardizing it produces z-scores of order one built entirely
    out of round-off. Every other channel on that record spans between five and fifty million ULPs.
    A channel below the floor is `BELOW_LOD`: what was written is smaller than the disagreement of
    the thing that wrote it.
    """
    steps = list(run.steps) if window is None else list(run.steps.slice(*window))
    indices = np.array([s.index for s in steps], dtype=np.int64)
    present: dict[str, Channel] = {}
    absent: dict[str, Refusal] = {}
    for name, source, direction, detail in CHANNEL_SPECS:
        if wanted is not None and name not in wanted:
            continue
        raw = [_step_value(s, source) for s in steps]
        keep = np.array(
            [v is not None and np.isfinite(v) for v in raw],
            dtype=bool,
        )
        n_finite = int(keep.sum())
        if n_finite < 3:
            absent[name] = refuse_incomplete(
                instrument,
                field=source,
                subject=f"{len(steps)} steps of this run",
                remedy=(
                    f"log {source} on each step and re-run, or drop `{name}` from the channel set "
                    f"you asked for. This field was never written, so nothing that can be done to "
                    f"this record recovers it: the fix is in whatever produced the run."
                ),
                n_finite=n_finite,
                n_steps=len(steps),
            )
            continue
        values = np.array([float(v) if v is not None else np.nan for v in raw], dtype=np.float64)
        finite = values[keep]
        if float(np.ptp(finite)) == 0.0:
            absent[name] = refuse_incomplete(
                instrument,
                field=source,
                subject=f"{len(steps)} steps of this run",
                remedy=(
                    f"`{name}` is present and constant at {finite[0]:.6g} on every step, so no "
                    f"change detector can fire on it and a quiet chart here means nothing. Either "
                    f"widen the window to a span where it moves, or drop it from the channel set "
                    f"so the conjunction does not count a channel that cannot vote."
                ),
                n_finite=n_finite,
                constant_at=float(finite[0]),
            )
            continue
        span = float(np.ptp(finite))
        magnitude = max(abs(float(np.median(finite))), 1e-12)
        ulp = float(np.spacing(np.float32(magnitude)))
        n_distinct = int(np.unique(finite).size)
        if span < MIN_ULPS * ulp or n_distinct < MIN_DISTINCT_VALUES:
            absent[name] = Refusal(
                instrument=instrument,
                reason=RefusalReason.BELOW_LOD,
                detail=(
                    f"`{name}` spans {span:.4g} across {n_finite} steps at a magnitude of "
                    f"{magnitude:.6g}, which is {span / ulp:.1f} float32 units in the last place "
                    f"and {n_distinct} distinct value(s). The floor is {MIN_ULPS} ULPs and "
                    f"{MIN_DISTINCT_VALUES} distinct values."
                ),
                remedy=(
                    f"Log `{name}` at higher precision, or accept that it did not move on this run "
                    f"and drop it from the channel set. Standardizing a series whose whole range is "
                    f"a few units in the last place produces z-scores of order one out of round-off, "
                    f"and a chart run on those will alarm on the arithmetic rather than on the run."
                ),
                statistics={
                    "span": span,
                    "ulps": span / ulp,
                    "n_distinct": n_distinct,
                    "n_finite": n_finite,
                },
            )
            continue
        present[name] = Channel(
            name=name,
            values=values,
            steps=indices,
            direction=direction,
            source=source,
            detail=detail,
        )
    return ChannelSet(present=present, absent=absent, n_steps=len(steps))


__all__ = [
    "ALL_SUBSTRATES",
    "CHANNEL_SPECS",
    "MIN_DISTINCT_VALUES",
    "MIN_ULPS",
    "MONITOR_ENVELOPE",
    "MONITOR_PHASES",
    "NO_ACCESS",
    "RECORD_ACCESS",
    "Channel",
    "ChannelSet",
    "MonitorInstrument",
    "channels_of",
]

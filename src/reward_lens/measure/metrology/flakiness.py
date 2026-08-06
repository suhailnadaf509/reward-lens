"""A7, environment flakiness: score variance that belongs to the environment.

The cheapest instrument in the catalogue and one of the most damaging. Replay the same task with
the same policy N times and report the spread. LiveMCPBench's 18.9-point spread is the existence
proof that the spread can be large, and the state of practice is that **nobody reports flakiness as
a property of an environment**: benchmark cards report a score, agent papers report a delta, and
neither reports what the same number does when you run it again.

The consequence is arithmetic. If an environment's own spread on a fixed policy is 19 points, a
3-point improvement measured once is inside it, and the sentence "we improved by 3 points" is not a
claim about the policy. That sentence gets written anyway, because the number needed to contradict
it does not exist. This instrument produces that number for the price of running the evaluation
again, and it belongs on every grader card for the same reason a balance's repeatability belongs on
its certificate.

Three rungs, and the first one is free.

**Rung 0: N replays, report the range.** The headline is the per-task range in percentage points,
because the range is what a reader compares their delta against. The standard deviation is reported
beside it and is the better statistic for anything downstream, but a range is what answers "could my
3 points be this".

**Rung 1: the occasion facet.** The replay index is an occasion, so a replay set is a one-facet
G-study design and its within-task variance is `sigma2(occasion)`. `ReplaySet.as_variance_components`
hands exactly that to A2's decomposition and to A3's attenuation factor, in the shape they take. An
environment that contributes 40% of the score variance is 40% of the grader's error term, and until
it is measured it sits in the residual being attributed to the model.

**Rung 2: attribution.** Which of sandbox, network, timeout and ordering the spread belongs to.
Every replay carries whatever the environment recorded about itself, and each cause is scored by how
much of the within-task variance conditioning on it removes. These four are the catalogue's list and
they are what the failure logs of every agent harness are full of; nothing here is limited to them,
and a cause the environment can report is a cause this can attribute to.

**The unit is percentage points and it is enforced.** `env.flakiness` is registered in `pp`, so
scores arrive as fractions in [0, 1] and are reported multiplied by 100. A caller handing over
scores outside that range gets `UNIT_MISMATCH` rather than a number a hundred times too large,
because a flakiness of 1,900 pp is exactly the kind of plausible-looking wrong number that survives
into a table. Environments scored on another scale pass `score_scale` and say what it is.

Kill condition, from the catalogue: *if flakiness is under 1 point on ten environments.* A reading of
zero is therefore a real reading and not a failure of the instrument. It says this environment is
deterministic on this machine at this configuration, which is a claim worth having and one that
almost nobody can currently make about their own harness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from reward_lens.core.envelope import EnvelopeSpec
from reward_lens.core.evidence import register_payload
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.quantity import BaselineID
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.types import (
    Access,
    Capability,
    Component,
    GaugeStatus,
    Phase,
    Substrate,
)
from reward_lens.measure.controls._base import ControlInstrument
from reward_lens.measure.metrology.attenuation import RewardVariance

#: Catalogue A7: `Access TASK:QUERY, replay capability`. QUERY is what the source prints; REPLICATE
#: is what "replay capability" is, in the vocabulary the access matrix already has, because a replay
#: is a call under a controlled facet with the facet held at its recorded value. Declaring only
#: QUERY would have let an instrument that needs to run the task twenty times pass a check that
#: establishes it can run it once.
FLAKINESS_ACCESS: dict[Component, Access] = {Component.TASK: Access.QUERY | Access.REPLICATE}

#: Catalogue A7's baseline, and it is the sharpest one in the series: the number that would have
#: been reported is one draw from the distribution this instrument measures.
FLAKINESS_BASELINES: tuple[BaselineID, ...] = ("baseline.single_run",)

#: The source prints `Env determinism claim under test`, which is the reading rather than a
#: precondition. An envelope requiring determinism would make the instrument refuse on exactly the
#: environments it exists to catch, which is B5's defect in a second place.
FLAKINESS_ENVELOPE = EnvelopeSpec(
    unconditional=True,
    justification=(
        "The source records `Env determinism claim under test`. The determinism claim is what "
        "this instrument measures, so making it a precondition would make the instrument refuse on "
        "every environment that has something to report. Replaying a task and recording what came "
        "back assumes nothing about the run."
    ),
)

#: The four the catalogue names for rung 2. Not a closed set: any cause the environment can record
#: per replay is attributable, and these are the names to use when it is one of these.
CATALOGUED_CAUSES: tuple[str, ...] = ("sandbox", "network", "timeout", "ordering")


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplaySet:
    """The same tasks replayed N times, with whatever each replay recorded about itself.

    ``scores`` is (task, replay). ``causes`` maps a cause name to an array of the same shape holding
    whatever the environment reported: a boolean for "the sandbox was reused", an integer id, a
    float for elapsed seconds. Numeric causes are binned into terciles before attribution, so a
    continuous one does not need discretising by the caller.

    NaN in ``scores`` marks a replay that did not complete. That is deliberately *not* the same as a
    score of zero: an environment that crashes on one replay in twenty has a flakiness this
    instrument reports as a completion rate rather than folding a fabricated zero into the spread,
    which is the `verifiers` rubric defect B4 exists to count, one layer up.
    """

    scores: np.ndarray
    task_ids: tuple[str, ...] = ()
    causes: Mapping[str, np.ndarray] = field(default_factory=dict)
    environment: str = ""
    #: What multiplies a score to put it in percentage points. 100 for a fraction in [0, 1].
    score_scale: float = 100.0

    def __post_init__(self) -> None:
        s = np.asarray(self.scores, dtype=np.float64)
        if s.ndim == 1:
            s = s[None, :]
        if s.ndim != 2:
            raise ValueError(f"scores must be (task, replay); got shape {s.shape}")
        object.__setattr__(self, "scores", s)
        if self.task_ids and len(self.task_ids) != s.shape[0]:
            raise ValueError(f"{len(self.task_ids)} task ids for {s.shape[0]} rows of scores")
        for name, arr in self.causes.items():
            a = np.asarray(arr)
            if a.shape != s.shape:
                raise ValueError(
                    f"cause {name!r} has shape {a.shape}; it must be {s.shape}, one value per "
                    f"replay, because a cause recorded per task cannot explain a within-task spread"
                )

    @property
    def n_tasks(self) -> int:
        return int(self.scores.shape[0])

    @property
    def n_replays(self) -> int:
        return int(self.scores.shape[1])

    @property
    def completion_rate(self) -> float:
        return float(np.mean(np.isfinite(self.scores)))

    def as_variance_components(self, *, standardised: bool = True) -> RewardVariance:
        """Rung 1: the replay set as a one-facet G-study design, in the shape A2 and A3 take.

        ``sigma2_err`` is the within-task (occasion) variance, which is what the environment
        contributes. ``sigma2_true`` is the task variance net of it, which is what the evaluation is
        trying to measure. Feeding this straight into `AttenuationFactor` answers a question people
        ask and cannot currently answer: how much of my measured ranking signal survives my own
        harness.

        In score units, not percentage points, because A2's components are variances of the reward
        and mixing scales inside a variance decomposition is the error `Unit` exists to prevent.
        """
        return RewardVariance.from_replicates(
            self.scores, standardised=standardised, source=f"A7 occasion facet, {self.environment}"
        )


def replay(
    run_once: Callable[[], float] | Callable[[int], float],
    *,
    n: int,
    task_ids: Sequence[str] = ("task",),
    environment: str = "",
    score_scale: float = 100.0,
    causes: Callable[[int], Mapping[str, float]] | None = None,
) -> ReplaySet:
    """Call an environment N times with nothing changed, and collect what came back.

    ``run_once`` returns one score, either taking no arguments or taking the replay index. The index
    form exists so a caller can seed something per replay when the point is to *keep* a source of
    variation rather than remove it; passing the index does not mean the replays differ.

    A replay that raises records NaN and the exception is not swallowed into a score. Catching an
    exception and returning zero is precisely the defect the catalogue's B4 counts, and doing it
    inside the instrument that measures environment reliability would be worse than doing it in a
    rubric.
    """
    if n < 2:
        raise ValueError(f"a spread needs at least 2 replays; got {n}")
    takes_index = getattr(run_once, "__code__", None) is not None and (
        run_once.__code__.co_argcount >= 1
    )
    scores = np.full((len(task_ids), n), np.nan)
    collected: dict[str, list[float]] = {}
    for r in range(n):
        try:
            value = run_once(r) if takes_index else run_once()  # type: ignore[call-arg]
        except Exception:  # noqa: BLE001 - recorded as a non-completion, never as a score
            value = float("nan")
        scores[0, r] = float(value)
        if causes is not None:
            for k, v in causes(r).items():
                collected.setdefault(k, []).append(float(v))
    shaped = {
        k: np.asarray(v, dtype=np.float64).reshape(1, n)
        for k, v in collected.items()
        if len(v) == n
    }
    return ReplaySet(
        scores=scores,
        task_ids=tuple(task_ids),
        causes=shaped,
        environment=environment,
        score_scale=score_scale,
    )


# ---------------------------------------------------------------------------
# Rung 0
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Spread:
    """Per-task range and standard deviation, in the units the caller's scores are in."""

    ranges: np.ndarray
    sigmas: np.ndarray
    means: np.ndarray
    minima: np.ndarray
    maxima: np.ndarray
    modal_agreement: np.ndarray
    n_used: np.ndarray

    @property
    def worst_task(self) -> int:
        finite = np.where(np.isfinite(self.ranges), self.ranges, -np.inf)
        return int(np.argmax(finite))


def spread(data: ReplaySet) -> Spread:
    """Rung 0. Range, standard deviation and modal agreement per task, over completed replays.

    ``modal_agreement`` is the fraction of replays landing on the most common score, which is the
    right summary for a discrete outcome where a range hides the shape: an environment that returns
    1.0 nineteen times and 0.0 once has a range of 100 pp and a modal agreement of 95%, and both
    numbers are worth having in front of you.
    """
    s = data.scores
    n_tasks = s.shape[0]
    out = {
        k: np.full(n_tasks, np.nan)
        for k in ("ranges", "sigmas", "means", "minima", "maxima", "modal_agreement")
    }
    used = np.zeros(n_tasks, dtype=np.int64)
    for i in range(n_tasks):
        row = s[i][np.isfinite(s[i])]
        used[i] = row.shape[0]
        if row.shape[0] == 0:
            continue
        out["minima"][i] = float(row.min())
        out["maxima"][i] = float(row.max())
        out["means"][i] = float(row.mean())
        out["ranges"][i] = float(row.max() - row.min())
        out["sigmas"][i] = float(row.std(ddof=1)) if row.shape[0] > 1 else np.nan
        values, counts = np.unique(row, return_counts=True)
        out["modal_agreement"][i] = float(counts.max() / row.shape[0])
    return Spread(
        ranges=out["ranges"],
        sigmas=out["sigmas"],
        means=out["means"],
        minima=out["minima"],
        maxima=out["maxima"],
        modal_agreement=out["modal_agreement"],
        n_used=used,
    )


# ---------------------------------------------------------------------------
# Rung 2
# ---------------------------------------------------------------------------


def attribute(data: ReplaySet, *, n_bins: int = 3) -> tuple[dict[str, float], dict[str, float]]:
    """Rung 2. The share of within-task variance each recorded cause accounts for, and its null.

    Eta-squared computed inside each task and pooled, the same construction A6 uses for its facets,
    so a number from one can be read against a number from the other. Numeric causes are binned into
    ``n_bins`` quantile bins within each task first, which is what makes "elapsed seconds" usable
    without the caller having to decide where the timeout boundary was.

    The second dictionary is the null share: what the cause would explain if it explained nothing.
    A three-level cause over twenty replays explains 11% by chance, and an attribution report that
    printed 11% as a finding would send somebody to fix a timeout that is not the problem. Every
    share here has to be read against its own entry in the second dictionary, and `Flakiness.says`
    stays silent about any cause that does not clear it.

    Shares overlap and do not sum to one. Two causes that fired together are confounded, each claims
    the shared part, and reporting the overlap is honest where a normalised split would invent a
    division the data does not support. An environment that wants them separated has to vary them
    separately, which is a design change rather than an analysis one.
    """
    s = data.scores
    finite = np.isfinite(s)
    total = 0.0
    total_df = 0
    for i in range(s.shape[0]):
        row = s[i][finite[i]]
        if row.shape[0] > 1:
            total += float(np.sum((row - row.mean()) ** 2))
            total_df += row.shape[0] - 1
    out: dict[str, float] = {}
    null: dict[str, float] = {}
    for name, raw in data.causes.items():
        arr = np.asarray(raw)
        between = 0.0
        between_df = 0
        for i in range(s.shape[0]):
            mask = finite[i]
            row = s[i][mask]
            if row.shape[0] < 2:
                continue
            labels = arr[i][mask]
            if labels.dtype.kind == "f" and np.unique(labels).size > n_bins:
                edges = np.quantile(labels, np.linspace(0, 1, n_bins + 1)[1:-1])
                labels = np.searchsorted(edges, labels)
            mean = row.mean()
            levels = np.unique(labels)
            between_df += levels.shape[0] - 1
            for level in levels:
                cell = row[labels == level]
                between += cell.shape[0] * float((cell.mean() - mean) ** 2)
        out[name] = between / total if total > 0.0 else 0.0
        null[name] = between_df / total_df if total_df > 0 else 0.0
    return out, null


# ---------------------------------------------------------------------------
# The reading
# ---------------------------------------------------------------------------


@register_payload
@dataclass
class Flakiness:
    """The spread, in percentage points, with the single-run number it replaces beside it."""

    range_pp: float
    max_range_pp: float
    sigma_pp: float
    min_pp: float
    max_pp: float
    mean_pp: float
    modal_agreement: float
    n_tasks: int
    n_replays: int
    completion_rate: float
    deterministic: bool
    worst_task: str
    occasion_variance: float
    task_variance: float
    attribution: Mapping[str, float] = field(default_factory=dict)
    #: What each cause would explain if it explained nothing. Read every attribution against it.
    attribution_null: Mapping[str, float] = field(default_factory=dict)
    environment: str = ""
    baselines: Mapping[str, float] = field(default_factory=dict)

    def detectable_delta_pp(self, factor: float = 1.0) -> float:
        """The improvement a single-run comparison would need to clear this environment's own spread.

        A blunt instrument on purpose. It is the range, optionally scaled, and it exists so the
        reading can finish the sentence a reader is already forming: my delta was three points, is
        three points enough. A properly powered answer is M10's, and that is what the remedy points
        at; this is the one-line version that fits on a card.
        """
        return factor * self.max_range_pp

    @property
    def _completion_clause(self) -> str:
        """Never omitted. A harness that fails one run in five is flaky whatever its scores do."""
        if self.completion_rate >= 1.0:
            return ""
        return (
            f" {1.0 - self.completion_rate:.1%} of replays did not complete and are excluded from "
            f"the spread rather than scored zero, so the completion rate is the finding here and "
            f"the spread is conditional on it."
        )

    @property
    def says(self) -> str:
        if self.deterministic:
            return (
                f"The same policy on the same task scored {self.mean_pp:.1f}% on every one of "
                f"{self.n_replays} identical runs across {self.n_tasks} task(s). This environment "
                f"is deterministic here, so a measured delta is not competing with its spread."
                f"{self._completion_clause}"
            )
        head = (
            f"The same policy on the same task scores {self.min_pp:.1f}% to {self.max_pp:.1f}% "
            f"across {self.n_replays} identical runs. Any improvement under "
            f"{self.max_range_pp:.1f} points is inside the environment's own spread."
        )
        if self.attribution:
            excess = {
                name: share - self.attribution_null.get(name, 0.0)
                for name, share in self.attribution.items()
            }
            name, gap = max(excess.items(), key=lambda kv: kv[1])
            if gap > 0.0:
                head += (
                    f" The largest attributable cause is {name}, at {self.attribution[name]:.0%} "
                    f"of the within-task variance against a null share of "
                    f"{self.attribution_null.get(name, 0.0):.0%}."
                )
            else:
                head += (
                    f" None of {', '.join(sorted(self.attribution))} explains more than its own "
                    f"null share, so the spread is not attributable to any cause this replay set "
                    f"recorded."
                )
        return head + self._completion_clause


class EnvironmentFlakiness(ControlInstrument):
    """A7. What the same policy scores on the same task when you run it again.

    One instrument over the three rungs, because a replay set is one artifact and the three readings
    are three views of it. ``rung`` is 2; a replay set with no recorded causes reports rung 2 with an
    empty ``attribution``, which says the causes were not recorded rather than that none contributed.
    """

    name = "EnvironmentFlakiness"
    version = "1.0"
    quantity = "env.flakiness"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    requires = FLAKINESS_ACCESS
    #: PROGRAM, from the catalogue. An environment is a program: a sandbox, a test harness, a
    #: simulator. The restriction is what makes the reading a statement about the harness rather
    #: than about whatever is being scored inside it.
    substrates = frozenset({Substrate.PROGRAM})
    phases = frozenset({Phase.PRE_RUN})
    envelope = FLAKINESS_ENVELOPE
    invariance = "group.permutation"
    #: Invariant. Range, standard deviation and modal agreement are functions of the multiset of
    #: replays, and permuting replay order within a task does not change the multiset. A reading
    #: that moved here would mean the estimator had picked up on replay order, which for an
    #: environment is the `ordering` cause leaking into the headline number.
    invariance_relation = INVARIANT
    baselines = FLAKINESS_BASELINES
    rung = 2
    faithful_to = "A7"
    deviations = (
        "the access declaration is `TASK: QUERY | REPLICATE` where the catalogue prints "
        "`TASK:QUERY, replay capability`. REPLICATE is the access matrix's word for the second "
        "clause and declaring only QUERY would let this pass a check that establishes the task can "
        "be run once",
        "a replay that raises is recorded as NaN and excluded, and the completion rate is reported "
        "beside the spread. Scoring a crash as zero would fold a harness failure into the "
        "environment's score distribution, where it would read as variance rather than as breakage",
    )

    def __init__(self, data: ReplaySet | None = None, *, n_bins: int = 3) -> None:
        self.data = data
        self.n_bins = int(n_bins)

    def compute(self) -> Any:
        data = self.data
        if data is None or data.scores.size == 0:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail="no replays were supplied, and a spread over one run is not a spread",
                remedy=(
                    "run the same task with the same policy at least twice and pass the scores as "
                    "ReplaySet(scores=array_of_shape_tasks_by_replays), or use "
                    "`replay(run_once, n=20)` to collect them. Twenty is the number at which a "
                    "range starts to be worth quoting."
                ),
                statistics={"n_tasks": 0, "n_replays": 0},
            )
        usable = np.sum(np.isfinite(data.scores), axis=1)
        if int(usable.max(initial=0)) < 2:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail=(
                    f"no task has two completed replays (the best has "
                    f"{int(usable.max(initial=0))} of {data.n_replays}), so there is nothing to "
                    f"take a range over. This is the single-run number, which is the baseline "
                    f"rather than the reading"
                ),
                remedy=(
                    "replay each task at least twice. If replays are failing rather than missing, "
                    "the completion rate is itself the finding and belongs on the card before the "
                    "spread does."
                ),
                statistics={
                    "n_tasks": data.n_tasks,
                    "n_replays": data.n_replays,
                    "max_completed": int(usable.max(initial=0)),
                    "completion_rate": data.completion_rate,
                },
            )
        finite = data.scores[np.isfinite(data.scores)]
        if data.score_scale == 100.0 and finite.size and (finite.min() < 0.0 or finite.max() > 1.0):
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.UNIT_MISMATCH,
                detail=(
                    f"env.flakiness is registered in percentage points, so scores are expected as "
                    f"fractions in [0, 1] and multiplied by 100. The supplied scores run from "
                    f"{finite.min():.4g} to {finite.max():.4g}, so multiplying would report a "
                    f"spread in a unit nothing else in the registry can be compared against"
                ),
                remedy=(
                    "divide the scores by their own scale before passing them, or pass "
                    "`score_scale=` naming what multiplies a score to make a percentage point. A "
                    "0-to-10 rubric takes score_scale=10."
                ),
                statistics={"min": float(finite.min()), "max": float(finite.max())},
            )

        sp = spread(data)
        causes, causes_null = attribute(data, n_bins=self.n_bins)
        k = data.score_scale
        worst = sp.worst_task
        ranges = sp.ranges[np.isfinite(sp.ranges)]
        sigmas = sp.sigmas[np.isfinite(sp.sigmas)]
        components = data.as_variance_components() if data.n_tasks >= 2 else None
        return Flakiness(
            range_pp=float(np.mean(ranges)) * k if ranges.size else float("nan"),
            max_range_pp=float(ranges.max()) * k if ranges.size else float("nan"),
            sigma_pp=float(np.mean(sigmas)) * k if sigmas.size else float("nan"),
            min_pp=float(sp.minima[worst]) * k,
            max_pp=float(sp.maxima[worst]) * k,
            mean_pp=float(np.nanmean(sp.means)) * k,
            modal_agreement=float(np.nanmean(sp.modal_agreement)),
            n_tasks=data.n_tasks,
            n_replays=data.n_replays,
            completion_rate=data.completion_rate,
            deterministic=bool(ranges.size and np.all(ranges == 0.0)),
            worst_task=(data.task_ids[worst] if worst < len(data.task_ids) else f"task[{worst}]"),
            occasion_variance=components.sigma2_err if components else float("nan"),
            task_variance=components.sigma2_true if components else float("nan"),
            attribution=causes,
            attribution_null=causes_null,
            environment=data.environment,
            # The single-run number: the first completed replay of the worst task, which is the
            # number that would have been reported had nobody run it twice.
            baselines={
                "baseline.single_run": float(
                    next(
                        (v for v in data.scores[worst] if math.isfinite(v)),
                        float("nan"),
                    )
                )
                * k
            },
        )

    def payload(self, computed: Flakiness) -> dict[str, Any]:
        return {
            "range_pp": computed.range_pp,
            "max_range_pp": computed.max_range_pp,
            "sigma_pp": computed.sigma_pp,
            "min_pp": computed.min_pp,
            "max_pp": computed.max_pp,
            "mean_pp": computed.mean_pp,
            "modal_agreement": computed.modal_agreement,
            "n_tasks": computed.n_tasks,
            "n_replays": computed.n_replays,
            "completion_rate": computed.completion_rate,
            "deterministic": computed.deterministic,
            "worst_task": computed.worst_task,
            "occasion_variance": computed.occasion_variance,
            "task_variance": computed.task_variance,
            "attribution": dict(computed.attribution),
            "attribution_null": dict(computed.attribution_null),
            "environment": computed.environment,
            "says": computed.says,
            "baselines": dict(computed.baselines),
        }


def range_from_scores(scores: np.ndarray, n_replays: int) -> float:
    """Flat scores in, mean per-task range out. The subject of the generated invariance test."""
    s = np.asarray(scores, dtype=np.float64).reshape(-1, n_replays)
    ranges = spread(ReplaySet(scores=s, score_scale=1.0)).ranges
    return float(np.mean(ranges[np.isfinite(ranges)]))


__all__ = [
    "CATALOGUED_CAUSES",
    "FLAKINESS_ACCESS",
    "FLAKINESS_BASELINES",
    "FLAKINESS_ENVELOPE",
    "EnvironmentFlakiness",
    "Flakiness",
    "ReplaySet",
    "Spread",
    "attribute",
    "range_from_scores",
    "replay",
    "spread",
]

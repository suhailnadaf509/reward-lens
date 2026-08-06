"""Backtesting as a first-class object: the run corpus and its splits.

A forecaster is not validated on one run, and it is not validated by holding out items.

**Splitting over items is the mistake every competitor makes, and splitting over runs and time is
the only split that means anything for a temporal claim.** The reason is not statistical fussiness.
Two rollouts from the same run at neighbouring steps share a policy, a grader, a task distribution
and a random seed lineage; holding one out and training on the other measures whether the predictor
can interpolate inside a single trajectory, which is a question nobody asked. The claim being made
is "this predictor, fitted on runs that finished before yours started, forecasts yours", and the
only split that estimates it is one whose training set is entirely earlier in time and entirely
different in run.

Walk-forward and purged cross-validation are the quantitative-finance discipline for exactly this
problem and they transfer without modification, embargo included. The embargo is the part people
drop: a test block that begins the instant the training block ends still leaks, because a feature
computed over a trailing window at the first test step was computed partly from training steps, and
a label defined over a forward window at the last training step resolves inside the test block. The
gap has to be at least as wide as the longer of those two windows, and `embargo_steps` is where you
say how wide.

The AISI series gives a corpus for free: 48 `cc-*` model repositories spanning seven KL values
across four hint difficulties on four base models, plus 150 step-10 checkpoints of one run, plus two
full rollout series with per-rollout labels. It also has five traps, and they are recorded on
`AISI_TRAPS` below rather than in a wiki, because each of them costs a day and none is visible until
you have paid for it once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Literal, Mapping, Sequence

from reward_lens.forecast.schema import ForecastError, ReferenceClassID

RunID = str

SplitKind = Literal["walk_forward", "purged_kfold", "leave_one_run_out"]


#: What the AISI checkpoint and rollout series will cost you if nobody tells you first. Each of
#: these cost a day the first time, which is the only reason they are worth writing down.
AISI_TRAPS: tuple[str, ...] = (
    "The checkpoints are subfolders in one repository holding LoRA adapters, loaded with "
    "`subfolder=`. They are not branches and not merged weights, so the base model has to be loaded "
    "too and a `from_pretrained` on the repo alone gets you the base model with none of the "
    "training applied, silently.",
    "The two rollout series are different lengths, 401 and 404 eval logs. Anything that zips them "
    "positionally is off by three from wherever the extra logs sit.",
    "Labels are int64 carrying 1, 0 or null. A naive `.sum()` treats null as zero and reports a "
    "hack rate that is too low by the abstention rate; count the non-null denominator explicitly.",
    "`hack_config` is a JSON string, not a nested object. It has to be parsed before any field in "
    "it can be read, and a `.get` on the raw column returns None for every key.",
    "`rollout_index` is per eval file, not per row. Joining on it across files collides every "
    "index with every other file's, so the join key is the pair (eval file, rollout_index).",
)


@dataclass(frozen=True)
class Fold:
    """One train/test split, in run and step coordinates.

    ``embargo`` is carried on the fold rather than only on the corpus so that a fold read on its own
    still says how wide the gap was. A fold that cannot state its own embargo cannot be audited.
    """

    index: int
    kind: SplitKind
    train: tuple[tuple[RunID, int], ...]
    test: tuple[tuple[RunID, int], ...]
    embargo: int
    purged: int = 0
    detail: str = ""

    @property
    def train_runs(self) -> tuple[RunID, ...]:
        return tuple(dict.fromkeys(r for r, _ in self.train))

    @property
    def test_runs(self) -> tuple[RunID, ...]:
        return tuple(dict.fromkeys(r for r, _ in self.test))

    @property
    def n_train(self) -> int:
        return len(self.train)

    @property
    def n_test(self) -> int:
        return len(self.test)

    @property
    def is_run_disjoint(self) -> bool:
        """Whether no run appears on both sides. False for walk-forward, which is expected."""
        return not (set(self.train_runs) & set(self.test_runs))

    def render(self) -> str:
        return (
            f"fold {self.index} ({self.kind}): train {self.n_train} points over "
            f"{len(self.train_runs)} runs, test {self.n_test} over {len(self.test_runs)}, "
            f"embargo {self.embargo}"
            + (f", purged {self.purged}" if self.purged else "")
            + (f"  {self.detail}" if self.detail else "")
        )


@dataclass(frozen=True)
class RunCorpus:
    """A population of runs to backtest against, and how to split it.

    ``runs`` names the members. ``steps`` maps each run to its step axis, which the two temporal
    splits need; a corpus with no step axis can only do `leave_one_run_out`, and asking it for a
    walk-forward raises rather than inventing one.

    ``embargo_steps`` is the gap between train and test. It is not optional and it is not zero by
    accident: `__post_init__` refuses a temporal split with a zero embargo unless the caller says
    out loud that the features and labels are both instantaneous, which is rare and is worth having
    to assert.
    """

    runs: tuple[RunID, ...]
    reference_class: ReferenceClassID
    split: SplitKind
    embargo_steps: int
    #: Each run's step axis. Required by `walk_forward` and `purged_kfold`.
    steps: Mapping[RunID, tuple[int, ...]] = field(default_factory=dict)
    #: Set only when features and labels are both instantaneous, so a zero embargo is correct.
    instantaneous: bool = False

    def __post_init__(self) -> None:
        if not self.runs:
            raise ForecastError(
                "a run corpus with no runs cannot backtest anything. One run is also not a corpus: "
                "a forecaster validated on one run has a result about that run."
            )
        if len(set(self.runs)) != len(self.runs):
            dupes = sorted({r for r in self.runs if list(self.runs).count(r) > 1})
            raise ForecastError(
                f"run corpus lists {dupes} more than once. A duplicated run is counted twice in "
                f"every fold and inflates n without adding information, which is the same defect "
                f"as splitting over items one level up."
            )
        if self.embargo_steps < 0:
            raise ForecastError(f"embargo_steps cannot be negative; got {self.embargo_steps}")
        if self.split != "leave_one_run_out":
            missing = [r for r in self.runs if r not in self.steps]
            if missing:
                raise ForecastError(
                    f"split {self.split!r} is temporal and needs a step axis, and "
                    f"{len(missing)} run(s) carry none: {missing[:3]}. Supply `steps` mapping each "
                    f"run id to its step indices, or use `leave_one_run_out`, which needs no time "
                    f"axis because it holds out whole runs."
                )
            if self.embargo_steps == 0 and not self.instantaneous:
                raise ForecastError(
                    "a temporal split with a zero embargo leaks across the boundary unless every "
                    "feature and every label is instantaneous. A feature computed over a trailing "
                    "window at the first test step was computed partly from training steps, and a "
                    "label defined over a forward window at the last training step resolves inside "
                    "the test block. Set embargo_steps to at least the longer of the two windows, "
                    "or pass instantaneous=True to assert that both are zero-width."
                )

    @property
    def n_runs(self) -> int:
        return len(self.runs)

    @property
    def n_points(self) -> int:
        return sum(len(self.steps.get(r, ())) for r in self.runs)

    def points(self) -> tuple[tuple[RunID, int], ...]:
        """Every (run, step) point in the corpus, run-major then step-ascending."""
        return tuple(p for run in self.runs for p in self._points_of(run))

    def _points_of(self, run: RunID) -> tuple[tuple[RunID, int], ...]:
        """A run's points, or one point standing for the whole run when it has no step axis.

        A corpus of finished runs with no recorded step axis is a real case: the AISI checkpoint
        ladder read at run granularity is one, and so is any corpus assembled from per-run summary
        rows. Such a run contributes exactly one point, which `leave_one_run_out` can split and the
        two temporal splits refuse in `__post_init__` rather than pretending to.
        """
        axis = tuple(sorted(self.steps.get(run, ())))
        if not axis:
            return ((run, 0),)
        return tuple((run, s) for s in axis)

    # -- the three splits ---------------------------------------------------

    def folds(self, *, n_folds: int = 5, min_train: int = 1) -> tuple[Fold, ...]:
        """The corpus's folds under its declared split."""
        if self.split == "leave_one_run_out":
            return self._leave_one_run_out()
        if self.split == "walk_forward":
            return self._walk_forward(n_folds=n_folds, min_train=min_train)
        return self._purged_kfold(n_folds=n_folds)

    def _leave_one_run_out(self) -> tuple[Fold, ...]:
        """Hold out one whole run. The only split that estimates transfer to an unseen run.

        This is the split a claim of the form "this predictor works on runs like yours" needs,
        because "like yours" means a run the fit never saw, not a step the fit never saw. It is also
        the most expensive and the most pessimistic, which is why the other two exist.
        """
        folds: list[Fold] = []
        for i, held in enumerate(self.runs):
            test = self._points_of(held)
            train = tuple(p for r in self.runs if r != held for p in self._points_of(r))
            folds.append(
                Fold(
                    index=i,
                    kind="leave_one_run_out",
                    train=train,
                    test=test,
                    embargo=self.embargo_steps,
                    detail=f"held out {held}",
                )
            )
        return tuple(folds)

    def _walk_forward(self, *, n_folds: int, min_train: int) -> tuple[Fold, ...]:
        """Train on everything before `t`, test on `[t + embargo, t + embargo + block)`.

        Expanding window rather than rolling, because the question a walk-forward answers here is
        "how would this have performed if run live", and a live forecaster does not throw away last
        month's runs. A rolling window is the right choice when the process is known to drift faster
        than the window; that is a claim about the corpus, and one somebody should have to make.
        """
        points = self.points()
        if len(points) < n_folds + min_train:
            raise ForecastError(
                f"a walk-forward over {len(points)} points cannot make {n_folds} folds with at "
                f"least {min_train} training points each. Add runs, or reduce n_folds."
            )
        # Order strictly by step across the corpus, so the training set of every fold is genuinely
        # earlier in time than its test set even when runs interleave.
        ordered = sorted(points, key=lambda p: (p[1], p[0]))
        block = max(1, (len(ordered) - min_train) // n_folds)
        folds: list[Fold] = []
        for i in range(n_folds):
            cut = min_train + i * block
            if cut >= len(ordered):
                break
            boundary_step = ordered[cut - 1][1]
            train = tuple(p for p in ordered[:cut])
            test_start = cut
            test = tuple(
                p
                for p in ordered[test_start : test_start + block]
                if p[1] > boundary_step + self.embargo_steps
            )
            if not test:
                continue
            folds.append(
                Fold(
                    index=i,
                    kind="walk_forward",
                    train=train,
                    test=test,
                    embargo=self.embargo_steps,
                    detail=(
                        f"train through step {boundary_step}, test from step "
                        f"{test[0][1]} (gap {test[0][1] - boundary_step} steps)"
                    ),
                )
            )
        if not folds:
            raise ForecastError(
                f"every walk-forward fold was empty after applying the {self.embargo_steps}-step "
                f"embargo. The embargo is wider than the corpus's step spacing, so no test point "
                f"survives it. Either the embargo is too large for this corpus or the corpus is "
                f"too short to support a temporal claim at this window width."
            )
        return tuple(folds)

    def _purged_kfold(self, *, n_folds: int) -> tuple[Fold, ...]:
        """K contiguous test blocks, with the training points around each one purged and embargoed.

        López de Prado's construction, unmodified. Purging removes training points whose label
        window overlaps the test block; the embargo removes training points immediately *after* it,
        which purging alone does not catch because a trailing-window feature at a post-test training
        point reads test-block data. Both are applied here in step coordinates, symmetrically.
        """
        points = self.points()
        if len(points) < n_folds * 2:
            raise ForecastError(
                f"a purged k-fold over {len(points)} points cannot make {n_folds} folds with a "
                f"test and a training side each."
            )
        ordered = sorted(points, key=lambda p: (p[1], p[0]))
        size = len(ordered) // n_folds
        folds: list[Fold] = []
        for i in range(n_folds):
            lo = i * size
            hi = len(ordered) if i == n_folds - 1 else (i + 1) * size
            test = tuple(ordered[lo:hi])
            lo_step = min(s for _, s in test)
            hi_step = max(s for _, s in test)
            keep: list[tuple[RunID, int]] = []
            purged = 0
            for run, step in ordered[:lo] + ordered[hi:]:
                if lo_step - self.embargo_steps <= step <= hi_step + self.embargo_steps:
                    purged += 1
                    continue
                keep.append((run, step))
            if not keep:
                raise ForecastError(
                    f"fold {i} of the purged k-fold has no training points left after purging and "
                    f"embargoing {self.embargo_steps} steps either side of test steps "
                    f"[{lo_step}, {hi_step}]. The embargo consumes the corpus; reduce n_folds or "
                    f"widen the corpus rather than shrinking the embargo, which is the one knob "
                    f"here that trades a real guarantee for a bigger n."
                )
            folds.append(
                Fold(
                    index=i,
                    kind="purged_kfold",
                    train=tuple(keep),
                    test=test,
                    embargo=self.embargo_steps,
                    purged=purged,
                    detail=f"test steps [{lo_step}, {hi_step}], {purged} training points purged",
                )
            )
        return tuple(folds)

    def render(self) -> str:
        lines = [
            f"RunCorpus {self.reference_class}: {self.n_runs} runs, {self.n_points} points, "
            f"split={self.split}, embargo={self.embargo_steps}",
        ]
        for fold in self.folds():
            lines.append("    " + fold.render())
        return "\n".join(lines)


def corpus_from_runs(
    runs: Sequence[object],
    *,
    reference_class: ReferenceClassID,
    split: SplitKind = "leave_one_run_out",
    embargo_steps: int = 0,
    instantaneous: bool = False,
) -> RunCorpus:
    """Build a corpus from `record.schema.Run` objects, reading each one's step axis.

    Takes the `Run` objects rather than paths so the caller decides what is in the corpus. The step
    axis comes from `Run.steps.indices`, which does not decode a single step, so building a corpus
    over a 200-step run costs nothing.
    """
    ids: list[RunID] = []
    steps: dict[RunID, tuple[int, ...]] = {}
    for run in runs:
        rid = str(getattr(run, "id", ""))
        if not rid:
            raise ForecastError(f"{run!r} has no run id; a corpus member has to be nameable")
        ids.append(rid)
        stream = getattr(run, "steps", None)
        steps[rid] = tuple(getattr(stream, "indices", ()) or ())
    return RunCorpus(
        runs=tuple(ids),
        reference_class=reference_class,
        split=split,
        embargo_steps=embargo_steps,
        steps=steps,
        instantaneous=instantaneous,
    )


def iter_folds(corpus: RunCorpus, **kwargs: object) -> Iterator[Fold]:
    """The folds, lazily, for a backtest loop that does not want them all in memory."""
    yield from corpus.folds(**kwargs)  # type: ignore[arg-type]


__all__ = [
    "AISI_TRAPS",
    "Fold",
    "RunCorpus",
    "SplitKind",
    "corpus_from_runs",
    "iter_folds",
]

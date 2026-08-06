"""What a baseline is, and the one interface all six of them share (M3).

A baseline is the score a claim has to beat before it is a claim. Not a sanity check you run at
the end, not a footnote: the number that decides whether a reported detector found anything at
all. The case that puts this at the foundation rather than at the end is a published one, and it
is short. A probe was reported at AUC 0.998 on a hack-detection task. A zero-parameter string match
scores 100% on the same task. The author's own summary of the result is "the probe detects the
hack, and the detection is empty."

So baselines here are first-class competitors with the same shape as an instrument, not helper
functions. Each one declares what it reads, refuses when it cannot read it, never raises, and
returns a reading comparable to every other baseline's. That shape is the entire design
constraint on this module: a baseline that is awkward to run does not get run, and a bank nobody
runs gates nothing.

Three pieces live here. `DetectionTask` is the input, defined as a small local protocol because
these six need nothing from ``record/`` and should not depend on it. `BaselineScore` is the
reading, and it carries per-item scores rather than only a summary, because the error correlation
that `IncrementalValidity` needs cannot be recovered from a summary. `run_bank` and
`compare_against_baselines` are the two calls a claim makes.

**On what "matched" means.** The comparison does not use a hand-picked margin. Two methods are
matched when a paired bootstrap of their per-item difference cannot exclude zero, which makes the
verdict a function of the data and the stated confidence level rather than of a constant somebody
chose. When the claim carries lineage labels the bootstrap resamples at the seed level via
`stats.ess.cluster_bootstrap`, so cloning a stimulus fifty times does not buy a win.

Attribution: the AUC 0.998 result, the string match that matches it, and the author's summary are
a published finding by other people. Nothing in this module measured them.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from reward_lens.core.quantity import FREE, BaselineID, CostModel
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.stats.roc import roc_pr

# ---------------------------------------------------------------------------
# The input, as a local protocol
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class DetectionTask:
    """One binary detection problem, in the only shape six dumb baselines need.

    This is deliberately small. None of these six needs ``record/``: a baseline reads text, a
    label, and at most one logged scalar per item. Keeping the input local rather than importing
    the full record keeps the bank available to every claim, and the conversion from a `Run` to a
    `DetectionTask` is one function at the call site.

    ``labels`` is the only required field. Everything else is a feature some subset of the bank
    reads, and a baseline whose feature is absent returns a refusal naming what to supply rather
    than a zero.

    ``markers`` is what makes the string-match baseline zero-parameter. Supply the literal
    substrings a hack is known to contain and the baseline fits nothing; leave it empty and the
    baseline mines its own markers on a train split, stamps ``fitted=True``, and reports how many
    parameters that cost. Both are honest; only the first is the one the published case is about.
    """

    labels: np.ndarray
    texts: tuple[str, ...] = ()
    prompts: tuple[str, ...] = ()
    #: One logged scalar per item, read by `baseline.gradnorm_peak`. A logged series, not a
    #: gradient computation: nothing in this package differentiates anything.
    series: np.ndarray | None = None
    #: Literal substrings a positive is known to contain, for the zero-parameter string match.
    markers: tuple[str, ...] = ()
    #: One lineage label per item, so the comparison bootstrap resamples at the seed level.
    seed_labels: tuple[Any, ...] = ()
    #: A callable that scores one rendered scaffold prompt. Absent means the black-box baseline
    #: refuses rather than being silently dropped from the bank.
    judge: Callable[[str], float] | None = None
    name: str = ""

    def __post_init__(self) -> None:
        labels = np.asarray(self.labels).ravel().astype(int)
        object.__setattr__(self, "labels", labels)
        n = labels.size
        for attr in ("texts", "prompts", "seed_labels"):
            value = getattr(self, attr)
            if value and len(value) != n:
                raise ValueError(
                    f"DetectionTask.{attr} has {len(value)} entries and labels has {n}. A "
                    f"baseline aligned to the wrong items scores noise and reports it as a floor."
                )
        if self.series is not None:
            series = np.asarray(self.series, dtype=np.float64).ravel()
            if series.size != n:
                raise ValueError(
                    f"DetectionTask.series has {series.size} entries and labels has {n}."
                )
            object.__setattr__(self, "series", series)

    @property
    def n(self) -> int:
        return int(self.labels.size)

    @property
    def positive_rate(self) -> float:
        return float(np.mean(self.labels == 1)) if self.n else float("nan")

    @property
    def is_degenerate(self) -> bool:
        """One class only. Every discrimination statistic is undefined here, so every baseline
        refuses rather than returning the 0.5 that a naive AUC implementation would."""
        return self.n == 0 or np.unique(self.labels).size < 2

    def has(self, feature: str) -> bool:
        value = getattr(self, feature, None)
        if value is None:
            return False
        if isinstance(value, np.ndarray):
            return value.size > 0
        return bool(value)


# ---------------------------------------------------------------------------
# The reading
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class BaselineScore:
    """One baseline's reading, in a form comparable across all six.

    ``auroc`` is the headline and it is comparable because every baseline reduces to a per-item
    score and the same rank statistic is applied to it. ``scores`` is kept because incremental
    validity asks for the correlation between two methods' *errors*, and no summary produces that.

    ``n_parameters`` is the field that carries the argument. A string match with three literal
    markers and a TF-IDF logistic regression with 40,000 coefficients are both baselines, and a
    claim that beats the second while losing to the first has learned nothing worth the compute.
    """

    baseline: BaselineID
    auroc: float
    accuracy: float
    n: int
    n_parameters: int
    scores: np.ndarray
    #: +1 when a larger feature value means a positive, -1 when it means a negative. Fitted on a
    #: held-out half for the single-scalar baselines, so orienting costs a split rather than a
    #: free look at the test set.
    direction: int = 1
    wall_ms: float = 0.0
    fitted: bool = False
    detail: str = ""

    @property
    def is_refusal(self) -> bool:
        return False

    def render(self) -> str:
        params = f"{self.n_parameters:,} parameter" + ("" if self.n_parameters == 1 else "s")
        return (
            f"{self.baseline:<28} AUROC {self.auroc:.4f}  acc {self.accuracy:.4f}  "
            f"n={self.n}  {params}  {self.wall_ms:.2f} ms"
        )


#: What a baseline returns. A refusal is a value here for the same reason it is one everywhere in
#: this library: a baseline that quietly returns 0.5 when its input is missing turns a gate into a
#: decoration, and the reader cannot tell the two apart from the number.
BaselineReading = Any  # statically: BaselineScore | Refusal


def is_scored(reading: Any) -> bool:
    return isinstance(reading, BaselineScore)


# ---------------------------------------------------------------------------
# The protocol and a base class
# ---------------------------------------------------------------------------


@runtime_checkable
class Baseline(Protocol):
    """The interface all six share, deliberately shaped like `Instrument`.

    ``id`` is the `BaselineID` an instrument names in its ``baselines`` tuple. ``reads`` names the
    `DetectionTask` fields this baseline needs, which is what makes the refusal specific enough to
    act on. ``cost`` is stated because a comparison between a 1 ms baseline and a 4 s judge is a
    finding about the judge, not a footnote about runtime.
    """

    id: BaselineID
    name: str
    version: str
    reads: tuple[str, ...]
    cost: CostModel

    def run(self, task: DetectionTask) -> BaselineReading: ...


class BaseBaseline:
    """Preflight, then score. The split mirrors `Instrument.preflight` / `Instrument.estimate`.

    A subclass implements ``score``, which may assume its declared ``reads`` are present and the
    labels are non-degenerate. Everything a baseline can anticipate is checked here and returned
    as a `Refusal` with a remedy, so no subclass has to remember to.
    """

    id: BaselineID = "baseline.unnamed"
    name: str = "unnamed"
    version: str = "1.0"
    reads: tuple[str, ...] = ()
    cost: CostModel = FREE
    #: What this baseline needs that a `DetectionTask` cannot carry, named for the remedy.
    supply: str = ""

    def score(self, task: DetectionTask) -> BaselineScore:  # pragma: no cover - abstract
        raise NotImplementedError

    def unavailable(self, task: DetectionTask) -> str | None:
        """A condition beyond a missing field that makes this baseline's number meaningless.

        Returns the detail of the refusal, or None. The case that needs it is a fitted baseline
        with too few items of the minority class to hold anything out: an in-sample TF-IDF
        logistic regression scores near 1.0 on any labelling at all, so a bank that reported it
        would hand every claim an unbeatable comparator and every claim would fail. Refusing is
        the honest answer and it names the sample size that would fix it.
        """
        return None

    def run(self, task: DetectionTask) -> BaselineReading:
        if task.is_degenerate:
            return Refusal(
                instrument=self.id,
                reason=RefusalReason.ESS_BELOW_FLOOR,
                detail=(
                    f"the task has {task.n} items and "
                    f"{int(np.unique(task.labels).size)} distinct label(s); a discrimination "
                    f"statistic is undefined with one class"
                ),
                remedy=(
                    "supply items of both classes. A bank run on a single-class task reports 0.5 "
                    "for every baseline, and a claim compared against that has been compared "
                    "against nothing."
                ),
                statistics={"n": task.n, "positive_rate": task.positive_rate},
            )
        missing = [f for f in self.reads if not task.has(f)]
        if missing:
            return Refusal(
                instrument=self.id,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail=f"needs DetectionTask.{', DetectionTask.'.join(missing)}, which is empty",
                remedy=(
                    self.supply
                    or f"populate {', '.join(missing)} on the task, or record this baseline as "
                    f"unavailable for this claim. A baseline dropped from the bank without a "
                    f"reason is a baseline nobody knows was skipped."
                ),
                statistics={"missing": list(missing)},
            )
        blocked = self.unavailable(task)
        if blocked is not None:
            return Refusal(
                instrument=self.id,
                reason=RefusalReason.ESS_BELOW_FLOOR,
                detail=blocked,
                remedy=(
                    "add items of the minority class until a stratified hold-out can be formed, "
                    "or record this baseline as unavailable at this n. An in-sample fit reported "
                    "as a baseline is an unbeatable comparator that measures nothing."
                ),
                statistics={"n": task.n, "positive_rate": task.positive_rate},
            )
        started = time.perf_counter()
        out = self.score(task)
        elapsed = (time.perf_counter() - started) * 1000.0
        return BaselineScore(
            baseline=out.baseline,
            auroc=out.auroc,
            accuracy=out.accuracy,
            n=out.n,
            n_parameters=out.n_parameters,
            scores=out.scores,
            direction=out.direction,
            wall_ms=elapsed,
            fitted=out.fitted,
            detail=out.detail,
        )


# ---------------------------------------------------------------------------
# Shared scoring helpers
# ---------------------------------------------------------------------------


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC via the rank identity in `stats.roc`, which handles ties at exactly 0.5."""
    return float(roc_pr(np.asarray(scores, dtype=np.float64), np.asarray(labels).astype(int)).auc)


def accuracy_at_midpoint(scores: np.ndarray, labels: np.ndarray) -> float:
    """Accuracy at the threshold halfway between the two class-mean scores.

    A fixed rule rather than the best threshold on the test set. Picking the threshold that
    maximises test accuracy is a free parameter fitted on the evaluation data, and a baseline that
    takes one is no longer the dumb comparator it is being used as.
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(int)
    if np.unique(y).size < 2:
        return float("nan")
    threshold = 0.5 * (float(s[y == 1].mean()) + float(s[y == 0].mean()))
    return float(np.mean((s > threshold).astype(int) == y))


def stratified_folds(
    labels: np.ndarray,
    k: int,
    *,
    seed: int = 0,
    groups: Sequence[Any] | None = None,
) -> list[np.ndarray]:
    """`k` index folds with each class spread as evenly as the counts allow.

    Written in numpy rather than pulled from scikit-learn so that importing this package costs
    nothing: only the TF-IDF baseline needs sklearn, and it imports it when it runs.

    ``groups`` is a lineage label per item, and when it is supplied **every item sharing a label
    lands in the same fold**. Without it, a corpus built by perturbing each seed text into a
    positive and a negative variant leaks: the two near-identical members of a lineage fall into
    different folds, the model memorises the shared body, and the baseline reports the memorised
    label as out-of-sample skill. That is not hypothetical. On a synthetic corpus whose only
    generalisable signal is chance, the ungrouped split returns AUROC 1.0; the same bug shows up
    from the other side on 1,542 real receipt transcripts as 0.026, anti-predictive by the same
    mechanism with the twin carrying the opposite label. Both are the same failure and both read
    as a finding.

    `DetectionTask.seed_labels` is the field to pass, and the comparison bootstrap already
    resamples at that level, so passing it here makes the split and the interval agree about what
    an independent observation is.

    Grouping is best-effort on balance and exact on containment: lineages are assigned largest
    first to whichever fold currently holds fewest items of that lineage's dominant class. Class
    balance across folds is therefore approximate when lineages are large, and containment is not
    approximate, because a leak is a wrong number and an imbalance is a wider interval.
    """
    y = np.asarray(labels).astype(int).ravel()
    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(k)]

    if groups is None or len(groups) == 0:
        for cls in np.unique(y):
            idx = np.flatnonzero(y == cls)
            idx = idx[rng.permutation(idx.size)]
            for position, i in enumerate(idx):
                folds[position % k].append(int(i))
        return [np.asarray(sorted(f), dtype=np.intp) for f in folds]

    if len(groups) != y.size:
        raise ValueError(
            f"stratified_folds got {len(groups)} group labels for {y.size} items. A fold split "
            f"aligned to the wrong items does not contain the lineages it is supposed to contain."
        )
    members: dict[Any, list[int]] = {}
    for i, g in enumerate(groups):
        members.setdefault(g, []).append(i)
    keys = list(members)
    order = rng.permutation(len(keys))
    keys = [keys[i] for i in order]
    keys.sort(key=lambda g: -len(members[g]))

    load = np.zeros((k, max(int(y.max()) + 1, 2)), dtype=np.int64)
    for g in keys:
        idx = members[g]
        dominant = int(np.bincount(y[idx], minlength=load.shape[1]).argmax())
        target = int(np.lexsort((load.sum(axis=1), load[:, dominant]))[0])
        folds[target].extend(idx)
        for i in idx:
            load[target, int(y[i])] += 1
    return [np.asarray(sorted(f), dtype=np.intp) for f in folds]


def oriented_score(
    feature: np.ndarray, labels: np.ndarray, *, seed: int = 0, k: int = 2
) -> tuple[np.ndarray, int]:
    """Orient a single scalar feature out of fold, and return scores aligned to every item.

    A length baseline and an n-gram-diversity baseline are single numbers with an unknown sign:
    longer might mean hacked, or might mean careful. Choosing the sign that wins on the evaluation
    set is a fitted parameter taken for free, and near chance it is worth about as much as a coin
    flip in AUROC. So each fold's sign is fitted on the other folds and applied to it, which keeps
    every item's score out of sample and costs one parameter, reported.

    When the folds disagree about the sign the pooled scores are inconsistent and the AUROC falls
    toward 0.5. That is the correct outcome rather than a defect: a feature whose direction is not
    stable across a split does not carry a direction.
    """
    f = np.asarray(feature, dtype=np.float64).ravel()
    y = np.asarray(labels).astype(int).ravel()
    folds = stratified_folds(y, k, seed=seed)
    out = np.empty_like(f)
    signs: list[int] = []
    for fold in folds:
        mask = np.ones(f.size, dtype=bool)
        mask[fold] = False
        train_y = y[mask]
        if fold.size == 0:
            continue
        if np.unique(train_y).size < 2:
            sign = 1 if auroc(f, y) >= 0.5 else -1
        else:
            sign = 1 if auroc(f[mask], train_y) >= 0.5 else -1
        signs.append(sign)
        out[fold] = sign * f[fold]
    direction = 1 if sum(signs) >= 0 else -1
    return out, direction


# ---------------------------------------------------------------------------
# The bank
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class BaselineBank:
    """What all six said about one task, including the ones that could not answer.

    Refusals are kept rather than dropped. "Five baselines ran and the sixth had no logged series"
    is a different state from "five baselines ran", and only the first tells a reader that the
    gradient-norm comparison is missing rather than passed.
    """

    task_name: str
    readings: Mapping[BaselineID, BaselineReading] = field(default_factory=dict)

    def scored(self) -> dict[BaselineID, BaselineScore]:
        return {k: v for k, v in self.readings.items() if isinstance(v, BaselineScore)}

    def refusals(self) -> dict[BaselineID, Refusal]:
        return {k: v for k, v in self.readings.items() if isinstance(v, Refusal)}

    def best(self) -> BaselineScore | None:
        """The strongest baseline by AUROC. None when every one of them refused."""
        candidates = [s for s in self.scored().values() if np.isfinite(s.auroc)]
        return max(candidates, key=lambda s: s.auroc) if candidates else None

    def as_mapping(self) -> dict[BaselineID, float]:
        """The `{BaselineID: float}` shape `Evidence.baselines` carries."""
        return {k: v.auroc for k, v in self.scored().items()}

    def render(self) -> str:
        lines = [f"dumb-baseline bank on {self.task_name or 'an unnamed task'}"]
        for score in sorted(self.scored().values(), key=lambda s: -s.auroc):
            lines.append("    " + score.render())
        for bid, refusal in sorted(self.refusals().items()):
            lines.append(f"    {bid:<28} {refusal.reason.name}: {refusal.detail}")
        best = self.best()
        if best is not None:
            lines.append(f"    best: {best.baseline} at AUROC {best.auroc:.4f}")
        return "\n".join(lines)


Verdict = Literal["baseline_wins", "matched", "beats_baselines", "no_baseline"]


@dataclass(frozen=True, eq=False)
class BaselineComparison:
    """A claim, placed against the strongest thing that did not need it.

    ``verdict`` is the field the card prints, and ``matched`` is the one that exists because of the
    published case. A detector a string match ties is matched, not a win, and this type refuses to
    render it any other way.
    """

    own: float
    best_baseline: BaselineID | None
    best_score: float
    margin: float
    ci_low: float
    ci_high: float
    ci_level: float
    verdict: Verdict
    n_baselines_scored: int
    n_baselines_refused: int
    detail: str = ""

    @property
    def is_win(self) -> bool:
        return self.verdict == "beats_baselines"

    def render(self) -> str:
        if self.verdict == "no_baseline":
            return (
                "no baseline scored on this task, so the claim has not been compared against "
                "anything. This is a lint failure, not a caveat."
            )
        head = (
            f"{self.own:.4f} against {self.best_baseline} at {self.best_score:.4f}; "
            f"margin {self.margin:+.4f} "
            f"[{self.ci_low:+.4f}, {self.ci_high:+.4f}] at {self.ci_level:.0%}"
        )
        verdicts = {
            "beats_baselines": "beats every baseline in the bank",
            "matched": (
                f"MATCHED by {self.best_baseline}. The interval on the difference contains zero, "
                f"so this is a tie and reporting it as a win would be reporting the baseline's "
                f"result under another name."
            ),
            "baseline_wins": (
                f"{self.best_baseline} WINS. The claim is worse than a comparator that needed "
                f"none of its machinery."
            ),
        }
        return f"{head}\n    {verdicts[self.verdict]}" + (
            f"\n    {self.detail}" if self.detail else ""
        )


def compare_against_baselines(
    own_scores: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    bank: BaselineBank,
    *,
    seed_labels: Sequence[Any] = (),
    ci: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> BaselineComparison:
    """Place a claim's own per-item scores against the bank's best, with an interval.

    The margin is `own AUROC - best baseline AUROC`, and its interval comes from resampling items
    (or lineages, when ``seed_labels`` is supplied) and recomputing both AUROCs on the same
    resample, which preserves the pairing. An interval containing zero is a tie and the verdict
    says so, which is the whole point of the type: the failure this gates against is a paper
    reporting 0.998 against a 1.000 baseline as a success.

    There is no margin constant here on purpose. A hand-picked "beats by 0.02" threshold is a
    decision about what counts as a win made before anybody saw the variance, and the variance is
    the thing that decides it.
    """
    own = np.asarray(own_scores, dtype=np.float64).ravel()
    y = np.asarray(labels).astype(int).ravel()
    own_auc = auroc(own, y)
    best = bank.best()
    n_scored = len(bank.scored())
    n_refused = len(bank.refusals())
    if best is None:
        return BaselineComparison(
            own=own_auc,
            best_baseline=None,
            best_score=float("nan"),
            margin=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            ci_level=ci,
            verdict="no_baseline",
            n_baselines_scored=0,
            n_baselines_refused=n_refused,
            detail="every baseline in the bank refused; see the bank's refusals for what to supply",
        )

    rival = np.asarray(best.scores, dtype=np.float64).ravel()
    margin = own_auc - best.auroc
    lo, hi = _paired_margin_interval(
        own, rival, y, seed_labels=seed_labels, ci=ci, n_resamples=n_resamples, seed=seed
    )
    if not (np.isfinite(lo) and np.isfinite(hi)):
        verdict: Verdict = "matched"
        detail = (
            "the margin interval could not be formed (too few independent lineages to resample), "
            "so this is reported as matched rather than as a win"
        )
    elif lo > 0.0:
        verdict, detail = "beats_baselines", ""
    elif hi < 0.0:
        verdict, detail = "baseline_wins", ""
    else:
        verdict, detail = "matched", ""
    return BaselineComparison(
        own=own_auc,
        best_baseline=best.baseline,
        best_score=best.auroc,
        margin=margin,
        ci_low=lo,
        ci_high=hi,
        ci_level=ci,
        verdict=verdict,
        n_baselines_scored=n_scored,
        n_baselines_refused=n_refused,
        detail=detail,
    )


def _paired_margin_interval(
    own: np.ndarray,
    rival: np.ndarray,
    labels: np.ndarray,
    *,
    seed_labels: Sequence[Any],
    ci: float,
    n_resamples: int,
    seed: int,
) -> tuple[float, float]:
    """Percentile interval on the AUROC difference, resampling lineages when they are declared.

    Both AUROCs are recomputed on the same resampled index set, so the correlation between two
    methods scored on the same items is preserved. Resampling them independently would widen the
    interval by exactly the amount the pairing removes, which is the difference between a test
    that can see a 0.002 gap and one that cannot.

    With very few lineages the interval is declined rather than reported. The rule is derived from
    the confidence level rather than picked: a bootstrap over `K` clusters has `C(2K-1, K)`
    distinct resamples, and resolving a tail of mass `(1 - ci) / 2` needs at least `2 / (1 - ci)`
    of them. At 95% that is 40 resamples, so `K = 5` is the floor and `K = 4` is not. Two
    lineages give three distinct resamples and a percentile interval that looks confident because
    it has nothing to vary over, which is precisely the failure the lineage machinery exists to
    prevent.
    """
    n = labels.size
    labels_list = list(seed_labels) if seed_labels else list(range(n))
    clusters: dict[Any, list[int]] = {}
    for i, lab in enumerate(labels_list):
        clusters.setdefault(lab, []).append(i)
    keys = list(clusters)
    k = len(keys)
    if k < 2 or math.comb(2 * k - 1, k) < 2.0 / max(1.0 - ci, 1e-9):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(keys), size=(n_resamples, len(keys)))
    replicates = np.empty(n_resamples, dtype=np.float64)
    index_of = [np.asarray(clusters[k], dtype=np.intp) for k in keys]
    for i in range(n_resamples):
        idx = np.concatenate([index_of[j] for j in draws[i]])
        yy = labels[idx]
        if np.unique(yy).size < 2:
            replicates[i] = np.nan
            continue
        replicates[i] = auroc(own[idx], yy) - auroc(rival[idx], yy)
    finite = replicates[np.isfinite(replicates)]
    if finite.size < 10:
        return float("nan"), float("nan")
    alpha = (1.0 - ci) / 2.0
    return float(np.quantile(finite, alpha)), float(np.quantile(finite, 1.0 - alpha))


__all__ = [
    "BaseBaseline",
    "Baseline",
    "BaselineBank",
    "BaselineComparison",
    "BaselineReading",
    "BaselineScore",
    "DetectionTask",
    "Verdict",
    "accuracy_at_midpoint",
    "auroc",
    "compare_against_baselines",
    "is_scored",
    "oriented_score",
    "stratified_folds",
]

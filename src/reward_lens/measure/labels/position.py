"""L5 `labels.position_prior`: a localiser scored against a position-stratified null.

A localiser is asked to point at the one place in a trajectory where something went wrong. The
usual way to score it is to pool every candidate position from every trajectory, rank them by the
localiser's statistic, and read the area under the ROC curve. That pooled statistic scores the
localiser against a **uniform prior over positions**: it asks whether a labelled position outranks a
randomly drawn unlabelled one, and every unlabelled position in the corpus is equally likely to be
the one drawn.

That prior is almost never right. Labelled positions cluster: an earliest-error step is not
uniformly distributed over a solution, and a critical decision in a trajectory is not uniformly
distributed over an episode. If the localiser's own statistic carries any position trend, and a
step score read off a process reward model carries a large one, then the pooled AUC is partly
measuring the agreement between two position distributions rather than the localiser's ability to
point at anything.

Conditioning fixes it. `stratified_auc` compares each labelled position only against unlabelled
positions in the same stratum, so whatever position alone predicts is removed by construction. What
survives is the discrimination the localiser adds on top of knowing where in the trajectory it is,
and that is the number worth publishing.

**What this instrument settles, which is the reason it exists.** An AUC below 0.5 has three
explanations and they call for opposite responses. It can be noise, in which case the interval
covers 0.5 and there is nothing to report. It can be a position confound, in which case
conditioning moves the number toward 0.5 and the original reading was an artifact. Or the
discrimination can be real and the score's sign can be backwards, in which case conditioning leaves
the number where it is or pushes it further from 0.5, and `1 - AUC` is the reading. Nothing in a
pooled AUC distinguishes the three. Everything in this instrument does, and the rule is fixed in
`_verdict` before any data is seen rather than chosen after.

**The same three apply above 0.5 and the mirror case is the common one**, because a localiser is
usually reported above chance. A pooled 0.83 with position alone at 0.84 and a conditioned 0.63 is
a localiser that does discriminate and a published number that overstates it by two thirds of its
distance from chance. `_verdict` names that case separately from the clean one; it did not always,
and the ordering bug that hid it is written up there.

Kill condition, from the catalogue: **n/a.** Both outcomes are results. A localiser that survives
conditioning has a real number and a localiser that does not has an explained one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from reward_lens.core.envelope import EnvelopeSpec
from reward_lens.core.evidence import Uncertainty, register_payload
from reward_lens.core.invariance import Relation
from reward_lens.core.quantity import BiasStatement, CostModel, EstimatorEntry, register_estimator
from reward_lens.core.reading import Reading, Refusal, RefusalReason
from reward_lens.core.types import Access, AccessMatrix, Capability, Component
from reward_lens.measure.base import Context
from reward_lens.measure.labels._common import (
    ALL_PHASES,
    ALL_SUBSTRATES,
    LabelsInstrument,
    cluster_bootstrap_ci,
    emit_with_reference,
    stratified_auc,
)
from reward_lens.stats.roc import roc_pr

#: What the stratified reading is compared against. Not a threshold in an envelope: the AUC of a
#: statistic that carries no information is 0.5 by construction, and this is that constant given a
#: name so the comparisons below read as arithmetic rather than as a tuning choice.
CHANCE = 0.5

#: Bins of normalised position for the default stratification. Ten because the pair count has to
#: survive the split: at the corpus sizes a localisation study reaches, a hundred bins leaves most
#: strata with no positive or no negative and the statistic is then computed on a fraction of the
#: data without saying so. The exact-position stratification below is the strict alternative and it
#: runs alongside rather than instead.
DEFAULT_BINS = 10


@register_payload
@dataclass(frozen=True)
class PositionPrior:
    """A localiser's AUC under a uniform prior over positions and under a position-stratified null.

    ``uniform_auc`` is the pooled statistic **computed on the same within-item standardised scores
    the stratified statistic uses**, which is what makes ``confound_size`` a statement about
    conditioning rather than about two different score transforms. That is worth stating plainly
    because it is not what "the pooled AUC" means in a paper: a localisation study that pools raw
    step scores reports ``uniform_auc_raw``, carried here beside it. On `VERIF-PRM` the two are
    0.2821 standardised and 0.2816 raw, a difference of 5.9e-4, and the standardised one is the
    number `VERIF-PRM` itself stored. Neither is quietly substituted for the other.

    ``stratified_auc`` is the same statistic conditioned on the position stratum.
    ``position_only_auc`` is what a localiser that reads nothing but the position achieves, and it
    is the size of the confound being removed.

    ``exact_position_auc`` repeats the conditioning on the strictest possible strata: identical step
    index inside an identically-long trajectory. ``exact_position_check`` is the position-only AUC
    under those strata and it must be 0.5, because position is constant within such a stratum and
    every comparison is a tie. It is carried rather than asserted because a stratification that
    silently failed to control what it claims to control is exactly the failure this instrument
    exists to catch, and a self-check that never appears in the output is a self-check nobody reads.
    """

    uniform_auc: float
    stratified_auc: float
    position_only_auc: float
    exact_position_auc: float
    exact_position_check: float
    n_items: int
    n_candidates: int
    n_strata: int
    n_pairs: int
    n_bins: int
    ci_low: float = float("nan")
    ci_high: float = float("nan")
    ci_level: float = 0.95
    verdict: str = ""
    interpretation: str = ""
    standardised_within_item: bool = True
    source: str = ""
    #: The pooled AUC on the raw stored scores, with no within-item standardisation. Equal to
    #: ``uniform_auc`` when ``standardise=False``. It is here because the pooled number a
    #: localisation study publishes is this one, and reporting the standardised pooled AUC under
    #: that name would make two different statistics share a label.
    uniform_auc_raw: float = float("nan")
    #: The cluster-bootstrap interval on ``confound_size``, both halves computed on the same
    #: resampled items so their correlation is carried. This is what decides whether the verdict
    #: names a position confound, and it exists so that decision is not a chosen threshold.
    confound_ci: tuple[float, float] = (float("nan"), float("nan"))

    @property
    def inverted_stratified_auc(self) -> float:
        """The same statistic under the opposite score orientation.

        Arithmetic, not a claim: the AUC of `-s` is `1 - AUC(s)` exactly. It is carried because a
        below-chance reading that survives conditioning is a reading whose sign is the open
        question, and making the reader do the subtraction is how a sign error survives a review.
        """
        return 1.0 - self.stratified_auc

    @property
    def confound_size(self) -> float:
        """How far the pooled reading moved once position was controlled.

        ``confound_ci`` is its interval, from the same cluster bootstrap, and the verdict turns on
        whether that interval excludes zero rather than on the sign of this number alone.
        """
        return self.stratified_auc - self.uniform_auc

    def render(self) -> str:
        raw = (
            ""
            if not np.isfinite(self.uniform_auc_raw)
            or abs(self.uniform_auc_raw - self.uniform_auc) < 5e-5
            else f" (on raw scores, unstandardised: {self.uniform_auc_raw:.4f})"
        )
        return (
            f"uniform prior over positions: AUC {self.uniform_auc:.4f}{raw}\n"
            f"position-stratified null:     AUC {self.stratified_auc:.4f} "
            f"[{self.ci_low:.4f}, {self.ci_high:.4f}] at {self.ci_level:.0%}\n"
            f"position alone:               AUC {self.position_only_auc:.4f}\n"
            f"what conditioning moved:      {self.confound_size:+.4f} "
            f"[{self.confound_ci[0]:+.4f}, {self.confound_ci[1]:+.4f}]\n"
            f"exact-position strata:        AUC {self.exact_position_auc:.4f} "
            f"(control check {self.exact_position_check:.4f}, must be {CHANCE})\n"
            f"{self.n_items} items, {self.n_candidates} candidates, {self.n_pairs:,} pairs in "
            f"{self.n_strata} strata\n"
            f"{self.verdict}: {self.interpretation}"
        )

    def __canonical__(self) -> dict[str, Any]:
        return {
            "uniform_auc": self.uniform_auc,
            "uniform_auc_raw": self.uniform_auc_raw,
            "stratified_auc": self.stratified_auc,
            "position_only_auc": self.position_only_auc,
            "exact_position_auc": self.exact_position_auc,
            "exact_position_check": self.exact_position_check,
            "n_items": self.n_items,
            "n_candidates": self.n_candidates,
            "n_strata": self.n_strata,
            "n_pairs": self.n_pairs,
            "n_bins": self.n_bins,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "confound_ci": list(self.confound_ci),
            "ci_level": self.ci_level,
            "verdict": self.verdict,
            "standardised_within_item": self.standardised_within_item,
            "source": self.source,
        }


@dataclass(frozen=True)
class LocalisationSeries:
    """One localiser's per-position statistic over a corpus of variable-length items.

    Ragged, in the layout the campaign store already uses: ``values`` is every position's statistic
    concatenated, ``offsets`` is the `N+1` boundaries, and ``labels[i]`` is the index of item `i`'s
    labelled position or a negative number when the item carries none.

    ``higher_is_positive`` is the orientation and it is required rather than defaulted. Half of what
    this instrument exists to adjudicate is a sign, and a default orientation is a sign somebody did
    not choose.
    """

    values: np.ndarray
    offsets: np.ndarray
    labels: np.ndarray
    higher_is_positive: bool
    name: str = ""

    def __post_init__(self) -> None:
        v = np.asarray(self.values, dtype=np.float64).ravel()
        o = np.asarray(self.offsets, dtype=np.int64).ravel()
        lab = np.asarray(self.labels, dtype=np.int64).ravel()
        object.__setattr__(self, "values", v)
        object.__setattr__(self, "offsets", o)
        object.__setattr__(self, "labels", lab)
        if o.size != lab.size + 1:
            raise ValueError(
                f"offsets has {o.size} entries and labels has {lab.size}; a ragged series needs "
                f"exactly one more boundary than it has items."
            )
        if o.size and int(o[-1]) != v.size:
            raise ValueError(
                f"offsets end at {int(o[-1])} and values holds {v.size} entries. A localiser "
                f"aligned to the wrong positions scores noise and reports it as a floor."
            )

    @property
    def n_items(self) -> int:
        return int(self.labels.size)


@dataclass(frozen=True)
class _Item:
    """One item's candidates, kept whole so the bootstrap can resample items rather than positions."""

    score: np.ndarray
    flag: np.ndarray
    position: np.ndarray
    n_steps: int
    index: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))


def _items(series: LocalisationSeries, *, standardise: bool) -> tuple[list[_Item], int]:
    """Split the ragged series into per-item candidate blocks, dropping the unlabelled ones.

    Within-item standardisation is on by default and it is not cosmetic. Items live on their own
    score scales, and pooling raw scores lets a between-item level difference (a hard item scored
    low throughout) masquerade as within-item localisation. An item whose statistic is constant
    standardises to zeros and contributes ties at chance, which is the correct contribution from an
    item that expresses no preference between its own positions.
    """
    out: list[_Item] = []
    skipped = 0
    for i, label in enumerate(series.labels):
        lo, hi = int(series.offsets[i]), int(series.offsets[i + 1])
        if label < 0:
            continue
        step = series.values[lo:hi].astype(np.float64)
        if label >= step.size:
            skipped += 1
            continue
        if standardise:
            sd = float(step.std())
            step = (step - float(step.mean())) / sd if sd > 0.0 else np.zeros_like(step)
        flag = np.zeros(step.size, dtype=np.int64)
        flag[int(label)] = 1
        n = step.size
        out.append(
            _Item(
                score=step if series.higher_is_positive else -step,
                flag=flag,
                position=np.arange(n, dtype=np.float64) / max(n - 1, 1),
                n_steps=n,
                index=np.arange(n, dtype=np.int64),
            )
        )
    return out, skipped


def _bin(position: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    return np.clip(np.digitize(position, edges[1:-1]), 0, n_bins - 1)


def _verdict(
    uniform: float,
    stratified: float,
    position_only: float,
    ci_low: float,
    ci_high: float,
    confound_ci: tuple[float, float] = (float("nan"), float("nan")),
) -> tuple[str, str]:
    """Which explanation for the pooled reading the numbers support.

    The rule is fixed here rather than chosen per run, and it turns on two comparisons: whether the
    interval on the conditioned reading covers chance, and which way conditioning moved the reading.

    - covers chance: nothing survives conditioning, so there is no reading to explain.
    - moved toward chance: position was part of the reading, and how much is the gap between the
      pooled and the conditioned number.
    - stayed away from chance: position was not the reading. The discrimination is real, and if it
      points below chance then the orientation of the score is the remaining explanation, which is a
      claim about the readout rather than about the localiser.

    **The two comparisons are read in that order, and they used to be read in the other one.** The
    sign test ran first, so `stratified > CHANCE` returned "localises" before the confound test was
    reached and a confound that *inflates* an above-chance reading could never be named. That is not
    a corner: it is the mirror image of the case this instrument was built for, and it is the more
    common one, because a localiser is usually reported above chance. Reproduced on 400 synthetic
    items whose statistic rises with position and carries a small genuine signal on top: pooled
    0.8291, position alone 0.8400, conditioned 0.6298 with a 95% interval of [0.5949, 0.6685]. The
    old rule called that "localises" and said "the discrimination is its own rather than the
    position prior's", when five sixths of the distance from chance to the pooled number was the
    position prior's.

    Five verdicts rather than four, because the above-chance confound is not the same statement as
    the below-chance one. Above chance the localiser does discriminate and the published number
    overstates by how much; below chance what survives is discrimination with the sign backwards.
    Collapsing the two into one string would have made "position confound" mean "your localiser is
    not real" in one direction and "your localiser is real but smaller" in the other.

    **"Moved toward chance" is judged against the bootstrap interval on the move**, not against a
    threshold somebody chose. `confound_ci` is the cluster-bootstrap interval on
    `stratified - uniform`, computed on the same resamples as the interval on the reading itself,
    and the confound is named only when that interval excludes zero. Without it a move of 0.008 on
    a reading of 0.87 gets the same verdict as a move of 0.20 on a reading of 0.83, and the first
    of those is a resampling artifact. When the interval is unavailable the sign of the move is
    used on its own, which is the old behaviour and is stated here rather than left implicit.
    """
    covers_chance = ci_low <= CHANCE <= ci_high
    moved_toward = abs(stratified - CHANCE) < abs(uniform - CHANCE)
    if np.isfinite(confound_ci[0]) and np.isfinite(confound_ci[1]):
        moved_toward = moved_toward and not (confound_ci[0] <= 0.0 <= confound_ci[1])
    if covers_chance:
        return (
            "indistinguishable from chance",
            "conditioned on position the localiser's interval covers 0.5, so nothing here "
            "discriminates and the pooled reading was position and noise.",
        )
    if stratified > CHANCE:
        if moved_toward:
            return (
                "localises, and position inflated the pooled reading",
                f"conditioning moved the reading from {uniform:.4f} to {stratified:.4f}, toward "
                f"chance, and a localiser scoring {position_only:.4f} on position alone is what "
                f"moved it. What survives is real: the interval on {stratified:.4f} excludes 0.5, "
                f"so the localiser does outrank a same-position competitor. It is the pooled "
                f"number that is the overstatement, and {stratified:.4f} is the one to publish.",
            )
        return (
            "localises",
            f"the localiser outranks a same-position competitor {stratified:.4f} of the time, and "
            f"conditioning did not move the reading toward chance, so the discrimination is its "
            f"own rather than the position prior's.",
        )
    if moved_toward:
        return (
            "position confound",
            f"conditioning moved the reading from {uniform:.4f} toward chance, and a localiser "
            f"scoring {position_only:.4f} on position alone is what moved it. The pooled number "
            f"was substantially a statement about where labelled positions sit. What is left at "
            f"{stratified:.4f} still excludes 0.5, so a smaller inverted signal survives the "
            f"conditioning; under the opposite score orientation it reads {1.0 - stratified:.4f}.",
        )
    return (
        "below chance, and not from position",
        f"conditioning moved the reading from {uniform:.4f} to {stratified:.4f}, away from "
        f"chance, so position was not what put it below 0.5. A statistic this far from chance is "
        f"discriminating; under the opposite score orientation the same comparison reads "
        f"{1.0 - stratified:.4f}. Check the readout's sign before reporting this as a failure to "
        f"localise.",
    )


def rescore_against_position(
    series: LocalisationSeries,
    *,
    n_bins: int = DEFAULT_BINS,
    standardise: bool = True,
    level: float = 0.95,
    n_boot: int = 400,
    seed: int = 0,
    source: str = "",
) -> PositionPrior:
    """The whole instrument as a function: pooled AUC, position-only AUC, and the conditioned AUC.

    The interval is a cluster bootstrap over items rather than over candidates. Positions inside one
    trajectory are not independent observations of anything, and resampling them would give an
    interval roughly `sqrt(candidates/items)` times too narrow, which at typical corpus shapes is a
    factor of three.
    """
    items, skipped = _items(series, standardise=standardise)
    if not items:
        raise ValueError(
            "no item carries a labelled position, so there is nothing to localise. Every item's "
            "label is negative or out of range."
        )
    # The pooled AUC on the raw stored scores, which is the number a localisation study publishes.
    # Computed on a second pass with the standardisation off rather than inferred, because it is a
    # different statistic and the difference is exactly what the reader is entitled to see.
    raw_items, _ = _items(series, standardise=False) if standardise else (items, 0)

    score = np.concatenate([it.score for it in items])
    flag = np.concatenate([it.flag for it in items])
    position = np.concatenate([it.position for it in items])
    exact = np.concatenate([it.n_steps * 1_000_000 + it.index for it in items])
    strata = _bin(position, n_bins)

    uniform = float(roc_pr(score, flag).auc)
    uniform_raw = float(
        roc_pr(
            np.concatenate([it.score for it in raw_items]),
            np.concatenate([it.flag for it in raw_items]),
        ).auc
    )
    position_only = float(roc_pr(position, flag).auc)
    conditioned, n_strata, n_pairs = stratified_auc(score, flag, strata)
    exact_auc, _, _ = stratified_auc(score, flag, exact)
    exact_check, _, _ = stratified_auc(position, flag, exact)

    def _statistic(sample: Sequence[_Item]) -> float:
        s = np.concatenate([it.score for it in sample])
        f = np.concatenate([it.flag for it in sample])
        p = np.concatenate([it.position for it in sample])
        value, _, _ = stratified_auc(s, f, _bin(p, n_bins))
        return value

    def _confound(sample: Sequence[_Item]) -> float:
        """`stratified - uniform` on one resample: how far conditioning moved the reading.

        Both halves on the same resampled items, which is the point. An interval built from two
        separate bootstraps of the two statistics would ignore their correlation and come out much
        too wide, and these two are strongly correlated because they are the same comparisons
        counted under two groupings.
        """
        s = np.concatenate([it.score for it in sample])
        f = np.concatenate([it.flag for it in sample])
        p = np.concatenate([it.position for it in sample])
        value, _, _ = stratified_auc(s, f, _bin(p, n_bins))
        return value - float(roc_pr(s, f).auc)

    ci_low, ci_high = cluster_bootstrap_ci(_statistic, items, n_boot=n_boot, level=level, seed=seed)
    confound_ci = cluster_bootstrap_ci(_confound, items, n_boot=n_boot, level=level, seed=seed)
    verdict, interpretation = _verdict(
        uniform, conditioned, position_only, ci_low, ci_high, confound_ci
    )
    if skipped:
        interpretation += (
            f" {skipped} item(s) named a labelled position outside their own candidate range and "
            f"were dropped."
        )
    return PositionPrior(
        uniform_auc=uniform,
        uniform_auc_raw=uniform_raw,
        stratified_auc=conditioned,
        position_only_auc=position_only,
        exact_position_auc=exact_auc,
        exact_position_check=exact_check,
        n_items=len(items),
        n_candidates=int(score.size),
        n_strata=n_strata,
        n_pairs=n_pairs,
        n_bins=n_bins,
        ci_low=ci_low,
        ci_high=ci_high,
        confound_ci=confound_ci,
        ci_level=level,
        verdict=verdict,
        interpretation=interpretation,
        standardised_within_item=standardise,
        source=source or series.name,
    )


# ---------------------------------------------------------------------------
# Reading the stored artifacts
# ---------------------------------------------------------------------------


def load_step_scores(
    store_dir: str | Path,
    *,
    observable: str,
    slice_name: str,
    roster_key: str | None = None,
    higher_is_positive: bool = False,
    sidecar_dirs: Sequence[str | Path] = (),
) -> LocalisationSeries:
    """A ragged step-score series out of an evidence store, opened read-only.

    Through `record.convert.store.CampaignStore`, which is the sanctioned read path: it opens the
    store with `readonly=True` so `append` raises, and it checks for blinded payloads before
    decoding permissively. Nothing here writes, and the store this was built against is the only
    copy of an experiment that cost real money.

    ``higher_is_positive`` defaults to False because the convention this instrument was built
    against scores a bad step low. It is still an argument with no default at the call sites that
    matter, and getting it wrong is precisely the thing the instrument reports on.
    """
    from reward_lens.record.convert.store import CampaignStore

    store = CampaignStore(store_dir, sidecar_dirs=sidecar_dirs)
    store.assert_no_blind_payloads()
    rows = [
        r
        for r in store.by_observable(observable)
        if r.slice_name == slice_name and (roster_key is None or r.roster_key == roster_key)
    ]
    if not rows:
        available = sorted({r.slice_name for r in store.by_observable(observable)})
        raise KeyError(
            f"no row of {observable!r} at slice {slice_name!r}"
            + (f" for {roster_key!r}" if roster_key else "")
            + f"; the store carries slices {available}"
        )
    if len(rows) > 1:
        raise KeyError(
            f"{len(rows)} rows of {observable!r} match slice {slice_name!r}; name a roster_key so "
            f"the series is unambiguous rather than silently taking the first."
        )
    value = store.value(rows[0])
    fields_ = value if isinstance(value, dict) else {f: getattr(value, f) for f in value.__dict__}
    return LocalisationSeries(
        values=np.asarray(fields_["values"], dtype=np.float64),
        offsets=np.asarray(fields_["offsets"], dtype=np.int64),
        labels=np.asarray(fields_["labels"], dtype=np.int64),
        higher_is_positive=higher_is_positive,
        name=f"{observable}[{slice_name}]",
    )


# ---------------------------------------------------------------------------
# The instrument
# ---------------------------------------------------------------------------

#: The stored artifacts and the labels that go with them. Nothing is queried and nothing is run:
#: the whole instrument is a re-reading of numbers already on disk, which is why it costs nothing.
ACCESS_STORED: AccessMatrix = {
    Component.RECORD: Access.RECORD,
    Component.GOLD: Access.RECORD,
}

#: A re-scoring of a recorded statistic. The twelve envelope conditions are all properties of a
#: training run, and this instrument reads an archive: no regime can make a rank statistic over
#: fixed numbers wrong. Declared explicitly rather than left empty, which `EnvelopeSpec` refuses.
POSITION_ENVELOPE = EnvelopeSpec(
    unconditional=True,
    justification=(
        "a rank statistic recomputed over stored numbers. It counts what is in the archive and "
        "asserts nothing about the process that produced it, so no regime condition can make the "
        "count wrong. The one precondition that does bite, that the labelled position and the "
        "candidate positions come from the same item, is a structural property of the ragged "
        "layout and is checked in LocalisationSeries rather than in a regime reading."
    ),
)


class PositionStratifiedNull(LabelsInstrument):
    """L5 `labels.position_prior`: score a localiser against the position distribution it faces.

    Kill condition, from the catalogue: **n/a.** Both outcomes are results, and that is not a
    formality here. A conditioned reading that lands at chance says the pooled number was an
    artifact, which is worth publishing; a conditioned reading that survives says the localiser
    discriminates, which is worth publishing; and a conditioned reading that moves *away* from
    chance says the sign of the readout is the open question, which is the most useful of the three
    because it is the cheapest to check.

    The instrument runs on stored artifacts and costs nothing beyond reading them. It is the one
    place in this series where the honest answer was already sitting in an archive and the only
    reason it had not been produced is that nobody had asked the pooled AUC what prior it was
    scoring against.
    """

    name = "PositionStratifiedNull"
    version = "1.0"
    quantity = "labels.position_prior"
    capabilities = Capability.NONE
    requires = ACCESS_STORED
    substrates = ALL_SUBSTRATES
    phases = ALL_PHASES
    envelope = POSITION_ENVELOPE
    invariance = "none"
    invariance_relation = Relation("invariant")
    baselines = (
        "the position-only localiser, which reads nothing but where in the trajectory a candidate "
        "sits and is the confound this instrument removes",
        "the pooled AUC under a uniform prior over positions, which is the number every "
        "localisation study currently reports",
    )
    rung = 0
    faithful_to = "the Mann-Whitney rank identity, conditioned on a stratum"
    deviations = (
        "the stratified statistic is a pair-count weighted average of within-stratum "
        "Mann-Whitney statistics, which is the conditional AUC. It is not a partial AUC and it is "
        "not a covariate-adjusted ROC curve; those estimate different things and would need a "
        "model of the score's dependence on position, which this deliberately does not fit.",
        "normalised position is the stratifying variable by default. A localiser whose confound is "
        "with absolute position rather than relative position is controlled by the exact-position "
        "stratification, which runs alongside and is reported, not by the default.",
        "the interval is a percentile cluster bootstrap over items. It carries no bias correction, "
        "so it is honest about sampling variation and not about estimator bias.",
    )

    def __init__(
        self,
        series: LocalisationSeries | None = None,
        *,
        n_bins: int = DEFAULT_BINS,
        standardise: bool = True,
        level: float = 0.95,
        n_boot: int = 400,
        seed: int = 0,
    ) -> None:
        self.series = series
        self.n_bins = n_bins
        self.standardise = standardise
        self.level = level
        self.n_boot = n_boot
        self.seed = seed

    def measure(self, ctx: Context) -> Any:
        series = self.series if self.series is not None else ctx.stats.get("series")
        prior = rescore_against_position(
            series,
            n_bins=self.n_bins,
            standardise=self.standardise,
            level=self.level,
            n_boot=self.n_boot,
            seed=self.seed,
        )
        return emit_with_reference(
            ctx,
            prior,
            quantity=self.quantity,
            uncertainty=Uncertainty(
                n=prior.n_items,
                ci_low=prior.ci_low,
                ci_high=prior.ci_high,
                ci_level=prior.ci_level,
                method="cluster bootstrap over items, percentile",
            ),
            baselines={
                "baseline.position_only": prior.position_only_auc,
                "baseline.uniform_prior": prior.uniform_auc,
            },
            subject_extra={"localiser": prior.source or "unnamed"},
        )

    def estimate(self, ctx: Context | None = None) -> Reading:
        ctx = ctx or Context(readout="score")
        series = self.series if self.series is not None else ctx.stats.get("series")
        if series is None:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail=(
                    "no localisation series was supplied, so there is no stored statistic to "
                    "re-score against a position-stratified null."
                ),
                remedy=(
                    "pass a LocalisationSeries to the constructor, or load one from an evidence "
                    "store with labels.position.load_step_scores(store_dir, observable=..., "
                    "slice_name=...). This instrument reads an archive and computes nothing new, "
                    "so the only thing it can be short of is the archive."
                ),
            )
        items, _ = _items(series, standardise=self.standardise)
        if not items:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.LABEL_QUALITY_UNKNOWN,
                detail=(
                    f"none of the {series.n_items} items carries a labelled position inside its "
                    f"own candidate range, so there is no answer key to score the localiser "
                    f"against and any AUC would be undefined rather than low."
                ),
                remedy=(
                    "supply labels as the index of each item's labelled position, using a negative "
                    "number for items that carry none. An index at or past an item's candidate "
                    "count is a misalignment between the labels and the series, not a hard case."
                ),
                statistics={"n_items": series.n_items},
            )
        return super().estimate(ctx)


# ---------------------------------------------------------------------------
# The ladder
# ---------------------------------------------------------------------------

_REGISTERED = False


def register() -> None:
    """Register L5's single rung. Idempotent, so importing the package twice is safe."""
    global _REGISTERED
    if _REGISTERED:
        return
    register_estimator(
        EstimatorEntry(
            quantity="labels.position_prior",
            impl="labels.position_prior.stratified",
            requires=ACCESS_STORED,
            envelope=POSITION_ENVELOPE,
            rung=0,
            bias=BiasStatement(
                direction="approximately_unbiased",
                why=(
                    "a rank statistic conditioned on the stratifying variable. It inherits the "
                    "residual confounding of whatever the strata fail to separate, so with coarse "
                    "bins it retains a fraction of the position effect and is biased toward the "
                    "pooled reading. The exact-position stratification bounds that fraction and is "
                    "reported alongside."
                ),
            ),
            cost=CostModel(cpu_seconds=5.0, note="a re-read of stored numbers and a bootstrap"),
            run=rescore_against_position,
        )
    )
    _REGISTERED = True


__all__ = [
    "ACCESS_STORED",
    "CHANCE",
    "DEFAULT_BINS",
    "POSITION_ENVELOPE",
    "LocalisationSeries",
    "PositionPrior",
    "PositionStratifiedNull",
    "load_step_scores",
    "register",
    "rescore_against_position",
]

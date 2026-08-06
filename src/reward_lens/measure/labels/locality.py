"""L4 `labels.fs_signal_locality`: is the tell at the end of the episode, or spread across it.

An agent that will falsely claim success is doing something detectable before it makes the claim.
The question is where. If the tell is only in the closing message then detecting it is a matter of
reading the last paragraph, and the whole episode is decoration. If the tell is present in the
episode with the closing message removed, then the behaviour has a trajectory-long signature and a
monitor can catch it before the claim is made, which is the only version of the result that is
operationally useful.

The text-level answer is published: excluding the closing message scores 0.924 against 0.934 for the
closing message alone. Both high, and the first only slightly lower, so the tell is distributed
rather than terminal. **Nobody has asked the same question of the residual stream.** That is one
well-defined experiment against a published comparison number, either outcome is publishable, and it
is registered here as rung 1 with no implementation so the capability report names it as an open
target rather than leaving it a sentence in a design document.

The measurement is a comparison of two detectors that differ only in what text they see:

    closing_only        the closing message, and nothing else
    excluding_closing   the episode, with the closing message removed
    whole               both, as the upper reference

Three fits of the same estimator on the same items with the same folds, so the difference between
the AUCs is a difference in the text and not in the method. The estimator is `baseline.tfidf`,
which is the mandatory comparator for any transcript-level detector on this distinction and is
therefore also the right instrument for asking where in the transcript the signal is.

Kill condition, from the catalogue: **n/a.**
"""

from __future__ import annotations

from dataclasses import dataclass
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
    LabelsInstrument,
    emit_with_reference,
    label_quality_refusal,
)
from reward_lens.stats.baselines.base import BaselineScore, DetectionTask, is_scored
from reward_lens.stats.baselines.text import TfidfLogisticRegression

#: What counts as the closing message when the caller does not supply one. The last paragraph, on
#: a blank-line split. A default worth stating rather than hiding: an episode with no blank line is
#: entirely its own closing message under this rule, which makes the comparison vacuous, and
#: `split_closing` reports how many episodes ended up that way so the vacuity is visible.
CLOSING_SEPARATOR = "\n\n"


@dataclass(frozen=True)
class TranscriptSplit:
    """One corpus, cut two ways, with the count of episodes the cut failed on."""

    closing: tuple[str, ...]
    excluding: tuple[str, ...]
    whole: tuple[str, ...]
    n_unsplittable: int
    rule: str

    @property
    def n(self) -> int:
        return len(self.whole)


def split_closing(
    texts: Sequence[str],
    closings: Sequence[str] | None = None,
    *,
    separator: str = CLOSING_SEPARATOR,
) -> TranscriptSplit:
    """Cut each transcript into its closing message and everything before it.

    Pass ``closings`` when the corpus knows where its own closing message is, which it usually does:
    an agent trace has a final assistant turn and guessing at it from whitespace is worse than
    reading it. The whitespace rule is the fallback and it is reported as the rule that was used.

    An episode the rule cannot split contributes an empty "excluding" string. That is the honest
    representation of "there was nothing before the closing message", and the count of such episodes
    travels with the split because a corpus where most episodes are unsplittable cannot answer this
    question at all.
    """
    if closings is not None:
        if len(closings) != len(texts):
            raise ValueError(f"{len(closings)} closings for {len(texts)} transcripts")
        excluding: list[str] = []
        for whole, close in zip(texts, closings):
            excluding.append(
                whole[: -len(close)].rstrip() if close and whole.endswith(close) else whole
            )
        unsplittable = sum(1 for e in excluding if not e.strip())
        return TranscriptSplit(
            closing=tuple(closings),
            excluding=tuple(excluding),
            whole=tuple(texts),
            n_unsplittable=unsplittable,
            rule="supplied closing message per episode",
        )
    closing: list[str] = []
    excluding = []
    for whole in texts:
        head, sep, tail = whole.rpartition(separator)
        if sep:
            excluding.append(head)
            closing.append(tail)
        else:
            excluding.append("")
            closing.append(whole)
    return TranscriptSplit(
        closing=tuple(closing),
        excluding=tuple(excluding),
        whole=tuple(texts),
        n_unsplittable=sum(1 for e in excluding if not e.strip()),
        rule=f"last block after {separator!r}",
    )


@register_payload
@dataclass(frozen=True)
class SignalLocality:
    """Where in a transcript the tell lives, as three AUCs from one estimator on one set of items.

    ``delta`` is `excluding_closing - closing_only`. Near zero with both AUCs high means the tell is
    distributed: the episode carries it and so does the closing message. A large negative delta with
    `excluding_closing` near chance means the tell is terminal, and a monitor reading the episode
    before the claim is made has nothing to work with.
    """

    closing_only_auc: float
    excluding_closing_auc: float
    whole_auc: float
    delta: float
    n: int
    n_positive: int
    n_unsplittable: int
    split_rule: str
    estimator: str
    verdict: str = ""
    interpretation: str = ""
    corpus: str = ""

    def render(self) -> str:
        return (
            f"{self.corpus or 'an unnamed corpus'}: {self.n} episodes, {self.n_positive} positive\n"
            f"  closing message only        AUC {self.closing_only_auc:.4f}\n"
            f"  episode without the closing AUC {self.excluding_closing_auc:.4f}\n"
            f"  whole episode               AUC {self.whole_auc:.4f}\n"
            f"  delta (excluding - closing) {self.delta:+.4f}\n"
            f"  {self.verdict}: {self.interpretation}"
        )

    def __canonical__(self) -> dict[str, Any]:
        return {
            "closing_only_auc": self.closing_only_auc,
            "excluding_closing_auc": self.excluding_closing_auc,
            "whole_auc": self.whole_auc,
            "delta": self.delta,
            "n": self.n,
            "n_positive": self.n_positive,
            "n_unsplittable": self.n_unsplittable,
            "split_rule": self.split_rule,
            "estimator": self.estimator,
            "verdict": self.verdict,
            "corpus": self.corpus,
        }


#: How far from chance an arm has to land before it counts as carrying signal. Not tuned: 0.55 is
#: the same operating threshold the shipped verification kill condition uses, and reusing it rather
#: than picking a new one keeps one number in the library instead of two.
SIGNAL_FLOOR = 0.55


def _verdict(closing: float, excluding: float) -> tuple[str, str]:
    """Distributed, terminal, or neither. The rule is fixed before any corpus is seen."""
    if excluding < SIGNAL_FLOOR and closing < SIGNAL_FLOOR:
        return (
            "no tell",
            f"neither half reaches {SIGNAL_FLOOR}, so there is nothing here to localise and the "
            f"comparison between the halves is a comparison of two noise levels.",
        )
    if excluding >= SIGNAL_FLOOR and closing >= SIGNAL_FLOOR:
        return (
            "distributed",
            f"both halves carry the tell ({excluding:.3f} without the closing message against "
            f"{closing:.3f} with only the closing message), so a monitor reading the episode "
            f"before the claim is made has something to work with.",
        )
    if excluding >= SIGNAL_FLOOR:
        return (
            "in the episode, not the closing message",
            f"the episode without its closing message scores {excluding:.3f} and the closing "
            f"message alone scores {closing:.3f}, so the closing message is not where the tell is.",
        )
    return (
        "terminal",
        f"only the closing message carries the tell ({closing:.3f} against {excluding:.3f}). "
        f"Detection here is reading the last paragraph, and nothing in the episode warns of it in "
        f"advance.",
    )


def signal_locality(
    texts: Sequence[str],
    labels: Sequence[int] | np.ndarray,
    *,
    closings: Sequence[str] | None = None,
    separator: str = CLOSING_SEPARATOR,
    corpus: str = "",
    seed_labels: Sequence[Any] = (),
) -> SignalLocality | Refusal:
    """Three fits of one estimator on one set of items, differing only in what text they see.

    The estimator is the TF-IDF logistic regression from the baseline bank, scored out of fold, so
    every AUC is a generalisation estimate. Using the mandatory comparator as the measuring
    instrument here is deliberate: the question is about where the *text* signal is, and a method
    that fits nothing but word counts answers it without a model's opinion in the middle.
    """
    y = np.asarray(labels).ravel().astype(int)
    if len(texts) != y.size:
        raise ValueError(f"{len(texts)} transcripts and {y.size} labels")
    split = split_closing(texts, closings, separator=separator)
    estimator = TfidfLogisticRegression()

    scores: dict[str, float] = {}
    for arm, arm_texts in (
        ("closing_only", split.closing),
        ("excluding_closing", split.excluding),
        ("whole", split.whole),
    ):
        task = DetectionTask(labels=y, texts=arm_texts, seed_labels=tuple(seed_labels), name=arm)
        reading = estimator.run(task)
        if not is_scored(reading):
            return Refusal(
                instrument="SignalLocality",
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail=(
                    f"the {arm} arm could not be scored: {getattr(reading, 'detail', reading)}"
                ),
                remedy=(
                    "supply more items, or more items of the minority class. Every arm here is "
                    "scored out of fold, and an in-sample TF-IDF fit reaches near 1.0 on any "
                    "labelling at all, so a comparison between an out-of-fold arm and an in-sample "
                    "one would be a comparison of two different things."
                ),
            )
        assert isinstance(reading, BaselineScore)
        scores[arm] = float(reading.auroc)

    verdict, interpretation = _verdict(scores["closing_only"], scores["excluding_closing"])
    if split.n_unsplittable:
        interpretation += (
            f" {split.n_unsplittable} of {split.n} episodes could not be split by the rule "
            f"{split.rule!r} and contributed an empty excluding-closing text, which drags that arm "
            f"toward chance independently of where the tell is."
        )
    return SignalLocality(
        closing_only_auc=scores["closing_only"],
        excluding_closing_auc=scores["excluding_closing"],
        whole_auc=scores["whole"],
        delta=scores["excluding_closing"] - scores["closing_only"],
        n=split.n,
        n_positive=int((y == 1).sum()),
        n_unsplittable=split.n_unsplittable,
        split_rule=split.rule,
        estimator=estimator.id,
        verdict=verdict,
        interpretation=interpretation,
        corpus=corpus,
    )


# ---------------------------------------------------------------------------
# The instrument
# ---------------------------------------------------------------------------

ACCESS_TRANSCRIPTS: AccessMatrix = {
    Component.RECORD: Access.RECORD,
    Component.GOLD: Access.RECORD,
}

LOCALITY_ENVELOPE = EnvelopeSpec(
    unconditional=True,
    justification=(
        "three out-of-fold fits of one estimator on one fixed set of transcripts. The comparison "
        "is between two views of the same text with the same items and the same folds, so no "
        "regime of the process that produced the transcripts can make the difference between them "
        "wrong. The precondition that does bite, that the closing message boundary is right, is "
        "carried on the split as the rule that was used and the count it failed on."
    ),
)


class SignalLocalityText(LabelsInstrument):
    """L4 rung 0: where in a transcript the tell is, measured on text.

    Kill condition, from the catalogue: **n/a.**

    Rung 1 is the same question asked of the residual stream and it is registered with no
    implementation. That registration is the point of listing it here: a quantity whose higher rung
    is specified and unbuilt reads in the capability report as an open research target with a
    stated cost, which is what the design asks for and what a sentence in a document cannot deliver.
    """

    name = "SignalLocalityText"
    version = "1.0"
    quantity = "labels.fs_signal_locality"
    capabilities = Capability.NONE
    requires = ACCESS_TRANSCRIPTS
    envelope = LOCALITY_ENVELOPE
    invariance = "none"
    invariance_relation = Relation("invariant")
    baselines = (
        "the whole-episode arm, which is the upper reference both halves are read against",
        "baseline.tfidf",
    )
    rung = 0
    faithful_to = "an ablation over the text a fixed estimator is allowed to read"
    deviations = (
        "the three arms share items and folds but are three separate fits, so the difference "
        "between their AUCs carries the fit's own variance. With few items that variance can "
        "exceed the difference and the verdict is then reading noise; the item count travels with "
        "the reading so a reader can see it.",
        "the default closing-message rule is the last block after a blank line, which is a "
        "guess about document structure. A corpus that knows its own turn boundaries should pass "
        "them and the rule used is recorded either way.",
        "an unsplittable episode contributes an empty excluding-closing text rather than being "
        "dropped, which biases that arm toward chance. Dropping them instead would change the "
        "item set between arms and break the comparison, so the bias is reported rather than "
        "traded for a worse one.",
    )

    def __init__(
        self,
        texts: Sequence[str] = (),
        labels: Sequence[int] | np.ndarray = (),
        *,
        closings: Sequence[str] | None = None,
        separator: str = CLOSING_SEPARATOR,
        corpus: str = "",
        seed_labels: Sequence[Any] = (),
    ) -> None:
        self.texts = tuple(texts)
        self.labels = np.asarray(labels).ravel().astype(int) if len(labels) else np.zeros(0, int)
        self.closings = closings
        self.separator = separator
        self.corpus = corpus or "unnamed corpus"
        self.seed_labels = tuple(seed_labels)

    def measure(self, ctx: Context) -> Any:
        result = signal_locality(
            self.texts,
            self.labels,
            closings=self.closings,
            separator=self.separator,
            corpus=self.corpus,
            seed_labels=self.seed_labels,
        )
        if isinstance(result, Refusal):
            return result
        return emit_with_reference(
            ctx,
            result,
            quantity=self.quantity,
            uncertainty=Uncertainty(n=result.n, method="out-of-fold AUROC per arm"),
            baselines={"baseline.tfidf.whole_episode": result.whole_auc},
            subject_extra={"corpus": self.corpus, "split_rule": result.split_rule},
        )

    def estimate(self, ctx: Context | None = None) -> Reading:
        ctx = ctx or Context(readout="score")
        if len(self.texts) == 0 or self.labels.size == 0:
            return label_quality_refusal(
                self.name,
                what=(
                    f"corpus {self.corpus!r} carries no labelled transcripts, so there is no tell "
                    f"to localise"
                ),
                remedy=(
                    "supply transcripts and a binary label per transcript. The label this "
                    "instrument was designed around is whether the agent falsely claimed success, "
                    "which is an audit column rather than the verifier's scalar."
                ),
                corpus=self.corpus,
                n=len(self.texts),
            )
        if np.unique(self.labels).size < 2:
            return label_quality_refusal(
                self.name,
                what=(
                    f"corpus {self.corpus!r} has one class only ({int(self.labels.sum())} positive "
                    f"of {self.labels.size}), so every discrimination statistic is undefined"
                ),
                remedy=(
                    "supply a corpus carrying both classes. A single-class corpus makes the AUC "
                    "undefined rather than low, and an implementation that returned 0.5 here would "
                    "be inventing an observation."
                ),
                corpus=self.corpus,
                n_positive=int(self.labels.sum()),
                n=int(self.labels.size),
            )
        return super().estimate(ctx)


_REGISTERED = False


def register() -> None:
    """Register L4's two rungs: the text one that runs and the white-box one that does not."""
    global _REGISTERED
    if _REGISTERED:
        return
    register_estimator(
        EstimatorEntry(
            quantity="labels.fs_signal_locality",
            impl="labels.fs_signal_locality.r0_text",
            requires=ACCESS_TRANSCRIPTS,
            envelope=LOCALITY_ENVELOPE,
            rung=0,
            bias=BiasStatement(
                direction="downward",
                why=(
                    "a bag of words reads no order and no structure, so a tell carried by the "
                    "sequence of actions rather than by their vocabulary is invisible to it. Both "
                    "arms are floors, and the arm with less text to work with is the lower floor."
                ),
            ),
            cost=CostModel(cpu_seconds=1.0, note="three out-of-fold TF-IDF fits"),
            run=signal_locality,
        )
    )
    # Rung 1 is specified and not built. `run=None` is how that is recorded: the
    # capability report prints the rung, its access, and its cost, and says it has no
    # implementation. This is the open white-box experiment, registered so it is nameable.
    register_estimator(
        EstimatorEntry(
            quantity="labels.fs_signal_locality",
            impl="labels.fs_signal_locality.r1_residual_stream",
            requires={Component.POLICY: Access.FORWARD, Component.GOLD: Access.RECORD},
            envelope=LOCALITY_ENVELOPE,
            rung=1,
            bias=BiasStatement(
                direction="unknown",
                why=(
                    "no residual-stream version of this ablation has been run by anyone, so there "
                    "is no measured bias to state and inventing a direction would be a guess "
                    "wearing a field name."
                ),
            ),
            cost=CostModel(
                gpu_seconds=3600.0,
                note=(
                    "a forward pass over every episode, twice, with the closing message masked in "
                    "one arm, plus a probe fit per layer"
                ),
            ),
            run=None,
        )
    )
    _REGISTERED = True


__all__ = [
    "ACCESS_TRANSCRIPTS",
    "CLOSING_SEPARATOR",
    "LOCALITY_ENVELOPE",
    "SIGNAL_FLOOR",
    "SignalLocality",
    "SignalLocalityText",
    "TranscriptSplit",
    "register",
    "signal_locality",
    "split_closing",
]

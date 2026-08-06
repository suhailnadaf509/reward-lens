"""L3 `labels.false_success_rate`, `labels.corrupt_success_rate`: a verifier wrong in both directions.

A programmatic verifier emits a scalar and the scalar is treated as ground truth. On tau2-bench both
of its errors have been measured on the same benchmark: 36% of the zeros are trajectories where the
agent claimed success, and 27 to 78% of the ones are trajectories that violated the procedure and
reached the right end state anyway. A verifier that is wrong in one direction is a noisy label. A
verifier that is wrong in *both* directions on the same benchmark is not a label at all, because the
two errors do not cancel and no scalar correction recovers the truth.

Two rates, and they are not two views of one thing:

    false_success_rate    among the verifier's zeros, the fraction where the agent claimed it had
                          succeeded. This is a disagreement between the verifier and the agent's own
                          account, and it is the population an honesty monitor has to work on.
    corrupt_success_rate  among the verifier's ones, the fraction that reached the right end state
                          by a route the task forbade. This is the population a capability
                          evaluation is silently crediting.

**The mandatory comparator, and the reason it is mandatory.** On the false-success distinction, LLM
judges came out *anti-correlated with truth*: AUROC 0.18 to 0.30 across 25 judge-condition cells. A
TF-IDF logistic regression on the same transcripts reached 0.849, and an XGBoost on AppWorld reached
0.953, at 1.19 ms against roughly 4,000 ms. Any transcript-level detector proposed here is compared
against `baseline.tfidf` before it is reported, and a detector that costs a forward pass and does not
clear a bag of words has not earned the forward pass.

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
from reward_lens.core.reading import Reading
from reward_lens.core.types import Capability
from reward_lens.measure.base import Context
from reward_lens.measure.labels._common import (
    ACCESS_LABELLED_CORPUS,
    LabelsInstrument,
    Proportion,
    emit_with_reference,
    label_quality_refusal,
    wilson_interval,
)
from reward_lens.stats.baselines import run_bank
from reward_lens.stats.baselines.base import (
    BaselineBank,
    BaselineComparison,
    DetectionTask,
    compare_against_baselines,
)

#: The comparator every transcript-level detector in this series is placed against. Named as a
#: constant so an instrument cannot quietly run a weaker bank and report the same verdict.
MANDATORY_BASELINE = "baseline.tfidf"


@dataclass(frozen=True)
class Verdict:
    """One trajectory, as the verifier scored it and as an audit found it.

    ``verifier_score`` is the scalar the harness emitted, 0 or 1. The other three are audit columns
    and every one of them may be absent, because the whole finding here is that benchmarks ship the
    scalar and nothing else. `None` means nobody looked, and the instrument refuses on the rate that
    needs the column rather than treating the absence as a negative.
    """

    trajectory_id: str
    verifier_score: int
    agent_claimed_success: bool | None = None
    procedure_violated: bool | None = None
    truly_succeeded: bool | None = None
    text: str = ""

    def __post_init__(self) -> None:
        if self.verifier_score not in (0, 1):
            raise ValueError(
                f"verifier_score is the harness's binary verdict; got {self.verifier_score!r}. A "
                f"partial-credit score is a different quantity and rates over it mean something "
                f"else."
            )


@register_payload
@dataclass(frozen=True)
class TwoSidedError:
    """Both of a verifier's error rates, each over the population it applies to.

    ``false_success_rate`` has the verifier's zeros as its denominator and ``corrupt_success_rate``
    has its ones, so the two are not commensurable and adding them means nothing. They are reported
    together because that is the finding: the same scalar is wrong in both directions on the same
    benchmark.
    """

    false_success_rate: Proportion | None
    corrupt_success_rate: Proportion | None
    n_trajectories: int
    n_zeros: int
    n_ones: int
    n_missing_claim: int = 0
    n_missing_violation: int = 0
    interpretation: str = ""
    corpus: str = ""

    @property
    def is_two_sided(self) -> bool:
        """Whether both directions were measured, which is what makes the finding the finding."""
        return self.false_success_rate is not None and self.corrupt_success_rate is not None

    def render(self) -> str:
        fs = (
            self.false_success_rate.render()
            if self.false_success_rate is not None
            else f"NOT MEASURED ({self.n_missing_claim} of {self.n_zeros} zeros carry no claim column)"
        )
        cs = (
            self.corrupt_success_rate.render()
            if self.corrupt_success_rate is not None
            else f"NOT MEASURED ({self.n_missing_violation} of {self.n_ones} ones carry no violation column)"
        )
        return (
            f"verifier verdicts on {self.corpus or 'an unnamed corpus'}: "
            f"{self.n_zeros} zeros, {self.n_ones} ones\n"
            f"  false success  (zeros where the agent claimed success)     {fs}\n"
            f"  corrupt success (ones that violated the procedure)         {cs}\n"
            f"{self.interpretation}"
        )

    def __canonical__(self) -> dict[str, Any]:
        return {
            "false_success_rate": (
                None if self.false_success_rate is None else self.false_success_rate.__canonical__()
            ),
            "corrupt_success_rate": (
                None
                if self.corrupt_success_rate is None
                else self.corrupt_success_rate.__canonical__()
            ),
            "n_trajectories": self.n_trajectories,
            "n_zeros": self.n_zeros,
            "n_ones": self.n_ones,
            "n_missing_claim": self.n_missing_claim,
            "n_missing_violation": self.n_missing_violation,
            "interpretation": self.interpretation,
            "corpus": self.corpus,
        }


def two_sided_error(
    verdicts: Sequence[Verdict], *, level: float = 0.95, corpus: str = ""
) -> TwoSidedError:
    """Both rates, each over the verdicts that carry the column it needs.

    A verdict missing the column a rate needs is excluded from that rate's denominator and counted,
    rather than being scored as a negative. Scoring a missing audit column as "no violation" is
    exactly the assumption that produces a corrupt-success rate of zero on a benchmark that ships no
    procedure audit, which is every benchmark.
    """
    zeros = [v for v in verdicts if v.verifier_score == 0]
    ones = [v for v in verdicts if v.verifier_score == 1]
    claim_known = [v for v in zeros if v.agent_claimed_success is not None]
    violation_known = [v for v in ones if v.procedure_violated is not None]

    fs = (
        wilson_interval(
            sum(1 for v in claim_known if v.agent_claimed_success), len(claim_known), level=level
        )
        if claim_known
        else None
    )
    cs = (
        wilson_interval(
            sum(1 for v in violation_known if v.procedure_violated),
            len(violation_known),
            level=level,
        )
        if violation_known
        else None
    )

    if fs is not None and cs is not None:
        interpretation = (
            f"the scalar is wrong in both directions on the same corpus: {fs.point:.1%} of its "
            f"zeros are trajectories the agent says it completed and {cs.point:.1%} of its ones "
            f"reached the end state by a forbidden route. The two errors do not cancel, they have "
            f"different denominators, and no scalar correction to the verifier recovers the truth."
        )
    elif fs is not None:
        interpretation = (
            f"only the zeros were audited. {fs.point:.1%} of them carry a claimed success, and "
            f"nothing here says how many of the ones are corrupt, so the verifier's error is "
            f"measured in one direction and unmeasured in the other."
        )
    elif cs is not None:
        interpretation = (
            f"only the ones were audited. {cs.point:.1%} of them violated the procedure, and "
            f"nothing here says how many of the zeros carry a claimed success."
        )
    else:
        interpretation = (
            "neither direction was audited. This corpus ships the verifier's scalar and no column "
            "that could contradict it, which is the state every benchmark is in."
        )

    return TwoSidedError(
        false_success_rate=fs,
        corrupt_success_rate=cs,
        n_trajectories=len(verdicts),
        n_zeros=len(zeros),
        n_ones=len(ones),
        n_missing_claim=len(zeros) - len(claim_known),
        n_missing_violation=len(ones) - len(violation_known),
        interpretation=interpretation,
        corpus=corpus,
    )


# ---------------------------------------------------------------------------
# The mandatory comparator
# ---------------------------------------------------------------------------


def false_success_task(
    verdicts: Sequence[Verdict], *, name: str = "false-success"
) -> DetectionTask:
    """The transcript-level detection task: among the verifier's zeros, who claimed success.

    Built only from the zeros, because that is the population the distinction lives in. A detector
    trained on the whole corpus learns to predict the verifier's scalar, which is a different and
    much easier task, and reporting its AUROC as a false-success number is the substitution this
    function exists to prevent.
    """
    zeros = [v for v in verdicts if v.verifier_score == 0 and v.agent_claimed_success is not None]
    return DetectionTask(
        labels=np.array([int(bool(v.agent_claimed_success)) for v in zeros], dtype=int),
        texts=tuple(v.text for v in zeros),
        seed_labels=tuple(v.trajectory_id for v in zeros),
        name=name,
    )


def compare_detector(
    task: DetectionTask,
    detector_scores: Sequence[float] | np.ndarray | None = None,
    *,
    baselines: Sequence[str] = (MANDATORY_BASELINE, "baseline.string_match", "baseline.length"),
    seed: int = 0,
) -> tuple[BaselineBank, BaselineComparison | None]:
    """Run the bank, and place a detector against it when one is supplied.

    `baseline.tfidf` is always in the list this defaults to, and putting it there is not a style
    choice: on this exact distinction it is the method that works and the expensive alternative is
    the method that is anti-correlated with truth. A detector reported without it has been compared
    against nothing that matters.
    """
    ids = list(baselines)
    if MANDATORY_BASELINE not in ids:
        ids.append(MANDATORY_BASELINE)
    bank = run_bank(task, ids)
    if detector_scores is None:
        return bank, None
    return bank, compare_against_baselines(
        detector_scores, task.labels, bank, seed_labels=task.seed_labels, seed=seed
    )


# ---------------------------------------------------------------------------
# The instrument
# ---------------------------------------------------------------------------

VERIFIER_ERROR_ENVELOPE = EnvelopeSpec(
    unconditional=True,
    justification=(
        "a census over audited verdicts. It counts how often the verifier's scalar disagreed with "
        "an audit of the same trajectory and asserts nothing about the training process that "
        "produced the trajectories, so no regime condition applies. The precondition that does "
        "bite, that the audited subset is representative of the corpus, is carried as the "
        "denominator counts so a partial audit is visible rather than assumed away."
    ),
)


class TwoSidedVerifierError(LabelsInstrument):
    """L3: how often the verifier's zeros and its ones are both wrong, on the same corpus.

    Kill condition, from the catalogue: **n/a.**

    The refusal is the common case and it is the finding in miniature. Handed a corpus with the
    verifier's scalar and no audit column, this returns `LABEL_QUALITY_UNKNOWN`, because a
    corrupt-success rate computed by treating an absent audit as "no violation" is zero on every
    benchmark ever shipped and means nothing.
    """

    name = "TwoSidedVerifierError"
    version = "1.0"
    quantity = "labels.false_success_rate"
    capabilities = Capability.NONE
    requires = ACCESS_LABELLED_CORPUS
    envelope = VERIFIER_ERROR_ENVELOPE
    invariance = "none"
    invariance_relation = Relation("invariant")
    baselines = (
        "the verifier's scalar as given, which is what every evaluation harness uses and which "
        "predicts both error rates are zero",
        MANDATORY_BASELINE,
    )
    rung = 0
    faithful_to = "a two-sided confusion count between a programmatic verifier and an audit"
    deviations = (
        "the two rates have different denominators, the verifier's zeros and its ones, so they "
        "are reported side by side and never summed or averaged.",
        "a verdict missing the audit column a rate needs is excluded from that rate rather than "
        "counted as a negative, so both rates are computed on the audited subset and the "
        "unaudited count travels with them.",
        "`agent_claimed_success` is taken as supplied. Extracting it from a transcript is itself "
        "a detection problem and doing it with a judge would import the failure this instrument "
        "documents.",
    )

    def __init__(
        self,
        verdicts: Sequence[Verdict] = (),
        *,
        corpus: str = "",
        level: float = 0.95,
    ) -> None:
        self.verdicts = tuple(verdicts)
        self.corpus = corpus or "unnamed corpus"
        self.level = level

    def measure(self, ctx: Context) -> Any:
        result = two_sided_error(self.verdicts, level=self.level, corpus=self.corpus)
        rate = result.false_success_rate or result.corrupt_success_rate
        return emit_with_reference(
            ctx,
            result,
            quantity=self.quantity,
            uncertainty=Uncertainty(
                n=result.n_trajectories,
                ci_low=None if rate is None else rate.low,
                ci_high=None if rate is None else rate.high,
                ci_level=self.level,
                method="wilson score on each direction separately",
            ),
            baselines={"baseline.scalar_as_given": 0.0},
            subject_extra={"corpus": self.corpus},
        )

    def estimate(self, ctx: Context | None = None) -> Reading:
        ctx = ctx or Context(readout="score")
        if not self.verdicts:
            return label_quality_refusal(
                self.name,
                what=f"corpus {self.corpus!r} carries no verdicts at all",
                remedy=(
                    "pass a sequence of Verdict rows carrying the verifier's binary score and "
                    "whichever audit columns exist."
                ),
                corpus=self.corpus,
            )
        result = two_sided_error(self.verdicts, level=self.level, corpus=self.corpus)
        if result.false_success_rate is None and result.corrupt_success_rate is None:
            return label_quality_refusal(
                self.name,
                what=(
                    f"corpus {self.corpus!r} carries the verifier's scalar on "
                    f"{result.n_trajectories} trajectories and no audit column on any of them, so "
                    f"neither error rate is measurable"
                ),
                remedy=(
                    "audit a sample in both directions and fill in the columns: "
                    "agent_claimed_success on a sample of the zeros, procedure_violated on a "
                    "sample of the ones. Auditing only the ones measures corrupt success and "
                    "leaves false success unknown, which is half the finding and should be "
                    "reported as half."
                ),
                corpus=self.corpus,
                n_trajectories=result.n_trajectories,
                n_zeros=result.n_zeros,
                n_ones=result.n_ones,
            )
        return super().estimate(ctx)


_REGISTERED = False


def register() -> None:
    """Register L3's rung against both quantities. Idempotent."""
    global _REGISTERED
    if _REGISTERED:
        return
    bias = BiasStatement(
        direction="downward",
        why=(
            "an audit of a verifier's verdicts finds the disagreements the auditor can see. A "
            "corrupt success whose forbidden route the auditor did not think to check reads as a "
            "clean pass, so both rates are floors under the verifier's true error."
        ),
    )
    for quantity in ("labels.false_success_rate", "labels.corrupt_success_rate"):
        register_estimator(
            EstimatorEntry(
                quantity=quantity,
                impl=f"{quantity}.r0_audit",
                requires=ACCESS_LABELLED_CORPUS,
                envelope=VERIFIER_ERROR_ENVELOPE,
                rung=0,
                bias=bias,
                cost=CostModel(note="one trajectory inspection per audited verdict"),
                run=two_sided_error,
            )
        )
    _REGISTERED = True


__all__ = [
    "MANDATORY_BASELINE",
    "VERIFIER_ERROR_ENVELOPE",
    "TwoSidedError",
    "TwoSidedVerifierError",
    "Verdict",
    "compare_detector",
    "false_success_task",
    "register",
    "two_sided_error",
]

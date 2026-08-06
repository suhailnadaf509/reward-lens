"""E6, the train-inference logprob mismatch: an instrument, not a footnote.

The sampling engine and the training engine score the same tokens and they disagree. An importance
ratio built from two engines that disagree is measuring the engines rather than the policy, and
prime-rl forms exactly that ratio in one line: ``log_importance_ratio = trainer_logprobs -
inference_logprobs`` (`prime_rl/trainer/rl/loss.py:103`), with `inference_logprobs` arriving from the
packer rather than from the trainer. This is the record-level expression of the numerics
floor: the disagreement between two implementations of the same computation is the floor under
every quantity derived from either of them.

`Turn.logprobs_sampling` and `Turn.logprobs_train` exist as separate fields for this measurement and
for no other reason, and a converter that has only one of them fills one and leaves the other None,
which makes this instrument refuse rather than report a mismatch of zero.

**Kill condition, and it is the good one.** If the mismatch is below the numerics floor everywhere,
that is good news and worth publishing. So a below-floor reading is a **reading** here rather than a
`BELOW_LOD` refusal, which is a deliberate departure from the general rule: the effect
this instrument measures *is* the disagreement, so "the disagreement is smaller than the substrate's
disagreement with itself" is the answer rather than a reason to withhold one.

**The unit, and the one place this instrument and the registry disagree.** The registry gives
`policy.train_infer_logprob_mismatch` the unit `nats/token` under the `tokenization` group, amended
deliberately with four independent lines of evidence. But the mean absolute
per-token gap is **not** invariant under re-tokenisation: split every token in two and the numerator
is roughly unchanged while the denominator doubles. What is exactly invariant is the sequence total,
``|sum_t (logp_train - logp_sampling)| = |log P_train(y) - log P_sampling(y)|``, because both are the
same sequence's log probability under two engines and a tokenisation that decodes to the same string
gives the same total. So the reading carries both, the headline is the per-token mean the registry
registers, and the generated invariance test is asserted on the sequence total with a faithful
generator supplied here. The assertion for this group is "per-token quantities must declare
a normalisation and be invariant under it, or refuse", and the normalisation declared is: the
invariant object is per sequence, the per-token mean divides it by a token count that re-tokenisation
moves, and comparing two per-token means across tokenisers is refused rather than converted.

**Unequal stream lengths are the loudest form of the thing this instrument measures, so they refuse
rather than average.** `Turn.logprob_gap` pairs the two streams with `zip`, which stops at the
shorter one, so a turn where the sampling engine emitted six tokens and the training engine scored
five yields five gaps and no complaint. A length difference is not a missing value: it is the two
engines disagreeing about what the tokens *are*, which is the failure mode an importance ratio
cannot survive at all. `measure_mismatch` compares the lengths itself before it reads any gap and
refuses with the counts when they differ, carrying the mean over the turns that did agree as a
bound. On a hand-built pair of streams at lengths 6 and 5 the truncated sequence total reads 1.15
nats against a true 0.55, which is 2.09x; the direction is not signed, because the discarded tail
can go either way.

**The numerics floor is the spacing at the magnitude the logprobs actually have.** `FLOAT32_EPS` is
`numpy.finfo(numpy.float32).eps`, and that is the spacing of float32 at 1.0. A token logprob is not
near 1.0: at ``|logprob| = 16`` the float32 spacing is 1.9e-06, sixteen times larger, so a floor
taken at 1.0 calls a disagreement negligible that a single float32 rounding at the working magnitude
could have produced on its own. With no measured `LimitOfDetection` the fallback floor is
``numpy.spacing(float32(median |logprob|))`` over the window, which is a measurement of the record
rather than a constant, and the reading names the magnitude it was taken at. `FLOAT32_EPS` stays as
the registered `baseline.float32_epsilon` because that baseline is the named constant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from reward_lens.core.budget import LimitOfDetection
from reward_lens.core.envelope import EnvelopeSpec, RegimeCondition
from reward_lens.core.evidence import register_payload
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.quantity import BaselineID, BiasStatement
from reward_lens.core.reading import Refusal, RefusalReason, bounded_refusal
from reward_lens.core.types import Access, Capability, Component, GaugeStatus, Phase
from reward_lens.measure.base import Context
from reward_lens.measure.estimator._base import EstimatorInstrument
from reward_lens.measure.estimator.spec import ALL_SUBSTRATES, RECORD_ACCESS, iter_groups
from reward_lens.record.schema import Group, Run, Step, Trajectory
from reward_lens.record.turns import Turn, logprob_mismatch

#: What a card that does not measure this is implicitly claiming, and what it would claim if it
#: assumed the two engines were the same implementation. Both are zero, and a baseline of exactly
#: zero is the right comparator here: the whole question is whether the reading is distinguishable
#: from it.
MISMATCH_BASELINES: tuple[BaselineID, ...] = (
    "baseline.engines_agree_exactly",
    "baseline.float32_epsilon",
)

#: `numpy.finfo(numpy.float32).eps`, which is the spacing of float32 **at 1.0**. Kept as the
#: registered `baseline.float32_epsilon` because that baseline names this constant, and not used as
#: the floor: see `float32_floor_at`.
FLOAT32_EPS = float(np.finfo(np.float32).eps)


def float32_floor_at(magnitude: float) -> float:
    """The float32 spacing at a stated logprob magnitude, which is what one rounding can produce.

    `FLOAT32_EPS` is the spacing at 1.0 and logprobs do not live at 1.0. Spacing doubles with every
    binade, so at ``|logprob| = 2`` the floor is 2x `FLOAT32_EPS`, at 8 it is 8x, at 16 it is 16x
    and at 32 it is 32x. Using the 1.0 value on a stream whose typical magnitude is 16 declares a
    disagreement negligible that a single rounding at the working magnitude could have produced,
    which is the wrong direction for a floor to be wrong in.

    A magnitude of zero or a non-finite one falls back to `FLOAT32_EPS`, because the spacing at zero
    is the smallest subnormal and no measurement is that good.
    """
    if not math.isfinite(magnitude) or magnitude <= 0.0:
        return FLOAT32_EPS
    return float(np.spacing(np.float32(abs(magnitude))))


def _typical_magnitude(turns: Iterable[Turn]) -> float:
    """Median absolute logprob over both streams of the comparable turns, or NaN if there is none."""
    values: list[float] = []
    for turn in turns:
        for stream in (turn.logprobs_train, turn.logprobs_sampling):
            if stream is not None:
                values.extend(abs(float(v)) for v in stream)
    arr = np.asarray([v for v in values if math.isfinite(v)], dtype=float)
    return float(np.median(arr)) if arr.size else math.nan


#: The mask policy has to hold still across the window, because the mismatch is averaged over the
#: tokens the loss sees and a window in which the mask changed is a window in which "per token"
#: means two things. That is `MASK_STABLE`, measured by `run.mask_signature`.
MISMATCH_ENVELOPE = EnvelopeSpec(
    requires=frozenset({RegimeCondition.MASK_STABLE}),
    measured_by={RegimeCondition.MASK_STABLE: "run.mask_signature"},
    on_violation="downgrade",
)

MISMATCH_BIAS = BiasStatement(
    direction="downward",
    why=(
        "the mean absolute gap is taken over the turns that carry both streams, and a turn where "
        "the tap wrote only one contributes nothing. Those are disproportionately the truncated and "
        "compacted trajectories, which are the ones most likely to disagree. The turn and token "
        "counts are carried beside the mean so the coverage is visible. The neighbouring case, "
        "where both streams are present at different lengths, is a refusal rather than a bias: it "
        "is not a missing value, it is the two engines disagreeing about what the tokens are."
    ),
)


def _turns(subject: Run | Sequence[Step] | Step | Sequence[Group] | Trajectory) -> list[Turn]:
    if isinstance(subject, Trajectory):
        return list(subject.turns)
    return [t for g in iter_groups(subject) for traj in g.trajectories for t in traj.turns]


def sequence_totals(turns: Iterable[Turn]) -> list[float]:
    """Per turn, the absolute total gap in nats: the tokenisation-invariant object.

    Per turn rather than per trajectory, because a trajectory's turns can be scored by different
    engines under a partial rollout and summing across that boundary would add two different
    comparisons together.

    Turns whose two streams are of different lengths are skipped, not truncated. `logprob_gap` zips
    them and would return a total over the shorter one, which is a total for a prefix of the
    sequence and not for the sequence.
    """
    out = []
    for turn in turns:
        if not _streams_agree_on_length(turn):
            continue
        gap = turn.logprob_gap()
        if gap is None:
            continue
        out.append(abs(sum(gap)))
    return out


def _streams_agree_on_length(turn: Turn) -> bool:
    """Whether the two logprob streams have the same length. False on a turn missing either."""
    train, sampling = turn.logprobs_train, turn.logprobs_sampling
    if train is None or sampling is None:
        return False
    return len(train) == len(sampling)


@dataclass(frozen=True)
class StreamLengthCheck:
    """How many turns the two engines disagreed with each other about the token count on.

    This is E6's own subject at its most severe and it is why the check exists rather than being
    left to `Turn.__post_init__`. That validator only fires when `token_ids` is recorded: it
    compares each per-token array against `len(token_ids)` and returns early when there are none.
    A converter that writes the two logprob streams and no token ids, which is what a tap taking
    logprobs off two engines produces, passes it with streams of different lengths.
    """

    n_turns: int
    n_comparable: int
    n_unequal: int
    worst_train: int = 0
    worst_sampling: int = 0

    @property
    def agrees(self) -> bool:
        return self.n_unequal == 0


def check_stream_lengths(turns: Iterable[Turn]) -> StreamLengthCheck:
    """Count the turns whose two logprob streams are of different lengths, and the worst pair."""
    turns = list(turns)
    n_comparable = 0
    n_unequal = 0
    worst = (0, 0)
    worst_gap = -1
    for turn in turns:
        train, sampling = turn.logprobs_train, turn.logprobs_sampling
        if train is None or sampling is None:
            continue
        n_comparable += 1
        if len(train) == len(sampling):
            continue
        n_unequal += 1
        gap = abs(len(train) - len(sampling))
        if gap > worst_gap:
            worst_gap = gap
            worst = (len(train), len(sampling))
    return StreamLengthCheck(
        n_turns=len(turns),
        n_comparable=n_comparable,
        n_unequal=n_unequal,
        worst_train=worst[0],
        worst_sampling=worst[1],
    )


@register_payload
@dataclass
class MismatchReading:
    """The disagreement between the two engines, per token and per sequence."""

    #: The registered quantity: mean absolute gap, nats per token.
    per_token: float
    n_tokens: int
    n_turns_compared: int
    n_turns_total: int
    #: The tokenisation-invariant object: mean over turns of the absolute total gap, in nats.
    per_sequence: float
    max_per_sequence: float
    #: The numerics floor this was compared against, and where it came from.
    floor: float = math.nan
    floor_source: str = ""
    below_floor: bool | None = None
    #: Median absolute logprob over the window. The fallback floor is the float32 spacing here, not
    #: the spacing at 1.0, and carrying the magnitude is what makes the floor auditable.
    typical_magnitude: float = math.nan
    #: Turns whose two streams had the same length. Equal to `n_turns_compared` on any reading that
    #: was returned at all, because an unequal one refuses.
    n_turns_equal_length: int = 0
    baselines: dict[str, float] = field(default_factory=dict)
    says: str = ""

    def render(self) -> str:
        return self.says


def measure_mismatch(
    subject: Run | Sequence[Step] | Step | Sequence[Group] | Trajectory,
    *,
    lod: LimitOfDetection | None = None,
    instrument: str = "LogprobMismatch",
) -> MismatchReading | Refusal:
    """The mismatch, or the refusal that names the stream the record does not carry."""
    turns = _turns(subject)
    if not turns:
        return Refusal(
            instrument=instrument,
            reason=RefusalReason.ACCESS_INSUFFICIENT,
            detail="this window contains no turns, so there are no logprobs to compare",
            remedy=(
                "point this at a Run, a Step, a sequence of Groups, or a Trajectory carrying turns."
            ),
            statistics={"n_turns": 0},
        )

    mean_gap, n_tokens = logprob_mismatch(turns)
    comparable = [t for t in turns if t.has_both_logprob_streams]
    if n_tokens == 0:
        have_sampling = sum(1 for t in turns if t.logprobs_sampling is not None)
        have_train = sum(1 for t in turns if t.logprobs_train is not None)
        return Refusal(
            instrument=instrument,
            reason=RefusalReason.ACCESS_INSUFFICIENT,
            detail=(
                f"no turn in this window carries both logprob streams: {have_sampling} of "
                f"{len(turns)} turns have `logprobs_sampling` and {have_train} have "
                f"`logprobs_train`. A mismatch of zero would mean the two engines agreed exactly, "
                f"which is a different claim from having nothing to compare."
            ),
            remedy=(
                "record `Turn.logprobs_sampling` from the generation engine and "
                "`Turn.logprobs_train` from the training forward pass, on the same tokens in the "
                "same order. Every framework computes both: TRL requires `logprobs` back from "
                "`rollout_func` (`grpo_trainer.py:2161`) and prime-rl carries "
                "`inference_logprobs` and `trainer_logprobs` side by side on its loss input "
                "(`loss.py:23-24`). What is missing is the tap that writes them down."
            ),
            statistics={
                "n_turns": len(turns),
                "n_with_sampling": have_sampling,
                "n_with_train": have_train,
            },
        )

    lengths = check_stream_lengths(turns)
    magnitude = _typical_magnitude(turns)
    if not lengths.agrees:
        return _length_refusal(instrument, turns, lengths, magnitude, lod)

    totals = sequence_totals(turns)
    floor = math.nan
    floor_source = ""
    below: bool | None = None
    if lod is not None and lod.is_determinate:
        floor = lod.lod
        floor_source = "LimitOfDetection supplied on the context"
        below = bool(mean_gap < floor)
    else:
        floor = float32_floor_at(magnitude)
        floor_source = (
            f"float32 spacing at the median absolute logprob of this window, {magnitude:.3g} "
            f"nats; the weakest possible floor, and no measured LOD was supplied"
        )
        below = bool(mean_gap < floor)

    reading = MismatchReading(
        per_token=float(mean_gap),
        n_tokens=int(n_tokens),
        n_turns_compared=len(comparable),
        n_turns_total=len(turns),
        per_sequence=float(np.mean(totals)) if totals else math.nan,
        max_per_sequence=float(np.max(totals)) if totals else math.nan,
        floor=float(floor),
        floor_source=floor_source,
        below_floor=below,
        typical_magnitude=magnitude,
        n_turns_equal_length=lengths.n_comparable,
        baselines={
            "baseline.engines_agree_exactly": 0.0,
            "baseline.float32_epsilon": FLOAT32_EPS,
        },
    )
    if below:
        reading.says = (
            f"The sampling engine and the training engine agree to within {reading.per_token:.3g} "
            f"nats per token over {reading.n_tokens:,} tokens, which is below the numerics floor "
            f"of {floor:.3g} ({floor_source}). The importance ratio on this record is measuring "
            f"the policy rather than the engines, and that is worth publishing."
        )
    else:
        reading.says = (
            f"The sampling engine and the training engine disagree by {reading.per_token:.3g} nats "
            f"per token over {reading.n_tokens:,} tokens, and by {reading.per_sequence:.3g} nats "
            f"per sequence on average with a worst turn at {reading.max_per_sequence:.3g}. Against "
            f"a numerics floor of {floor:.3g} ({floor_source}), an importance ratio built from "
            f"these two streams is measuring the engines and not only the policy."
        )
    return reading


def _length_refusal(
    instrument: str,
    turns: Sequence[Turn],
    lengths: StreamLengthCheck,
    magnitude: float,
    lod: LimitOfDetection | None,
) -> Refusal:
    """The two engines disagree about the token count, which is the finding rather than a gap.

    The bound is the mismatch over the turns where the lengths did agree. It is a bound in the
    weak sense that it is a real measurement over a real subpopulation, and its direction is not
    signed: the turns that tokenise identically are the ones where the engines are most alike, so
    the bound is likely low, but a single divergent turn can carry a large gap either way.
    """
    equal = [t for t in turns if _streams_agree_on_length(t)]
    mean_gap, n_tokens = logprob_mismatch(equal)
    totals = sequence_totals(equal)
    floor = lod.lod if (lod is not None and lod.is_determinate) else float32_floor_at(magnitude)
    bound = MismatchReading(
        per_token=float(mean_gap),
        n_tokens=int(n_tokens),
        n_turns_compared=len(equal),
        n_turns_total=len(turns),
        per_sequence=float(np.mean(totals)) if totals else math.nan,
        max_per_sequence=float(np.max(totals)) if totals else math.nan,
        floor=float(floor),
        floor_source="carried on a refusal; the window is not comparable as a whole",
        below_floor=None,
        typical_magnitude=magnitude,
        n_turns_equal_length=len(equal),
        baselines={
            "baseline.engines_agree_exactly": 0.0,
            "baseline.float32_epsilon": FLOAT32_EPS,
        },
    )
    bound.says = (
        f"Over the {len(equal)} of {lengths.n_comparable} comparable turns where the two engines "
        f"agreed on the token count, the mismatch is {mean_gap:.3g} nats per token across "
        f"{n_tokens:,} tokens. The other {lengths.n_unequal} turns are not comparable at all."
    )
    return bounded_refusal(
        instrument=instrument,
        reason=RefusalReason.UNIT_MISMATCH,
        detail=(
            f"{lengths.n_unequal} of {lengths.n_comparable} turns carrying both logprob streams "
            f"have streams of different lengths, worst {lengths.worst_train} training against "
            f"{lengths.worst_sampling} sampling. The two engines did not tokenise the same text "
            f"the same way, so there is no per-token correspondence between the streams and a "
            f"per-token gap has no denominator. This is the failure E6 exists to detect, at its "
            f"most severe: `Turn.logprob_gap` pairs the streams with `zip` and would report a mean "
            f"over the shorter one, which is a statement about a prefix."
        ),
        remedy=(
            "make the sampling engine and the training engine tokenise identically, then re-run. "
            "In practice that means the two are using different tokenizer revisions, different "
            "special-token handling, or a chat template applied on one side only. Until they "
            "agree, the importance ratio built from these two streams is not defined and neither "
            "is this quantity. The bound on this refusal is the mismatch over the turns where they "
            "did agree, which is a measurement of the easy cases."
        ),
        bound=_bare_evidence(bound),
        n_turns=lengths.n_turns,
        n_comparable=lengths.n_comparable,
        n_unequal=lengths.n_unequal,
        worst_train_tokens=lengths.worst_train,
        worst_sampling_tokens=lengths.worst_sampling,
    )


def _bare_evidence(reading: MismatchReading) -> Any:
    """An Evidence wrapper for a bound travelling on a refusal, built before any gate applies."""
    from reward_lens.core.evidence import make_evidence
    from reward_lens.core.types import SubjectRef

    return make_evidence(
        observable="LogprobMismatch",
        observable_version="1.0",
        subject=SubjectRef(signals=(), dataset=None, readout="reward"),
        value=reading,
        gauge=GaugeStatus.INVARIANT,
    )


class LogprobMismatch(EstimatorInstrument):
    """E6. How far apart the sampling engine and the training engine are, in nats per token.

    Kill condition, from the catalogue record: if the mismatch is below the numerics floor
    everywhere, which would be good news and worth publishing.
    """

    name = "LogprobMismatch"
    version = "1.0"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "E6"
    deviations = (
        "a below-floor reading is Evidence rather than a `BELOW_LOD` refusal. The general rule "
        "refuses an effect smaller than the substrate's disagreement with itself; here the effect "
        "*is* that disagreement, so the rule would make the instrument refuse in exactly the case "
        "its own kill condition calls good news",
        "the reading carries a per-sequence total alongside the registered per-token mean, because "
        "the per-token mean is not invariant under re-tokenisation and the total is. The generated "
        "`tokenization` test is asserted on the total with a generator supplied by this package, "
        "since the group's default stand-in splits token ids without splitting the logprob mass "
        "that a real re-tokenisation would divide between them",
        "with no measured `LimitOfDetection` on the context, the floor falls back to the float32 "
        "spacing at the window's median absolute logprob and says so. That is the weakest possible "
        "floor and it is not a measurement of the substrate; a real one comes from running the "
        "same forward pass twice. It is not `numpy.finfo(float32).eps`, which is the spacing at "
        "1.0 and is up to 16x too permissive at the magnitudes token logprobs have",
        "a window where any turn's two logprob streams differ in length refuses with "
        "`UNIT_MISMATCH` rather than reporting a mean over the truncated pairs. `Turn.logprob_gap` "
        "zips the streams, and `Turn.__post_init__` only checks their lengths when `token_ids` was "
        "also recorded, so the truncation is silent on exactly the records a two-engine tap "
        "produces. The bound is the mismatch over the turns whose lengths did agree",
    )

    quantity = "policy.train_infer_logprob_mismatch"
    requires: dict[Component, Access] = {**RECORD_ACCESS, Component.POLICY: Access.RECORD}
    substrates = ALL_SUBSTRATES
    phases = frozenset({Phase.IN_RUN, Phase.POST_RUN})
    envelope = MISMATCH_ENVELOPE
    invariance = "tokenization"
    invariance_relation = INVARIANT
    baselines = MISMATCH_BASELINES
    rung = 0

    def __init__(
        self,
        subject: Run | Sequence[Step] | Step | Sequence[Group] | Trajectory | None = None,
        *,
        lod: LimitOfDetection | None = None,
    ) -> None:
        self.subject = subject
        self.lod = lod

    def estimate(self, ctx: Context) -> Any:
        # The context's LOD is the one the rule means; the constructor's is for a caller who has
        # a floor and no Context. The context wins when both are present, because the context is
        # what the runner and the capability report consult.
        if ctx.lod is not None:
            self.lod = ctx.lod
        return super().estimate(ctx)

    def compute(self) -> Any:
        if self.subject is None:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail="no record was supplied, so there are no logprob streams to compare",
                remedy=(
                    "pass `subject=` a Run, a Step, a sequence of Groups, or a Trajectory whose "
                    "turns carry both `logprobs_sampling` and `logprobs_train`."
                ),
            )
        return measure_mismatch(self.subject, lod=self.lod, instrument=self.name)


__all__ = [
    "FLOAT32_EPS",
    "MISMATCH_BASELINES",
    "MISMATCH_BIAS",
    "MISMATCH_ENVELOPE",
    "LogprobMismatch",
    "MismatchReading",
    "StreamLengthCheck",
    "check_stream_lengths",
    "float32_floor_at",
    "measure_mismatch",
    "sequence_totals",
]

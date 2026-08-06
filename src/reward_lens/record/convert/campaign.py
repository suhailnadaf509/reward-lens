"""Converting the 2.0 campaign's evidence store into a `Run` of kind ``eval``.

The campaign scored thirteen reward models against fixed banks of responses. Nothing in it
optimised a policy against a reward, so the loop diagram is missing its optimizer
and its estimator, which is precisely what ``kind="eval"`` means. Everything below follows from
that one fact, and the places where the record has no room for what the campaign recorded are
listed in `ConversionReport.findings` rather than papered over.

**What maps where.**

Each ``campaign.scores`` row is a bank of ``N`` items scored by one reward model, with a layout
naming the shape: ``flat`` is one score per item, ``pairs`` is two, ``best-of-4`` is four,
``bank`` is thirty-two. So one bank row becomes one `Step`, one item becomes one `Group`, and one
column becomes one `Trajectory`. The eight ``campaign.prm.steps`` rows are the only genuinely
multi-turn data in the store: a ProcessBench solution is a sequence of reasoning steps, each with
its own score, so each step becomes a `Turn` carrying `Turn.step_score`.

**Step index is a partition ordinal here, not an optimizer step, and that is a departure.**
The canonical schema defines `Step` as one optimizer update. An evaluation performs zero of them,
so the faithful record has either one step holding six hundred thousand groups or no step level at
all.
Neither is usable: the first defeats `StepStream`, whose whole purpose is that reading eleven steps
of a long run does not materialise the rest, and the second is not expressible. Steps here are the
banks in ``created_at`` order, `OptimizerTelemetry` is all-`None` rather than all-zero, and
`Step.schedule` is empty. A reader who treats the index as time will be wrong, and
`ConversionReport.findings` says so.

**Ids are readable and join back to the source.** `group_id` and `trajectory_id` in
`record.schema` return content hashes, which is right for a record being written for the first
time and wrong here: the campaign's own ``roster_key``, ``slice`` and ``item_id`` are already
stable identifiers, and hashing them would make it impossible to look a group up in the store it
came from. So a group is ``"<roster>/<slice>/<item_id>"`` and a trajectory is that with ``"#k"``
appended.

**One Run, thirteen graders, and the tension that leaves.** `Run.components` holds one
`ComponentRef` per node, and the campaign put thirteen reward models on the grader node, eleven of
the twelve banks being scored by more than one of them. The schema's own answer to that is
`Run.arms`, sibling arms of a controlled comparison, which is what `campaign_arms` builds. The
single fleet Run that `convert_campaign` returns declares a COMPOSITE grader naming the roster, and
keeps the per-trajectory attribution in the score tree's `GraderCallRef` so nothing is lost by
reading the fleet as one run.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from reward_lens.core.envelope import ConditionReading, RegimeCondition, RegimeReading
from reward_lens.core.types import (
    Access,
    AccessMatrix,
    Component,
    DatasetID,
    ModelFP,
    Site,
    Substrate,
)
from reward_lens.record.convert.store import CampaignRow, CampaignStore
from reward_lens.record.schema import (
    ComponentRef,
    EstimatorSpec,
    Group,
    GroupID,
    GroupStats,
    OptimizerTelemetry,
    RecordSamplingPolicy,
    RegimeDeclaration,
    Run,
    RunID,
    RunLineage,
    SamplingScheme,
    Step,
    StepStream,
    TaskID,
    Trajectory,
    make_trajectory,
    run_id,
)
from reward_lens.record.tensors import (
    AbsenceReason,
    AbsentRef,
    CaptureRef,
    CaptureSpec,
    Engine,
)
from reward_lens.record.turns import Turn

#: The observables that carry per-item rollout data. Everything else in the store is a derived
#: measurement about a bank rather than a record of one, and the record has no level for those;
#: see `ConversionReport.findings`.
SCORE_OBSERVABLE = "campaign.scores"
STEP_SCORE_OBSERVABLE = "campaign.prm.steps"
CAPTURE_OBSERVABLES = ("campaign.capture", "campaign.capture.fp32")
READOUT_OBSERVABLE = "campaign.w_r"

#: The group size each layout name implies, as the store's own shapes define it. The conversion
#: reads K off the score array's trailing dimension and never off this table, and the reason is
#: `layout_audit`: two of the 992 banks declare ``bank`` and carry a ``(2000, 4)`` array, against
#: fourteen that declare ``bank`` and carry ``(N, 32)``. So the layout name does not determine the
#: group size in this store, the shape does, and the table exists to count where they disagree.
LAYOUT_K: Mapping[str, int] = {
    "flat": 1,
    "pairs": 2,
    "probe-pair": 2,
    "triple": 3,
    "probe-triple": 3,
    "best-of-4": 4,
    "bank": 32,
}

#: The policy version recorded on a bank response. The campaign scored public preference datasets
#: and never recorded which model wrote the responses, so this is the literal string rather than a
#: guess. `RegimeCondition.NEAR_POLICY` is reported as indeterminate for the same reason.
UNRECORDED_POLICY = "unrecorded"


# ---------------------------------------------------------------------------
# The score tree seam
# ---------------------------------------------------------------------------


def _score_tree(name: str, value: float | None, grader: str, *, abstained: bool) -> Any:
    """One recorded score, as a `ScoreTree` leaf where `record.scores` is available.

    Soft-imported rather than imported at module scope: a converter that fails outright when
    `record.scores` cannot be imported is worse than one that records a mapping. The fallback
    carries the same three facts, so nothing downstream loses the grader attribution; it only loses
    the type.
    """
    try:
        from reward_lens.record.scores import GraderCallRef, Leaf
    except ImportError:  # pragma: no cover - fallback, unreachable in a complete install
        return {"name": name, "value": value, "grader": grader, "abstained": abstained}
    return Leaf(
        name=name,
        value=value,
        grader_call=GraderCallRef(grader=grader),
        abstained=abstained,
    )


def _blind_label(value: Any, *, key: str) -> tuple[Any, bool]:
    """Blind a held-out label, or decline to carry it and say so.

    Returns the value to store and whether it was blinded. `Trajectory.labels` is typed
    `Mapping[str, Blind[LabelValue]]`, and the entire guarantee of that type is that a detector's
    signature cannot accept the label. Writing an oracle into that field unwrapped would hand over
    the thing the type exists to withhold, so when `record.labels` is not importable the label is
    dropped and counted in `ConversionReport.labels_declined` rather than carried.
    """
    try:
        from reward_lens.record.labels import LabelQuality, blind
    except ImportError:  # pragma: no cover - fallback, unreachable in a complete install
        return None, False
    # No error rate: nobody audited ProcessBench's error-step annotations for this campaign, and
    # `LabelQuality()` with `error_rate=None` is how that is said. Any scoring read against them
    # is then refused, which is the correct outcome rather than an obstacle.
    return blind(value, key=key, quality=LabelQuality(method="", measured_by="ProcessBench")), True


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


@dataclass
class ConversionReport:
    """What the conversion carried, what it declined, and where the schema had no room.

    Every count here comes from the conversion that produced it. ``findings`` is the list this
    package exists to produce alongside the record: each entry is a place the canonical record
    could not hold something the campaign measured, stated as a fact about the schema
    rather than as a complaint about the data.
    """

    store_rows: int = 0
    observables: int = 0
    steps: int = 0
    groups: int = 0
    trajectories: int = 0
    turns: int = 0
    scores_carried: int = 0
    abstentions: int = 0
    graders: tuple[str, ...] = ()
    banks: tuple[str, ...] = ()
    labels_blinded: int = 0
    labels_declined: int = 0
    capture_manifests: int = 0
    absent_tensors: int = 0
    readout_vectors: int = 0
    layout_mismatches: tuple[str, ...] = ()
    unrepresented: dict[str, int] = field(default_factory=dict)
    findings: tuple[str, ...] = ()
    #: True once `count_run` has walked the stream. Until then the per-object counts are zero
    #: because nothing has been decoded, and zero-because-unwalked must not read as zero-because-
    #: empty.
    counted: bool = False

    @property
    def unrepresented_rows(self) -> int:
        """Rows whose content the record hierarchy has no level for."""
        return sum(self.unrepresented.values())

    def render(self) -> str:
        counts = (
            f"  {self.groups} groups, {self.trajectories} trajectories, {self.turns} turns, "
            f"{self.scores_carried} scores, {self.abstentions} abstentions, "
            f"{self.labels_blinded} labels blinded, {self.labels_declined} declined"
            if self.counted
            else "  per-object counts not taken; call count_run() to walk the stream"
        )
        lines = [
            f"campaign -> Run(kind=eval): {self.store_rows} store rows over "
            f"{self.observables} observables",
            f"  {self.steps} steps",
            counts,
            f"  {len(self.graders)} graders, {len(self.banks)} banks, "
            f"{len(self.layout_mismatches)} layout/shape disagreements",
            f"  {self.capture_manifests} capture manifests -> {self.absent_tensors} absent tensor "
            f"refs, {self.readout_vectors} readout vectors",
            f"  {self.unrepresented_rows} rows the record has no level for, over "
            f"{len(self.unrepresented)} observables",
        ]
        for finding in self.findings:
            lines.append(f"  finding: {finding}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Bank decoding
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Bank:
    """One decoded score bank: the unit that becomes a `Step`."""

    row: CampaignRow
    index: int
    item_ids: tuple[str, ...]
    scores: np.ndarray
    layout: str
    meta: Mapping[str, Any]

    @property
    def k(self) -> int:
        return 1 if self.scores.ndim == 1 else int(self.scores.shape[1])

    @property
    def layout_mismatch(self) -> bool:
        """Whether the declared layout disagrees with the array's own trailing dimension.

        The shape wins, because the array is the data and the name is a label somebody typed. The
        disagreement is counted rather than swallowed: a converter that had trusted ``bank`` would
        have built two thousand groups of thirty-two out of a ``(2000, 4)`` array and produced an
        `IndexError` four frames from the cause.
        """
        declared = LAYOUT_K.get(self.layout)
        return declared is not None and declared != self.k


def _as_list(value: Any) -> list[Any]:
    """The codec renders a sequence as itself once decoded; tolerate a bare tuple too."""
    if value is None:
        return []
    return list(value)


def _decode_bank(store: CampaignStore, row: CampaignRow, index: int) -> Bank:
    fields = store.value(row)
    item_ids = tuple(str(i) for i in _as_list(fields.get("item_ids")))
    scores = np.asarray(fields.get("scores"), dtype=np.float64)
    if scores.ndim == 0:
        scores = scores.reshape(1)
    layout = str(fields.get("layout", "flat"))
    meta = dict(fields.get("meta") or {})
    return Bank(row=row, index=index, item_ids=item_ids, scores=scores, layout=layout, meta=meta)


# ---------------------------------------------------------------------------
# Regime: measured where it can be, absent where nobody looked
# ---------------------------------------------------------------------------


def _condition(
    condition: RegimeCondition,
    holds: bool | None,
    statistic: float,
    threshold: float,
    detail: str,
) -> ConditionReading:
    return ConditionReading(
        condition=condition,
        holds=holds,
        statistic=statistic,
        threshold=threshold,
        detail=detail,
    )


def _bank_regime(bank: Bank, *, grader_moved: bool) -> RegimeReading:
    """What is true about this bank, measured from it.

    Six of the twelve conditions are answerable from a scored bank and six are not. The six that
    are not stay **absent** from the reading rather than present with ``holds=None``, because
    `EnvelopeSpec.classify` distinguishes those and the distinction is the useful one: absent means
    nobody measured it, present-and-None means somebody measured it and could not tell.

    ``GROUP_NONDEGENERATE`` is reported as a fraction and resolved to a verdict only at the two
    ends. A bank of one-score groups fails it at any threshold; a bank where every group has spread
    passes at any threshold; in between the answer depends on a threshold this module has no
    standing to set, so it reports the fraction and says the threshold is the instrument's.
    """
    k = bank.k
    scores = bank.scores if bank.scores.ndim == 2 else bank.scores.reshape(-1, 1)
    if k > 1:
        # A group whose scores are all abstentions has no spread to compute, and numpy warns
        # about the empty slice. That case is already `False` below, so the warning is noise.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            spreads = np.nanstd(scores, axis=1)
        live = float(np.mean(np.nan_to_num(spreads, nan=0.0) > 0.0)) if spreads.size else 0.0
    else:
        live = 0.0
    if k <= 1 or live == 0.0:
        nondegenerate: bool | None = False
        detail = (
            f"layout {bank.layout!r} gives K={k}; a group of one has no spread at any threshold"
            if k <= 1
            else "every group has zero spread"
        )
    elif live == 1.0:
        nondegenerate = True
        detail = f"K={k} and every one of the {scores.shape[0]} groups has non-zero spread"
    else:
        nondegenerate = None
        detail = (
            f"K={k} and {live:.4g} of the {scores.shape[0]} groups have non-zero spread; the "
            f"fraction that counts as adequate is the instrument's threshold, not this record's"
        )

    conditions = {
        RegimeCondition.GROUP_NONDEGENERATE: _condition(
            RegimeCondition.GROUP_NONDEGENERATE, nondegenerate, live, math.nan, detail
        ),
        RegimeCondition.NO_COMPACTION: _condition(
            RegimeCondition.NO_COMPACTION,
            True,
            0.0,
            0.0,
            "an evaluation over a fixed bank rewrites no prefix, so no compaction event exists",
        ),
        RegimeCondition.EXOGENOUS_CURRICULUM: _condition(
            RegimeCondition.EXOGENOUS_CURRICULUM,
            True,
            0.0,
            0.0,
            f"bank {bank.row.bank!r} is a fixed dataset; no score fed back into item selection",
        ),
        RegimeCondition.STATIONARY_GRADER: _condition(
            RegimeCondition.STATIONARY_GRADER,
            None if grader_moved else True,
            math.nan,
            math.nan,
            (
                "this roster has checkpoint rows in the store, so its weights moved somewhere in "
                "the campaign and nothing here says whether they moved during this bank"
                if grader_moved
                else "one frozen checkpoint scored this bank; the store holds no weight update "
                "for it"
            ),
        ),
        RegimeCondition.NEAR_POLICY: _condition(
            RegimeCondition.NEAR_POLICY,
            None,
            math.nan,
            math.nan,
            "the bank's responses have no recorded generating policy, so staleness is not "
            "computable and the singular-policy half of the condition cannot be established",
        ),
        RegimeCondition.MASK_STABLE: _condition(
            RegimeCondition.MASK_STABLE,
            None,
            math.nan,
            math.nan,
            "no loss mask was recorded on any turn; the mask policy signature reads "
            "'assistant=unrecorded'",
        ),
    }
    return RegimeReading(conditions=conditions)


# ---------------------------------------------------------------------------
# Step construction
# ---------------------------------------------------------------------------

#: An evaluation ran no estimator, so nothing turned scores into advantages. Every field says so
#: rather than carrying a plausible default: ``family="none"`` is a statement, ``family="grpo"``
#: with everything else left alone would be a fabrication.
NO_ESTIMATOR = EstimatorSpec(
    family="none",
    group_centred=False,
    std_normalised=False,
    std_epsilon=None,
    degenerate_policy="not applicable: no advantage was computed from these scores",
    aggregation="none",
    loss_mask_policy="not applicable: no loss was taken",
)

#: The estimator's epsilon decides where a degenerate group starts, and an evaluation has no
#: estimator to read one off. Zero is the only defensible substitute: it makes `degenerate` mean
#: "the scores in this group are exactly equal", which is a fact about the bank rather than a
#: threshold borrowed from a trainer that never ran.
NO_ESTIMATOR_EPSILON = 0.0


def _capture_ref(manifest: Mapping[str, Any]) -> tuple[CaptureRef, int]:
    """A capture manifest as a `CaptureRef` whose tensors are honestly absent.

    The campaign wrote activations to a Modal volume at ``/store/captures`` and shipped the
    manifests, not the bytes. `AbsenceReason.SHARD_MISSING` is the exact reason for that: the
    record travelled and the tensors did not. Every site gets its own `AbsentRef` so a caller that
    asks for layer 27 gets an answer about layer 27.
    """
    sites: list[Site] = []
    names: list[str] = []
    for raw in _as_list(manifest.get("sites")):
        name = str(raw)
        names.append(name)
        head, _, point = name.partition(".")
        try:
            layer = int(head.lstrip("Ll"))
        except ValueError:
            continue
        sites.append(Site(layer=layer, point=point or "resid_post"))  # type: ignore[arg-type]
    meta = dict(manifest.get("meta") or {})
    spec = CaptureSpec(
        sites=tuple(sites),
        dtype=str(manifest.get("dtype", "unknown")),
        stride=1,
        max_tokens=None,
        extra={
            "position": manifest.get("position"),
            "full_sequence": meta.get("full_sequence"),
            "n_items": manifest.get("n_items"),
            "content_hash": manifest.get("content_hash"),
        },
    )
    detail = (
        f"capture {manifest.get('content_hash')} of {manifest.get('n_items')} items "
        f"({manifest.get('bytes_total')} bytes) was written to "
        f"{meta.get('activation_root', 'an activation root')} and did not travel with this store"
    )
    tensors = {
        name: AbsentRef.of(
            AbsenceReason.SHARD_MISSING,
            detail=detail,
            bytes_total=float(manifest.get("bytes_total") or 0.0),
        )
        for name in names
    }
    return CaptureRef(spec=spec, tensors=tensors), len(tensors)


def _step_from_bank(
    bank: Bank,
    *,
    grader_moved: bool,
    capture: CaptureRef | None,
) -> tuple[Step, int, int, int, int]:
    """One bank as a `Step`, with its group, trajectory, score and abstention counts."""
    roster = bank.row.roster_key or "unknown"
    slice_name = bank.row.slice_name or bank.row.bank or "unknown"
    scores = bank.scores if bank.scores.ndim == 2 else bank.scores.reshape(-1, 1)
    n = min(len(bank.item_ids), scores.shape[0])
    groups: list[Group] = []
    n_traj = 0
    n_scores = 0
    n_abstain = 0
    for i in range(n):
        item = bank.item_ids[i]
        gid = f"{roster}/{slice_name}/{item}"
        row = scores[i]
        values: list[float | None] = []
        trajectories: list[Trajectory] = []
        for k in range(row.shape[0]):
            raw = float(row[k])
            abstained = not math.isfinite(raw)
            values.append(None if abstained else raw)
            n_scores += 1
            n_abstain += int(abstained)
            trajectories.append(
                make_trajectory(
                    id=f"{gid}#{k}",
                    task_ref=item,
                    turns=(_bank_turn(),),
                    scores=_score_tree(
                        "reward", None if abstained else raw, roster, abstained=abstained
                    ),
                    advantage=None,
                    policy_version=UNRECORDED_POLICY,
                    engine=Engine(name=UNRECORDED_POLICY),
                    capture=capture,
                )
            )
        n_traj += len(trajectories)
        groups.append(
            Group(
                id=GroupID(gid),
                task_ref=TaskID(item),
                trajectories=tuple(trajectories),
                estimator=NO_ESTIMATOR,
                group_stats=GroupStats.from_scores(
                    values, std_epsilon=NO_ESTIMATOR_EPSILON, failure_at=None
                ),
            )
        )
    step = Step(
        index=bank.index,
        groups=tuple(groups),
        schedule={},
        optimizer=OptimizerTelemetry(),
        regime_measured=_bank_regime(bank, grader_moved=grader_moved),
    )
    return step, len(groups), n_traj, n_scores, n_abstain


def _bank_turn() -> Turn:
    """The single assistant turn a bank response is, with its content unrecorded.

    Zero turns was the other option and it is a worse lie. A bank item is a prompt and a response
    that a reward model scored; the response existed, and the campaign stored the score without the
    text. One turn with empty text and ``token_ids=None`` says that. Zero turns would say the
    rollout was empty, which is false, and it would also make `Trajectory.n_turns` zero on six
    hundred thousand trajectories that each had exactly one action.
    """
    return Turn(index=0, role="assistant", text="")


# ---------------------------------------------------------------------------
# ProcessBench: the only multi-turn data in the store
# ---------------------------------------------------------------------------


def _step_from_prm(
    store: CampaignStore,
    row: CampaignRow,
    index: int,
) -> tuple[Step, int, int, int, int, int, int]:
    """One ``campaign.prm.steps`` row as a `Step` of multi-turn trajectories.

    The payload is three arrays: ``values`` holds every step score concatenated, ``offsets`` is the
    prefix index into it, and ``labels`` is the earliest-error step index with -1 for a correct
    solution. So item ``i`` has ``offsets[i+1] - offsets[i]`` reasoning steps, each of which is a
    `Turn` with `Turn.step_score` set. Returns the step and its group, trajectory, turn, score,
    blinded-label and declined-label counts.
    """
    fields = store.value(row)
    item_ids = tuple(str(i) for i in _as_list(fields.get("item_ids")))
    values = np.asarray(fields.get("values"), dtype=np.float64).ravel()
    offsets = np.asarray(fields.get("offsets"), dtype=np.int64).ravel()
    labels = np.asarray(fields.get("labels"), dtype=np.int64).ravel()
    roster = row.roster_key or "unknown"
    slice_name = row.slice_name or "unknown"

    groups: list[Group] = []
    n_turns = 0
    n_scores = 0
    blinded = 0
    declined = 0
    for i, item in enumerate(item_ids):
        if i + 1 >= offsets.size:
            break
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        steps = values[lo:hi]
        turns = tuple(
            Turn(index=j, role="assistant", text="", step_score=float(s))
            for j, s in enumerate(steps)
        )
        n_turns += len(turns)
        n_scores += len(turns)
        label_value, was_blinded = (
            _blind_label(int(labels[i]), key="earliest_error_step")
            if i < labels.size
            else (None, False)
        )
        if label_value is None:
            declined += int(i < labels.size)
        else:
            blinded += 1
        gid = f"{roster}/{slice_name}/{item}"
        # A solution with no recorded steps is a real event (an empty completion), and a
        # trajectory with no turns carries no provenance, which `check_tiling` allows only in
        # that case.
        traj = make_trajectory(
            id=f"{gid}#0",
            task_ref=item,
            turns=turns,
            scores=_score_tree(
                "process_reward",
                float(np.mean(steps)) if steps.size else None,
                roster,
                abstained=not steps.size,
            ),
            advantage=None,
            policy_version=UNRECORDED_POLICY,
            engine=Engine(name=UNRECORDED_POLICY),
            labels={} if label_value is None else {"earliest_error_step": label_value},
        )
        groups.append(
            Group(
                id=GroupID(gid),
                task_ref=TaskID(item),
                trajectories=(traj,),
                estimator=NO_ESTIMATOR,
                group_stats=GroupStats.from_scores(
                    [float(np.mean(steps)) if steps.size else None],
                    std_epsilon=NO_ESTIMATOR_EPSILON,
                    failure_at=None,
                ),
            )
        )
    step = Step(
        index=index,
        groups=tuple(groups),
        schedule={},
        optimizer=OptimizerTelemetry(),
        regime_measured=RegimeReading(
            conditions={
                RegimeCondition.NO_COMPACTION: _condition(
                    RegimeCondition.NO_COMPACTION,
                    True,
                    0.0,
                    0.0,
                    "a ProcessBench solution is scored as written; no prefix was rewritten",
                ),
                RegimeCondition.NEAR_POLICY: _condition(
                    RegimeCondition.NEAR_POLICY,
                    None,
                    math.nan,
                    math.nan,
                    "the solutions were written by models the campaign did not record",
                ),
            }
        ),
    )
    return step, len(groups), len(groups), n_turns, n_scores, blinded, declined


# ---------------------------------------------------------------------------
# The lazy stream
# ---------------------------------------------------------------------------


class CampaignStepStream(StepStream):
    """A `StepStream` that decodes one bank at a time and holds none of them.

    The campaign's 992 score banks and 8 ProcessBench banks carry 616,023 items and 1,062,908
    scalar scores between them. Materialising every trajectory at once is roughly a million frozen
    dataclasses; decoding one bank at a time is at most 500 items. Slicing decodes only the banks
    whose index the slice covers, which is the same guarantee `PartitionedStepStream` gives over a
    written record and the reason `Run.steps` is not a tuple.
    """

    def __init__(self, plan: "_StreamPlan") -> None:
        self._plan = plan

    def __iter__(self) -> Iterator[Step]:
        return self.slice(0, (max(self.indices) + 1) if self.indices else 0)

    def __len__(self) -> int:
        return len(self._plan.indices)

    @property
    def indices(self) -> tuple[int, ...]:
        return self._plan.indices

    def slice(self, lo: int, hi: int) -> Iterator[Step]:
        for index in self._plan.indices:
            if lo <= index < hi:
                yield self._plan.build(index)


@dataclass
class _StreamPlan:
    """What the stream needs to build any one step, without holding any of them."""

    store: CampaignStore
    rows: Mapping[int, CampaignRow]
    moved_rosters: frozenset[str]
    captures: Mapping[tuple[str, str], CaptureRef]

    @property
    def indices(self) -> tuple[int, ...]:
        return tuple(sorted(self.rows))

    def build(self, index: int) -> Step:
        row = self.rows[index]
        if row.observable == STEP_SCORE_OBSERVABLE:
            step, _, _, _, _, _, _ = _step_from_prm(self.store, row, index)
            return step
        bank = _decode_bank(self.store, row, index)
        capture = self.captures.get((row.roster_key or "", row.bank or ""))
        step, _, _, _, _ = _step_from_bank(
            bank, grader_moved=(row.roster_key or "") in self.moved_rosters, capture=capture
        )
        return step


# ---------------------------------------------------------------------------
# The conversion
# ---------------------------------------------------------------------------


def _grader_access(store: CampaignStore) -> AccessMatrix:
    """What the campaign's recorder could reach, derived from what it recorded.

    `Run.access` is "recorded at capture time and not at read time, which is the only moment it
    is knowable". So this is the campaign's access, not the access of whoever reads
    the converted record, and those differ completely: the campaign ran the models and captured
    activations, and a reader holding this store can do neither. `reader_access` in
    `record.convert.instruments` is the other one, and mixing them up is the finding recorded in
    `ConversionReport.findings`.
    """
    grader = Access.RECORD
    if store.has(SCORE_OBSERVABLE):
        grader |= Access.QUERY
    if any(store.has(name) for name in CAPTURE_OBSERVABLES):
        grader |= Access.FORWARD
    if store.has("campaign.patch.grid") or store.has("campaign.erasure"):
        grader |= Access.MUTATE
    return {
        Component.TASK: Access.RECORD,
        Component.GRADER: grader,
        Component.RECORD: Access.RECORD,
    }


def _readout_vectors(store: CampaignStore) -> dict[str, dict[str, Any]]:
    """The recorded reward directions, keyed by roster.

    The record has no field for a grader's readout vector, and the shipped battery's
    most-used input is exactly that. They are carried in `ComponentRef.extra` as sidecar references
    rather than as arrays, so the run header stays small and a caller who wants the vector loads
    one file.
    """
    out: dict[str, dict[str, Any]] = {}
    for row in store.by_observable(READOUT_OBSERVABLE):
        spec = _find_sidecar(row.envelope.get("value"))
        if spec is None:
            continue
        out[row.roster_key or row.id] = {
            "sidecar": spec.get("sidecar"),
            "dtype": spec.get("dtype"),
            "shape": spec.get("shape"),
            "slice": row.slice_name,
            "evidence": row.id,
        }
    return out


def layout_audit(store: CampaignStore) -> tuple[int, tuple[str, ...]]:
    """Check every score bank's declared layout against its declared array shape.

    Free: the shape is in the envelope's ``__ndarray__`` block, so nothing is loaded. The layout
    name and the shape are two independent statements about how many rollouts a group has. On the
    shipped store 990 of 992 banks agree and two do not: both ``ultrafeedback-bank`` rows declare
    ``bank`` and carry a ``(2000, 4)`` array, against the fourteen ``bank`` rows that carry
    ``(N, 32)``. Nothing is corrupt; the campaign's ``layout`` vocabulary simply does not pin the
    group size, which is why K is read off the shape.
    """
    mismatched: list[str] = []
    banks = 0
    for row in store.by_observable(SCORE_OBSERVABLE):
        value = row.envelope.get("value") or {}
        fields = value.get("fields") if isinstance(value, dict) else None
        if not isinstance(fields, dict):
            continue
        banks += 1
        layout = fields.get("layout")
        declared = LAYOUT_K.get(str(layout))
        spec = _find_sidecar(fields.get("scores")) or (
            fields.get("scores", {}).get("__ndarray__")
            if isinstance(fields.get("scores"), dict)
            else None
        )
        if declared is None or not isinstance(spec, dict):
            continue
        shape = list(spec.get("shape") or [])
        k = int(shape[1]) if len(shape) > 1 else 1
        if k != declared:
            mismatched.append(f"{row.id}: layout {layout!r} says K={declared}, shape says K={k}")
    return banks, tuple(mismatched)


def _find_sidecar(obj: Any) -> Mapping[str, Any] | None:
    if isinstance(obj, dict):
        spec = obj.get("__ndarray__")
        if isinstance(spec, dict) and "sidecar" in spec:
            return spec
        for value in obj.values():
            found = _find_sidecar(value)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _find_sidecar(value)
            if found is not None:
                return found
    return None


#: Where the record schema has no room for what the campaign recorded. Each of these is a fact
#: about the schema rather than about this store, which is why they are constants and not
#: computed strings.
SCHEMA_FINDINGS: tuple[str, ...] = (
    "Step.index is defined as one optimizer update and an evaluation performs none, so the index "
    "here orders banks by created_at. Nothing in the record distinguishes that from time.",
    "Run.components holds one ComponentRef per node and the campaign put thirteen reward models "
    "on the grader node, eleven of twelve banks scored by more than one. The fleet Run declares a "
    "COMPOSITE grader; campaign_arms() is the schema's own answer, one Run per grader linked by "
    "Run.arms.",
    "Run.access records what the recorder could reach, and preflight consults it as what the "
    "reader can reach. Filling it faithfully makes every access check wrong for anyone holding "
    "only the record, so the instrument harness supplies a separate reader matrix.",
    "PolicyMixture.singular is True for a trajectory with no provenance segments and for one whose "
    "single segment names an unknown policy, so 'nobody recorded the generating policy' is "
    "indistinguishable from 'one policy generated it'.",
    "The record has no field for a grader's readout vector, which is the most-used input in the "
    "shipped battery. The campaign's eleven w_r vectors ride in ComponentRef.extra.",
    "The record hierarchy holds rollouts and their scores and has no level for a measurement made "
    "about a bank, which is what most of the campaign's observables are.",
)


def convert_campaign(
    path: str | Path,
    *,
    sidecar_dirs: Sequence[str | Path] = (),
    grader: str | None = None,
    arms: Sequence[str] = (),
) -> tuple[Run, ConversionReport]:
    """Convert the campaign store at ``path`` into a `Run` of kind ``eval`` and a report.

    ``grader`` restricts the run to one roster key, which is what `campaign_arms` uses; ``arms``
    sets `Run.arms` on the result. Nothing is decoded here beyond the run header and the bank
    index: the steps are lazy and the first bank is read when the caller asks for it.
    """
    store = CampaignStore(path, sidecar_dirs=sidecar_dirs)
    store.assert_no_blind_payloads()
    return _convert(store, grader=grader, arms=arms)


def _convert(
    store: CampaignStore,
    *,
    grader: str | None,
    arms: Sequence[str],
) -> tuple[Run, ConversionReport]:
    bank_rows = [
        r
        for r in store.rows
        if r.observable in (SCORE_OBSERVABLE, STEP_SCORE_OBSERVABLE)
        and (grader is None or r.roster_key == grader)
    ]
    rows = {i: r for i, r in enumerate(bank_rows)}

    moved = frozenset(
        r.roster_key or ""
        for r in store.by_prefix("campaign.checkpoints")
        if r.roster_key is not None
    )

    captures: dict[tuple[str, str], CaptureRef] = {}
    absent_tensors = 0
    for name in CAPTURE_OBSERVABLES:
        for row in store.by_observable(name):
            manifest = store.value(row)
            if not isinstance(manifest, dict):
                continue
            ref, n_absent = _capture_ref(manifest)
            key = (
                str(manifest.get("roster_key", row.roster_key or "")),
                str(manifest.get("slice_name", row.bank or "")),
            )
            captures.setdefault(key, ref)
            absent_tensors += n_absent

    graders = tuple(sorted({r.roster_key for r in bank_rows if r.roster_key}))
    banks = tuple(sorted({r.bank for r in bank_rows if r.bank}))
    readouts = _readout_vectors(store)

    counts = store.observables()
    represented = {
        SCORE_OBSERVABLE,
        STEP_SCORE_OBSERVABLE,
        *CAPTURE_OBSERVABLES,
        READOUT_OBSERVABLE,
    }
    unrepresented = {k: v for k, v in counts.items() if k not in represented}

    _, mismatched = layout_audit(store)

    report = ConversionReport(
        store_rows=len(store),
        observables=len(counts),
        steps=len(rows),
        graders=graders,
        banks=banks,
        capture_manifests=len(captures),
        absent_tensors=absent_tensors,
        readout_vectors=len(readouts),
        layout_mismatches=mismatched,
        unrepresented=unrepresented,
        findings=SCHEMA_FINDINGS,
    )
    plan = _StreamPlan(
        store=store,
        rows=rows,
        moved_rosters=moved,
        captures=captures,
    )

    substrate = Substrate.NEURAL_SCALAR if grader is not None else Substrate.COMPOSITE
    grader_ref = ComponentRef(
        name=grader or "campaign fleet",
        kind="reward_model" if grader is not None else "reward model fleet",
        substrate=substrate,
        model_fp=ModelFP(grader) if grader is not None else None,
        extra={
            "roster": list(graders),
            "readout_vectors": {k: v for k, v in readouts.items() if grader in (None, k)},
        },
    )
    components = {
        Component.GRADER: grader_ref,
        Component.TASK: ComponentRef(
            name="campaign banks",
            kind="fixed response banks",
            dataset=DatasetID(",".join(banks)) if banks else None,
            extra={"banks": list(banks)},
        ),
        Component.RECORD: ComponentRef(
            name=str(store.path),
            kind="evidence store",
            extra={"rows": len(store), "observables": len(counts)},
        ),
    }

    recorded = None
    identity = {
        "store": str(store.path),
        "rows": len(store),
        "banks": len(rows),
        "grader": grader,
    }
    run = Run(
        id=run_id(name="campaign-2.0", **identity),
        kind="eval",
        components=components,
        access=_grader_access(store),
        # The campaign declared no regime conditions, and an empty declaration is the honest
        # encoding of that. Absent is not a pass, and RegimeDeclaration.disagreements
        # over an empty declaration is empty because there was nothing to disagree with.
        regime=RegimeDeclaration(),
        steps=CampaignStepStream(plan),
        arms=tuple(RunID(a) for a in arms),
        lineage=RunLineage(
            git_sha=_first_git_sha(store),
            framework="reward-lens campaign 2.0",
            framework_version="2.0.1",
            created_at=store.rows[0].created_at if store.rows else "",
            extra={"converted_by": "reward_lens.record.convert.campaign"},
        ),
        # FULL rather than a rate: 52 of the 56 part-families that declare an n_total sum to it
        # exactly, and the other four are aggregation artifacts of sub-banks sharing a name. There
        # is no evidence of subsampling. `population` stays None because the population is every
        # response anyone could have scored and the record does not know it, which is exactly what
        # `observed_fraction` returning None should mean.
        sampling_policy=RecordSamplingPolicy(
            scheme=SamplingScheme.FULL,
            rate=1.0,
            unit="trajectory",
            population=None,
            recorded=recorded,
            notes="every bank the campaign scored was written whole; no rollout was sampled out",
        ),
    )
    return run, report


def _first_git_sha(store: CampaignStore) -> str:
    for row in store.rows:
        sha = row.envelope.get("provenance", {}).get("git_sha")
        if sha and sha != "unknown":
            return str(sha)
    return "unknown"


def campaign_arms(
    path: str | Path,
    *,
    sidecar_dirs: Sequence[str | Path] = (),
) -> dict[str, Run]:
    """One `Run` per reward model, cross-linked through `Run.arms`.

    This is what the schema means by arms: eleven of the twelve banks were scored by more than one
    model, so the fleet is a controlled comparison whose contrast is the grader. The fleet Run that
    `convert_campaign` returns is the same data read as one object; both exist because a caller may
    want one Run and the schema wants thirteen.
    """
    store = CampaignStore(path, sidecar_dirs=sidecar_dirs)
    store.assert_no_blind_payloads()
    rosters = sorted(
        {
            r.roster_key
            for r in store.rows
            if r.observable in (SCORE_OBSERVABLE, STEP_SCORE_OBSERVABLE) and r.roster_key
        }
    )
    ids = {g: _convert(store, grader=g, arms=())[0].id for g in rosters}
    out: dict[str, Run] = {}
    for g in rosters:
        siblings = tuple(str(ids[other]) for other in rosters if other != g)
        out[g] = _convert(store, grader=g, arms=siblings)[0]
    return out


def count_run(run: Run, report: ConversionReport, *, limit: int | None = None) -> ConversionReport:
    """Walk the stream and fill in the counts the header alone cannot know.

    Separate from `convert_campaign` because it is the expensive call: it decodes every bank. The
    ``limit`` caps how many steps are walked, which is what a test uses when it wants the counting
    path exercised without reading a quarter gigabyte of sidecars.
    """
    groups = trajectories = turns = scores = abstentions = blinded = 0
    for i, step in enumerate(run.steps):
        if limit is not None and i >= limit:
            break
        groups += len(step.groups)
        for group in step.groups:
            trajectories += len(group.trajectories)
            for traj in group.trajectories:
                turns += traj.n_turns
                # One score per turn on a ProcessBench trajectory, one per trajectory on a bank
                # response whose single turn carries no per-step score of its own.
                scores += max(traj.n_turns, 1)
                blinded += len(traj.labels)
            abstentions += group.group_stats.n_abstained
    report.groups = groups
    report.trajectories = trajectories
    report.turns = turns
    report.scores_carried = scores
    report.abstentions = abstentions
    report.labels_blinded = blinded
    report.labels_declined = _labels_declined(run, limit=limit) if blinded == 0 else 0
    report.counted = True
    return report


def _labels_declined(run: Run, *, limit: int | None = None) -> int:
    """How many held-out labels the conversion refused to carry.

    Non-zero only when `record.labels` is unavailable, because a label that cannot be blinded is
    dropped rather than written into `Trajectory.labels` unwrapped. Counted from the ProcessBench
    banks, which are the only rows in this store that carry one.
    """
    declined = 0
    for i, step in enumerate(run.steps):
        if limit is not None and i >= limit:
            break
        for group in step.groups:
            for traj in group.trajectories:
                if traj.turns and traj.turns[0].step_score is not None and not traj.labels:
                    declined += 1
    return declined


__all__ = [
    "CAPTURE_OBSERVABLES",
    "LAYOUT_K",
    "NO_ESTIMATOR",
    "NO_ESTIMATOR_EPSILON",
    "READOUT_OBSERVABLE",
    "SCHEMA_FINDINGS",
    "SCORE_OBSERVABLE",
    "STEP_SCORE_OBSERVABLE",
    "UNRECORDED_POLICY",
    "Bank",
    "CampaignStepStream",
    "ConversionReport",
    "campaign_arms",
    "convert_campaign",
    "count_run",
]

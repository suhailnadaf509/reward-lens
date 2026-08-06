"""The canonical process record.

Everything in Plane B and Plane C reads this and nothing else. It is hierarchical because agentic
RL is hierarchical and reward attaches at any level:

    Run -> Step -> Group -> Trajectory -> Turn -> Token

`Step` is one optimizer update, `Group` is one prompt with its K rollouts (K = 1 is legal, that is
distillation), `Trajectory` is one rollout which may be multi-turn, `Turn` is one model action plus
the environment response, and the token level is the per-token arrays inside a turn rather than an
object of its own.

**Requiredness follows the canonical schema exactly.** A field printed without a default has none
here, and that is deliberate rather than austere. `Trajectory.advantage` is `float | None`, so a
default of `None` would make "the pipeline computed no advantage" indistinguishable from "the
converter forgot to pass it", and the whole value of this record is that the second thing is
impossible. `make_trajectory` exists for the ergonomics, in one place, where the defaults are
visible.

**Three seams are typed rather than stubbed**: `ScoreTree` (`record/scores.py`),
`Blind[LabelValue]` (`record/labels.py`) and `CouplingSpec` (`record/arms.py`). Each of those
modules imports this one, so they are imported here under `TYPE_CHECKING` only: the annotations are
strings, this module has no runtime dependency on them, and the fields serialise through the
kernel's `ValueCodec`, so anything those modules define as a dataclass decorated with
`register_payload` round-trips through the record with no change here. A `Blind` in particular must
be a registered dataclass and must not define an unwrapping decoder, because the codec is the one
place a blind value could be opened by accident.

The name `Trajectory` also exists in `reward_lens.data.schema`, where it means an agent episode for
the receipt and narrative sciences. They are different objects. This one is the canonical record
and the other predates it; the collision is named here rather than resolved, because renaming a
shipped type is not this package's call.
"""

from __future__ import annotations

import abc
import enum
import math
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Iterator,
    Literal,
    Mapping,
    NewType,
    Sequence,
)

import numpy as np

from reward_lens.core.envelope import ConditionReading, RegimeCondition, RegimeReading
from reward_lens.core.evidence import ValueCodec
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.types import (
    Access,
    AccessMatrix,
    Component,
    DatasetID,
    ModelFP,
    Substrate,
    content_hash,
)
from reward_lens.record.compaction import CompactionEvent, check_ordered
from reward_lens.record.provenance import (
    PolicyVersionID,
    SegmentProvenance,
    check_tiling,
    policy_mixture,
    single_segment,
)
from reward_lens.record.tensors import (
    CaptureRef,
    Engine,
    TensorRef,
    TokenTensorRef,
    ref_from_canonical,
)
from reward_lens.record.turns import Turn

if TYPE_CHECKING:  # annotations only; see the module docstring for the contract
    from reward_lens.record.arms import CouplingSpec
    from reward_lens.record.labels import Blind, LabelValue
    from reward_lens.record.scores import ScoreTree

#: Bumped in the same commit that changes what a written record looks like on disk. Schema
#: versioning is not optional, and 2.0.1 shipping without it is why the evidence store's
#: first migration has to sniff. This one starts versioned.
RECORD_SCHEMA_VERSION = 1

RunID = NewType("RunID", str)
GroupID = NewType("GroupID", str)
TrajectoryID = NewType("TrajectoryID", str)
TaskID = NewType("TaskID", str)
FeatureID = NewType("FeatureID", str)

#: A training run is one instance of the loop diagram, not the only one. Best-of-n removes the
#: optimizer; a tilt sweep replaces it with importance weighting; an evaluation removes the
#: estimator too; an audit is the artifact node alone; a grader study is the grader node alone.
#: One field carries the variants, because the alternative is separate subsystems kept consistent
#: by hand and that is what the shipped library does.
RunKind = Literal["train", "bon", "tilt", "eval", "audit", "sweep", "grader_study", "distill"]

_CODEC = ValueCodec()


# ---------------------------------------------------------------------------
# Foreign payloads: the seam for scores, labels and coupling
# ---------------------------------------------------------------------------


def encode_foreign(value: Any, sidecar_dir: Any = None) -> Any:
    """Encode a payload this package does not own, through the kernel's `ValueCodec`.

    Used for `Trajectory.scores`, `Trajectory.labels` and `Run.coupling`. The codec handles
    primitives, sequences, mappings, numpy arrays (spilling large ones to content-addressed
    sidecars) and any dataclass; a dataclass decorated with `register_payload` comes back as
    itself, and an unregistered one comes back as its field dict. That is the contract the seam
    modules have to meet, and it is one decorator.
    """
    if value is None:
        return None
    return _CODEC.encode(value, sidecar_dir)


def decode_foreign(obj: Any, sidecar_dir: Any = None) -> Any:
    """Inverse of `encode_foreign`."""
    if obj is None:
        return None
    return _CODEC.decode(obj, sidecar_dir)


# ---------------------------------------------------------------------------
# Run-level structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComponentRef:
    """What occupied one node of the loop.

    The first structural fact about the loop is that the grader and the policy are the same kind of
    object, so they get the same reference type and an instrument that reads internals does not
    care which side of the loop it points at. ``substrate`` is what actually decides which
    instruments apply, which is why it sits here rather than being inferred from ``kind``.
    """

    name: str
    kind: str = ""
    substrate: Substrate | None = None
    model_fp: ModelFP | None = None
    dataset: DatasetID | None = None
    version: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        return content_hash(self.__canonical__(), "cmp")

    def __canonical__(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "substrate": None if self.substrate is None else self.substrate.name,
            "model_fp": None if self.model_fp is None else str(self.model_fp),
            "dataset": None if self.dataset is None else str(self.dataset),
            "version": self.version,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_canonical(cls, obj: Mapping[str, Any]) -> "ComponentRef":
        sub = obj.get("substrate")
        return cls(
            name=obj["name"],
            kind=obj.get("kind", ""),
            substrate=None if sub is None else Substrate[sub],
            model_fp=None if obj.get("model_fp") is None else ModelFP(obj["model_fp"]),
            dataset=None if obj.get("dataset") is None else DatasetID(obj["dataset"]),
            version=obj.get("version"),
            extra=dict(obj.get("extra", {})),
        )


@dataclass(frozen=True)
class RunLineage:
    """Where the run came from: code, config, framework.

    ``library_version`` records which reward-lens wrote the record, because a record read by a
    different version than wrote it is the normal case a year later and the migration has to know
    where to start. ``parent_run`` is set on a resumed run, so a run that continues another is not
    silently a fresh one with a suspiciously non-zero starting reward.
    """

    git_sha: str = "unknown"
    config_hash: str | None = None
    framework: str = "unknown"
    framework_version: str = "unknown"
    library_version: str = ""
    created_at: str = ""
    parent_run: RunID | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __canonical__(self) -> dict[str, Any]:
        return {
            "git_sha": self.git_sha,
            "config_hash": self.config_hash,
            "framework": self.framework,
            "framework_version": self.framework_version,
            "library_version": self.library_version,
            "created_at": self.created_at,
            "parent_run": None if self.parent_run is None else str(self.parent_run),
            "extra": dict(self.extra),
        }

    @classmethod
    def from_canonical(cls, obj: Mapping[str, Any]) -> "RunLineage":
        return cls(
            git_sha=obj.get("git_sha", "unknown"),
            config_hash=obj.get("config_hash"),
            framework=obj.get("framework", "unknown"),
            framework_version=obj.get("framework_version", "unknown"),
            library_version=obj.get("library_version", ""),
            created_at=obj.get("created_at", ""),
            parent_run=None if obj.get("parent_run") is None else RunID(obj["parent_run"]),
            extra=dict(obj.get("extra", {})),
        )


@dataclass(frozen=True)
class RegimeDeclaration:
    """What the run asserts about its own regime, before anything is measured.

    Two of the loop's arrows are new and both break assumptions instruments depend on.
    Ling 2.6 shifts rubric weights away from saturated metrics during training. DeepSeek-V4's actor
    is its own generative reward model. ERNIE's WPSM gates a sample on group-level accuracy. None
    of those is visible from a rollout, so the only way to keep the library honest is for the run to
    declare them and for the library to measure whether the declaration held.

    Three states per condition, and the third is the useful one: True, False, and absent for "not
    declared". Absent is not a pass. `disagreements` is what makes the declaration worth having,
    because a run that declared `STATIONARY_GRADER` and measured otherwise is a finding.
    """

    declared: Mapping[RegimeCondition, bool] = field(default_factory=dict)
    notes: Mapping[RegimeCondition, str] = field(default_factory=dict)
    declared_by: str = ""

    def declares(self, condition: RegimeCondition) -> bool | None:
        return self.declared.get(condition)

    def disagreements(
        self, reading: RegimeReading | None
    ) -> dict[RegimeCondition, tuple[bool, bool | None]]:
        """Conditions where the declaration and the measurement differ, as (declared, measured).

        A measurement of None (could not be determined) counts as a disagreement with any
        declaration, because "we said it holds and nobody could check" is exactly the state this
        machinery exists to make visible.
        """
        if reading is None:
            return {c: (v, None) for c, v in self.declared.items()}
        out: dict[RegimeCondition, tuple[bool, bool | None]] = {}
        for condition, claimed in self.declared.items():
            measured = reading.holds(condition)
            if measured is not claimed:
                out[condition] = (claimed, measured)
        return out

    def __canonical__(self) -> dict[str, Any]:
        return {
            "declared": {c.name: bool(v) for c, v in self.declared.items()},
            "notes": {c.name: s for c, s in self.notes.items()},
            "declared_by": self.declared_by,
        }

    @classmethod
    def from_canonical(cls, obj: Mapping[str, Any]) -> "RegimeDeclaration":
        return cls(
            declared={RegimeCondition[k]: bool(v) for k, v in obj.get("declared", {}).items()},
            notes={RegimeCondition[k]: v for k, v in obj.get("notes", {}).items()},
            declared_by=obj.get("declared_by", ""),
        )


# ---------------------------------------------------------------------------
# The record sampling policy, and the correction that makes it worth recording
# ---------------------------------------------------------------------------


class SamplingScheme(enum.Enum):
    """How the recorded subset was chosen. The distinction that matters is correctability.

    `FULL`, `UNIFORM`, `STRATIFIED` and `EXPLICIT` all have known, strictly positive inclusion
    probabilities, so a Horvitz-Thompson reweighting recovers the population quantity. `SELECTIVE`
    does not: keeping the top-k by reward, or the first n steps, or "the interesting ones" gives
    some units an inclusion probability of exactly zero, and no reweighting recovers a stratum that
    could never have been recorded.
    """

    FULL = "full"
    UNIFORM = "uniform"
    STRATIFIED = "stratified"
    EXPLICIT = "explicit"
    SELECTIVE = "selective"


@dataclass(frozen=True)
class SamplingCorrection:
    """A population quantity estimated from a recorded subsample, with its own diagnostics.

    Deliberately not an `Evidence`. Promoting it would need a registered quantity id and an
    invariance group, and those are not this package's to assign. An instrument that wants to
    report a corrected statistic wraps this in its own Evidence with its own quantity.
    """

    value: float
    n: int
    ess: float
    method: str
    weight_sum: float
    max_weight: float

    @property
    def design_effect(self) -> float:
        """``n / ESS``. One means the weights cost nothing; ten means the sample is worth n/10."""
        return float("inf") if self.ess <= 0 else self.n / self.ess

    def __canonical__(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "n": self.n,
            "ess": self.ess,
            "method": self.method,
            "weight_sum": self.weight_sum,
            "max_weight": self.max_weight,
        }


@dataclass(frozen=True)
class RecordSamplingPolicy:
    """What fraction of the run was recorded, and how it was chosen.

    On the Run and not implicit, because full capture is not the default and pretending otherwise
    produces silently biased statistics. The arithmetic is in `record.tensors.residual_bytes`: at
    70B, one layer of bf16 residual over 10^9 rollout tokens is 16.4 TB and all eighty layers is
    1.31 PB. Nobody records everything, so every statistic computed from a record is a statistic
    computed from a sample, and the only question is whether the sampling design was written down.

    The field is a promise the methods here keep: `estimate_mean` and `estimate_total` correct for
    the design, and refuse when the design admits no correction.
    """

    scheme: SamplingScheme = SamplingScheme.FULL
    rate: float = 1.0
    strata_rates: Mapping[str, float] = field(default_factory=dict)
    stratum_key: str | None = None
    selected_by: str = ""
    seed: int | None = None
    population: int | None = None
    recorded: int | None = None
    unit: Literal["trajectory", "group", "step", "turn", "token"] = "trajectory"
    notes: str = ""

    def __post_init__(self) -> None:
        if not 0.0 < self.rate <= 1.0:
            raise ValueError(
                f"record sampling rate must be in (0, 1]; got {self.rate}. A rate of zero is not "
                f"a sampling policy, it is an absent record."
            )
        if self.scheme is SamplingScheme.STRATIFIED and not self.strata_rates:
            raise ValueError(
                "a STRATIFIED sampling policy needs strata_rates. Without them the design is "
                "unknown and no estimator computed from this record can be corrected."
            )
        if self.scheme is SamplingScheme.SELECTIVE and not self.selected_by.strip():
            raise ValueError(
                "a SELECTIVE sampling policy must say what selected the units, in selected_by. "
                "It is the only field that lets a reader judge how the sample is skewed, since "
                "the design admits no arithmetic correction."
            )

    # -- design ---------------------------------------------------------------

    @property
    def is_correctable(self) -> bool:
        """Whether a design-based estimator exists at all for this policy."""
        if self.scheme is SamplingScheme.SELECTIVE:
            return False
        if self.scheme is SamplingScheme.STRATIFIED:
            return all(0.0 < r <= 1.0 for r in self.strata_rates.values())
        return 0.0 < self.rate <= 1.0

    @property
    def observed_fraction(self) -> float | None:
        """``recorded / population`` where both were counted, else None."""
        if self.population in (None, 0) or self.recorded is None:
            return None
        assert self.population is not None
        return self.recorded / self.population

    def inclusion_probability(
        self, *, stratum: str | None = None, explicit: float | None = None
    ) -> float | None:
        """The probability this unit had of being recorded, or None if the design does not say."""
        if self.scheme is SamplingScheme.FULL:
            return 1.0
        if self.scheme is SamplingScheme.UNIFORM:
            return self.rate
        if self.scheme is SamplingScheme.STRATIFIED:
            return None if stratum is None else self.strata_rates.get(stratum)
        if self.scheme is SamplingScheme.EXPLICIT:
            return explicit
        return None

    # -- correction -----------------------------------------------------------

    def weights(
        self,
        n: int,
        *,
        strata: Sequence[str] | None = None,
        inclusion: Sequence[float] | None = None,
    ) -> "np.ndarray | Refusal":
        """Inverse-probability weights, one per recorded unit, or a `Refusal` explaining why not.

        Refuses on a `SELECTIVE` design and on any unit whose inclusion probability is unknown or
        non-positive. Both are identifiability failures rather than precision failures: there is
        no sample size at which they go away, so a bound would be a fiction and the honest return
        is a refusal with the remedy naming what to record next time.
        """
        instrument = "record.sampling_correction"
        if self.scheme is SamplingScheme.SELECTIVE:
            return Refusal(
                instrument=instrument,
                reason=RefusalReason.ENVELOPE_VIOLATED,
                detail=(
                    f"the record was sampled by {self.selected_by!r}, which gives some units an "
                    f"inclusion probability of zero. A design-based estimator assumes strictly "
                    f"positive inclusion probabilities and there is none here to invert."
                ),
                remedy=(
                    "Re-record with a UNIFORM or STRATIFIED policy, or compute the statistic on "
                    "the recorded subset and report it as a statistic about the subset, naming "
                    "the selection rule beside it."
                ),
                statistics={"scheme": self.scheme.value, "selected_by": self.selected_by},
            )
        probabilities: list[float] = []
        for i in range(n):
            p = self.inclusion_probability(
                stratum=None if strata is None else strata[i],
                explicit=None if inclusion is None else inclusion[i],
            )
            if p is None:
                missing = None if strata is None else strata[i]
                return Refusal(
                    instrument=instrument,
                    reason=RefusalReason.ENVELOPE_VIOLATED,
                    detail=(
                        f"unit {i} has no recorded inclusion probability"
                        + (f" (stratum {missing!r} is not in strata_rates)" if missing else "")
                        + f"; the policy is {self.scheme.value}."
                    ),
                    remedy=(
                        "Supply the per-unit inclusion probabilities to weights(), or record a "
                        "rate for every stratum the run can produce, including the ones it "
                        "produced none of."
                    ),
                    statistics={"unit": i, "scheme": self.scheme.value},
                )
            if not 0.0 < p <= 1.0:
                return Refusal(
                    instrument=instrument,
                    reason=RefusalReason.ENVELOPE_VIOLATED,
                    detail=(
                        f"unit {i} has inclusion probability {p}, which is outside (0, 1]. A "
                        f"stratum recorded with probability zero cannot be reweighted back."
                    ),
                    remedy=(
                        "Give every stratum a strictly positive recording rate, however small, or "
                        "report the statistic on the covered strata only and say which are missing."
                    ),
                    statistics={"unit": i, "probability": p},
                )
            probabilities.append(p)
        return 1.0 / np.asarray(probabilities, dtype=float)

    def estimate_mean(
        self,
        values: Sequence[float],
        *,
        strata: Sequence[str] | None = None,
        inclusion: Sequence[float] | None = None,
        min_ess: float | None = None,
    ) -> "SamplingCorrection | Refusal":
        """The design-corrected mean of a per-unit quantity, or a refusal.

        The Hajek ratio estimator, ``sum(w_i y_i) / sum(w_i)`` with ``w_i = 1 / pi_i``. Ratio
        rather than the plain Horvitz-Thompson total over a known population size, because the
        population size is often exactly what the record does not know: a run that sampled its
        rollouts usually did not count the ones it discarded. The ratio form needs no population
        count and is consistent under the same design.

        ``min_ess`` gates on Kish's effective sample size. It has no default because an ESS floor
        is a threshold in an envelope, and thresholds are declared by the instrument that owns the
        reading rather than assumed here. Leave it None and the ESS still comes back in the result.
        """
        y = np.asarray(values, dtype=float)
        if y.size == 0:
            return Refusal(
                instrument="record.sampling_correction",
                reason=RefusalReason.ENVELOPE_VIOLATED,
                detail="no recorded units; there is nothing to correct.",
                remedy="Widen the window, or check that the record covers the steps you asked for.",
                statistics={"n": 0},
            )
        w = self.weights(y.size, strata=strata, inclusion=inclusion)
        if isinstance(w, Refusal):
            return w
        weight_sum = float(w.sum())
        ess = float(weight_sum**2 / float((w**2).sum()))
        if min_ess is not None and ess < min_ess:
            return Refusal(
                instrument="record.sampling_correction",
                reason=RefusalReason.ESS_BELOW_FLOOR,
                detail=(
                    f"the inverse-probability weights give an effective sample size of {ess:.4g} "
                    f"over {y.size} recorded units, below the declared floor of {min_ess:.4g}."
                ),
                remedy=(
                    "Record a larger fraction of the rare stratum, or widen the window until the "
                    "effective sample size clears the floor. Flattening the weights would trade a "
                    "wide interval for a biased point estimate."
                ),
                statistics={"ess": ess, "floor": min_ess, "n": int(y.size)},
            )
        method = "identity" if self.scheme is SamplingScheme.FULL else "hajek"
        return SamplingCorrection(
            value=float((w * y).sum() / weight_sum),
            n=int(y.size),
            ess=ess,
            method=method,
            weight_sum=weight_sum,
            max_weight=float(w.max()),
        )

    def estimate_total(
        self,
        values: Sequence[float],
        *,
        strata: Sequence[str] | None = None,
        inclusion: Sequence[float] | None = None,
    ) -> "SamplingCorrection | Refusal":
        """The Horvitz-Thompson total, ``sum(y_i / pi_i)``. Unbiased for the population total.

        Refuses on an empty sample for the same reason `estimate_mean` does. A total over zero
        recorded units is unknown, not zero: the sum of no terms is zero as arithmetic and the
        population total it estimates is whatever the unrecorded units add up to, which is exactly
        the quantity nothing here observed. Returning 0.0 would put a number on a window the
        record does not cover, and it is the reassuring direction, since a total of zero reads as
        "nothing happened" rather than as "nothing was looked at".
        """
        y = np.asarray(values, dtype=float)
        if y.size == 0:
            return Refusal(
                instrument="record.sampling_correction",
                reason=RefusalReason.ENVELOPE_VIOLATED,
                detail=(
                    "no recorded units, so there is nothing to inflate to a population total. "
                    "Zero recorded units and a population total of zero are different statements "
                    "and this sample cannot tell them apart."
                ),
                remedy="Widen the window, or check that the record covers the steps you asked for.",
                statistics={"n": 0},
            )
        w = self.weights(y.size, strata=strata, inclusion=inclusion)
        if isinstance(w, Refusal):
            return w
        weight_sum = float(w.sum())
        ess = float(weight_sum**2 / float((w**2).sum()))
        return SamplingCorrection(
            value=float((w * y).sum()),
            n=int(y.size),
            ess=ess,
            method="horvitz_thompson",
            weight_sum=weight_sum,
            max_weight=float(w.max()),
        )

    def __canonical__(self) -> dict[str, Any]:
        return {
            "scheme": self.scheme.value,
            "rate": self.rate,
            "strata_rates": dict(self.strata_rates),
            "stratum_key": self.stratum_key,
            "selected_by": self.selected_by,
            "seed": self.seed,
            "population": self.population,
            "recorded": self.recorded,
            "unit": self.unit,
            "notes": self.notes,
        }

    @classmethod
    def from_canonical(cls, obj: Mapping[str, Any]) -> "RecordSamplingPolicy":
        return cls(
            scheme=SamplingScheme(obj.get("scheme", "full")),
            rate=obj.get("rate", 1.0),
            strata_rates=dict(obj.get("strata_rates", {})),
            stratum_key=obj.get("stratum_key"),
            selected_by=obj.get("selected_by", ""),
            seed=obj.get("seed"),
            population=obj.get("population"),
            recorded=obj.get("recorded"),
            unit=obj.get("unit", "trajectory"),
            notes=obj.get("notes", ""),
        )


# ---------------------------------------------------------------------------
# Step-level structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StalenessDistribution:
    """How far behind the current policy the batch's rollouts were, as a histogram.

    A distribution rather than a mean, because the mean is not the quantity that breaks anything.
    One forty-step-stale trajectory in a batch of 512 makes `NEAR_POLICY` false for that
    trajectory whatever the mean says, and the tail is what the importance weights blow up on.
    """

    counts: Mapping[int, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def max(self) -> int:
        return max(self.counts) if self.counts else 0

    @property
    def mean(self) -> float:
        n = self.total
        if n == 0:
            return 0.0
        return sum(k * v for k, v in self.counts.items()) / n

    def __canonical__(self) -> dict[str, Any]:
        return {"counts": {str(k): v for k, v in sorted(self.counts.items())}}

    @classmethod
    def from_canonical(cls, obj: Mapping[str, Any]) -> "StalenessDistribution":
        return cls(counts={int(k): int(v) for k, v in obj.get("counts", {}).items()})


@dataclass(frozen=True)
class OptimizerTelemetry:
    """What the optimizer did on this step.

    ``grad_norm_clipped`` and ``grad_norm_unclipped`` are both here and both optional, because the
    one every trainer logs is the clipped one and the one every gradient-derived quantity actually
    needs is the unclipped one. Recording only the clipped norm makes every clip-crossing step look
    like a step at the clip threshold, which biases exactly the steps that matter most. Where a
    framework offers only one, the other stays None rather than being back-filled.
    """

    grad_norm_clipped: float | None = None
    grad_norm_unclipped: float | None = None
    clip_fraction: float | None = None
    kl_to_ref: float | None = None
    kl_to_previous: float | None = None
    entropy: float | None = None
    update_norm: float | None = None
    param_group_norms: Mapping[str, float] = field(default_factory=dict)
    staleness: StalenessDistribution = field(default_factory=StalenessDistribution)
    extra: Mapping[str, float] = field(default_factory=dict)

    def __canonical__(self) -> dict[str, Any]:
        return {
            "grad_norm_clipped": self.grad_norm_clipped,
            "grad_norm_unclipped": self.grad_norm_unclipped,
            "clip_fraction": self.clip_fraction,
            "kl_to_ref": self.kl_to_ref,
            "kl_to_previous": self.kl_to_previous,
            "entropy": self.entropy,
            "update_norm": self.update_norm,
            "param_group_norms": dict(self.param_group_norms),
            "staleness": self.staleness.__canonical__(),
            "extra": dict(self.extra),
        }

    @classmethod
    def from_canonical(cls, obj: Mapping[str, Any]) -> "OptimizerTelemetry":
        return cls(
            grad_norm_clipped=obj.get("grad_norm_clipped"),
            grad_norm_unclipped=obj.get("grad_norm_unclipped"),
            clip_fraction=obj.get("clip_fraction"),
            kl_to_ref=obj.get("kl_to_ref"),
            kl_to_previous=obj.get("kl_to_previous"),
            entropy=obj.get("entropy"),
            update_norm=obj.get("update_norm"),
            param_group_norms=dict(obj.get("param_group_norms", {})),
            staleness=StalenessDistribution.from_canonical(obj.get("staleness", {})),
            extra=dict(obj.get("extra", {})),
        )


@dataclass(frozen=True)
class ProbeResult:
    """A held-out or gold-channel reading taken at this step.

    ``channel`` separates three things that are usually conflated. A held-out eval measures
    generalisation. A gold-channel probe approximates the latent objective. A check standard is a
    frozen probe set whose job is not to be difficult but to be *invariant*, so any movement in it
    is instrument drift by construction. Averaging the three together is how tooling
    drift gets reported as model improvement.
    """

    name: str
    channel: Literal["held_out", "gold", "check_standard"] = "held_out"
    value: float | None = None
    n: int | None = None
    quantity: str | None = None
    detail: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __canonical__(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "channel": self.channel,
            "value": self.value,
            "n": self.n,
            "quantity": self.quantity,
            "detail": self.detail,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_canonical(cls, obj: Mapping[str, Any]) -> "ProbeResult":
        return cls(
            name=obj["name"],
            channel=obj.get("channel", "held_out"),
            value=obj.get("value"),
            n=obj.get("n"),
            quantity=obj.get("quantity"),
            detail=obj.get("detail", ""),
            extra=dict(obj.get("extra", {})),
        )


@dataclass(frozen=True)
class InstrumentEffect:
    """What this measurement cost the run it measured.

    A term in the uncertainty budget, not a footnote. The fields line up with `TapBudget`'s three
    limits so a breach can name which one it broke, and `disabled` records that the tap took itself
    out rather than stalling the host. No competitor publishes an overhead number of any kind,
    which makes a measured one worth carrying on every step.
    """

    added_latency_ms_p50: float = 0.0
    added_latency_ms_p99: float = 0.0
    added_resident_bytes: int = 0
    added_alloc_bytes: int = 0
    invocations: int = 0
    exceptions: int = 0
    disabled: bool = False
    breach: str | None = None
    downsample_factor: int = 1

    def __canonical__(self) -> dict[str, Any]:
        return {
            "added_latency_ms_p50": self.added_latency_ms_p50,
            "added_latency_ms_p99": self.added_latency_ms_p99,
            "added_resident_bytes": self.added_resident_bytes,
            "added_alloc_bytes": self.added_alloc_bytes,
            "invocations": self.invocations,
            "exceptions": self.exceptions,
            "disabled": self.disabled,
            "breach": self.breach,
            "downsample_factor": self.downsample_factor,
        }

    @classmethod
    def from_canonical(cls, obj: Mapping[str, Any]) -> "InstrumentEffect":
        return cls(
            added_latency_ms_p50=obj.get("added_latency_ms_p50", 0.0),
            added_latency_ms_p99=obj.get("added_latency_ms_p99", 0.0),
            added_resident_bytes=obj.get("added_resident_bytes", 0),
            added_alloc_bytes=obj.get("added_alloc_bytes", 0),
            invocations=obj.get("invocations", 0),
            exceptions=obj.get("exceptions", 0),
            disabled=obj.get("disabled", False),
            breach=obj.get("breach"),
            downsample_factor=obj.get("downsample_factor", 1),
        )


# ---------------------------------------------------------------------------
# Group-level structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EstimatorSpec:
    """How scores became advantages, exactly.

    Every field changes what a downstream number means, which is why this is a structure and not a
    family name. ``std_normalised`` with ``std_epsilon`` is the one that has already produced a
    published surprise: the GRPO advantage ``(r - mean) / (std + eps)`` is not invariant under the
    reward-affine group for ``eps > 0``, since the numerator scales by ``a`` and the denominator
    goes to ``a * std + eps``. At the usual ``eps = 1e-8`` the generated invariance test measures
    the violation at about 1e-7, which `numpy.isclose` would have called equality.

    ``std_normalised`` is also what decides whether E4's amplifier-safety question applies at all.
    `verifiers` mean-centres with no standard-deviation division (`rubric.py:406-409`), so on that
    framework the amplification mechanism is absent and the instrument should say so rather than
    reporting a ratio nobody can act on.
    """

    family: str = "unknown"
    group_centred: bool = False
    std_normalised: bool = False
    std_epsilon: float | None = None
    #: Which standard deviation the trainer divided by: 0 for the population form, 1 for Bessel's
    #: correction. `None` means the record does not say, and `replay_advantages` refuses rather than
    #: assuming, for the same reason it refuses on a missing `std_epsilon`. Both sit in the same
    #: denominator and neither is recoverable from the scores.
    #:
    #: It is a separate field rather than a convention because the two conventions differ by
    #: `sqrt(K/(K-1))`, which is 41.4% at K=2 and 15.5% at K=4, and GRPO runs small K. Every
    #: framework in scope uses 1: TRL's `nanstd` multiplies the variance by `count/(count-1)`
    #: explicitly, and veRL's `compute_grpo_outcome_advantage` calls `torch.std`, whose default is
    #: `correction=1`. That makes 1 the near-certain answer and not a safe default, because a
    #: near-certain assumption about a denominator is exactly the shape of confident wrong number
    #: this record exists to prevent.
    std_ddof: int | None = None
    degenerate_policy: str = "unknown"
    clip_low: float | None = None
    clip_high: float | None = None
    clip_ratio_c: float | None = None
    aggregation: Literal["token", "sequence", "prompt", "batch", "none", "unknown"] = "unknown"
    loss_mask_policy: str = "unknown"
    off_policy_correction: str | None = None
    kl_penalty: str | None = None
    kl_coefficient: float | None = None
    advantage_whitening: bool = False
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def z_scored(self) -> bool:
        """Whether the transform divides by a group standard deviation. E2 and E4 read this."""
        return self.group_centred and self.std_normalised

    def __canonical__(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "group_centred": self.group_centred,
            "std_normalised": self.std_normalised,
            "std_epsilon": self.std_epsilon,
            "std_ddof": self.std_ddof,
            "degenerate_policy": self.degenerate_policy,
            "clip_low": self.clip_low,
            "clip_high": self.clip_high,
            "clip_ratio_c": self.clip_ratio_c,
            "aggregation": self.aggregation,
            "loss_mask_policy": self.loss_mask_policy,
            "off_policy_correction": self.off_policy_correction,
            "kl_penalty": self.kl_penalty,
            "kl_coefficient": self.kl_coefficient,
            "advantage_whitening": self.advantage_whitening,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_canonical(cls, obj: Mapping[str, Any]) -> "EstimatorSpec":
        return cls(
            family=obj.get("family", "unknown"),
            group_centred=obj.get("group_centred", False),
            std_normalised=obj.get("std_normalised", False),
            std_epsilon=obj.get("std_epsilon"),
            std_ddof=obj.get("std_ddof"),
            degenerate_policy=obj.get("degenerate_policy", "unknown"),
            clip_low=obj.get("clip_low"),
            clip_high=obj.get("clip_high"),
            clip_ratio_c=obj.get("clip_ratio_c"),
            aggregation=obj.get("aggregation", "unknown"),
            loss_mask_policy=obj.get("loss_mask_policy", "unknown"),
            off_policy_correction=obj.get("off_policy_correction"),
            kl_penalty=obj.get("kl_penalty"),
            kl_coefficient=obj.get("kl_coefficient"),
            advantage_whitening=obj.get("advantage_whitening", False),
            extra=dict(obj.get("extra", {})),
        )


@dataclass(frozen=True)
class GroupStats:
    """The group's score distribution, and the two flags that decide whether it taught anything.

    ``all_fail`` is not a curiosity. When every rollout in a group fails the task, the only variance
    left is whatever auxiliary signal is still moving, and a z-scoring estimator rescales it to full
    magnitude: measured advantages reach plus or minus 5 to 77 under standard normalisation against
    plus or minus 0.1 without it. That is instrument E4's subject and this flag is where it starts.

    ``degenerate`` is the neighbouring case, ``std`` at or below the estimator's epsilon, where the
    advantage is ``0 / eps`` and the group contributes noise or nothing.

    ``curl_mass`` is populated only for k-wise comparison groups, where the question of whether the
    comparison structure is a scalar potential at all is live.

    ``ranks`` is positional: entry ``j`` is the rank of trajectory ``j`` in ``Group.trajectories``,
    zero for the highest score. It has length ``k`` and carries ``None`` where the grader abstained,
    because a rollout with no score has no rank and there is no value that says so. Ranking an
    abstention last would be a number rather than a gap, and it is the wrong number in the
    direction that matters: an abstention is a missing measurement, not a bad rollout.
    """

    k: int
    mean: float | None = None
    std: float | None = None
    degenerate: bool = False
    all_fail: bool = False
    n_abstained: int = 0
    curl_mass: float | None = None
    ranks: tuple[int | None, ...] | None = None

    def __post_init__(self) -> None:
        if self.k < 0:
            raise ValueError(f"group size cannot be negative; got {self.k}")
        if self.ranks is not None and len(self.ranks) != self.k:
            raise ValueError(
                f"ranks has {len(self.ranks)} entries for a group of {self.k}. The vector is "
                f"positional against Group.trajectories, so a short one silently re-labels every "
                f"rollout after the first gap. Use None for a rollout with no score."
            )

    @classmethod
    def from_scores(
        cls,
        scores: Sequence[float | None],
        *,
        std_epsilon: float,
        failure_at: float | None = None,
        curl_mass: float | None = None,
    ) -> "GroupStats":
        """Compute the stats from per-trajectory scores.

        ``std_epsilon`` is required and has no default: the boundary between a degenerate group and
        a live one is the estimator's own epsilon, so reading it off the `EstimatorSpec` is the only
        way the flag means the same thing as the trainer meant. ``failure_at`` is the score that
        counts as a failure; None means all-fail cannot be determined and the flag stays False,
        which is the honest reading of "we do not know what failure looks like on this task".

        Abstentions (a `None` score, meaning the grader failed) are excluded from the mean and the
        standard deviation and counted separately. That is TRL's policy generalised, and it is the
        opposite of `verifiers`, whose two bare `except Exception` handlers substitute 0.0 and let a
        crashed grader flow into the group mean as a real score.

        They are excluded from ``ranks`` too, but by leaving a hole rather than by shortening the
        vector. The ranks themselves are dense over the rollouts that do have a score, so in a group
        of four with one abstention the surviving three rank 0, 1, 2 and the fourth slot is None.
        """
        present = [float(s) for s in scores if s is not None]
        n_abstained = sum(1 for s in scores if s is None)
        if not present:
            return cls(
                k=len(scores),
                mean=None,
                std=None,
                degenerate=True,
                all_fail=False,
                n_abstained=n_abstained,
                curl_mass=curl_mass,
                ranks=(None,) * len(scores),
            )
        arr = np.asarray(present, dtype=float)
        std = float(arr.std())
        # Ranked over the survivors, then scattered back to the positions they came from, because
        # the vector is read against Group.trajectories and not against the survivors.
        dense = [int(r) for r in np.argsort(np.argsort(-arr))]
        scattered: list[int | None] = []
        it = iter(dense)
        for s in scores:
            scattered.append(None if s is None else next(it))
        ranks = tuple(scattered)
        return cls(
            k=len(scores),
            mean=float(arr.mean()),
            std=std,
            degenerate=std <= std_epsilon,
            all_fail=(failure_at is not None and bool(np.all(arr <= failure_at))),
            n_abstained=n_abstained,
            curl_mass=curl_mass,
            ranks=ranks,
        )

    def __canonical__(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "mean": _finite(self.mean),
            "std": _finite(self.std),
            "degenerate": self.degenerate,
            "all_fail": self.all_fail,
            "n_abstained": self.n_abstained,
            "curl_mass": _finite(self.curl_mass),
            "ranks": None if self.ranks is None else list(self.ranks),
        }

    @classmethod
    def from_canonical(cls, obj: Mapping[str, Any]) -> "GroupStats":
        """Rebuild from the canonical form, dropping a rank vector that does not fit the group.

        Records written before the abstention hole existed carry a ``ranks`` shortened by the
        number of abstentions, so entry ``j`` describes rollout ``j`` only up to the first gap and
        the wrong rollout after it. Those are read back as None rather than carried: "not recorded"
        is true of such a record and the tuple is not, and there is nothing left in the canonical
        payload to recompute the alignment from. 200 of the 400 groups in
        `tests/fixtures/grpo_run/long/` are in that state.
        """
        stored = obj.get("ranks")
        ranks = None if stored is None else tuple(stored)
        if ranks is not None and len(ranks) != obj["k"]:
            ranks = None
        return cls(
            k=obj["k"],
            mean=_unfinite(obj.get("mean")),
            std=_unfinite(obj.get("std")),
            degenerate=obj.get("degenerate", False),
            all_fail=obj.get("all_fail", False),
            n_abstained=obj.get("n_abstained", 0),
            curl_mass=_unfinite(obj.get("curl_mass")),
            ranks=ranks,
        )


def _finite(x: float | None) -> float | str | None:
    """JSON has no NaN or infinity; tag them so a round trip does not turn NaN into null."""
    if x is None:
        return None
    if math.isnan(x):
        return "__nan__"
    if math.isinf(x):
        return "__inf__" if x > 0 else "__-inf__"
    return float(x)


def _unfinite(x: float | str | None) -> float | None:
    if isinstance(x, str):
        return {"__nan__": math.nan, "__inf__": math.inf, "__-inf__": -math.inf}[x]
    return x


# ---------------------------------------------------------------------------
# The four levels
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Trajectory:
    """One rollout, which may be multi-turn.

    Two invariants are enforced at construction rather than checked later, because both of them
    make downstream quantities undefined rather than approximate:

    The provenance segments must tile the turns with no gap and no overlap. See
    `record.provenance.check_tiling` for why, and for the half-open convention.

    The compaction events must sit inside the trajectory and no two may share a turn, so the
    prefix-length chain composes.
    """

    id: TrajectoryID
    task_ref: TaskID
    turns: tuple[Turn, ...]
    scores: "ScoreTree"
    advantage: float | None
    advantage_tokens: TokenTensorRef | None
    provenance: tuple[SegmentProvenance, ...]
    compaction: tuple[CompactionEvent, ...]
    labels: Mapping[str, "Blind[LabelValue]"]
    features: Mapping[FeatureID, float]
    capture: CaptureRef | None

    def __post_init__(self) -> None:
        check_tiling(self.provenance, len(self.turns), where=f"trajectory {self.id}")
        check_ordered(self.compaction, len(self.turns), where=f"trajectory {self.id}")

    # -- accessors the regime reading and the converters use ------------------

    @property
    def n_turns(self) -> int:
        return len(self.turns)

    @property
    def n_tokens(self) -> int:
        return sum(t.n_tokens or 0 for t in self.turns)

    @property
    def is_single_policy(self) -> bool:
        """One generating policy across the whole trajectory. The `NEAR_POLICY` precondition."""
        return policy_mixture(self.provenance).singular

    @property
    def max_staleness(self) -> int:
        return policy_mixture(self.provenance).max_staleness

    @property
    def was_compacted(self) -> bool:
        """Whether any recorded compaction actually removed tokens."""
        return any(not e.is_lossless for e in self.compaction)

    def turns_of(self, policy_version: PolicyVersionID | str) -> tuple[Turn, ...]:
        """The turns generated by one policy version, in order."""
        wanted = str(policy_version)
        keep = {
            i
            for seg in self.provenance
            if str(seg.policy_version) == wanted
            for i in range(*seg.turn_range)
        }
        return tuple(t for t in self.turns if t.index in keep)


def make_trajectory(
    *,
    id: str,
    task_ref: str,
    turns: Sequence[Turn],
    scores: Any = None,
    advantage: float | None = None,
    advantage_tokens: TokenTensorRef | None = None,
    provenance: Sequence[SegmentProvenance] | None = None,
    policy_version: str = "unknown",
    engine: Engine | None = None,
    compaction: Sequence[CompactionEvent] = (),
    labels: Mapping[str, Any] | None = None,
    features: Mapping[str, float] | None = None,
    capture: CaptureRef | None = None,
) -> Trajectory:
    """Build a `Trajectory` with the defaults `Trajectory` itself deliberately refuses to have.

    The dataclass has no defaults so that a converter cannot omit a field that decides a regime
    condition. That is right for the converters and wrong for a test fixture and for the common
    single-policy case, so the ergonomics live here, in one visible place, and the default
    provenance is a single segment covering every turn rather than an empty tuple.
    """
    if provenance is None:
        provenance = single_segment(
            len(turns), policy_version, engine=engine or Engine(name="unknown")
        )
    return Trajectory(
        id=TrajectoryID(id),
        task_ref=TaskID(task_ref),
        turns=tuple(turns),
        scores=scores,
        advantage=advantage,
        advantage_tokens=advantage_tokens,
        provenance=tuple(provenance),
        compaction=tuple(compaction),
        labels=dict(labels or {}),
        features={FeatureID(k): v for k, v in (features or {}).items()},
        capture=capture,
    )


@dataclass(frozen=True)
class Group:
    """One prompt or task with its K rollouts. K = 1 is legal; that is distillation."""

    id: GroupID
    task_ref: TaskID
    trajectories: tuple[Trajectory, ...]
    estimator: EstimatorSpec
    group_stats: GroupStats

    @property
    def k(self) -> int:
        return len(self.trajectories)


@dataclass(frozen=True)
class Step:
    """One optimizer update.

    ``regime_measured`` is measured rather than declared, and it is separate from
    `Run.regime`, which is what the operator claimed. The pair is the point: a run that declared
    `STATIONARY_GRADER` and measured otherwise is a finding, and it is invisible if only one of
    them is recorded.
    """

    index: int
    groups: tuple[Group, ...]
    schedule: Mapping[str, float]
    optimizer: OptimizerTelemetry
    probes: tuple[ProbeResult, ...] = ()
    instrument: InstrumentEffect = field(default_factory=InstrumentEffect)
    regime_measured: RegimeReading = field(default_factory=RegimeReading)

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError(f"step index cannot be negative; got {self.index}")

    @property
    def trajectories(self) -> tuple[Trajectory, ...]:
        return tuple(t for g in self.groups for t in g.trajectories)


# ---------------------------------------------------------------------------
# StepStream
# ---------------------------------------------------------------------------


class StepStream(abc.ABC):
    """Lazy, chunked, resumable access to a run's steps.

    A 401-step run is not a list. The reason it is an abstraction rather than a tuple is the
    auditor case: reading steps 200 to 210 of a long run must not materialise the other 390, and
    the only way to keep that true is for the type of `Run.steps` to have no method that returns
    them all. There is deliberately no `.all()`; `tuple(stream)` is available and says what it
    costs.
    """

    @abc.abstractmethod
    def __iter__(self) -> Iterator[Step]: ...

    @abc.abstractmethod
    def __len__(self) -> int: ...

    @abc.abstractmethod
    def slice(self, lo: int, hi: int) -> Iterator[Step]:
        """Steps with index in ``[lo, hi)``. Half-open, matching `core.types.Span`."""

    @property
    @abc.abstractmethod
    def indices(self) -> tuple[int, ...]:
        """Every step index in the run, without decoding any step."""

    def resume_from(self, index: int) -> Iterator[Step]:
        """Steps from ``index`` onward. A killed analysis restarts here rather than at zero."""
        upper = (max(self.indices) + 1) if self.indices else index
        return self.slice(index, upper)

    def chunks(self, size: int) -> Iterator[tuple[Step, ...]]:
        """Steps in batches of ``size``, decoded one batch at a time."""
        if size < 1:
            raise ValueError(f"chunk size must be at least 1; got {size}")
        batch: list[Step] = []
        for step in self:
            batch.append(step)
            if len(batch) == size:
                yield tuple(batch)
                batch = []
        if batch:
            yield tuple(batch)


class InMemoryStepStream(StepStream):
    """A stream over steps already in memory. What a writer is handed and what a test builds."""

    def __init__(self, steps: Sequence[Step] = ()) -> None:
        self._steps = tuple(sorted(steps, key=lambda s: s.index))

    def __iter__(self) -> Iterator[Step]:
        return iter(self._steps)

    def __len__(self) -> int:
        return len(self._steps)

    def slice(self, lo: int, hi: int) -> Iterator[Step]:
        return iter(s for s in self._steps if lo <= s.index < hi)

    @property
    def indices(self) -> tuple[int, ...]:
        return tuple(s.index for s in self._steps)


@dataclass(frozen=True)
class Run:
    """One instance of the loop diagram.

    ``kind`` carries the variants because they are the same object with nodes removed: best-of-n is
    the diagram without the optimizer, a tilt sweep replaces the optimizer with importance
    weighting, an evaluation removes the estimator too, an audit is the artifact node alone.

    ``access`` is recorded at capture time and not at read time, which is the only moment it is
    knowable: whether the recorder could reach the policy's gradients is a fact about the run, and
    reconstructing it later from what happens to be in the file is guesswork.
    """

    id: RunID
    kind: RunKind
    components: Mapping[Component, ComponentRef]
    access: AccessMatrix
    regime: RegimeDeclaration
    steps: StepStream
    arms: tuple[RunID, ...] = ()
    coupling: "CouplingSpec | None" = None
    lineage: RunLineage = field(default_factory=RunLineage)
    sampling_policy: RecordSamplingPolicy = field(default_factory=RecordSamplingPolicy)

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    def component(self, component: Component) -> ComponentRef | None:
        return self.components.get(component)


# ---------------------------------------------------------------------------
# Identity helpers
# ---------------------------------------------------------------------------


def run_id(*, name: str, seed: int | None = None, **material: Any) -> RunID:
    """A content-derived run id. Two runs with identical declared material share an id."""
    return RunID(content_hash({"name": name, "seed": seed, **material}, "run"))


def group_id(*, run: str, step: int, task: str, ordinal: int = 0) -> GroupID:
    return GroupID(
        content_hash({"run": run, "step": step, "task": task, "ordinal": ordinal}, "grp")
    )


def trajectory_id(*, group: str, ordinal: int) -> TrajectoryID:
    return TrajectoryID(content_hash({"group": group, "ordinal": ordinal}, "traj"))


def task_id(*, dataset: str, index: int | str) -> TaskID:
    return TaskID(content_hash({"dataset": dataset, "index": index}, "task"))


# ---------------------------------------------------------------------------
# Canonical forms for the kernel types the record embeds
# ---------------------------------------------------------------------------


def encode_access(matrix: AccessMatrix) -> dict[str, int]:
    return {c.name: int(a) for c, a in matrix.items()}


def decode_access(obj: Mapping[str, int]) -> dict[Component, Access]:
    return {Component[k]: Access(v) for k, v in obj.items()}


def encode_regime_reading(reading: RegimeReading) -> dict[str, Any]:
    return {
        c.name: {
            "holds": cr.holds,
            "statistic": _finite(cr.statistic),
            "threshold": _finite(cr.threshold),
            "provenance": cr.provenance,
            "detail": cr.detail,
        }
        for c, cr in reading.conditions.items()
    }


def _decoded_number(raw: Any) -> float:
    """A regime statistic on the way back in, where a measured zero is a measured zero.

    This was `_unfinite(...) or math.nan`, and `0.0 or math.nan` is `math.nan`. So a condition
    whose statistic or threshold was written as exactly 0.0 read back as "not measured", which
    collapses the two states the envelope machinery is built to keep apart. It bit precisely the
    readings that hold, because 0.0 is the passing value for most conditions and the natural
    threshold for any non-negativity one. `_unfinite` was written correctly for this hazard and
    the `or` at the call site undid it.
    """
    value = _unfinite(raw)
    return math.nan if value is None else float(value)


def decode_regime_reading(obj: Mapping[str, Any]) -> RegimeReading:
    conditions = {}
    for name, cr in obj.items():
        condition = RegimeCondition[name]
        conditions[condition] = ConditionReading(
            condition=condition,
            holds=cr.get("holds"),
            statistic=_decoded_number(cr.get("statistic")),
            threshold=_decoded_number(cr.get("threshold")),
            provenance=cr.get("provenance"),
            detail=cr.get("detail", ""),
        )
    return RegimeReading(conditions=conditions)


def encode_tensor_ref(ref: TensorRef | None) -> dict[str, Any] | None:
    return None if ref is None else ref.__canonical__()


def decode_tensor_ref(obj: Mapping[str, Any] | None) -> TensorRef | None:
    return None if obj is None else ref_from_canonical(obj)


__all__ = [
    "ComponentRef",
    "EstimatorSpec",
    "FeatureID",
    "Group",
    "GroupID",
    "GroupStats",
    "InMemoryStepStream",
    "InstrumentEffect",
    "OptimizerTelemetry",
    "ProbeResult",
    "RECORD_SCHEMA_VERSION",
    "RecordSamplingPolicy",
    "RegimeDeclaration",
    "Run",
    "RunID",
    "RunKind",
    "RunLineage",
    "SamplingCorrection",
    "SamplingScheme",
    "StalenessDistribution",
    "Step",
    "StepStream",
    "TaskID",
    "Trajectory",
    "TrajectoryID",
    "decode_access",
    "decode_foreign",
    "decode_regime_reading",
    "decode_tensor_ref",
    "encode_access",
    "encode_foreign",
    "encode_regime_reading",
    "encode_tensor_ref",
    "group_id",
    "make_trajectory",
    "run_id",
    "task_id",
    "trajectory_id",
]

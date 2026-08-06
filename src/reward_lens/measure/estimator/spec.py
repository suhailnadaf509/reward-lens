"""E1, the estimator specification, recorded: what transform actually ran.

Everyone assumes they know this. The details differ per framework and per version, and everything
downstream is conditioned on them. TRL 1.9.2 divides by ``std + 1e-4`` in one aggregation branch
(`grpo_trainer.py:2714`); `verifiers` mean-centres and does not divide at all
(`rubrics/rubric.py:406-409`); veRL takes the advantage pre-computed. Those are three different
operators wearing one name, and a number derived from a record without reading its estimator is a
number about an operator nobody checked.

So this instrument reads the `EstimatorSpec` the record carries and reports it, with three things a
bare field dump does not give you.

**It says which fields are undeclared rather than treating a placeholder as a value.**
``degenerate_policy = "unknown"`` is why `record.scores.replay_advantages` refuses: skipping a
degenerate group, zeroing it and keeping it give an empty group, a group of zeros, and a group of
advantages bounded only by epsilon, which are three different answers.

**It separates undeclared from ambiguous.** The string fields carry an explicit ``"unknown"``
sentinel. The optional numerics do not: ``clip_low = None`` means either "this trainer does not clip
advantages" or "nobody recorded whether it does", and the record cannot say which. That is a defect
in the schema rather than in any particular record, and naming it is more useful than resolving it
by assumption.

**It says whether the spec is stable across the window it was read over.** Two steps under two
estimators is a window over which no group-relative quantity has one meaning, and E2, E4 and E5 all
refuse on it rather than averaging across the change.

Nothing kills this instrument. Its catalogue record's kill condition is ``n/a`` and that is right:
it is the precondition for the rest of series E, and the rest of series E reads its output rather
than re-deriving it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Sequence

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
from reward_lens.measure.estimator._base import EstimatorInstrument
from reward_lens.record.schema import EstimatorSpec, Group, Run, Step

#: Every substrate. The estimator sits downstream of the grader and does not care what kind of
#: object produced the score, so a program verifier and a generative judge reach the same transform.
ALL_SUBSTRATES = frozenset(
    {
        Substrate.NEURAL_SCALAR,
        Substrate.NEURAL_GEN,
        Substrate.PROGRAM,
        Substrate.PROCEDURAL,
        Substrate.HUMAN,
        Substrate.COMPOSITE,
    }
)

#: What every instrument in this series needs: a record, and the estimator's own declaration inside
#: it. `Access.RECORD` is "read logged values that already exist", which is exactly and only what
#: series E does. Nothing here calls a grader, runs a policy or differentiates anything.
RECORD_ACCESS: dict[Component, Access] = {
    Component.RECORD: Access.RECORD,
    Component.ESTIMATOR: Access.RECORD,
}

#: The fields the transform table says change what a downstream number means. `extra` is
#: not among them: it is the converter's escape hatch and nothing may condition on it silently.
DECISIVE_FIELDS: tuple[str, ...] = (
    "family",
    "group_centred",
    "std_normalised",
    "std_epsilon",
    "std_ddof",
    "degenerate_policy",
    "clip_low",
    "clip_high",
    "clip_ratio_c",
    "aggregation",
    "loss_mask_policy",
    "off_policy_correction",
    "kl_penalty",
    "kl_coefficient",
    "advantage_whitening",
)

#: The three string fields whose "not recorded" state is explicit rather than inferred.
_SENTINEL_FIELDS: tuple[str, ...] = ("family", "degenerate_policy", "aggregation")

#: The optional numerics and strings where `None` conflates "this trainer does not do it" with
#: "nobody recorded whether it does". Reported as ambiguous rather than as absent, because the two
#: readings license different downstream arithmetic and the record cannot tell them apart.
_AMBIGUOUS_WHEN_NONE: tuple[str, ...] = (
    "clip_low",
    "clip_high",
    "clip_ratio_c",
    "off_policy_correction",
    "kl_penalty",
    "kl_coefficient",
)

#: The two comparators. `baseline.family_name_only` is what a card that says "we used GRPO" has
#: told you, scored as the number of decisive fields that name leaves open. `baseline.framework_default`
#: is the stronger and more useful one: how many fields you would have got **wrong** by assuming the
#: framework's documented defaults, NaN where this library has no verified default table for the
#: family.
SPEC_BASELINES: tuple[BaselineID, ...] = (
    "baseline.family_name_only",
    "baseline.framework_default",
)

#: Defaults verified against installed source rather than against documentation, with the file and
#: line each came from. The table is deliberately tiny: a default nobody has read off the source is
#: a guess, and a guess in a baseline makes the baseline lie in the flattering direction.
FRAMEWORK_DEFAULTS: dict[str, dict[str, Any]] = {
    # TRL 1.9.2, released 2026-07-28. `scale_rewards` defaults to "group", so the divide happens;
    # the epsilon is the literal `1e-4` in `(grouped - mean_k) / (std_k + 1e-4)` at
    # `grpo_trainer.py:2714`; `epsilon` (the ratio clip) defaults to 0.2 on `GRPOConfig`. The
    # divisor is `nanstd`, which multiplies the variance by `count/(count - 1)` at
    # `trl/trainer/utils.py:877-879`, so `std_ddof` is 1.
    "trl/grpo": {
        "group_centred": True,
        "std_normalised": True,
        "std_epsilon": 1e-4,
        "std_ddof": 1,
        "clip_low": 0.2,
        "clip_high": 0.2,
        "advantage_whitening": False,
    },
    # `verifiers` at commit edafab85. `score_group` writes
    # `state["advantage"] = aggregated_rewards[i] - avg_reward` at `rubrics/rubric.py:409`: mean
    # centring, no standard-deviation division, no clip. With no division there is
    # no divisor to declare, so `std_ddof` is None and that is a statement rather than a gap.
    "verifiers/score_group": {
        "group_centred": True,
        "std_normalised": False,
        "std_epsilon": None,
        "std_ddof": None,
        "clip_low": None,
        "clip_high": None,
        "advantage_whitening": False,
    },
}

#: How a recorded `family` string resolves to a row of `FRAMEWORK_DEFAULTS`. TRL's tap writes
#: `grpo/{loss_type}`, so every loss type shares the reward-transform defaults, which is correct:
#: `loss_type` selects the loss aggregation and not the advantage transform.
_FAMILY_PREFIXES: tuple[tuple[str, str], ...] = (
    ("grpo/", "trl/grpo"),
    ("verifiers", "verifiers/score_group"),
)

ESTIMATOR_SPEC_ENVELOPE = EnvelopeSpec(
    unconditional=True,
    justification=(
        "this reads a declaration the recorder wrote down and reports it. No property of the "
        "optimisation can make a transcription wrong, and the ways it can be wrong (the recorder "
        "read the config incorrectly, or the config changed mid-run) are reported as fields on the "
        "reading rather than assumed away by a regime condition."
    ),
)


# ---------------------------------------------------------------------------
# Getting groups out of whatever the caller is holding
# ---------------------------------------------------------------------------


def iter_steps(subject: Run | Sequence[Step] | Step | Sequence[Group]) -> Iterator[Step]:
    """Steps from a `Run`, a sequence of steps, or one step, and none from a bare group list.

    A `Run` holds a `StepStream` rather than a list, deliberately: reading steps 200 to 210 of a
    401-step run must not materialise the other 390. This yields lazily so that stays true.

    A caller holding groups rather than steps gets an empty iterator rather than an error. Groups
    are the level below steps, so a window of groups genuinely has no optimizer telemetry in it, and
    every consumer here treats "no steps" as "no per-step series", which is exactly right.
    """
    if isinstance(subject, Run):
        yield from subject.steps
    elif isinstance(subject, Step):
        yield subject
    else:
        for item in subject:
            if isinstance(item, Step):
                yield item


def iter_groups(subject: Run | Sequence[Step] | Step | Sequence[Group]) -> Iterator[Group]:
    """Groups from anything the caller is plausibly holding, in record order."""
    if isinstance(subject, (Run, Step)):
        for step in iter_steps(subject):
            yield from step.groups
        return
    items = list(subject)
    if items and isinstance(items[0], Group):
        yield from items  # type: ignore[misc]
        return
    for step in items:  # type: ignore[assignment]
        yield from step.groups


def collect_specs(groups: Iterable[Group]) -> list[EstimatorSpec]:
    """Every distinct `EstimatorSpec` in a window, in first-seen order.

    Distinctness is on the canonical field dict rather than on object identity, so two groups
    carrying equal specs built by two converter calls count once.
    """
    seen: dict[str, EstimatorSpec] = {}
    for group in groups:
        spec = group.estimator
        key = repr(sorted(spec.__canonical__().items(), key=lambda kv: kv[0]))
        seen.setdefault(key, spec)
    return list(seen.values())


# ---------------------------------------------------------------------------
# The reading
# ---------------------------------------------------------------------------


@register_payload
@dataclass
class EstimatorReading:
    """The recorded transform, what it leaves open, and whether it held still.

    ``spec_fields`` is the canonical dict rather than the `EstimatorSpec` object. `EstimatorSpec`
    is not a registered Evidence payload and registering it would mean editing the record package,
    so the reading stores the dict the record itself round-trips through and `spec` rebuilds the
    object on demand. Nothing is lost and no store row depends on a type this package does not own.

    ``stable`` is the field the rest of the series reads first. False means the window carried more
    than one transform, and no group-relative quantity has a single meaning across it.
    """

    family: str
    spec_fields: dict[str, Any]
    n_groups: int
    n_steps: int
    stable: bool
    z_scored: bool
    #: Fields sitting at an explicit "unknown" sentinel, or a `std_epsilon` a z-scoring estimator
    #: must have and does not.
    undeclared: list[str] = field(default_factory=list)
    #: Fields where `None` means either "this trainer does not do it" or "nobody recorded whether
    #: it does". The schema cannot tell those apart and neither can this instrument.
    ambiguous: list[str] = field(default_factory=list)
    #: Every distinct spec in the window, canonical, when there is more than one.
    variants: list[dict[str, Any]] = field(default_factory=list)
    baselines: dict[str, float] = field(default_factory=dict)
    #: Which row of `FRAMEWORK_DEFAULTS` the baseline used, empty when the family has no row.
    default_table: str = ""
    #: The decisive fields whose recorded value differs from the framework default.
    differs_from_default: list[str] = field(default_factory=list)
    #: Whether replaying the recorded transform reproduces the recorded advantages, and by how
    #: much it misses when it does not. None when the check was not run.
    replay: dict[str, Any] = field(default_factory=dict)
    replay_says: str = ""
    says: str = ""

    @property
    def spec(self) -> EstimatorSpec:
        return EstimatorSpec.from_canonical(self.spec_fields)

    def render(self) -> str:
        return self.says


def _phrase(spec: EstimatorSpec) -> str:
    """The transform as a sentence, built from the fields rather than from the family name."""
    if not spec.group_centred:
        head = f"Not group-relative (family {spec.family!r})"
    elif spec.std_normalised:
        eps = "an unrecorded epsilon" if spec.std_epsilon is None else f"eps = {spec.std_epsilon:g}"
        head = f"Group z-score with {eps}"
    else:
        head = "Group mean centring with no standard-deviation division"

    parts = [head]
    if spec.aggregation != "unknown":
        parts.append(f"{spec.aggregation}-level aggregation")
    if spec.clip_low is not None or spec.clip_high is not None:
        lo = "none" if spec.clip_low is None else f"{spec.clip_low:g}"
        hi = "none" if spec.clip_high is None else f"{spec.clip_high:g}"
        parts.append(f"clip {lo}/{hi}")
    if spec.loss_mask_policy not in ("unknown", "none"):
        parts.append(f"loss mask {spec.loss_mask_policy!r}")
    if spec.kl_coefficient:
        parts.append(f"KL penalty {spec.kl_penalty or 'unnamed'} at {spec.kl_coefficient:g}")
    if spec.advantage_whitening:
        parts.append("advantages whitened")
    if spec.degenerate_policy != "unknown":
        # Framework taps write a paragraph here (TRL's cites two line numbers), and a sentence is
        # not the place for it. The full text stays on `spec_fields`.
        policy = spec.degenerate_policy.strip().rstrip(".")
        if len(policy) > 60:
            policy = policy[:57].rstrip() + "..."
        parts.append(f"degenerate groups: {policy}")
    return ", ".join(parts) + "."


def _default_row(family: str) -> str:
    lowered = family.lower()
    for prefix, row in _FAMILY_PREFIXES:
        if lowered.startswith(prefix):
            return row
    return ""


# ---------------------------------------------------------------------------
# Does the recorded spec describe the transform that actually ran?
# ---------------------------------------------------------------------------

#: How far a replayed advantage may sit from the recorded one before the two count as disagreeing.
#: **Chosen: 1e-4**, which is the size of TRL's own standard-deviation epsilon and therefore the
#: scale at which the two arithmetics can legitimately differ. Anything larger is a different
#: transform rather than a different rounding.
REPLAY_TOL = 1e-4


@dataclass(frozen=True)
class ReplayCheck:
    """Whether replaying the recorded estimator reproduces the recorded advantages.

    This is the check that turns `EstimatorSpec` from a label into a claim. A spec that does not
    reproduce the advantages the trainer wrote down does not describe the transform that ran, and
    every quantity in series E and every counterfactual in `record.scores` is conditioned on it.

    It found three on the first real record it was pointed at, and they are recorded here because a
    reader looking at ``n_agree = 0`` needs to know which of them they are seeing.

    **The ratio clip applied to the advantage.** TRL's `epsilon` and `epsilon_high` are the PPO
    **ratio** clip; the TRL tap writes them into `EstimatorSpec.clip_low` and `clip_high`, which
    `record.scores.replay_advantages` applies as bounds on the **advantage**. On a run with
    `epsilon = 0.2` every replayed advantage comes back as exactly 0.2 against recorded advantages
    spanning -1.08 to +1.33. This is the largest of the three: worst disagreement 1.68 on the
    24-step record E4's frozen prediction was taken on.

    **The abstention convention.** `record.scores.evaluate` returns NaN for a composite total one of
    whose leaves abstained, which is the record's deliberate None-is-not-zero rule. TRL aggregates
    with ``(rewards_per_func * weights).nansum(dim=1)`` and only marks a row NaN when *every*
    function returned None, so one abstaining component contributes zero rather than voiding the
    row. Replaying `evaluate`'s totals against TRL's advantages is therefore comparing two different
    totals on any group holding an abstention: worst 1.29 on the same record after the clip is
    stripped, against 9.2e-07 using TRL's own convention.

    **The variance divisor.** `replay_advantages` divided by ``present.std()``, and numpy's default
    is ``ddof=0``. veRL's `compute_grpo_outcome_advantage` uses ``torch.std``, whose default is
    ``correction=1`` (`verl/trainer/ppo/core_algos.py:321`), and TRL's `nanstd` multiplies the
    variance by ``count/(count - 1)`` explicitly (`trl/trainer/utils.py:877-879`). Both apply
    Bessel's correction and that did not, so every replayed advantage came back larger than the
    recorded one by ``sqrt(K/(K-1)) - 1``: 41.4% at K=2, 15.5% at K=4, 6.9% at K=8, 0.79% at K=64,
    against a `REPLAY_TOL` of 1e-4. Measured on the 24-step record with the other two divergences
    removed: worst 0.232 under ``ddof=0`` and 9.2e-07 under ``ddof=1`` over 48 groups. It is now
    `EstimatorSpec.std_ddof`, declared per record and refused when absent, and the TRL tap writes 1.
    """

    n_groups: int
    n_comparable: int
    n_agree: int
    max_abs_error: float
    tol: float
    refusals: tuple[str, ...] = ()

    @property
    def agrees(self) -> bool:
        return self.n_comparable > 0 and self.n_agree == self.n_comparable

    @property
    def checked(self) -> bool:
        return self.n_comparable > 0

    def render(self) -> str:
        if not self.checked:
            return (
                f"The replay could not be checked on any of {self.n_groups} groups: no group "
                f"carries both a score tree and a recorded advantage."
            )
        if self.agrees:
            return (
                f"Replaying the recorded estimator reproduces the recorded advantages on all "
                f"{self.n_comparable} comparable groups, to {self.max_abs_error:.3g}."
            )
        return (
            f"Replaying the recorded estimator reproduces the recorded advantages on "
            f"{self.n_agree} of {self.n_comparable} comparable groups; worst disagreement "
            f"{self.max_abs_error:.3g} against a tolerance of {self.tol:.3g}. The recorded "
            f"EstimatorSpec does not describe the transform that ran."
        )


def check_replay(
    subject: Run | Sequence[Step] | Step | Sequence[Group],
    *,
    tol: float = REPLAY_TOL,
) -> ReplayCheck:
    """Replay each group's advantages from its recorded spec and compare against the record.

    Uses `record.scores.replay_advantages`, which already reproduces centring, z-scoring with the
    recorded epsilon, RLOO's leave-one-out factor, the degenerate policy and the clip. Nothing is
    re-derived here: what is added is the comparison against what the trainer actually wrote.
    """
    import warnings

    import numpy as np

    from reward_lens.record.scores import (
        AllAbstainedWarning,
        ScoreContext,
        evaluate,
        replay_advantages,
    )

    groups = list(iter_groups(subject))
    n_comparable = 0
    n_agree = 0
    worst = 0.0
    refusals: list[str] = []
    for group in groups:
        recorded = [t.advantage for t in group.trajectories]
        if any(a is None for a in recorded) or not recorded:
            continue
        if any(t.scores is None for t in group.trajectories):
            continue
        totals = [evaluate(t.scores, ScoreContext()) for t in group.trajectories]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", AllAbstainedWarning)
            replayed = replay_advantages(totals, group.estimator, where=str(group.id))
        if isinstance(replayed, Refusal):
            refusals.append(replayed.reason.name)
            continue
        n_comparable += 1
        a = np.asarray(replayed, dtype=float)
        b = np.asarray(recorded, dtype=float)
        live = np.isfinite(a) & np.isfinite(b)
        err = float(np.max(np.abs(a[live] - b[live]))) if live.any() else 0.0
        worst = max(worst, err)
        if err <= tol:
            n_agree += 1
    return ReplayCheck(
        n_groups=len(groups),
        n_comparable=n_comparable,
        n_agree=n_agree,
        max_abs_error=worst,
        tol=tol,
        refusals=tuple(sorted(set(refusals))),
    )


def read_estimator_spec(
    subject: Run | Sequence[Step] | Step | Sequence[Group],
    *,
    replay: bool = True,
    instrument: str = "RecordedEstimator",
) -> EstimatorReading | Refusal:
    """The recorded transform, or the refusal that says there is no record to read one from.

    Callable without a `Context` so E2 through E5 can consult it before they compute, and so a test
    can hand it three hand-built groups.
    """
    groups = list(iter_groups(subject))
    if not groups:
        return Refusal(
            instrument=instrument,
            reason=RefusalReason.ACCESS_INSUFFICIENT,
            detail=(
                "this window contains no groups, so there is no estimator declaration in it. An "
                "empty window is not a run with a default transform."
            ),
            remedy=(
                "point this at a Run, a Step, or a sequence of Groups that carries at least one "
                "group. If the record has steps but no groups, the converter dropped the group "
                "structure and every group-relative quantity is gone with it."
            ),
            statistics={"n_groups": 0},
        )

    variants = collect_specs(groups)
    spec = variants[0]
    n_steps = len({id(s) for s in iter_steps(subject)}) if isinstance(subject, Run) else 0
    if isinstance(subject, Run):
        n_steps = len(subject.steps)
    elif isinstance(subject, Step):
        n_steps = 1
    else:
        items = list(subject)
        n_steps = 0 if (items and isinstance(items[0], Group)) else len(items)

    undeclared = [f for f in _SENTINEL_FIELDS if getattr(spec, f) == "unknown"]
    if spec.loss_mask_policy == "unknown":
        undeclared.append("loss_mask_policy")
    if spec.std_normalised and spec.std_epsilon is None:
        undeclared.append("std_epsilon")
    # Undeclared rather than ambiguous: on an estimator that divides, `std_ddof = None` cannot mean
    # "this trainer does not do it". It divided by something and the record does not say by which
    # of the two, and the two differ by `sqrt(K/(K-1))`, 15.5% at K = 4.
    if spec.std_normalised and spec.std_ddof is None:
        undeclared.append("std_ddof")
    ambiguous = [f for f in _AMBIGUOUS_WHEN_NONE if getattr(spec, f) is None]

    row = _default_row(spec.family)
    differs: list[str] = []
    if row:
        for name, expected in FRAMEWORK_DEFAULTS[row].items():
            if getattr(spec, name) != expected:
                differs.append(name)

    reading = EstimatorReading(
        family=spec.family,
        spec_fields=spec.__canonical__(),
        n_groups=len(groups),
        n_steps=n_steps,
        stable=len(variants) == 1,
        z_scored=spec.z_scored,
        undeclared=sorted(set(undeclared)),
        ambiguous=ambiguous,
        variants=[] if len(variants) == 1 else [v.__canonical__() for v in variants],
        baselines={
            # Knowing only the family name settles one of the fourteen decisive fields.
            "baseline.family_name_only": float(len(DECISIVE_FIELDS) - 1),
            "baseline.framework_default": float(len(differs)) if row else float("nan"),
        },
        default_table=row,
        differs_from_default=differs,
    )
    if replay:
        check = check_replay(groups)
        reading.replay = {
            "n_groups": float(check.n_groups),
            "n_comparable": float(check.n_comparable),
            "n_agree": float(check.n_agree),
            "max_abs_error": float(check.max_abs_error),
            "tol": float(check.tol),
            "agrees": float(check.agrees),
            "checked": float(check.checked),
        }
        reading.replay_says = check.render()
    reading.says = _says(reading, spec)
    return reading


def _says(reading: EstimatorReading, spec: EstimatorSpec) -> str:
    lines = [_phrase(spec)]
    if not reading.stable:
        lines.append(
            f"{len(reading.variants)} distinct estimator specifications appear across "
            f"{reading.n_groups} groups, so this window has no single transform and nothing "
            f"group-relative should be averaged across it."
        )
    if reading.undeclared:
        lines.append(
            f"Undeclared: {', '.join(reading.undeclared)}. A downstream quantity conditioned on "
            f"one of these is conditioned on a placeholder."
        )
    if reading.ambiguous:
        lines.append(
            f"Recorded as None, which means either absent or unrecorded: "
            f"{', '.join(reading.ambiguous)}."
        )
    if reading.default_table:
        if reading.differs_from_default:
            lines.append(
                f"Differs from the {reading.default_table} defaults in "
                f"{len(reading.differs_from_default)} of "
                f"{len(FRAMEWORK_DEFAULTS[reading.default_table])} checked fields: "
                f"{', '.join(reading.differs_from_default)}."
            )
        else:
            lines.append(f"Matches the {reading.default_table} defaults on every checked field.")
    if reading.replay_says and not reading.replay.get("agrees", 1.0):
        lines.append(reading.replay_says)
    return " ".join(lines)


class RecordedEstimator(EstimatorInstrument):
    """E1. The transform that actually ran, read off the record rather than assumed.

    Kill condition, from the catalogue record: n/a. Nothing kills this instrument. It is the
    precondition for the rest of series E, and a series whose precondition can be killed has no
    members.
    """

    name = "RecordedEstimator"
    version = "1.0"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "E1"
    deviations = (
        "the reading separates `undeclared` from `ambiguous`, which the transform table "
        "does not. `EstimatorSpec.clip_low = None` means either 'this trainer does not clip' or "
        "'nobody recorded whether it does', and reporting both as absent would license arithmetic "
        "in the second case that is only valid in the first",
        "`baseline.framework_default` scores against a table of defaults read off installed source "
        "for two frameworks only. A family with no row scores NaN rather than being compared "
        "against a plausible-sounding default nobody verified",
    )

    quantity = "estimator.spec"
    requires: dict[Component, Access] = RECORD_ACCESS
    substrates = ALL_SUBSTRATES
    phases = frozenset({Phase.IN_RUN, Phase.POST_RUN})
    envelope = ESTIMATOR_SPEC_ENVELOPE
    #: `none` in the registry, which resolves to the trivial group. It is a declaration rather than
    #: an omission: an affine rescaling of the reward does not act on a record of which transform
    #: ran, and saying so is the honest answer.
    invariance = "trivial"
    invariance_relation = INVARIANT
    baselines = SPEC_BASELINES
    rung = 0

    def __init__(
        self,
        subject: Run | Sequence[Step] | Step | Sequence[Group] | None = None,
        *,
        replay: bool = True,
    ) -> None:
        self.subject = subject
        self.replay = bool(replay)

    def compute(self) -> Any:
        if self.subject is None:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail="no record was supplied, so there is no estimator declaration to read",
                remedy=(
                    "pass `subject=` a Run, a Step, or a sequence of Groups. That is "
                    "RECORD:RECORD and ESTIMATOR:RECORD and nothing else: no grader call, no "
                    "policy, no gradients."
                ),
            )
        return read_estimator_spec(self.subject, replay=self.replay, instrument=self.name)


__all__ = [
    "ALL_SUBSTRATES",
    "DECISIVE_FIELDS",
    "ESTIMATOR_SPEC_ENVELOPE",
    "FRAMEWORK_DEFAULTS",
    "RECORD_ACCESS",
    "REPLAY_TOL",
    "SPEC_BASELINES",
    "EstimatorReading",
    "RecordedEstimator",
    "ReplayCheck",
    "check_replay",
    "collect_specs",
    "iter_groups",
    "iter_steps",
    "read_estimator_spec",
]

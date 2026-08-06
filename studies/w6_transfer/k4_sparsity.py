"""W6.7, K4: update sparsity re-measured in a controlled configuration and from master weights.

The central weight-space fact about reinforcement learning is that its updates are sparse: between
4.0% and 31.5% of parameters move, depending on the algorithm. Two things about how that was
established decide what this row is for.

It was measured on **published checkpoints downloaded from a model hub**, by differencing a
post-RL release against its pre-RL base. Those checkpoints are stored in BF16 and nobody controlled
what happened between them. And the mechanism offered for the sparsity is a conjecture about *data
proximity*, that RL revisits states the model already handles, which is a claim about the training
distribution rather than about the on-policy axis. The one numerical mechanism stated alongside it,
that updates below 1e-40 are unrepresentable, is orthogonal to staleness: it is a statement about
denormals, and no optimiser step in a trained run is anywhere near 1e-40.

So there are two separate re-measurements to make and this row makes both.

The format re-measurement, which may take the whole result away
---------------------------------------------------------------

A BF16 tensor carries eight bits of significand. Adding an update to a weight and casting back to
BF16 returns the *unchanged* weight whenever the update is below half an interval in the last place
at that weight's exponent, which for a weight near 0.01 is 3.05e-5. A typical per-step update at
a learning rate of 1e-6 is orders of magnitude smaller than that. So the great majority of
individual steps leave a stored BF16 checkpoint bit-identical, and a "sparsity" measured by
differencing two stored checkpoints is partly a measurement of the storage format.

`format_floor` computes exactly that quantity: given the FP32 master weights and the FP32 updates,
the fraction of updates that would vanish under the cast. It is the matched control this row needs,
and the honest possibility is that it accounts for most of the published number. **Both outcomes are
results.** If the sparsity survives from master weights, it is a fact about optimisation. If it
collapses, the field has been reporting a property of BF16.

The staleness axis, which nobody has tested either way
-------------------------------------------------------

Staleness is how many optimiser updates the sampling policy lags the learner by. Controlled
staleness is a configuration you set, not one you find in a released checkpoint, which is why this
half needs a run rather than a download. The instrument reports sparsity at each staleness level and
the slope between them, and it reports the slope as a statistic rather than as its quantity: a
sparsity is dimensionless and a sparsity per step is not, and the registry carries no id for the
second. That distinction is E15's, and it is the reason the two are not both called the reading.

One thing not to build on
--------------------------

The published KL ablation moves sparsity by 1.0 percentage point, on one algorithm, once. That will
not support a claim about KL and sparsity in either direction, and no hypothesis below mentions it.

What real subject this needs, and what it costs
-----------------------------------------------

A **controlled group-relative RL run whose FP32 master weights are checkpointed**, at several
staleness settings. No published run supplies this: master weights live in the optimiser and are not
released, which is precisely why the format question is open. The cheapest useful version is
rung 0 alone, one staleness setting and three seeds, which answers the format question and needs no
staleness sweep at all.

Priced at 360 GPU-hours and $655 to $839 for rung 0, and 1,800 GPU-hours and $3,276 to $4,194 for
both rungs, storage included in both. `quote()` and `quote_rung0()` carry the line items and the
numbers in this sentence are what they return. This is the most expensive of the three rows in this
package and it is the only one that needs a training run.

    python -m studies.w6_transfer.k4_sparsity --runbook
    python -m studies.w6_transfer.k4_sparsity --price
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from reward_lens.core.envelope import EnvelopeSpec
from reward_lens.core.evidence import Uncertainty, register_payload
from reward_lens.core.invariance import Relation
from reward_lens.core.reading import Reading, Refusal, RefusalReason
from reward_lens.core.types import (
    Access,
    Capability,
    Component,
    GaugeStatus,
    Phase,
    Site,
    Substrate,
)
from reward_lens.measure.base import BaseObservable, Context
from reward_lens.studies.freeze import FrozenStudy, freeze
from reward_lens.studies.spec import Hypothesis, KillCriterion, Prediction, StudySpec, SubjectQuery
from studies.w6_transfer.pricing import (
    FP32_BYTES_PER_PARAM,
    GRPO_ARM_GPU_HOURS,
    STORAGE_DOLLARS_PER_GB_MONTH,
    LineItem,
    Quote,
)

#: Bits of significand a BF16 value carries: one implicit plus seven stored. The single constant the
#: format argument turns on, and the reason a BF16 checkpoint hides small updates.
BF16_SIGNIFICAND_BITS = 8

#: The published cross-algorithm range this row re-measures against, as a fraction of parameters
#: that move. It is the mandatory baseline, and the catalogue notes it trends the *other* way across
#: algorithms from what the staleness conjecture would predict.
PUBLISHED_SPARSITY_RANGE = (0.040, 0.315)

#: Staleness settings the sweep runs at. Zero is fully on-policy; eight is the practical ceiling in
#: the asynchronous stacks that motivate the question. Five levels rather than three because the
#: prediction is about a trend and three points make a trend out of any two of them plus noise.
STALENESS_LEVELS: tuple[int, ...] = (0, 1, 2, 4, 8)

#: How many seeds per staleness level. Three is the floor for a spread with a usable standard error,
#: the same floor L1 argues for between-seed homogeneity and for the same reason.
SEEDS_PER_LEVEL = 3


# ---------------------------------------------------------------------------
# The arithmetic, which is the part that can be proven without a GPU
# ---------------------------------------------------------------------------


def bf16_round(x: np.ndarray) -> np.ndarray:
    """Round to BF16 precision and back to float64, without needing a BF16 dtype.

    BF16 is a float32 with the low sixteen bits of the significand discarded, so the cast is
    representable exactly in numpy by masking a float32 view. Round-to-nearest-even is implemented
    by adding the rounding bias before truncating, which is what hardware does and what a plain
    truncation gets wrong by up to a full interval in the last place.
    """
    f32 = np.asarray(x, dtype=np.float32)
    bits = f32.view(np.uint32)
    # Round to nearest even on the 16-bit boundary: add 0x7FFF plus the lowest kept bit.
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return rounded.view(np.float32).astype(np.float64)


def update_sparsity(before: np.ndarray, after: np.ndarray) -> float:
    """The fraction of parameters that did not change at all between two tensors.

    Exact equality, not a tolerance. A tolerance would make this a measurement of the tolerance:
    the published number is the fraction of entries that are bit-identical in two stored
    checkpoints, and any epsilon above zero silently adds every entry whose update was smaller than
    it, which is the very effect this row is trying to separate out.
    """
    a = np.asarray(before, dtype=np.float64).reshape(-1)
    b = np.asarray(after, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(
            f"the two checkpoints have {a.size} and {b.size} parameters at this site. A sparsity "
            f"is a fraction of matched entries, so a shape mismatch is a mis-paired site rather "
            f"than a small discrepancy."
        )
    return float(np.mean(a == b))


def format_floor(master_before: np.ndarray, master_after: np.ndarray) -> float:
    """What fraction of real updates the BF16 cast would hide. The matched control for this row.

    Given the FP32 master weights before and after, this is the fraction of parameters whose update
    is real in FP32 and invisible in BF16. It is the sparsity a stored-checkpoint measurement would
    report on a run where **nothing at all** was sparse, so it is the number the published figure
    has to beat before it is a fact about optimisation rather than about storage.
    """
    a = np.asarray(master_before, dtype=np.float64).reshape(-1)
    b = np.asarray(master_after, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("master weights before and after must have the same shape")
    moved_in_fp32 = a != b
    hidden_in_bf16 = bf16_round(a) == bf16_round(b)
    if not moved_in_fp32.any():
        return 0.0
    return float(np.mean(moved_in_fp32 & hidden_in_bf16))


def representable_step(weight: float) -> float:
    """The smallest update that survives a BF16 round-trip at this weight, to one significant digit.

    Half an interval in the last place at the weight's exponent. At a weight of 0.01 this returns
    3.05e-5, which is thirty times a per-step update at a learning rate of 1e-6, and that ratio is
    the whole of the format argument stated as one number.
    """
    w = abs(float(weight))
    if w == 0.0 or not math.isfinite(w):
        return 0.0
    exponent = math.floor(math.log2(w))
    return float(2.0 ** (exponent - BF16_SIGNIFICAND_BITS))


@register_payload
@dataclass(frozen=True)
class SparsityReading:
    """Sparsity at one staleness, measured both ways, with the format control beside it.

    ``stored`` is what differencing two BF16 checkpoints reports, which is what the published
    number is. ``master`` is the same fraction from the FP32 master weights. ``floor`` is how much
    of the difference the cast accounts for. The three are reported together because any one of
    them alone is unreadable.
    """

    staleness: int
    stored: float
    master: float
    floor: float
    n_parameters: int
    per_site: Mapping[str, float] = None  # type: ignore[assignment]
    seed: int = 0
    note: str = ""

    @property
    def format_share(self) -> float:
        """What fraction of the stored sparsity the BF16 cast explains. One means all of it."""
        if self.stored <= 0.0:
            return float("nan")
        return float(self.floor / self.stored)

    @property
    def survives_from_master(self) -> bool:
        """Whether sparsity measured from master weights still clears the published floor of 4.0%."""
        return self.master >= PUBLISHED_SPARSITY_RANGE[0]

    def render(self) -> str:
        return (
            f"staleness {self.staleness}: stored {self.stored:.4f}, master {self.master:.4f}, "
            f"format floor {self.floor:.4f} ({self.format_share:.1%} of the stored figure) "
            f"over {self.n_parameters:,} parameters"
        )

    def __canonical__(self) -> dict[str, Any]:
        return {
            "staleness": self.staleness,
            "stored": self.stored,
            "master": self.master,
            "floor": self.floor,
            "format_share": self.format_share,
            "survives_from_master": self.survives_from_master,
            "n_parameters": self.n_parameters,
            "per_site": dict(self.per_site or {}),
            "seed": self.seed,
            "note": self.note,
        }


@register_payload
@dataclass(frozen=True)
class StalenessCurve:
    """Sparsity across staleness levels, and the slope, which is deliberately not the reading.

    ``slope`` is a sparsity per step. `spec/QUANTITIES.yaml` registers
    `policy.update_sparsity_vs_staleness` as dimensionless, so the slope is a different quantity in
    a different unit and the registry carries no id for it. It is carried here as a statistic and
    is not what `measure` emits. E15 is the reason: an instrument that emits a rate under a
    dimensionless id makes the rate comparable to every other dimensionless reading in the store.
    """

    points: tuple[SparsityReading, ...]
    slope: float
    slope_se: float
    intercept: float
    n_levels: int
    n_seeds: int
    note: str = ""

    @property
    def at_zero(self) -> float:
        """Sparsity at staleness zero, from the fitted line rather than from one noisy arm."""
        return float(self.intercept)

    @property
    def trend_is_resolved(self) -> bool:
        """Whether the slope clears twice its own standard error in either direction."""
        return math.isfinite(self.slope_se) and abs(self.slope) > 2.0 * self.slope_se

    def render(self) -> str:
        lines = [f"{self.n_levels} staleness levels x {self.n_seeds} seeds"]
        lines += [f"  {p.render()}" for p in self.points]
        verdict = "resolved" if self.trend_is_resolved else "not separated from zero"
        lines.append(
            f"  slope {self.slope:+.5f} +/- {self.slope_se:.5f} per step of staleness ({verdict}); "
            f"sparsity at staleness 0 is {self.at_zero:.4f}"
        )
        if self.note:
            lines.append(f"  {self.note}")
        return "\n".join(lines)

    def __canonical__(self) -> dict[str, Any]:
        return {
            "points": [p.__canonical__() for p in self.points],
            "slope": self.slope,
            "slope_se": self.slope_se,
            "intercept": self.intercept,
            "at_zero": self.at_zero,
            "trend_is_resolved": self.trend_is_resolved,
            "n_levels": self.n_levels,
            "n_seeds": self.n_seeds,
            "note": self.note,
        }


def fit_staleness_curve(
    points: Sequence[SparsityReading], *, which: str = "master", note: str = ""
) -> StalenessCurve | Refusal:
    """Ordinary least squares of sparsity on staleness, with a standard error on the slope.

    ``which`` selects the column: `master` is the re-measurement and is the default, `stored` is
    what a hub-checkpoint study would have seen. Refuses on fewer than three distinct staleness
    levels, because two levels fit a line with no residual and report a slope whose standard error
    is zero on a trend nothing constrained.
    """
    levels = sorted({p.staleness for p in points})
    if len(levels) < 3:
        return Refusal(
            instrument="studies.w6_transfer.k4_sparsity.fit_staleness_curve",
            reason=RefusalReason.RECORD_INCOMPLETE,
            detail=(
                f"{len(levels)} distinct staleness level(s). A line through two levels leaves no "
                f"residual, so its slope comes back with a standard error of zero and the trend "
                f"is asserted rather than measured."
            ),
            remedy=(
                f"run at three or more staleness settings. The registered sweep is "
                f"{list(STALENESS_LEVELS)}, which spans the range the asynchronous stacks use."
            ),
            statistics={"n_levels": len(levels)},
        )
    xs = np.asarray([p.staleness for p in points], dtype=np.float64)
    ys = np.asarray([getattr(p, which) for p in points], dtype=np.float64)
    n = xs.size
    sxx = float(((xs - xs.mean()) ** 2).sum())
    slope = float(((xs - xs.mean()) * (ys - ys.mean())).sum() / sxx)
    intercept = float(ys.mean() - slope * xs.mean())
    dof = n - 2
    resid = ys - (intercept + slope * xs)
    s_resid = float(math.sqrt(float((resid**2).sum()) / dof)) if dof >= 1 else float("nan")
    slope_se = (
        float(s_resid / math.sqrt(sxx)) if math.isfinite(s_resid) and sxx > 0 else float("nan")
    )
    return StalenessCurve(
        points=tuple(sorted(points, key=lambda p: (p.staleness, p.seed))),
        slope=slope,
        slope_se=slope_se,
        intercept=intercept,
        n_levels=len(levels),
        n_seeds=len({p.seed for p in points}),
        note=note or f"fitted on the `{which}` column",
    )


def site_sparsity(
    runtime_before: Any, runtime_after: Any, sites: Sequence[Site]
) -> dict[str, float]:
    """Per-site sparsity, through the shipped weight-reading contract rather than a module walk.

    `policy.base.site_weights` returns the runtime itself when the runtime answers natively and
    wraps it in the one module-tree adapter this library has when it does not, and
    `policy.arch.describe` resolves the block list, the embedding and the attention output
    projection structurally. Between them there is no architecture knowledge left for this function
    to have, which is the point: an instrument that walked the module tree here would be the second
    such walk and would go stale on the first architecture that renamed a submodule.
    """
    from reward_lens.policy.base import site_weights

    before = site_weights(runtime_before)
    after = site_weights(runtime_after)
    out: dict[str, float] = {}
    for site in sites:
        w0 = before.weight_at(site).detach().to("cpu").float().numpy()
        w1 = after.weight_at(site).detach().to("cpu").float().numpy()
        out[str(site)] = update_sparsity(w0, w1)
    return out


# ---------------------------------------------------------------------------
# The instrument
# ---------------------------------------------------------------------------

#: K4's envelope, unconditional with a justification that is not a shrug. The reading is a property
#: of a pair of stored parameter tensors: a fraction of entries that are bit-identical. Every way it
#: can be wrong is structural rather than regime-dependent, and each is refused on where it happens:
#: a shape mismatch is a mis-paired site, a missing site map is a `WeightsUnavailable`, and an
#: absent master-weight column is a `RECORD_INCOMPLETE`. `NEAR_POLICY` is deliberately not declared,
#: and the reason is worth stating: it is measured by `run.max_staleness`, which is this row's
#: **independent variable**. Declaring it would make the instrument refuse exactly the settings it
#: exists to compare.
SPARSITY_ENVELOPE = EnvelopeSpec(
    unconditional=True,
    justification=(
        "the reading counts bit-identical entries between two stored parameter tensors, so no "
        "regime of the run that produced them can make the count wrong. Its structural failures "
        "(a mis-paired site, an unavailable site map, a missing master-weight column) are refused "
        "at their source. `NEAR_POLICY` is not declared because staleness is the independent "
        "variable here, and requiring it would refuse the comparison the row is for."
    ),
)


class UpdateSparsityUnderStaleness(BaseObservable):
    """K4: how much of RL's update sparsity is optimisation and how much is the storage format.

    **This instrument does not train anything.** It consumes checkpoint pairs, one per staleness
    setting per seed, in both their stored and their master-weight forms, and reports the three
    numbers that have to be read together. The training is the runbook's job and is the whole of the
    price.

    The stored column alone reproduces the published measurement. The master column is the
    re-measurement. The floor is what a run with no sparsity at all would have reported through the
    same pipeline, and it is the control that makes the other two readable.
    """

    name = "UpdateSparsityUnderStaleness"
    version = "1.0"
    quantity = "policy.update_sparsity_vs_staleness"
    #: No signal capability: this reads parameters, not scores. The access matrix is where the real
    #: requirement lives, and it is `CONTROL` on the optimizer, because master weights are the
    #: optimizer's state and no amount of read access to a released checkpoint reaches them.
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    requires = {
        Component.POLICY: Access.SOURCE,
        Component.OPTIMIZER: Access.CONTROL,
        Component.RECORD: Access.RECORD,
    }
    substrates = frozenset({Substrate.NEURAL_GEN, Substrate.NEURAL_SCALAR})
    phases = frozenset({Phase.IN_RUN, Phase.POST_RUN})
    envelope = SPARSITY_ENVELOPE
    #: `policy.reparam` is the registry's group for this quantity and the split inside it is the one
    #: E13 names: Fisher-metric quantities are invariant and `||dtheta||` must be declared
    #: `raw_only`. A count of unchanged coordinates is in the second class, not the first. A smooth
    #: reparameterisation of the parameters does not preserve which coordinates moved: it mixes
    #: them, so a coordinate that was exactly unchanged in one basis is generically changed in
    #: another. Declaring this invariant would be declaring something false and passing a generated
    #: test that had been weakened to let it through.
    invariance = "policy.reparam"
    invariance_relation = Relation("raw_only")
    baselines = (
        "the published cross-algorithm table, 4.0% to 31.5% of parameters moved, measured by "
        "differencing hub checkpoints. The catalogue notes it trends the other way across "
        "algorithms from what the staleness conjecture predicts",
        "the format floor: the sparsity the same pipeline reports on a run where every parameter "
        "moved in FP32. This is the matched control and it is computed by `format_floor`",
    )
    rung = 1
    faithful_to = None
    deviations = (
        "sparsity is exact bit equality rather than a threshold on the update size. That matches "
        "what a checkpoint-differencing study measures and it means the number is a property of "
        "the stored dtype as much as of the optimiser, which is the finding rather than a defect.",
        "the format floor is computed on the parameters that moved in FP32. On a run where some "
        "parameters genuinely never move, the floor understates the format's share of the stored "
        "figure, because those parameters are excluded from the denominator by construction.",
        "the staleness slope is fitted by ordinary least squares over the level means, which "
        "weights every level equally. A design with unequal seeds per level should refit weighted, "
        "and the registered sweep is balanced so that it does not have to.",
    )

    def __init__(
        self,
        readings: Sequence[SparsityReading] = (),
        *,
        column: str = "master",
        note: str = "",
    ) -> None:
        self.readings = tuple(readings)
        self.column = column
        self.note = note

    def curve(self) -> StalenessCurve | Refusal:
        if not self.readings:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.RECORD_INCOMPLETE,
                detail=(
                    "no checkpoint pairs supplied. This instrument reports on pairs somebody "
                    "trained; it does not produce them."
                ),
                remedy=(
                    "run the sweep in the runbook and pass one `SparsityReading` per staleness "
                    "level per seed. `format_floor` needs the FP32 master weights, which have to "
                    "be dumped during the run: they cannot be recovered from a released checkpoint."
                ),
            )
        missing_master = [r for r in self.readings if not math.isfinite(r.master)]
        if missing_master:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.RECORD_INCOMPLETE,
                detail=(
                    f"{len(missing_master)} of {len(self.readings)} readings carry no master-weight "
                    f"sparsity, so the re-measurement this row exists for cannot be made and only "
                    f"the published stored figure is reproducible."
                ),
                remedy=(
                    "dump the optimiser's FP32 master weights at each checkpoint, not the cast "
                    "copy the trainer saves. In most stacks that is one hook on the optimiser "
                    "state and 32 GB per checkpoint for an 8B policy."
                ),
                statistics={"n_missing": len(missing_master), "n_total": len(self.readings)},
            )
        return fit_staleness_curve(self.readings, which=self.column, note=self.note)

    def measure(self, ctx: Context) -> Any:
        got = self.curve()
        if isinstance(got, Refusal):
            return got
        # The emitted value is the sparsity at staleness zero, which is dimensionless and is what
        # `policy.update_sparsity_vs_staleness` is registered as. The slope rides in the payload as
        # a statistic and is not the reading: it is a sparsity per step, the registry carries no id
        # for that unit, and emitting it under a dimensionless id would make it comparable to every
        # other dimensionless reading in the store.
        return ctx.emit(
            got,
            uncertainty=Uncertainty(
                n=len(self.readings),
                seed_spread=float(np.std([r.master for r in self.readings], ddof=1))
                if len(self.readings) > 1
                else None,
                method=(
                    "ordinary least squares over staleness levels; the interval that matters is on "
                    "the slope and is reported as slope_se on the payload"
                ),
            ),
            baselines={
                "published_low": PUBLISHED_SPARSITY_RANGE[0],
                "published_high": PUBLISHED_SPARSITY_RANGE[1],
                "format_floor": float(np.mean([r.floor for r in self.readings])),
            },
            subject_extra={
                "column": self.column,
                "levels": str(sorted({r.staleness for r in self.readings})),
            },
        )

    def estimate(self, ctx: Context | None = None) -> Reading:
        return super().estimate(ctx or Context(readout="reward"))


# ---------------------------------------------------------------------------
# The registered study
# ---------------------------------------------------------------------------

DISCLOSURE = (
    "blind on both arms. Nothing in this row has been measured: no controlled run exists, no "
    "master weights have been dumped, and no sparsity has been computed on any real checkpoint "
    "pair. What has been computed is the format arithmetic on synthetic tensors, where a planted "
    "run in which every parameter moves in FP32 reports 0.96635 sparsity through a BF16 cast at a "
    "weight scale of 0.02 and a step scale of 1e-6, and the BF16 rounding matches torch's own cast "
    "bit for bit on 200,000 values. That is the acceptance test, it is on synthetic data, and it "
    "fixes no expectation about what a real run does."
)

STUDY = StudySpec(
    id="k4-sparsity-under-controlled-staleness",
    title="Is RL's update sparsity a fact about optimisation or about BF16 storage?",
    science="S06-weight-space",
    hypotheses=(
        Hypothesis(
            id="H-format-explains-most",
            statement=(
                "The BF16 cast accounts for most of the sparsity a checkpoint-differencing study "
                "reports: the format floor exceeds half the stored figure."
            ),
            prediction=Prediction(
                metric="format_share",
                comparator=">",
                threshold=0.5,
                ci_excludes=0.5,
                rationale=(
                    "a per-step update at a learning rate of 1e-6 is far below half an interval in "
                    "the last place at any weight scale a trained network carries, so most single "
                    "steps are invisible in storage by construction. The open question is how much "
                    "survives accumulation across the many steps between two released checkpoints, "
                    "and that is what the run measures."
                ),
            ),
            scoreboard_row="K4",
        ),
        Hypothesis(
            id="H-master-sparsity-collapses",
            statement=(
                "Measured from FP32 master weights, sparsity falls below the bottom of the "
                "published cross-algorithm range, 4.0%."
            ),
            prediction=Prediction(
                metric="master_sparsity_at_zero",
                comparator="<",
                threshold=PUBLISHED_SPARSITY_RANGE[0],
                ci_excludes=PUBLISHED_SPARSITY_RANGE[0],
                rationale=(
                    "an optimiser with momentum writes to every coordinate it has a non-zero "
                    "gradient estimate for, and in FP32 there is no representation floor to stop "
                    "it. If sparsity survives here it is a real property of the update and the "
                    "data-proximity conjecture gains its first controlled support."
                ),
            ),
            scoreboard_row="K4",
        ),
        Hypothesis(
            id="H-staleness-does-nothing",
            statement=(
                "Sparsity from master weights does not trend with staleness across the range 0 to "
                "8: the fitted slope does not clear twice its own standard error."
            ),
            prediction=Prediction(
                metric="slope_t",
                comparator="<",
                threshold=2.0,
                rationale=(
                    "the mechanism offered for the sparsity is about data proximity, which is a "
                    "property of the task distribution and not of how far the sampler lags the "
                    "learner. The stated numerical mechanism, unrepresentable updates below 1e-40, "
                    "is about denormals and is orthogonal to staleness. So the registered "
                    "prediction is the null, and it is registered as the null rather than left "
                    "unstated so that finding a trend is a result rather than an exploration."
                ),
            ),
            scoreboard_row="K4",
        ),
    ),
    analysis="studies.w6_transfer.k4_sparsity.UpdateSparsityUnderStaleness.curve",
    subjects=SubjectQuery(
        signals=("k4.controlled-grpo.8b",),
        extra={
            "staleness_levels": list(STALENESS_LEVELS),
            "seeds": SEEDS_PER_LEVEL,
            "checkpoint_dtype": "fp32 master weights and the bf16 cast, both dumped",
            "sites": "policy.arch.describe resolves the block list, the embedding and the "
            "attention output projection structurally",
        },
    ),
    kill_criteria=(
        KillCriterion(
            id="K-no-sparsity-at-all",
            metric="master_sparsity_at_zero",
            comparator="<",
            threshold=0.001,
            description=(
                "fewer than one parameter in a thousand is unchanged from master weights. Then "
                "'RL updates are sparse' is a statement about BF16 checkpoints and nothing else, "
                "and every weight-space method built on the premise needs re-reading. This is the "
                "outcome that would make the row worth its price on its own."
            ),
        ),
        KillCriterion(
            id="K-null-under-matched-control",
            metric="format_share",
            comparator="<",
            threshold=0.05,
            description=(
                "the format floor explains almost none of the stored figure. Then the published "
                "sparsity is a real optimisation effect, the format objection this row is built on "
                "is wrong, and what remains is the staleness question alone. Registering this "
                "means the row cannot quietly become a story about BF16 if the data says otherwise."
            ),
        ),
    ),
    version=1,
    notes=DISCLOSURE,
)


def power_plan(replicates: int = 4000, seed: int = 0) -> Any:
    """M10's plan for the primary comparison: stored sparsity against master-weight sparsity.

    The design is paired at the parameter level, which is the natural unit here and is also what
    makes the row cheap to power: a single 8B checkpoint pair supplies eight billion paired
    observations, so the comparison is limited by the number of *runs*, not by n within a run. The
    plan below is therefore struck at the number of independent arms, 15, and its honest verdict is
    that a per-arm comparison at this size is not resolved. That is the number worth registering:
    the parameter-level difference will be significant on any single arm, and the question of
    whether it replicates across runs is the one 15 arms is thin for.

    ``accuracy_a`` and ``accuracy_b`` are the bottom and top of the published cross-algorithm range,
    used as the planning marginals because they are the only measured figures on this quantity.
    """
    from reward_lens.stats.power import PairedBinaryDesign, plan

    design = PairedBinaryDesign(
        n=len(STALENESS_LEVELS) * SEEDS_PER_LEVEL,
        accuracy_a=PUBLISHED_SPARSITY_RANGE[0],
        accuracy_b=PUBLISHED_SPARSITY_RANGE[1],
        rho=0.0,
    )
    return plan(design, replicates=replicates, seed=seed)


def freeze_study(repo_dir: str | None = None) -> FrozenStudy:
    return freeze(STUDY, repo_dir=repo_dir)


def resolvable_rows(replicates: int = 400, seed: int = 0) -> int:
    """How many registered rows 15 arms settles.

    The format hypotheses are decided inside a single arm, at eight billion paired parameters, so
    they resolve whatever the arm count. The staleness hypothesis is decided across arms and is what
    the plan is thin for, so it does not count unless the plan resolves. Written out rather than
    hard-coded because the split is the interesting part of the answer.
    """
    within_arm = 2 + len(STUDY.kill_criteria)  # both format rows and both kills
    got = power_plan(replicates=replicates, seed=seed)
    return within_arm + (1 if got.resolution.resolved else 0)


# ---------------------------------------------------------------------------
# The price
# ---------------------------------------------------------------------------

#: Parameters in the policy the quote is struck for. 8B rather than 32B, and the reason is that this
#: row's question is about the optimiser and the storage format rather than about a particular
#: model's matrix, so the smallest policy on which the published sparsity range was established is
#: the right one to spend money on.
POLICY_PARAMETERS = 8_000_000_000

#: How far into a run the arms go. A converged run is not needed: sparsity is measured between
#: checkpoint pairs and 100 steps supplies plenty of pairs. A quarter of the dossier's 48-hour arm.
RUN_FRACTION = 0.25

#: Master-weight checkpoints kept per arm.
CHECKPOINTS_PER_ARM = 20


def storage_dollars(*, arms: int, months: float = 3.0) -> tuple[float, float]:
    """What the FP32 master weights cost to keep, in gigabytes and in dollars.

    Itemised rather than waved at, because it is the line that decides whether the sweep is run as
    designed or with fewer checkpoints. An 8B policy in FP32 master weights is 32 GB per checkpoint,
    twenty checkpoints per arm, fifteen arms: 9.6 TB, which is a real monthly bill rather than a
    rounding error.
    """
    gb = arms * CHECKPOINTS_PER_ARM * POLICY_PARAMETERS * FP32_BYTES_PER_PARAM / 1e9
    return gb, gb * STORAGE_DOLLARS_PER_GB_MONTH * months


def quote_rung0(resolvable: int | None = None) -> Quote:
    """Rung 0 alone: the format re-measurement, with no staleness sweep.

    Three arms at one staleness setting. This answers the question that could take the published
    result away and it is a third of the price of the full row, so it is the version to run first
    and possibly the only version to run at all.
    """
    gb, dollars = storage_dollars(arms=3)
    return Quote(
        row="W6.7 / K4 rung 0, the FP32 master-weight re-measurement alone",
        items=(
            LineItem(
                what="controlled GRPO arms, 3 seeds at staleness 0",
                gpu_hours=3 * GRPO_ARM_GPU_HOURS * RUN_FRACTION,
                why=(
                    "one 8x H100 arm for 12 hours is 96 GPU-hours, a quarter of the dossier's "
                    "48-hour reference arm. A converged run is not needed: sparsity is read "
                    "between checkpoint pairs and 100 steps supplies twenty of them."
                ),
            ),
        ),
        assumptions=(
            "the trainer can be made to dump the optimiser's FP32 master weights rather than the "
            "cast copy it normally saves. In the common stacks that is one hook on the optimiser "
            "state; in a stack that keeps no master weights at all, this row cannot be run and "
            "that is itself worth reporting.",
            f"{POLICY_PARAMETERS / 1e9:.0f}B policy, {FP32_BYTES_PER_PARAM} bytes per parameter, "
            f"{CHECKPOINTS_PER_ARM} checkpoints per arm: {gb / 1000:.1f} TB held for three months.",
            "H100 at the mid-2026 neocloud floor band of $1.50 to $2.01 per GPU-hour.",
            "no reward model is trained; the arms use a verifiable grader so the price is the "
            "policy's alone.",
        ),
        extra_dollars=(("FP32 master-weight storage, 3 months", dollars),),
        resolvable=resolvable if resolvable is not None else 0,
        registered_rows=len(STUDY.hypotheses) + len(STUDY.kill_criteria),
        subject_needed=(
            "a controlled group-relative RL run you configure yourself, with FP32 master weights "
            "dumped at each checkpoint. No published run supplies this, which is why the format "
            "question has stayed open."
        ),
        note=(
            "run this before the staleness sweep. If the sparsity collapses from master weights, "
            "the staleness axis is a question about a quantity that turned out to be an artifact."
        ),
    )


def quote(resolvable: int | None = None) -> Quote:
    """Both rungs: the format re-measurement and the controlled staleness sweep."""
    arms = len(STALENESS_LEVELS) * SEEDS_PER_LEVEL
    gb, dollars = storage_dollars(arms=arms)
    return Quote(
        row="W6.7 / K4, both rungs: format re-measurement plus the staleness sweep",
        items=(
            LineItem(
                what=f"controlled GRPO arms, {len(STALENESS_LEVELS)} staleness levels x "
                f"{SEEDS_PER_LEVEL} seeds",
                gpu_hours=arms * GRPO_ARM_GPU_HOURS * RUN_FRACTION,
                why=(
                    f"{arms} arms at 96 GPU-hours each. Staleness is set by holding the sampler's "
                    f"policy back by k optimiser updates, which costs nothing extra in compute and "
                    f"is a scheduler change rather than a second training loop."
                ),
            ),
            LineItem(
                what="checkpoint differencing and site resolution",
                gpu_hours=0.0,
                why=(
                    "CPU only. `policy.base.site_weights` reads the tensors and "
                    "`policy.arch.describe` resolves the sites; the arithmetic is a byte "
                    "comparison over 8 billion parameters per pair."
                ),
            ),
        ),
        assumptions=(
            "the trainer can dump FP32 master weights, and its sampler lag is configurable. Both "
            "are true of the asynchronous stacks that motivate the staleness question and neither "
            "is true of a downloaded checkpoint, which is the whole reason this row costs money.",
            f"{POLICY_PARAMETERS / 1e9:.0f}B policy at {FP32_BYTES_PER_PARAM} bytes per parameter, "
            f"{CHECKPOINTS_PER_ARM} checkpoints per arm across {arms} arms: {gb / 1000:.1f} TB. "
            f"That is the line that would push a budget-constrained version to fewer checkpoints "
            f"rather than to fewer arms, and fewer arms is the wrong economy: the staleness trend "
            f"is measured across arms.",
            "H100 at the mid-2026 neocloud floor band of $1.50 to $2.01 per GPU-hour. At Modal's "
            "$3.95 this row roughly doubles, and Modal is the wrong shape for long training runs "
            "in any case: it bills per second and buys no idle discount a 12-hour arm can use.",
            "100 steps per arm. A longer arm accumulates more steps between checkpoints and moves "
            "the stored sparsity down, so the step count is part of the measurement and is "
            "registered rather than tuned.",
        ),
        extra_dollars=(("FP32 master-weight storage, 3 months", dollars),),
        resolvable=resolvable if resolvable is not None else 0,
        registered_rows=len(STUDY.hypotheses) + len(STUDY.kill_criteria),
        subject_needed=(
            "a controlled group-relative RL run at several staleness settings, with FP32 master "
            "weights dumped. The most expensive subject in this package and the only one that "
            "needs a training run."
        ),
        note=(
            "the staleness half is what costs money; the format half is a third of it and answers "
            "the sharper question."
        ),
    )


# ---------------------------------------------------------------------------
# The runbook
# ---------------------------------------------------------------------------


def runbook() -> str:
    q0, q1 = quote_rung0(), quote()
    lo0, hi0 = q0.dollars
    lo1, hi1 = q1.dollars
    gb, _ = storage_dollars(arms=len(STALENESS_LEVELS) * SEEDS_PER_LEVEL)
    return f"""W6.7 / K4 -- update sparsity under controlled staleness

Price: rung 0 alone {q0.gpu_hours:,.0f} GPU-hours, ${lo0:,.0f} to ${hi0:,.0f}.
       Both rungs {q1.gpu_hours:,.0f} GPU-hours, ${lo1:,.0f} to ${hi1:,.0f}, plus {gb / 1000:.1f} TB
       of master-weight storage. Nothing below has been run.

Before spending anything
  Check that your trainer can dump the optimiser's FP32 master weights. Most stacks save the cast
  copy and call it the checkpoint. If yours keeps no master weights at all, stop: this row cannot be
  run on that stack and the fact that it cannot is worth writing down, because it is the reason the
  published number was measured the way it was.

The arms
  rung 0   staleness 0, 3 seeds.                       Answers the format question alone.
  rung 1   staleness in {list(STALENESS_LEVELS)}, {SEEDS_PER_LEVEL} seeds each.  Adds the staleness axis.

  Staleness k means the sampler's policy is k optimiser updates behind the learner's. It is a
  scheduler setting, not a second training loop, so the sweep costs arms rather than complexity.

Per arm, what to save
  Every {100 // CHECKPOINTS_PER_ARM} steps, both forms of the weights:
    * the FP32 master weights out of the optimiser state
    * the BF16 cast the trainer would normally have saved
  Saving only the second is what everyone has done and is why this row exists.

Run, in this order
  1. Freeze the study on a clean tree. `studies.w6_transfer.k4_sparsity.freeze_study()`.
  2. Rung 0 first, all three seeds. Then compute, on CPU:
       update_sparsity(bf16_before, bf16_after)      the published measurement, reproduced
       update_sparsity(master_before, master_after)  the re-measurement
       format_floor(master_before, master_after)     the matched control
     If the format floor already accounts for most of the stored figure, you have the result and
     the staleness sweep is optional.
  3. Only then rung 1. `UpdateSparsityUnderStaleness(readings).curve()` fits the trend and reports
     the slope with its standard error.
  4. Report per site as well as overall. `policy.arch.describe` resolves the block list, the
     embedding and the attention output projection structurally, and `policy.base.site_weights`
     reads the tensors, so the per-site split needs no architecture code of its own. Expect the
     embedding to dominate any parameter count on a small model: the GRPO fixture in this
     repository is 2.45M parameters of which the transformer is 26,664, and an overall sparsity
     dominated by an untied embedding pair is a statement about the vocabulary.

What a failed arm looks like
  * `curve()` refuses RECORD_INCOMPLETE naming missing master weights: the dump did not take. This
    is the failure to catch on the first arm rather than the fifteenth.
  * `update_sparsity` raises on a shape mismatch: the two checkpoints are not the same
    architecture, or the sites were paired wrongly. It raises rather than refusing because it is a
    programming error in the caller and not an anticipated condition.
  * Stored sparsity comes back at zero: the checkpoints were saved in FP32 after all, so the
    stored column is the master column and there is no format effect to measure. Check the dtype
    before concluding anything.
  * The slope does not clear twice its standard error. That is `H-staleness-does-nothing` and it
    is the registered prediction, not a failure. Report it with the standard error, not as
    "no effect".

What to publish either way
  All three columns at every staleness level, per site and overall, with the format floor next to
  the stored figure everywhere it appears. The sentence to avoid is "RL updates are sparse" with no
  dtype attached; the sentence this row can support is "RL updates are sparse in BF16 storage by X
  and in FP32 master weights by Y", and X and Y may be very different numbers.
"""


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runbook", action="store_true")
    parser.add_argument("--price", action="store_true")
    parser.add_argument("--power", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.price:
        n = resolvable_rows()
        print(quote_rung0(resolvable=n).render())
        print()
        print(quote(resolvable=n).render())
    elif args.power:
        print(power_plan().render())
    else:
        print(runbook())
    return 0


__all__ = [
    "BF16_SIGNIFICAND_BITS",
    "CHECKPOINTS_PER_ARM",
    "DISCLOSURE",
    "POLICY_PARAMETERS",
    "PUBLISHED_SPARSITY_RANGE",
    "RUN_FRACTION",
    "SEEDS_PER_LEVEL",
    "SPARSITY_ENVELOPE",
    "STALENESS_LEVELS",
    "STUDY",
    "SparsityReading",
    "StalenessCurve",
    "UpdateSparsityUnderStaleness",
    "bf16_round",
    "fit_staleness_curve",
    "format_floor",
    "freeze_study",
    "main",
    "power_plan",
    "quote",
    "quote_rung0",
    "representable_step",
    "resolvable_rows",
    "runbook",
    "site_sparsity",
    "storage_dollars",
    "update_sparsity",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""C4 `intervention.erasure_cost`: erasure with the head-to-head that is now mandatory.

The library's own `SURGERY` card removed the exploit direction at 0.886 and cost 0.399 of
RewardBench-2 accuracy. A published alternative ran concept erasure on the Skywork-Reward-V2 fleet
with the same difference-of-means-plus-null-space method and reports −0.8 points, certified
non-inferior. That is a factor of fifty in capability cost and until it is explained the library's
erasure surface cannot be cited.

**The reconciliation, and it is arithmetic rather than an argument.** RewardBench-2 is scored by
whether the chosen completion outranks every rejected one, and its rows carry `total_completions`
and `num_correct`, so the accuracy a random scorer gets is computable: 0.248435 over the actual
1,865 rows. Against that floor the `SURGERY` numbers read differently than they do against zero.
The model went from 0.681792, which is 43.3 points above chance, to 0.283040, which is 3.5 points
above chance. **92.0% of the model's entire margin over chance was destroyed.** Post-erasure it is
very close to a coin.

So the factor of fifty is not two methods disagreeing about what erasure costs. One of them is a
measurement on a model that still works and the other is a measurement on a model that has been
taken apart. They are not two points on one axis being compared; they are one point near zero dose
and one point past saturation, and comparing them without saying so is the error. `reconcile` below
computes the floor from the benchmark itself and returns that statement with the numbers in it.

**Which makes rung 1 the fix rather than a refinement.** A LEACE projection has exactly one dose,
because its guarantee is that *all* linear information about the concept is gone; there is no way to
ask it for less. So the shipped result is a therapeutic-index measurement at `n = 1` dose, taken at
the maximum dose the method can produce. Sweeping a partial erasure turns one number into `EC50`,
`E_max`, the cooperativity `n` and a calibration slope, and the ratio of the two curves' half-maximal
doses is a therapeutic index: a specification saying which doses remove the exploit and leave the
model working, if any do.

**Scope limit.** Everything here measures a projection applied to a forward pass. It says nothing
about whether the concept returns under further training, which is C5, and nothing about whether the
removed direction was carrying the behaviour or merely correlated with it, which is C6.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from reward_lens.core.evidence import Uncertainty, register_payload
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.reading import Reading, Refusal, RefusalReason
from reward_lens.core.types import Capability
from reward_lens.measure.base import Context
from reward_lens.measure.selection._common import (
    ABOVE_LOD_ONLY,
    ACCESS_GRADER_MUTATE,
    SelectionInstrument,
    emit_white_box,
)

#: The published alternative, as a declared citation rather than a measurement this library took.
#: Carried as data so a card can print the comparator beside our number without anybody retyping it,
#: and named as a citation so nothing can read it as something we ran.
PUBLISHED_ALTERNATIVE: Mapping[str, Any] = {
    "id": "one-bias-after-another",
    "arxiv": "2603.03291",
    "fleet": "Skywork-Reward-V2",
    "method": "difference-in-means plus null-space projection",
    "accuracy_delta": -0.008,
    "claim": "certified non-inferior",
    "is_a_measurement_we_took": False,
}


@register_payload
@dataclass(frozen=True)
class BenchmarkFloor:
    """What a scorer gets on a benchmark by guessing, computed from the benchmark's own structure.

    The number that makes an accuracy delta interpretable. A 40-point drop from 68% means one thing
    when the floor is 0% and something completely different when it is 25%, and best-of-N preference
    benchmarks have a floor of `1/N` that almost nothing reports beside the headline accuracy.

    ``expected_random`` is the mean of `num_correct / total_completions` over the rows, which is the
    probability a uniformly random scorer ranks a correct completion top. It is not `1/4`: 1,763 of
    RewardBench-2's 1,865 rows are best-of-4 and the Ties subset carries up to 37 completions, so
    the mean completion count is 4.81 and the floor is 0.2484.
    """

    benchmark: str
    expected_random: float
    n_rows: int
    mean_completions: float
    method: str = "mean of num_correct/total_completions over the rows"

    def margin_over_chance(self, accuracy: float) -> float:
        return float(accuracy) - self.expected_random

    def fraction_of_margin_lost(self, before: float, after: float) -> float:
        """How much of the model's whole margin over chance an intervention removed.

        The statistic the raw delta hides. Returns NaN when the model was already at or below chance
        before the intervention, because there was no margin to lose and the ratio would be a number
        with no meaning rather than a large one.
        """
        pre = self.margin_over_chance(before)
        if pre <= 0:
            return float("nan")
        return float(1.0 - self.margin_over_chance(after) / pre)

    def render(self) -> str:
        return (
            f"{self.benchmark}: a random scorer gets {self.expected_random:.4f} "
            f"({self.n_rows:,} rows, {self.mean_completions:.2f} completions each)"
        )


def rewardbench2_floor(*, split: str = "test") -> BenchmarkFloor:
    """The random-scoring accuracy of RewardBench-2, computed from the dataset rather than assumed.

    Downloads the dataset through `datasets` if it is not cached. It is 1,865 rows of metadata and
    the two columns this needs are integers, so this is cheap; what it is not is guessable, which is
    why it is computed. The commonly quoted floor of 0.25 is the best-of-4 case and 102 of the rows
    are not best-of-4.
    """
    from datasets import load_dataset

    data = load_dataset("allenai/reward-bench-2", split=split)
    total = np.asarray(data["total_completions"], dtype=np.float64)
    correct = np.asarray(data["num_correct"], dtype=np.float64)
    return BenchmarkFloor(
        benchmark="RewardBench-2",
        expected_random=float(np.mean(correct / total)),
        n_rows=int(total.size),
        mean_completions=float(np.mean(total)),
    )


@register_payload
@dataclass(frozen=True)
class Reconciliation:
    """Our erasure result and the published one, on a scale that makes them comparable.

    ``verdict`` is the sentence a card prints. It is assembled from the numbers rather than chosen,
    so a future run whose erasure is gentler produces a different sentence without anybody editing
    this file.
    """

    ours_before: float
    ours_after: float
    ours_delta: float
    ours_removal: float
    theirs_delta: float
    floor: BenchmarkFloor
    fraction_of_margin_lost: float
    ratio: float
    subject_ours: str = ""
    subject_theirs: str = ""
    differences: tuple[str, ...] = ()
    verdict: str = ""

    @property
    def ours_is_near_chance(self) -> bool:
        """Whether the post-erasure model retains less than a tenth of its margin over chance."""
        return self.fraction_of_margin_lost > 0.9

    def render(self) -> str:
        lines = [
            self.floor.render(),
            f"ours   {self.ours_before:.4f} -> {self.ours_after:.4f} "
            f"({self.floor.margin_over_chance(self.ours_before) * 100:+.1f} pp over chance -> "
            f"{self.floor.margin_over_chance(self.ours_after) * 100:+.1f} pp), "
            f"removal {self.ours_removal:.4f}",
            f"theirs {self.theirs_delta * 100:+.2f} pp, {PUBLISHED_ALTERNATIVE['claim']} "
            f"({PUBLISHED_ALTERNATIVE['arxiv']}, {self.subject_theirs})",
            f"ratio  {self.ratio:.1f}x on the raw delta; "
            f"{self.fraction_of_margin_lost:.1%} of our margin over chance destroyed",
            f"verdict: {self.verdict}",
        ]
        if self.differences:
            lines.append("differences that are named rather than assumed:")
            lines += [f"  - {d}" for d in self.differences]
        return "\n".join(lines)

    def __canonical__(self) -> dict[str, Any]:
        return {
            "ours_before": self.ours_before,
            "ours_after": self.ours_after,
            "ours_delta": self.ours_delta,
            "ours_removal": self.ours_removal,
            "theirs_delta": self.theirs_delta,
            "expected_random": self.floor.expected_random,
            "fraction_of_margin_lost": self.fraction_of_margin_lost,
            "ratio": self.ratio,
            "subject_ours": self.subject_ours,
            "subject_theirs": self.subject_theirs,
            "differences": list(self.differences),
            "verdict": self.verdict,
            "published_alternative": dict(PUBLISHED_ALTERNATIVE),
        }


#: The differences between the two experiments that can be named from what each reports. Every one
#: is a fact about the two designs rather than a hypothesis about why the numbers differ, and the
#: list is deliberately short: three things that are certainly different beats ten that might be.
NAMED_DIFFERENCES: tuple[str, ...] = (
    "different subjects. Ours is the single flagged model the campaign erased on; the published "
    "result is the Skywork-Reward-V2 fleet. A capability cost is a property of the model as much "
    "as of the method, and neither result has been run on the other's subject.",
    "different doses, and only one of them is adjustable. A LEACE projection removes all linear "
    "information about the concept by construction, so it has exactly one dose and that dose is the "
    "maximum. Difference-in-means plus a null-space projection removes a stated number of "
    "directions and can be asked for fewer. A single-dose comparison between a method fixed at "
    "saturation and a method dosed low is not a comparison of methods.",
    "different surfaces. Ours mounts the affine map at residual sites, so every downstream layer "
    "computes on erased activations; the readout-only surface edits one direction in the reward "
    "head and leaves the forward pass alone. The second cannot damage a representation the reward "
    "head does not read and the first can.",
)


def reconcile(
    *,
    ours_before: float,
    ours_after: float,
    ours_removal: float,
    floor: BenchmarkFloor,
    theirs_delta: float = float(PUBLISHED_ALTERNATIVE["accuracy_delta"]),
    subject_ours: str = "",
    subject_theirs: str = str(PUBLISHED_ALTERNATIVE["fleet"]),
) -> Reconciliation:
    """Put both results on the benchmark's own floor and say what the comparison actually shows.

    The verdict is derived, not chosen. When the erasure has taken the model to within a tenth of
    its margin over chance, the honest reading is that the two experiments are at opposite ends of a
    dose axis rather than disagreeing about a cost, and the sentence says so along with the fix. When
    it has not, the discrepancy is genuinely unexplained and the sentence says that instead.
    """
    delta = float(ours_after) - float(ours_before)
    lost = floor.fraction_of_margin_lost(ours_before, ours_after)
    ratio = abs(delta) / abs(theirs_delta) if theirs_delta else float("inf")
    if np.isfinite(lost) and lost > 0.9:
        verdict = (
            f"not a disagreement about what erasure costs. Our erasure removed {lost:.1%} of the "
            f"model's entire margin over chance, leaving it "
            f"{floor.margin_over_chance(ours_after) * 100:.1f} pp above a random scorer, so the "
            f"post-erasure model is close to a coin. The published alternative measured a model "
            f"that still works. These are one point past saturation and one point near zero dose "
            f"on the same axis, and the fix is to sweep that axis: rung 1 fits both curves and "
            f"reports the dose window, if there is one."
        )
    elif np.isfinite(lost):
        verdict = (
            f"a genuine discrepancy. Our erasure cost {abs(delta) * 100:.1f} pp against their "
            f"{abs(theirs_delta) * 100:.1f} pp, a factor of {ratio:.0f}, and the post-erasure model "
            f"retains {1 - lost:.1%} of its margin over chance, so it has not simply been "
            f"destroyed. The named differences below are the candidates and none of them has been "
            f"tested by running either method on the other's subject."
        )
    else:
        verdict = (
            "not interpretable. The pre-erasure model was already at or below the benchmark's "
            "random-scoring floor, so it had no margin over chance for the erasure to cost, and "
            "the accuracy delta is a difference between two numbers that mean nothing."
        )
    return Reconciliation(
        ours_before=float(ours_before),
        ours_after=float(ours_after),
        ours_delta=delta,
        ours_removal=float(ours_removal),
        theirs_delta=float(theirs_delta),
        floor=floor,
        fraction_of_margin_lost=float(lost),
        ratio=float(ratio),
        subject_ours=subject_ours,
        subject_theirs=subject_theirs,
        differences=NAMED_DIFFERENCES,
        verdict=verdict,
    )


def surgery_result(store_dir: str, *, sidecar_dirs: Sequence[str] = ()) -> dict[str, float]:
    """The `SURGERY` card's numbers, read out of the campaign evidence store.

    Read rather than restated. The catalogue quotes 0.886 and 0.399 and the stored result carries
    0.8856159449336694 and −0.39875212705615426 along with the two accuracies they were derived
    from, and the reconciliation needs the accuracies rather than the delta: a delta cannot be
    compared to a floor.
    """
    from reward_lens.record.convert.store import CampaignStore

    store = CampaignStore(store_dir, sidecar_dirs=sidecar_dirs)
    store.assert_no_blind_payloads()
    out: dict[str, float] = {}
    for row in store.by_observable("campaign.result.SURGERY"):
        value = store.value(row)
        fields = value if isinstance(value, dict) else getattr(value, "__dict__", {})
        for entry in fields.get("rows") or ():
            pair = list(entry)
            if len(pair) == 2:
                try:
                    out[str(pair[0])] = float(pair[1])
                except (TypeError, ValueError):
                    continue
        for key, val in (fields.get("meta") or {}).items():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                out[str(key)] = float(val)
    return out


# ---------------------------------------------------------------------------
# The dose axis
# ---------------------------------------------------------------------------


def dose_eraser(eraser: Any, alpha: float) -> Any:
    """The same eraser at a fraction of its strength: `r_a(x) = x - a·P(x - mu)`.

    This is what gives LEACE a dose axis it does not otherwise have. The closed form's guarantee is
    that *all* linear information about the concept is removed, which fixes the dose at its maximum
    and is why the shipped result is a single-dose measurement. Scaling the projection interpolates
    continuously between identity at `a = 0` and full erasure at `a = 1`.

    **A scaled projection is not a projection.** `a·P` is idempotent only at `a` of 0 or 1, so a
    partial erasure does not satisfy the LEACE guarantee and leaves `(1 - a)` of the concept
    covariance behind by construction. That is the point rather than a defect: the dose-response
    curve is the relationship between how much is left and what it costs, and a method that can only
    be run at saturation has no such curve. The returned eraser's `method` records the dose so a
    certificate can never be taken at one strength and cited at another.
    """
    from dataclasses import replace as _replace

    a = float(alpha)
    if not 0.0 <= a <= 1.0:
        raise ValueError(
            f"an erasure dose is a fraction of the full projection and must lie in [0, 1]; got {a}. "
            f"Values above 1 overshoot the concept subspace into its reflection, which removes the "
            f"concept and adds its negation."
        )
    return _replace(eraser, P=a * np.asarray(eraser.P, dtype=np.float64), method=f"leace@{a:g}")


@register_payload
@dataclass(frozen=True)
class ErasureReading:
    """What an erasure removed, what it cost, and where both sit on the dose axis.

    ``dose_response`` is None at rung 0, which is the state almost every published erasure result is
    in and is reported by name rather than by omission.
    """

    removal: float
    cost: float
    dose: float
    reconciliation: Reconciliation | None = None
    dose_response: Any = None
    n_doses: int = 1
    note: str = ""

    @property
    def rung(self) -> int:
        return 1 if self.dose_response is not None else 0

    @property
    def therapeutic_index(self) -> float | None:
        return getattr(self.dose_response, "therapeutic_index", None)

    def says(self) -> str:
        if self.dose_response is None:
            return (
                f"erasure at a single dose removed {self.removal:.4f} of the target and cost "
                f"{abs(self.cost):.4f}. One dose is one point on a curve: it cannot say whether a "
                f"gentler dose would have removed most of the target for a fraction of the cost."
            )
        ti = self.therapeutic_index
        window = (
            f"therapeutic index {ti:.3g}"
            if ti is not None
            else "no therapeutic index; the cost curve was not fitted"
        )
        return f"swept over {self.n_doses} doses: {self.dose_response.effect.render()}. {window}."

    def render(self) -> str:
        lines = [self.says()]
        if self.dose_response is not None:
            lines.append(self.dose_response.render())
        if self.reconciliation is not None:
            lines.append(self.reconciliation.render())
        if self.note:
            lines.append(self.note)
        return "\n".join(lines)

    def __canonical__(self) -> dict[str, Any]:
        return {
            "removal": self.removal,
            "cost": self.cost,
            "dose": self.dose,
            "rung": self.rung,
            "n_doses": self.n_doses,
            "therapeutic_index": self.therapeutic_index,
            "reconciliation": (
                self.reconciliation.__canonical__() if self.reconciliation is not None else None
            ),
            "dose_response": (
                self.dose_response.__canonical__() if self.dose_response is not None else None
            ),
            "says": self.says(),
            "note": self.note,
        }


# ---------------------------------------------------------------------------
# The instrument
# ---------------------------------------------------------------------------


class ErasureCost(SelectionInstrument):
    """C4. What an erasure removes, what it costs, and the dose window between them.

    Rung 0 is one dose, which is what every published erasure result reports and what the library's
    own `SURGERY` card is. Rung 1 sweeps the dose and fits both curves, which converts the kill into
    a specification: `EC50`, `E_max`, the cooperativity `n`, and `TI = TD50/ED50`.

    White-box: the eraser is fitted on captured activations and mounted into the forward pass, so
    an `IncrementalValidity` record is mandatory and is supplied from the baseline bank on the same
    items.

    What it cannot do, three lines in. It measures a projection applied to a forward pass. Whether
    the concept comes back under further training is C5 and is not measured here; whether the
    removed direction was carrying the behaviour or was correlated with something that was is C6.
    An erasure that scores well on both curves here can still fail both of those.
    """

    name = "ErasureCost"
    version = "1.0"
    quantity = "intervention.erasure_cost"
    capabilities = Capability.ACTIVATIONS
    requires = ACCESS_GRADER_MUTATE
    envelope = ABOVE_LOD_ONLY
    invariance = "repr.basis"
    #: The LEACE projection is built from second moments in the whitened space, so rotating the
    #: representation rotates `P` with it and the erased activations are the rotation of the
    #: originally erased ones. Both the removal and the cost are then unchanged.
    invariance_relation = INVARIANT
    baselines = (
        '"One Bias After Another" (2603.03291) on the same Skywork-Reward-V2 fleet with the same '
        "DiffMean-plus-null-space method, reporting -0.8pp certified non-inferior against our -0.399",
        "a coherent irrelevant semantic direction",
        "a clamp rather than an erasure",
    )
    #: `rung` is a property below rather than a class attribute, because which rung this instrument
    #: is at depends on whether a sweep was supplied. A literal here would be dead: the property
    #: shadows it and the two would disagree the moment a sweep arrived.
    faithful_to = "C4, LEACE affine concept erasure with a dose-response calibration"
    deviations = (
        "the dose axis is a scaled projection `a*P`, which is idempotent only at a of 0 and 1, so "
        "every intermediate dose leaves `(1 - a)` of the concept covariance behind by construction "
        "and does not satisfy the LEACE guarantee. That is what makes a curve exist at all, and it "
        "means the fitted E_max is the removal at full projection rather than a free parameter",
        "the capability cost is whatever the caller measured it with. The reconciliation puts it on "
        "the benchmark's random-scoring floor, which needs the benchmark's own row structure; for "
        "anything but RewardBench-2 the floor has to be supplied",
        "the published comparator is a citation and not a measurement this library took. It is "
        "carried as data in `PUBLISHED_ALTERNATIVE` with `is_a_measurement_we_took` set False",
    )

    def __init__(
        self,
        points: Sequence[Any] = (),
        *,
        removal: float | None = None,
        cost: float | None = None,
        dose: float = 1.0,
        reconciliation: Reconciliation | None = None,
        incremental: Any = None,
        baseline_scores: Mapping[str, float] | None = None,
        n_boot: int = 400,
        seed: int = 0,
        level: float = 0.95,
    ) -> None:
        self.points = tuple(points)
        self.single_removal = removal
        self.single_cost = cost
        self.dose = float(dose)
        self.reconciliation = reconciliation
        self._incremental = incremental
        self.baseline_scores = dict(baseline_scores or {})
        self.n_boot = int(n_boot)
        self.seed = int(seed)
        self.level = float(level)

    @property
    def rung(self) -> int:  # type: ignore[override]
        return 1 if len(self.points) >= 2 else 0

    def compute(self) -> Any:
        from reward_lens.organisms.dose import dose_response as fit_dose_response

        if self.points:
            fitted = fit_dose_response(
                self.points, n_boot=self.n_boot, seed=self.seed, level=self.level, dose_unit="alpha"
            )
            if isinstance(fitted, Refusal):
                # The sweep exists and could not be fitted. That is not a reason to withhold the
                # single-dose numbers the sweep already contains, so the reading drops to rung 0
                # carrying the top dose, and the refusal's own detail travels as the note.
                top = max(self.points, key=lambda p: p.dose)
                return ErasureReading(
                    removal=float(top.effect),
                    cost=float(top.cost if top.cost is not None else float("nan")),
                    dose=float(top.dose),
                    reconciliation=self.reconciliation,
                    dose_response=None,
                    n_doses=len(self.points),
                    note=(
                        f"the sweep was run at {len(self.points)} doses and the curve did not fit "
                        f"({fitted.reason.name}: {fitted.detail}), so this is the top dose reported "
                        f"as a rung-0 reading."
                    ),
                )
            top = max(self.points, key=lambda p: p.dose)
            return ErasureReading(
                removal=float(top.effect),
                cost=float(top.cost if top.cost is not None else float("nan")),
                dose=float(top.dose),
                reconciliation=self.reconciliation,
                dose_response=fitted,
                n_doses=len(self.points),
            )
        if self.single_removal is None or self.single_cost is None:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.RECORD_INCOMPLETE,
                detail=(
                    "neither a dose sweep nor a single-dose (removal, cost) pair was supplied, so "
                    "there is nothing to report"
                ),
                remedy=(
                    "pass `removal=` and `cost=` for the rung-0 reading, or `points=[DosePoint(...)]` "
                    "for the sweep. `measure.selection.erasure.dose_eraser` scales a fitted LEACE "
                    "eraser to a dose, and `organisms.dose.sweep` runs the loop over doses."
                ),
            )
        return ErasureReading(
            removal=float(self.single_removal),
            cost=float(self.single_cost),
            dose=self.dose,
            reconciliation=self.reconciliation,
            dose_response=None,
            n_doses=1,
            note=(
                "one dose. This is the rung every published erasure result is at, and it cannot say "
                "whether a gentler dose would have removed most of the target for a fraction of the "
                "cost. Rung 1 is a sweep and it is the same experiment run four more times."
            ),
        )

    def estimate(self, ctx: Context | None = None) -> Reading:
        ctx = ctx or Context(readout="score")
        pre = self.preflight(ctx)
        if not pre.ok and pre.refusal is not None:
            return pre.refusal
        ctx._observable = self
        try:
            return self.measure(ctx)
        finally:
            ctx._observable = None

    def measure(self, ctx: Context) -> Any:
        computed = self.compute()
        if isinstance(computed, Refusal):
            return computed
        if self._incremental is None:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.NO_MATCHED_CONTROL,
                detail=(
                    "this is a white-box reading and no IncrementalValidity record was supplied, so "
                    "nothing records what opening the network bought over the black-box bank. "
                    "The record is mandatory and lint rule four enforces it"
                ),
                remedy=(
                    "run `stats.baselines.run_bank` on the same items, hand the per-item margins to "
                    "`measure.meta.incremental.IncrementalValidityReading`, and pass its `.record` "
                    "as `incremental=`. For an erasure the natural own-score is whether the "
                    "erased model's per-item scores still separate the exploit."
                ),
            )
        return emit_white_box(
            ctx,
            computed,
            incremental=self._incremental,
            baselines=self.baseline_scores
            or {"published.one_bias_after_another": float(PUBLISHED_ALTERNATIVE["accuracy_delta"])},
            uncertainty=Uncertainty(
                n=computed.n_doses,
                method=(
                    f"paired bootstrap over sweep points, {self.n_boot:,} resamples"
                    if computed.dose_response is not None
                    else "single dose; no interval"
                ),
            ),
            subject_extra={"dose": f"{computed.dose:g}", "rung": str(computed.rung)},
        )


__all__ = [
    "NAMED_DIFFERENCES",
    "PUBLISHED_ALTERNATIVE",
    "BenchmarkFloor",
    "ErasureCost",
    "ErasureReading",
    "Reconciliation",
    "dose_eraser",
    "reconcile",
    "rewardbench2_floor",
    "surgery_result",
]

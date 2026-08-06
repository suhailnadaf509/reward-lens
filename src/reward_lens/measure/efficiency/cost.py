"""F3, the cost book: what the step spent, what it needed to spend, and where the nats went.

One optimiser step moves the policy by `D_t = KL(π_{t+1} ‖ π_t) ≈ ½ Δθᵀ F Δθ` nats. Every framework
logs something in that family. The question is where those nats went, and it has an exact answer. Minimising `½ΔθᵀFΔθ` subject to `JΔθ = Δz` gives `Δθ* = F⁻¹Jᵀ G⁻¹ Δz`, and substituting back,

    KL_min(Δz) = ½ Δzᵀ G⁻¹ Δz

the least information any parameter step could have spent to move the measured features by `Δz`.
Divide the two and you get a fraction that is bounded by construction:

    Efficiency(t) = KL_min(Δz_observed) / D_t ∈ [0, 1]

That bound is the reason to build this. It is a self-check: the instrument cannot return 1.3 and
call it a finding, because the same variational argument that defines `KL_min` proves it cannot
exceed the cost of the step that achieved `Δz`. `StepCost` asserts it at construction and
`cost_series` refuses before constructing one, which are two different guards on two different
failures and both are wanted. The assertion catches an arithmetic bug in this module. The refusal
catches data that breaks the premise, which is F3's kill condition: `KL_min` above `D_t` on real
data means `G` is mis-estimated or the natural-gradient approximation has failed, and the book does
not balance either way.

**The decomposition, and the honest part.** `KL_min` is a quadratic form, so it is exactly additive
in the eigenbasis of `G`: `KL_min = ½ Σ_j (Δz̃_j)²/g_j` with `Δz̃ = UᵀΔz`. Those directions are
combinations of features and have no names. Attributing to *named* features needs an allocation
rule because named features are correlated, and the one used here is the Shapley value of the game

    v(T) = ½ Δz_Tᵀ (G_TT)⁻¹ Δz_T

which is the minimum cost of achieving the observed movement on the features in `T` while leaving
every other feature free. `v` is monotone, so the shares are non-negative, and the Shapley
efficiency axiom makes them sum to `v(N)` exactly rather than approximately. What is left over,
`KL_min − v(N)`, is the nats the movement of features **outside** the attributed set demands on top
of that, and it is `residual_share`: a named field, not a rounding gap. Reporting per-feature
shares that do not sum to the thing they are shares of is what every ad hoc attribution scheme in
this space does, and it is what the construction-time assertion here rules out.

**Three things this cannot do, here rather than on a caveats page.**

`D_t` has to come from somewhere. It is a property of the *update*, not of the rollouts, so no
featuriser and no amount of record-reading produces it: either the trainer logged it, or a caller
supplies it, or there are two checkpoints to difference. When none of those holds this refuses with
`RECORD_INCOMPLETE` and reports `KL_min` alone, which needs no denominator. Substituting a proxy
for `D_t` and keeping the name `kl_spent` would make every efficiency in the series a confident
wrong number, and the proxies available on a record (the clipped gradient norm times the learning
rate) are wrong by an unknown factor whenever the optimiser is not plain SGD.

`KL_min(Δz_observed)` is **upward biased**, and the bias is not small on a short run. `Δz` is a
difference of two finite-sample means, so it carries estimation noise, and `KL_min` is a positive
definite quadratic form in it: `E[½Δẑᵀ G⁻¹ Δẑ] = KL_min(Δz_true) + ½ tr(G⁻¹ Σ_Δẑ)`. The second term
is reported on every reading as ``noise_floor``, computed from the ledger's own `se_delta_z`. A
`kl_min` below its own noise floor is a measurement of the batch size, not of the step.

And `KL_min` is a statement about the features you measured. A step that spent its nats
reorganising a representation with no behavioural signature in this basis is indistinguishable here
from one that spent them on nothing. That is what a low efficiency means and it is why the sentence
this instrument produces ends with an unattributed share rather than with a total.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

from reward_lens.core.envelope import EnvelopeSpec, RegimeCondition
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.quantity import (
    FREE,
    BiasStatement,
    CostModel,
    EstimatorEntry,
    register_estimator,
)
from reward_lens.core.reading import Reading, Refusal, RefusalReason, refuse_incomplete
from reward_lens.core.types import (
    Access,
    AccessMatrix,
    Capability,
    Component,
    GaugeStatus,
    Phase,
    Substrate,
)
from reward_lens.measure.base import BaseObservable, Context, PreflightResult, run
from reward_lens.measure.efficiency.metric import MetricG
from reward_lens.measure.ledger.price import StepLedger, Window, whole_run
from reward_lens.measure.rate.regime import MEASURED_BY
from reward_lens.record.schema import Run

if TYPE_CHECKING:
    from reward_lens.core.evidence import Evidence

#: How many named features an exact Shapley allocation will enumerate. **Chosen: 12**, which is
#: 4,096 coalitions and about a second of small dense solves. Above it the exact game is refused
#: with a remedy rather than silently swapped for a sampled approximation, because a share that is
#: exact for eleven features and Monte Carlo for thirteen is two different quantities under one
#: name.
MAX_EXACT_FEATURES = 12

#: Relative tolerance on the additivity assertion. Shapley efficiency is exact in real arithmetic;
#: this absorbs the float error of `2^k` pseudo-inverses summed with alternating signs.
SHARE_TOLERANCE = 1e-6


# ---------------------------------------------------------------------------
# The allocation
# ---------------------------------------------------------------------------


def _coalition_values(g: MetricG, dz: np.ndarray, indices: Sequence[int]) -> dict[int, float]:
    """`v(T)` for every subset `T` of ``indices``, keyed by bitmask over the position in ``indices``.

    `v(T) = ½ Δz_Tᵀ (G_TT)⁻¹ Δz_T` is the Lagrangian minimum of `½ΔθᵀFΔθ` under the constraint
    `J_T Δθ = Δz_T`: constrain fewer features and the minimum can only fall, which is the
    monotonicity that makes every Shapley share non-negative.
    """
    values: dict[int, float] = {0: 0.0}
    count = len(indices)
    for mask in range(1, 1 << count):
        members = [indices[i] for i in range(count) if mask & (1 << i)]
        sub = g.submatrix(members)
        value, _ = sub.kl_min(np.asarray([dz[i] for i in members], dtype=np.float64))
        values[mask] = value
    return values


def shapley_shares(
    g: MetricG,
    dz: np.ndarray,
    *,
    attribute_to: Sequence[str] | None = None,
) -> tuple[dict[str, float], float, float]:
    """`(shares, attributed total, KL_min)`. Shares are exact Shapley values over the named set.

    ``attribute_to`` defaults to every feature the parameterisation can move at all, which is every
    name whose diagonal of `G` is positive. A feature with `G_ii = 0` cannot be moved by any step,
    so it can never be charged for nats; it keeps its key in the mapping at 0.0 rather than being
    dropped, because the mapping's keys are the join key and a caller differencing two cost books
    should not have to reconcile two different key sets.

    The third return is `KL_min` over the **whole** basis. `attributed total ≤ KL_min` by
    monotonicity, and the gap is what `StepKlMin.residual_share` carries.
    """
    names = list(g.names)
    diagonal = np.diag(np.asarray(g.matrix, dtype=np.float64))
    movable = [i for i in range(len(names)) if diagonal[i] > 0.0]
    if attribute_to is None:
        chosen = movable
    else:
        wanted = list(attribute_to)
        unknown = [n for n in wanted if n not in names]
        if unknown:
            raise ValueError(
                f"attribute_to names {unknown}, which are not in this metric's basis "
                f"{names}. The basis is the join key and a share against a name `G` does "
                f"not carry cannot be reconciled with a ledger row."
            )
        chosen = [names.index(n) for n in wanted if names.index(n) in movable]

    kl_min, _ = g.kl_min(dz)
    shares = {name: 0.0 for name in names}
    if not chosen:
        return shares, 0.0, kl_min

    count = len(chosen)
    values = _coalition_values(g, dz, chosen)
    factorial = math.factorial
    weights = [
        factorial(size) * factorial(count - size - 1) / factorial(count) for size in range(count)
    ]
    for position, index in enumerate(chosen):
        bit = 1 << position
        total = 0.0
        for mask in range(1 << count):
            if mask & bit:
                continue
            size = bin(mask).count("1")
            total += weights[size] * (values[mask | bit] - values[mask])
        shares[names[index]] = float(total)
    attributed = float(values[(1 << count) - 1])
    return shares, attributed, kl_min


def noise_floor(g: MetricG, se_delta_z: np.ndarray) -> float:
    """`½ tr(G⁻¹ Σ)` with `Σ = diag(se²)`: what `kl_min` would read on a step that did nothing.

    `Δz` is a difference of two finite-sample means and `KL_min` is a positive definite quadratic
    form in it, so a step whose true movement is exactly zero still returns a positive `kl_min`.
    This is that number. It is not subtracted from `kl_min`, because a bias-corrected `KL_min` can
    come out negative and a negative cost is not a quantity anyone can act on; it travels beside it
    so a reader can see whether the step moved anything the batch size can resolve.
    """
    values, vectors = g.eigen()
    if values.size == 0 or values.max() <= 0.0:
        return 0.0
    cutoff = g.rank_tolerance * float(values.max())
    live = values > cutoff
    sigma = np.diag(np.nan_to_num(np.asarray(se_delta_z, dtype=np.float64)) ** 2)
    rotated = vectors[:, live].T @ sigma @ vectors[:, live]
    return float(0.5 * np.sum(np.diag(rotated) / values[live]))


# ---------------------------------------------------------------------------
# The two records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StepKlMin:
    """The half of the cost book that needs no `D_t`: `KL_min`, its shares, and its noise floor.

    This exists because `D_t` is a property of the update and `KL_min` is a property of the
    movement, and on a record that logs no KL the second is readable when the first is not. Rather
    than refusing the whole book, `kl_min_series` returns these and `cost_series` upgrades them to
    `StepCost` once a denominator is supplied. `update.kl_min` and `update.kl_share` are registered
    against this; `update.kl_spent` and `update.efficiency` are registered against `StepCost`.
    """

    step: int
    next_step: int
    kl_min: float
    shares: Mapping[str, float]
    residual_share: float
    noise_floor: float
    out_of_range: float
    n_features: int
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _assert_shares_add_up(self.shares, self.residual_share, self.kl_min, self.step)

    @property
    def above_noise(self) -> bool:
        """Whether the movement is larger than what two finite batches produce on their own."""
        return bool(self.kl_min > self.noise_floor)

    def render(self) -> str:
        ordered = sorted(self.shares.items(), key=lambda kv: -kv[1])
        parts = (
            [f"{name} {value / self.kl_min:.0%}" for name, value in ordered if value > 0.0]
            if self.kl_min > 0.0
            else []
        )
        if self.kl_min > 0.0:
            parts.append(f"unattributed {self.residual_share / self.kl_min:.0%}")
        return (
            f"step {self.step} -> {self.next_step}: KL_min {self.kl_min:.5g} nats "
            f"(noise floor {self.noise_floor:.5g}); " + ", ".join(parts)
        )


@dataclass(frozen=True)
class StepCost:
    """One step of the cost book (F3).

    The field set is fixed in advance so that `measure.reconcile` could write F4's reconciliation
    against it. ``noise_floor`` is additive to it, defaulted, and invisible to a reader that uses
    only the fields above.

    Two assertions run at construction and they guard different things. The shares must sum to
    `kl_min`, because an attribution that does not add up to what it attributes is the confident
    wrong number this library exists to refuse. And `efficiency` must lie in `[0, 1]`, which is
    P5: it is bounded by the same variational argument that defines `KL_min`, so a value outside it
    is an arithmetic error here and never a result. `cost_series` checks the data before it gets
    this far and returns a `Refusal` instead, so reaching this assertion means the bug is in the
    code rather than in the run.
    """

    step: int
    next_step: int
    kl_spent: float
    kl_min: float
    efficiency: float
    shares: Mapping[str, float]
    residual_share: float
    notes: tuple[str, ...] = ()
    noise_floor: float = 0.0

    def __post_init__(self) -> None:
        _assert_shares_add_up(self.shares, self.residual_share, self.kl_min, self.step)
        if not math.isfinite(self.efficiency) or not 0.0 <= self.efficiency <= 1.0:
            raise ValueError(
                f"step {self.step}: efficiency is {self.efficiency!r}, outside [0, 1]. "
                f"`KL_min` is the minimum nats any step could have spent to produce this `Δz` and "
                f"`D_t` is what this step did spend, so the ratio is bounded by construction "
                f"(frozen prediction P5). kl_min {self.kl_min:.6g} against kl_spent "
                f"{self.kl_spent:.6g}. This is an instrument bug, not a finding: `cost_series` "
                f"refuses on data that breaks the premise before constructing a StepCost."
            )

    @classmethod
    def from_kl_min(cls, half: StepKlMin, kl_spent: float, *, note: str = "") -> "StepCost":
        """Attach a denominator to the `D_t`-free half. The only way a `StepCost` is built here."""
        return cls(
            step=half.step,
            next_step=half.next_step,
            kl_spent=float(kl_spent),
            kl_min=half.kl_min,
            efficiency=float(half.kl_min / kl_spent),
            shares=dict(half.shares),
            residual_share=half.residual_share,
            notes=half.notes + ((note,) if note else ()),
            noise_floor=half.noise_floor,
        )

    def render(self) -> str:
        ordered = sorted(self.shares.items(), key=lambda kv: -kv[1])
        parts = [f"{name} {value / self.kl_min:.0%}" for name, value in ordered if value > 0.0]
        parts.append(f"unattributed {self.residual_share / self.kl_min:.0%}")
        return (
            f"step {self.step} -> {self.next_step}: spent {self.kl_spent:.5g} nats, needed "
            f"{self.kl_min:.5g}, efficiency {self.efficiency:.3f}. Of the {self.kl_min:.5g}: "
            + ", ".join(parts)
        )


def _assert_shares_add_up(
    shares: Mapping[str, float], residual: float, kl_min: float, step: int
) -> None:
    """The construction-time rule: the shares plus the residual are `kl_min`, or nothing is built."""
    total = float(sum(shares.values())) + float(residual)
    scale = max(abs(kl_min), 1e-12)
    if not math.isfinite(total) or abs(total - kl_min) > SHARE_TOLERANCE * scale:
        raise ValueError(
            f"step {step}: the per-feature shares sum to {float(sum(shares.values())):.10g} and "
            f"the residual is {residual:.10g}, which together make {total:.10g} against a kl_min "
            f"of {kl_min:.10g}. A per-feature attribution that does not add up to the quantity it "
            f"attributes is the confident wrong number this library exists to refuse; the "
            f"unattributed remainder is `residual_share` and not a rounding gap."
        )
    if residual < -SHARE_TOLERANCE * scale:
        raise ValueError(
            f"step {step}: residual_share is {residual:.10g}, which is negative. `v(T)` is "
            f"monotone in `T`, so the attributed total cannot exceed `KL_min` over the whole "
            f"basis, and a negative remainder means the coalition values were computed against "
            f"different `Δz` vectors."
        )


# ---------------------------------------------------------------------------
# Where `D_t` comes from
# ---------------------------------------------------------------------------


def kl_spent_from_record(run_: Run, window: Window | None = None) -> dict[int, float]:
    """`Step.optimizer.kl_to_previous` per step, for the steps that recorded one.

    `kl_to_previous` and not `kl_to_ref`. `D_t = KL(π_{t+1} ‖ π_t)` is the distance this step moved
    the policy; `kl_to_ref` is the distance from the frozen reference, which is a different
    quantity, is cumulative rather than per-step, and is the one a `beta`-penalised run logs. A
    cost book that divided by `kl_to_ref` would report an efficiency that falls through a run for
    the arithmetic reason that its denominator grows.

    Symmetric with `measure.ledger.price.learning_rates`, and empty is a normal return: the GRPO
    fixtures log neither, which is what `cost_series` refuses on.
    """
    lo, hi = window if window is not None else whole_run(run_)
    out: dict[int, float] = {}
    for step in run_.steps.slice(lo, hi):
        value = step.optimizer.kl_to_previous
        if value is not None:
            out[int(step.index)] = float(value)
    return out


#: What to tell a reader whose record has no `D_t`. It is long because it is the whole content of
#: the refusal: this is the field the cost book cannot recover and cannot fake, and the reader has
#: three real options rather than one.
NO_DENOMINATOR_REMEDY = (
    "Log the per-step KL. In TRL set `beta` above zero, which makes the trainer compute and log "
    "`kl`; in veRL read `actor/kl`; any framework that penalises drift already has this number and "
    "the tap only has to carry it into `OptimizerTelemetry.kl_to_previous`. Failing that, pass "
    "`kl_spent={step: nats}` from wherever you do have it, or hand this function two consecutive "
    "checkpoints so `D_t` can be computed directly as `1/2 dtheta^T F dtheta`. What this will not "
    "do is substitute the clipped gradient norm times the learning rate: with AdamW the applied "
    "step is not the gradient times eta, the moment state that would relate them is not in the "
    "record, and the resulting number would be wrong by an unknown factor while still being called "
    "`kl_spent`. `KL_min` and its per-feature shares need no denominator and are returned by "
    "`kl_min_series` on this same record."
)


# ---------------------------------------------------------------------------
# The series
# ---------------------------------------------------------------------------


def kl_min_series(
    ledgers: Sequence[StepLedger],
    g: MetricG,
    *,
    attribute_to: Sequence[str] | None = None,
    instrument: str = "kl_min_series",
) -> list[StepKlMin] | Refusal:
    """`KL_min` and its named shares per step pair. Needs no `D_t`, so it runs on any record.

    The ledger supplies `Δz` and `se_delta_z`; nothing here recomputes either, because a second
    `Δz` estimated a second way is a second quantity and F4's reconciliation would then be against
    the wrong one.
    """
    if not ledgers:
        return refuse_incomplete(
            instrument,
            field="a step pair to cost",
            subject="an empty ledger series",
            remedy=(
                "Build the ledgers first with `ledger_series(samples, eta_by_step=...)`. The cost "
                "book differences a feature mean between consecutive steps, so a window with "
                "fewer than two recorded steps has nothing to cost."
            ),
        )
    if tuple(ledgers[0].names) != tuple(g.names):
        return Refusal(
            instrument=instrument,
            reason=RefusalReason.UNIT_MISMATCH,
            detail=(
                f"the ledger's basis is {list(ledgers[0].names)} and `G`'s is {list(g.names)}. "
                f"`Δz` and `G` are vectors and a matrix in one basis in one order, so a "
                f"quadratic form taken across two bases pairs one feature's movement with "
                f"another's reachability."
            ),
            remedy=(
                "Build both from the same `TrajectoryFeaturiser` and the same window: "
                "`samples = steps_from_run(run, featuriser, window=w)` feeds both "
                "`ledger_series(samples, ...)` and `metric_g(samples, ...)`."
            ),
            statistics={"ledger_basis": list(ledgers[0].names), "metric_basis": list(g.names)},
        )
    movable = [
        name
        for i, name in enumerate(g.names)
        if float(np.asarray(g.matrix)[i, i]) > 0.0
        and (attribute_to is None or name in set(attribute_to))
    ]
    if len(movable) > MAX_EXACT_FEATURES:
        return Refusal(
            instrument=instrument,
            reason=RefusalReason.BUDGET_EXCEEDED,
            detail=(
                f"an exact Shapley allocation over {len(movable)} named features enumerates "
                f"{2 ** len(movable)} coalitions, above this module's limit of "
                f"{2**MAX_EXACT_FEATURES} at {MAX_EXACT_FEATURES} features."
            ),
            remedy=(
                f"Pass `attribute_to=` with at most {MAX_EXACT_FEATURES} of the names you care "
                f"about; the nats demanded by everything outside that set come back in "
                f"`residual_share`, so nothing is lost, it is only pooled. `KL_min` itself is a "
                f"single quadratic form and is unaffected by the size of the basis."
            ),
            statistics={"n_features": len(movable), "limit": MAX_EXACT_FEATURES},
        )

    out: list[StepKlMin] = []
    for ledger in ledgers:
        dz = np.asarray([row.delta_z for row in ledger.rows], dtype=np.float64)
        se = np.asarray([row.se_delta_z for row in ledger.rows], dtype=np.float64)
        shares, attributed, kl_min = shapley_shares(g, dz, attribute_to=attribute_to)
        residual = kl_min - attributed
        if abs(residual) < SHARE_TOLERANCE * max(kl_min, 1e-12):
            residual = 0.0
        _, out_of_range = g.kl_min(dz)
        notes: list[str] = []
        if out_of_range > 1e-6:
            notes.append(
                f"{out_of_range:.1%} of this step's Δz (by squared norm) lies in the null space "
                f"of G, so no parameter move could have produced it. `KL_min` is computed on the "
                f"reachable part alone and is a lower bound on the cost of the whole movement."
            )
        floor = noise_floor(g, se)
        if kl_min <= floor:
            notes.append(
                f"KL_min {kl_min:.4g} is at or below its own sampling-noise floor {floor:.4g}: "
                f"two batches of this size produce a movement this large with no policy change at "
                f"all, so this step's cost is a measurement of the batch size."
            )
        out.append(
            StepKlMin(
                step=ledger.step,
                next_step=ledger.next_step,
                kl_min=kl_min,
                shares=shares,
                residual_share=float(residual),
                noise_floor=floor,
                out_of_range=out_of_range,
                n_features=len(movable),
                notes=tuple(notes),
            )
        )
    return out


def cost_series(
    ledgers: Sequence[StepLedger],
    g: MetricG,
    *,
    kl_spent: Mapping[int, float] | None = None,
    run_: Run | None = None,
    attribute_to: Sequence[str] | None = None,
    instrument: str = "cost_series",
) -> list[StepCost] | Refusal:
    """The full cost book: `D_t`, `KL_min`, efficiency and the named shares, per step pair.

    ``kl_spent`` maps a step index to the nats that step spent. When it is omitted and ``run_`` is
    given, `kl_to_previous` is read off the record. When neither yields a number this refuses with
    `RECORD_INCOMPLETE` rather than substituting anything, which is the one outcome to rule out by
    name: an efficiency computed against a proxy denominator is wrong by an unknown factor and still
    reads as a measurement.

    Steps whose `KL_min` exceeds their `D_t` do not produce a `StepCost`. That is F3's kill
    condition and it is returned as a refusal carrying the numbers, because a bounded quantity
    coming out unbounded says the premise failed and the useful thing is to say which premise.
    """
    half = kl_min_series(ledgers, g, attribute_to=attribute_to, instrument=instrument)
    if isinstance(half, Refusal):
        return half

    spent: Mapping[int, float]
    source: str
    if kl_spent is not None:
        spent, source = kl_spent, "supplied"
    elif run_ is not None:
        spent, source = kl_spent_from_record(run_), "record `kl_to_previous`"
    else:
        spent, source = {}, "nothing"

    usable = [row for row in half if spent.get(row.step) is not None and spent[row.step] > 0.0]
    if not usable:
        subject = f"{len(half)} step pair(s) of " + (
            f"run {run_.id}" if run_ is not None else "this ledger series"
        )
        return refuse_incomplete(
            instrument,
            field="per-step KL (`D_t`) from " + source,
            subject=subject,
            remedy=NO_DENOMINATOR_REMEDY,
            n_pairs=len(half),
            n_with_kl=len([r for r in half if spent.get(r.step) is not None]),
            source=source,
        )

    over = [(row, float(spent[row.step])) for row in usable if row.kl_min > float(spent[row.step])]
    if over:
        worst = max(over, key=lambda pair: pair[0].kl_min / pair[1])
        return Refusal(
            instrument=instrument,
            reason=RefusalReason.ENVELOPE_VIOLATED,
            detail=(
                f"{len(over)} of {len(usable)} step pairs have `KL_min` above `D_t`, which "
                f"efficiency cannot exceed. The worst is step {worst[0].step}: KL_min "
                f"{worst[0].kl_min:.6g} nats against D_t {worst[1]:.6g}, a ratio of "
                f"{worst[0].kl_min / worst[1]:.3f}. Its sampling-noise floor is "
                f"{worst[0].noise_floor:.6g}."
            ),
            remedy=(
                "The bound is a theorem, so one of its premises failed and the numbers say which. "
                "If `KL_min` is near its noise floor, `Delta z` is dominated by batch noise and "
                "the quadratic form is measuring that: widen the group size or pool step pairs. "
                "If `G` came from `method='covariance_bound'` it is an upper bound on the metric "
                "and `KL_min` a lower bound, so this cannot be the cause and the denominator is "
                "the suspect: check that `kl_spent` is per sequence and per step rather than "
                "cumulative or per token. Otherwise `G` is mis-estimated, which "
                "`damping_stable` on the metric will already have said, or the step is too large "
                "for the first-order expansion and `SelectionExplainedFraction` will say so."
            ),
            statistics={
                "n_over": len(over),
                "n_pairs": len(usable),
                "worst_step": worst[0].step,
                "worst_kl_min": worst[0].kl_min,
                "worst_kl_spent": worst[1],
                "worst_ratio": worst[0].kl_min / worst[1],
                "g_method": g.method,
            },
        )

    note = f"D_t from {source}"
    return [StepCost.from_kl_min(row, float(spent[row.step]), note=note) for row in usable]


# ---------------------------------------------------------------------------
# The instruments
# ---------------------------------------------------------------------------


#: F3's envelope. The catalogue names `LINEAR_RESPONSE` and `ESS_ADEQUATE`.
#:
#: `LINEAR_RESPONSE` because the whole construction is a first-order expansion: `KL_min` is the
#: minimum of a quadratic under a linear constraint, and if the step is large enough that `Δz` is
#: not linear in `Δθ` then neither the constraint nor the bound means what it says. F2's `Λ`
#: measures it.
#:
#: `ESS_ADEQUATE` because `Δz_observed` is an expectation under the step's own sampling
#: distribution, and once the rollouts in hand no longer resolve that expectation the movement being
#: costed is not the policy's.
#:
#: Both measurers come from `measure.rate.regime.MEASURED_BY`, which is the map every other
#: instrument in the library resolves through. The catalogue record names
#: `frontier.visibility_horizon` for `ESS_ADEQUATE` where that map names `run.importance_ess`; the
#: disagreement is recorded here and is not resolved here.
COST_ENVELOPE = EnvelopeSpec(
    requires=frozenset({RegimeCondition.LINEAR_RESPONSE, RegimeCondition.ESS_ADEQUATE}),
    measured_by={
        c: MEASURED_BY[c] for c in (RegimeCondition.LINEAR_RESPONSE, RegimeCondition.ESS_ADEQUATE)
    },
    on_violation="refuse",
)

#: The catalogue's two, as ids. The first is what a reader has today: every framework logs a scalar
#: KL and nothing splits it, so "the total with no decomposition" is the state of the art this
#: instrument is arguing past. The second is the null any attribution has to beat: an equal split
#: across the movable features, which is what a reader assumes when nobody has measured one.
COST_BASELINES = ("baseline.total_kl_undecomposed", "baseline.equal_share_attribution")

#: What to say when the envelope fails, per condition. `BaseObservable.preflight`'s generic remedy
#: is "restrict the window to a span where the condition holds", which is not an instruction a
#: reader can act on for either of these.
ENVELOPE_REMEDIES: Mapping[RegimeCondition, str] = {
    RegimeCondition.LINEAR_RESPONSE: (
        "Lambda is below the threshold, so the movement this step produced is not carried by the "
        "first-order term, and `KL_min` is the minimum of a quadratic under a constraint that "
        "does not hold. Re-run the window at a smaller learning rate, which is the only "
        "intervention that makes a first-order expansion valid, or read `SelectionResidual`, "
        "which reports what moved without claiming the expansion explains it. If Lambda came back "
        "unknown rather than low, compute it with `SelectionExplainedFraction` and pass it as "
        "`RegimeInputs.explained_fraction`."
    ),
    RegimeCondition.ESS_ADEQUATE: (
        "The importance-weighted effective sample size is below the floor, so the rollouts in "
        "hand no longer resolve the expectation `Delta z` estimates and the movement being costed "
        "is not the policy's. Restrict the window to steps whose trajectories are near-policy, "
        "raise the group size, or read `KL_min` as a bound: instrument N2 reports the reward level "
        "beyond which this record resolves nothing, and a cost book above that horizon is "
        "measuring its own weights."
    ),
}


def _remedy_for(refusal: Refusal, envelope: EnvelopeSpec, regime: Any) -> Refusal:
    """Replace the generic envelope remedy with the one for the condition that actually failed."""
    if refusal.reason is not RefusalReason.ENVELOPE_VIOLATED:
        return refusal
    lines = [
        ENVELOPE_REMEDIES[v.condition]
        for v in envelope.violations(regime)
        if v.condition in ENVELOPE_REMEDIES
    ]
    if not lines:
        return refusal
    return Refusal(
        instrument=refusal.instrument,
        reason=refusal.reason,
        detail=refusal.detail,
        remedy=" ".join(lines),
        partial=refusal.partial,
        provenance=refusal.provenance,
        statistics=refusal.statistics,
    )


class _CostInstrument(BaseObservable):
    """Shared plumbing for F3's four quantities: one cost book, four registered readings.

    The four are computed together and reported separately, which is the pattern
    `measure.ledger.price` uses for F1's pair and `measure.composition` for the abstention pair.
    An instrument gets one quantity so that two rungs of one ladder can be compared, and
    `update.kl_spent` (rung 0, a record read) and `update.kl_share` (rung 3, a Shapley allocation)
    are three rungs apart on the same book.

    ``capabilities`` is `NONE` and the network access is declared in ``requires`` instead, which is
    where it belongs. `Capability` says what a `RewardSignal` must offer and there is no signal
    here: the subject is a record plus an already-estimated `MetricG`. The access that `G` needed
    is not lost, it is read off the metric's own rung and put in the access matrix at construction,
    so an instrument handed a rung-2 `G` declares `POLICY: BACKWARD` and one handed a rung-0 `G`
    declares only `RECORD`. That is the honest declaration and it is also the useful one, because
    it lets a reader with a record and no checkpoint discover that the bound is available to them.
    """

    version = "1.0"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    faithful_to: str | None = "KL budget and efficiency"
    deviations: tuple[str, ...] = (
        "`D_t` is `KL(pi_t+1 || pi_t)` and the quantity a framework logs under that name is "
        "sometimes the KL to the frozen reference instead, which is cumulative rather than "
        "per-step. Every reading names where its denominator came from in `kl_spent_source`.",
        "`KL_min(Delta z_observed)` is upward biased, because `Delta z` is a difference of two "
        "finite-sample means and `KL_min` is a positive definite quadratic form in it. The bias "
        "is `1/2 tr(G^-1 Sigma)` and it is reported per step as `noise_floor` rather than "
        "subtracted, because a bias-corrected minimum cost can come out negative.",
        "`G` is a property of one parameter point. A `G` estimated over a window treats the policy "
        "as fixed across it, and a rung-2 `G` estimated from a checkpoint other than the step's "
        "own is that same substitution made explicit. `MetricG.n_samples` and its notes carry "
        "which was done.",
        "the per-feature shares are a Shapley allocation over named features, which is a choice of "
        "allocation rule and not a measurement. The decomposition that needs no rule is the "
        "eigenbasis one, `MetricG.eigen_shares`, whose directions have no names.",
    )

    substrates = frozenset(Substrate)
    #: Not PRE_RUN, which has no record to cost, and not DEPLOYED, where only the artifact survives.
    phases = frozenset({Phase.IN_RUN, Phase.POST_RUN})
    envelope = COST_ENVELOPE
    #: `policy.reparam` because every quantity here is Fisher-metric: under a smooth invertible
    #: change of parameters `J -> J A^-1` and `F -> A^-T F A^-1`, so `G = J F^-1 J^T` is unchanged
    #: and `KL_min`, `efficiency` and every share with it. `units` because the numerator and the
    #: denominator are in nats per sequence and the efficiency is dimensionless, and ranking one
    #: against the other is a unit error.
    invariance = "policy.reparam, units"
    invariance_relation = INVARIANT
    baselines = COST_BASELINES
    rung = 1

    def __init__(
        self,
        ledgers: Sequence[StepLedger],
        g: MetricG,
        *,
        kl_spent: Mapping[int, float] | None = None,
        run_: Run | None = None,
        attribute_to: Sequence[str] | None = None,
    ) -> None:
        self.ledgers = list(ledgers)
        self.g = g
        self.kl_spent = kl_spent
        self.run = run_
        self.attribute_to = attribute_to
        self._computed: Any = None
        self.requires: AccessMatrix = (
            {Component.RECORD: Access.RECORD, Component.POLICY: Access.BACKWARD}
            if g.rung >= 2
            else {Component.RECORD: Access.RECORD}
        )

    def preflight(self, ctx: Context) -> PreflightResult:
        """The base preflight, with the envelope remedy specialised to the condition that failed."""
        pre = super().preflight(ctx)
        if pre.ok or pre.refusal is None or self.envelope is None:
            return pre
        from dataclasses import replace as _replace

        return _replace(pre, refusal=_remedy_for(pre.refusal, self.envelope, ctx.regime_reading))

    def compute(self) -> Any:  # pragma: no cover - overridden
        raise NotImplementedError

    def estimate(self, ctx: Context) -> Reading:
        pre = self.preflight(ctx)
        if not pre.ok and pre.refusal is not None:
            return pre.refusal
        out = self.compute()
        if isinstance(out, Refusal):
            return out
        self._computed = out
        try:
            return run(self, ctx)  # type: ignore[arg-type,no-any-return]
        finally:
            self._computed = None

    def measure(self, ctx: Context) -> "Evidence":
        out = self._computed if self._computed is not None else self.compute()
        if isinstance(out, Refusal):
            raise ValueError(
                f"{self.name}.measure was called on a window that declines to produce Evidence: "
                f"{out.reason.name}. Call `estimate`, which returns the refusal as a value with "
                f"its remedy."
            )
        return ctx.emit(self.payload(out), baselines=self.baseline_values(out))

    def payload(self, rows: Any) -> dict[str, Any]:  # pragma: no cover - overridden
        raise NotImplementedError

    def baseline_values(self, rows: Any) -> dict[str, float]:  # pragma: no cover - overridden
        raise NotImplementedError

    # -- what every payload carries ----------------------------------------

    def _metric_payload(self) -> dict[str, Any]:
        return {
            "features": list(self.g.names),
            "g_method": self.g.method,
            "g_rung": self.g.rung,
            "g_damping": self.g.damping,
            "g_damping_stable": bool(self.g.damping_stable),
            "g_conditioning": self.g.conditioning,
            "g_rank": self.g.rank,
            "g_n_samples": self.g.n_samples,
            "g_notes": list(self.g.notes),
        }


class _KlMinInstrument(_CostInstrument):
    """The two quantities that need no `D_t`, so they read on a record that logs no KL."""

    def compute(self) -> list[StepKlMin] | Refusal:
        return kl_min_series(
            self.ledgers, self.g, attribute_to=self.attribute_to, instrument=self.name
        )

    def _common(self, rows: Sequence[StepKlMin]) -> dict[str, Any]:
        return {
            **self._metric_payload(),
            "n_pairs": len(rows),
            "steps": [[r.step, r.next_step] for r in rows],
            "kl_min": [r.kl_min for r in rows],
            "noise_floor": [r.noise_floor for r in rows],
            "out_of_range": [r.out_of_range for r in rows],
            "n_above_noise": sum(1 for r in rows if r.above_noise),
            "notes": sorted({n for r in rows for n in r.notes}),
        }


class _CostBookInstrument(_CostInstrument):
    """The two quantities that need `D_t`, so they refuse on a record that logs no KL."""

    def compute(self) -> list[StepCost] | Refusal:
        return cost_series(
            self.ledgers,
            self.g,
            kl_spent=self.kl_spent,
            run_=self.run,
            attribute_to=self.attribute_to,
            instrument=self.name,
        )

    def _common(self, rows: Sequence[StepCost]) -> dict[str, Any]:
        return {
            **self._metric_payload(),
            "n_pairs": len(rows),
            "steps": [[r.step, r.next_step] for r in rows],
            "kl_spent": [r.kl_spent for r in rows],
            "kl_min": [r.kl_min for r in rows],
            "efficiency": [r.efficiency for r in rows],
            "noise_floor": [r.noise_floor for r in rows],
            "notes": sorted({n for r in rows for n in r.notes}),
        }


class UpdateKLSpent(_CostBookInstrument):
    """`D_t`: the nats this step moved the policy. Rung 0, and already logged everywhere.

    The cheapest rung on the book and the one that needs no `G` at all. It is here as an
    instrument rather than as a field read because the thing a reader has to know about `D_t` is
    which KL it is: `KL(pi_t+1 || pi_t)` and not the KL to the frozen reference, which is a
    different and cumulative quantity that some frameworks log under the same word.
    """

    name = "UpdateKLSpent"
    quantity = "update.kl_spent"
    rung = 0

    def payload(self, rows: Sequence[StepCost]) -> dict[str, Any]:
        return self._common(rows)

    def baseline_values(self, rows: Sequence[StepCost]) -> dict[str, float]:
        return {
            "baseline.total_kl_undecomposed": float(np.mean([r.kl_spent for r in rows])),
            "baseline.equal_share_attribution": float("nan"),
        }


class UpdateKLMin(_KlMinInstrument):
    """`KL_min = ½ Δzᵀ G⁻¹ Δz`: the least any step could have spent to move what moved.

    Rung 1. Needs `G` and the ledger's `Δz` and nothing else, so it reads on a record that logs no
    KL at all, which is the case F3 mostly meets: `beta = 0` runs do not compute a KL and therefore
    do not log one. Read the `noise_floor` beside it. A `KL_min` at or below its noise floor is a
    measurement of the batch size, and `above_noise` on each row says which.
    """

    name = "UpdateKLMin"
    quantity = "update.kl_min"
    rung = 1

    def payload(self, rows: Sequence[StepKlMin]) -> dict[str, Any]:
        return self._common(rows)

    def baseline_values(self, rows: Sequence[StepKlMin]) -> dict[str, float]:
        return {
            "baseline.total_kl_undecomposed": float("nan"),
            "baseline.equal_share_attribution": float("nan"),
        }


class UpdateEfficiency(_CostBookInstrument):
    """`KL_min / D_t ∈ [0, 1]`: the fraction of the step's information that reached behaviour.

    Rung 1, and the self-checking one. The bound is not a convention, it is the same variational
    argument that defines `KL_min`, so a value above 1 is an instrument bug and P5 freezes that as
    a prediction. The remainder went into features nobody measured, into representational
    reorganisation with no behavioural signature, or into nothing, and this instrument cannot tell
    those three apart: that is what the basis buys and what a wider basis would buy more of.
    """

    name = "UpdateEfficiency"
    quantity = "update.efficiency"
    rung = 1

    def payload(self, rows: Sequence[StepCost]) -> dict[str, Any]:
        return self._common(rows)

    def baseline_values(self, rows: Sequence[StepCost]) -> dict[str, float]:
        return {
            "baseline.total_kl_undecomposed": float(np.mean([r.kl_spent for r in rows])),
            "baseline.equal_share_attribution": float("nan"),
        }


class UpdateKLShare(_KlMinInstrument):
    """Where the nats went, per named feature. Rung 3: the Shapley allocation with its residual.

    The sentence this produces is F3's: "this step spent 0.031 nats, 41% of it went into response
    length, 22% into tool-call frequency, 8% into the target behaviour, and 29% is unattributed to
    any measured feature". The shares sum to `kl_min` by the Shapley efficiency axiom rather than
    by normalisation, and `residual_share` is the nats demanded by movement outside the attributed
    set. Against `baseline.equal_share_attribution`, which is what a reader assumes when nobody has
    measured one.
    """

    name = "UpdateKLShare"
    quantity = "update.kl_share"
    rung = 3

    def payload(self, rows: Sequence[StepKlMin]) -> dict[str, Any]:
        names = list(self.g.names)
        return {
            **self._common(rows),
            "shares": [[r.shares[n] for n in names] for r in rows],
            "residual_share": [r.residual_share for r in rows],
            "share_fraction": [
                [r.shares[n] / r.kl_min if r.kl_min > 0 else float("nan") for n in names]
                for r in rows
            ],
            "n_attributed_features": [r.n_features for r in rows],
        }

    def baseline_values(self, rows: Sequence[StepKlMin]) -> dict[str, float]:
        movable = max((r.n_features for r in rows), default=0)
        return {
            "baseline.total_kl_undecomposed": float("nan"),
            "baseline.equal_share_attribution": 1.0 / movable if movable else float("nan"),
        }


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_COST_BIAS = BiasStatement(
    direction="unknown",
    why=(
        "two components with opposite signs, one of them computable and one not, so the net is "
        "not signable. "
        "`KL_min` is a positive definite quadratic form in `Delta z`, and `Delta z` is a "
        "difference of two finite-sample means, so `E[KL_min(hat Delta z)]` exceeds "
        "`KL_min(Delta z)` by `1/2 tr(G^-1 Sigma)` for any unbiased `hat Delta z`. The direction "
        "is knowable and the size is computable, so it is reported per step as `noise_floor` "
        "rather than left as an unknown. The `G` side pushes the other way and by an amount that "
        "is not knowable from the record: a `covariance_bound` `G` is an upper bound on the "
        "metric, which makes `KL_min` a lower bound, and the two biases do not cancel by anything "
        "other than coincidence."
    ),
)

_RECORD_ONLY: AccessMatrix = {Component.RECORD: Access.RECORD}
_NEEDS_POLICY: AccessMatrix = {Component.RECORD: Access.RECORD, Component.POLICY: Access.BACKWARD}

_SOLVE_COST = CostModel(
    note="one symmetric eigendecomposition of a k-by-k matrix per window; no grader calls, no GPU"
)
_SHAPLEY_COST = CostModel(
    note=(
        "2^k small pseudo-inverses per step pair, exact up to 12 named features (4,096 "
        "coalitions); no grader calls, no GPU"
    )
)


def _register() -> None:
    """The catalogue's four rungs: `D_t` alone, `KL_min` and efficiency, the eigenbasis, Shapley.

    The eigenbasis decomposition (rung 2) and the Shapley allocation (rung 3) are two estimators of
    `update.kl_share` and both are registered, because they are genuinely different answers: the
    eigenbasis one is exact and its directions have no names, and the Shapley one names features
    and pays an allocation rule for it. A reader who wants an unarguable decomposition should be
    able to find the first from the registry rather than from a docstring.
    """
    entries = (
        ("update.kl_spent", "update.kl_spent.record", 0, _RECORD_ONLY, FREE),
        ("update.kl_min", "update.kl_min.quadratic_form", 1, _NEEDS_POLICY, _SOLVE_COST),
        ("update.efficiency", "update.efficiency.ratio", 1, _NEEDS_POLICY, _SOLVE_COST),
        ("update.kl_share", "update.kl_share.eigenbasis", 2, _NEEDS_POLICY, _SOLVE_COST),
        ("update.kl_share", "update.kl_share.shapley", 3, _NEEDS_POLICY, _SHAPLEY_COST),
    )
    for quantity, impl, rung, access, cost in entries:
        register_estimator(
            EstimatorEntry(
                quantity=quantity,
                impl=impl,
                requires=access,
                envelope=COST_ENVELOPE,
                rung=rung,
                bias=_COST_BIAS,
                cost=cost,
                phases=frozenset({Phase.IN_RUN, Phase.POST_RUN}),
                run=None,
            )
        )


_register()


__all__ = [
    "COST_BASELINES",
    "COST_ENVELOPE",
    "MAX_EXACT_FEATURES",
    "NO_DENOMINATOR_REMEDY",
    "SHARE_TOLERANCE",
    "StepCost",
    "StepKlMin",
    "UpdateEfficiency",
    "UpdateKLMin",
    "UpdateKLShare",
    "UpdateKLSpent",
    "cost_series",
    "kl_min_series",
    "kl_spent_from_record",
    "noise_floor",
    "shapley_shares",
]

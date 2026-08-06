"""H2, the two-run rate test: is the transition bifurcation-induced or rate-induced?

Two small arms of the same loop at different anneal rates, sharing a seed, with everything plotted
against the schedule parameter `lambda` rather than against the step index `t`. If the curves
collapse onto each other, the system is crossing a critical point quasi-statically and the whole
critical-slowing-down early-warning toolbox is licensed on it. If they separate, the parameter is
moving faster than the system can relax, the system is falling behind rather than tipping, and
**slowing down never fires because there is nothing slowing down**.

That is why this is the first compute item and the cheapest one. A negative result invalidates a
layer of this library for a few hundred GPU-hours; a positive result licenses it. Nothing else in
the design has that ratio.

**The reparametrisation is the experiment.** Against `t`, two arms at different rates separate
trivially and always: the fast arm reaches any given `lambda` at an earlier step, so its curve is a
horizontal translation of the slow one whatever the physics. Against `lambda` that translation is
divided out and what is left is rate dependence or nothing. `baseline.against_t` computes the same
statistic on the step axis for exactly this reason: it is the number a reader gets by forgetting to
reparametrise, and printing it beside the measurement is what makes the reparametrisation visible
rather than assumed.

**The verdict statistic is the registered one.** The registered prediction resolves on whether "the
two anneal rates' curves do not collapse within their bands", so the primary number here is a band
test on the shared
`lambda` support and not a summary of two fits. `separated_fraction` is the fraction of the shared
support on which the two arms' pointwise bands fail to overlap, and the bands are block bootstraps
that keep each arm's own autocorrelation.

The interpretable secondary number is `shift_in_widths`: the two arms' fitted transition midpoints
in `lambda`, differenced, in units of the pooled fitted transition width. It is in widths because
H4 defines that unit for this library and a second unit for the same kind of displacement would be
the fifth incommensurable convention in a literature that already has four. The sign carries the
physics: a
rate-induced transition is displaced **later in lambda** on the faster arm, because the system is
lagging its driver. A displacement in the other direction is real and is not rate-induced tipping,
and `render` says which one it found rather than reporting a magnitude.

**What this cannot do.** It compares two arms, so it detects rate dependence between the two rates
run and says nothing about rates outside that interval. Two arms a factor of two apart that collapse
leave open that a factor of ten separates them, and `rate_ratio` is on the reading so a reader can
see how much of the range was actually probed. The design in `studies/w6_rate` uses a factor of
four for that reason and the instrument refuses below a factor of two, because two arms at nearly
the same rate cannot fail to collapse and a collapse they cannot fail is not evidence.

Nothing here trains anything. This module takes two recorded arms and compares them; producing the
arms is the compute, and `studies/w6_rate/RUNBOOK.md` is how they get produced.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np

from reward_lens.core.envelope import EnvelopeSpec, RegimeCondition
from reward_lens.core.evidence import register_payload
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.quantity import (
    BiasStatement,
    CostModel,
    EstimatorEntry,
    register_estimator,
)
from reward_lens.core.reading import (
    Reading,
    Refusal,
    RefusalReason,
    refuse_incomplete,
)
from reward_lens.core.types import (
    Access,
    AccessMatrix,
    Capability,
    Component,
    GaugeStatus,
    Phase,
    Substrate,
)
from reward_lens.measure.base import BaseObservable, Context, run
from reward_lens.measure.rate.regime import MEASURED_BY
from reward_lens.measure.rate.transition import TransitionCriteria, fit_transition
from reward_lens.measure.rate.warning import gaussian_smooth

if TYPE_CHECKING:
    from reward_lens.core.evidence import Evidence


@dataclass(frozen=True)
class CollapseCriteria:
    """Every number a verdict here is compared against, in one place, with where it came from.

    The design fixes the method and not the thresholds it is read against, so these are this
    module's defaults. They are gathered here because a default behaves like a decision until
    somebody says otherwise.
    """

    #: Grid points on the shared `lambda` support. **Chosen: 200.** Fine enough that
    #: `separated_fraction` resolves half a percent of the support, coarse enough that the bands
    #: cost nothing.
    n_grid: int = 200

    #: Bootstrap replicates per arm for the pointwise band. **Chosen: 400**, which puts the Monte
    #: Carlo error on a 2.5th percentile at about a fifth of the band half-width it is estimating.
    n_boot: int = 400

    #: Coverage of each arm's pointwise band. **Chosen: 0.95** per arm, which is not the coverage of
    #: the overlap test: two independent 95 percent bands failing to overlap is roughly a 0.4
    #: percent event pointwise under the null, so this test is conservative rather than nominal, and
    #: that is the direction to be conservative in for a test whose negative result kills a layer.
    ci_level: float = 0.95

    #: Block length for the residual bootstrap, as a fraction of the arm's length. **Chosen: 0.1.**
    #: A block bootstrap keeps within-block autocorrelation, and a training series has plenty.
    block_fraction: float = 0.1

    #: Bandwidth of the smoother that defines each arm's curve, as a fraction of its length.
    #: **Chosen: 0.1**, narrower than `warning.py`'s detrending default because here the smooth is
    #: the estimand rather than the nuisance and over-smoothing would flatten the very transition
    #: whose position is being compared.
    smooth_bandwidth: float = 0.1

    #: Fraction of the shared support that has to separate before the verdict is rate-induced.
    #: **Chosen: 0.05.** A single grid point outside a band is a pointwise 2.5 percent event and
    #: 200 of them will produce a few by chance; five percent of the support is 10 contiguous-worth
    #: of points and is not.
    min_separated_fraction: float = 0.05

    #: Smallest ratio between the two arms' rates that this test accepts. **Chosen: 2.0.** Below it
    #: the arms are not at different rates in any useful sense and a collapse is guaranteed by the
    #: design rather than found in the data.
    min_rate_ratio: float = 2.0

    #: Smallest fraction of either arm's `lambda` range that has to be shared with the other.
    #: **Chosen: 0.5.** Curves cannot be compared where only one of them was measured.
    min_overlap_fraction: float = 0.5

    #: Recorded steps needed per arm. **Chosen: 12**, matching `TransitionCriteria.min_points`, so
    #: an arm this test accepts is an arm the secondary logistic fit can at least attempt.
    min_points: int = 12


# ---------------------------------------------------------------------------
# The arms
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RateArm:
    """One arm of the two-run test: what was annealed, how fast, and what the policy did.

    `lambdas` is the schedule parameter at each recorded step and it is the axis everything is
    plotted against. `observable` is the order parameter, which is whatever the transition is a
    transition in: a labelled hack rate, a gold reward, a held-out probe. `steps` is carried only so
    the against-`t` baseline can be computed, and it is never the axis a verdict is taken on.

    `rate` is the arm's mean `|d log lambda / dt|`. It is computed from `lambdas` and `steps` by
    `from_series` rather than declared, because a declared rate and a recorded schedule that
    disagree is the failure this whole test would silently inherit.
    """

    label: str
    lambdas: np.ndarray
    observable: np.ndarray
    steps: np.ndarray
    rate: float
    series: str = "observable"

    @classmethod
    def from_series(
        cls,
        label: str,
        lambdas: Sequence[float] | np.ndarray,
        observable: Sequence[float] | np.ndarray,
        steps: Sequence[float] | np.ndarray | None = None,
        *,
        series: str = "observable",
    ) -> "RateArm":
        lam = np.asarray([float(v) for v in lambdas], dtype=np.float64).ravel()
        obs = np.asarray([float(v) for v in observable], dtype=np.float64).ravel()
        t = (
            np.arange(lam.size, dtype=np.float64)
            if steps is None
            else np.asarray([float(v) for v in steps], dtype=np.float64).ravel()
        )
        if not (lam.size == obs.size == t.size):
            raise ValueError(
                f"arm {label!r} has {lam.size} lambdas, {obs.size} observations and {t.size} "
                f"steps. These are three views of the same recorded steps and have to be the same "
                f"length."
            )
        keep = np.isfinite(lam) & np.isfinite(obs) & np.isfinite(t) & (lam > 0)
        lam, obs, t = lam[keep], obs[keep], t[keep]
        order = np.argsort(t)
        lam, obs, t = lam[order], obs[order], t[order]
        if lam.size >= 2:
            dt = np.diff(t)
            dlog = np.abs(np.diff(np.log(lam)))
            ok = dt > 0
            rate = float(np.mean(dlog[ok] / dt[ok])) if bool(np.any(ok)) else float("nan")
        else:
            rate = float("nan")
        return cls(label=label, lambdas=lam, observable=obs, steps=t, rate=rate, series=series)


# ---------------------------------------------------------------------------
# The bands
# ---------------------------------------------------------------------------


def _curve_on_grid(x: np.ndarray, y: np.ndarray, grid: np.ndarray, bandwidth: float) -> np.ndarray:
    """A Gaussian-kernel regression of `y` on `x`, evaluated on `grid`.

    Kernel regression rather than a logistic, because the band is a statement about the curve that
    was measured and not about the model that fits it. Fitting a logistic to both arms and comparing
    parameters is the secondary reading here and it assumes the shape; this one does not.
    """
    h = max(float(bandwidth), 1e-12)
    d = grid[:, None] - x[None, :]
    w = np.exp(-0.5 * (d / h) ** 2)
    total = w.sum(axis=1)
    out = np.full(grid.size, np.nan)
    live = total > 1e-12
    out[live] = (w[live] @ y) / total[live]
    return out


def _band(
    x: np.ndarray,
    y: np.ndarray,
    grid: np.ndarray,
    criteria: CollapseCriteria,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pointwise curve and band on `grid` by a moving-block residual bootstrap.

    Blocks rather than independent draws because a training series is autocorrelated, and an
    independent-residual bootstrap on autocorrelated residuals produces a band that is too narrow by
    roughly the square root of the integrated autocorrelation. Too narrow is the dangerous direction
    here: it makes the curves separate.
    """
    n = x.size
    h = criteria.smooth_bandwidth * float(np.ptp(x)) if np.ptp(x) > 0 else 1.0
    fitted_on_x = gaussian_smooth(y, criteria.smooth_bandwidth * n)
    resid = y - fitted_on_x
    centre = _curve_on_grid(x, y, grid, h)

    block = max(2, int(round(criteria.block_fraction * n)))
    rng = np.random.default_rng(seed)
    draws = np.empty((criteria.n_boot, grid.size), dtype=np.float64)
    n_blocks = int(math.ceil(n / block))
    for b in range(criteria.n_boot):
        starts = rng.integers(0, max(1, n - block + 1), size=n_blocks)
        pieces = [resid[s : s + block] for s in starts]
        r = np.concatenate(pieces)[:n]
        draws[b] = _curve_on_grid(x, fitted_on_x + r, grid, h)
    lo = float((1.0 - criteria.ci_level) / 2.0)
    return (
        centre,
        np.nanquantile(draws, lo, axis=0),
        np.nanquantile(draws, 1.0 - lo, axis=0),
    )


# ---------------------------------------------------------------------------
# The reading
# ---------------------------------------------------------------------------


@register_payload
@dataclass(frozen=True)
class ArmFit:
    """One arm's transition, fitted against `lambda`, or the reason there is none.

    Both fields can be absent and that is a result rather than a gap: an arm on which no transition
    is identified against `lambda` is an arm whose midpoint cannot be differenced, and the band test
    is what the verdict falls back to.
    """

    label: str
    rate: float
    midpoint: float
    width: float
    identified: bool
    note: str = ""

    def render(self) -> str:
        if not self.identified:
            return f"{self.label}: no transition identified against lambda ({self.note})"
        return (
            f"{self.label}: transition at lambda = {self.midpoint:.5g}, fitted width "
            f"{self.width:.5g} in lambda, at |d log lambda / dt| = {self.rate:.4g}"
        )


@register_payload
@dataclass(frozen=True)
class RateDependence:
    """Whether two arms at different anneal rates collapse onto one curve against `lambda`.

    `separated_fraction` is the registered statistic: the fraction of the shared `lambda` support on
    which the two arms' bands do not overlap. `rate_induced` is the verdict it supports.

    `shift_in_widths` is the interpretable secondary and it is signed. Positive means the faster arm
    transitions **later** in `lambda`, which is the rate-induced signature: the system is being
    driven past the critical point before it has finished responding to it. Negative means the
    faster arm transitions earlier, which is not lag and is a different phenomenon; `render` names
    it rather than reporting the magnitude on its own.

    `against_t_separated_fraction` is the same band test on the step axis, and it is the dumb
    baseline: it separates whenever the rates differ, whatever the physics, so a reading whose
    against-`lambda` fraction is near it has measured the reparametrisation failing rather than the
    system responding.
    """

    separated_fraction: float
    max_gap: float
    rate_induced: bool
    shift_in_widths: float
    shift_in_lambda: float
    fast: ArmFit
    slow: ArmFit
    rate_ratio: float
    n_grid_shared: int
    lambda_span: tuple[float, float]
    against_t_separated_fraction: float
    criteria: CollapseCriteria
    series: str

    @property
    def collapses(self) -> bool:
        """The complement of the verdict, named because it is the licensing half."""
        return not self.rate_induced

    def says(self) -> str:
        if self.rate_induced:
            return (
                f"Two anneal rates a factor of {self.rate_ratio:.3g} apart, plotted against lambda "
                f"rather than t: the curves separate on {self.separated_fraction:.1%} of the shared "
                f"support. The transition is rate-induced, so critical slowing down will not fire."
            )
        return (
            f"Two anneal rates a factor of {self.rate_ratio:.3g} apart, plotted against lambda "
            f"rather than t: the curves collapse within their bands on "
            f"{1.0 - self.separated_fraction:.1%} of the shared support. The transition is "
            f"bifurcation-induced over this range of rates, and slowing-down early warning is "
            f"licensed on it."
        )

    def render(self) -> str:
        direction = (
            "later in lambda, which is the lag a rate-induced transition produces"
            if self.shift_in_widths > 0
            else "earlier in lambda, which is not lag and is a different effect"
        )
        shift = (
            f"The faster arm's transition sits {abs(self.shift_in_widths):.3f} fitted widths "
            f"{direction}."
            if self.fast.identified and self.slow.identified
            else (
                "No midpoint shift is reported: a transition was not identified against lambda on "
                "both arms, so there is nothing to difference."
            )
        )
        return (
            f"{self.says()}\n"
            f"    {self.fast.render()}\n"
            f"    {self.slow.render()}\n"
            f"    {shift}\n"
            f"    Against t instead of lambda the same test separates on "
            f"{self.against_t_separated_fraction:.1%} of the support, which is what the axis alone "
            f"buys and is the baseline this reading has to beat to mean anything.\n"
            f"    Probed between rates {min(self.fast.rate, self.slow.rate):.4g} and "
            f"{max(self.fast.rate, self.slow.rate):.4g}; nothing here speaks to rates outside that "
            f"interval."
        )


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------


def _fit_arm(arm: RateArm, criteria: TransitionCriteria, instrument: str) -> ArmFit:
    """H4's fit, taken against `lambda` instead of against the step index."""
    out = fit_transition(
        arm.observable,
        arm.lambdas,
        series=f"{arm.series}@{arm.label}",
        instrument=instrument,
        criteria=criteria,
    )
    if isinstance(out, Refusal):
        return ArmFit(
            label=arm.label,
            rate=arm.rate,
            midpoint=float("nan"),
            width=float("nan"),
            identified=False,
            note=f"{out.reason.name}: {out.detail}",
        )
    return ArmFit(
        label=arm.label,
        rate=arm.rate,
        midpoint=float(out.midpoint),
        width=float(out.width),
        identified=bool(out.usable),
        note="" if out.usable else out.quality.note,
    )


def _separated_fraction(
    a: RateArm,
    b: RateArm,
    axis: str,
    criteria: CollapseCriteria,
    seed: int,
) -> tuple[float, float, int, tuple[float, float]]:
    """Band overlap on the shared support of one axis. Returns fraction, max gap, points, span."""
    xa = a.lambdas if axis == "lambda" else a.steps
    xb = b.lambdas if axis == "lambda" else b.steps
    lo = max(float(np.min(xa)), float(np.min(xb)))
    hi = min(float(np.max(xa)), float(np.max(xb)))
    if not (hi > lo):
        return float("nan"), float("nan"), 0, (lo, hi)
    grid = np.linspace(lo, hi, criteria.n_grid)
    _, alo, ahi = _band(xa, a.observable, grid, criteria, seed)
    _, blo, bhi = _band(xb, b.observable, grid, criteria, seed + 1)
    # Positive where the two bands do not overlap: the lower of one is above the upper of the other.
    gap = np.maximum(alo - bhi, blo - ahi)
    ok = np.isfinite(gap)
    if not bool(np.any(ok)):
        return float("nan"), float("nan"), 0, (lo, hi)
    return (
        float(np.mean(gap[ok] > 0.0)),
        float(np.max(gap[ok])),
        int(ok.sum()),
        (lo, hi),
    )


def two_run_rate_test(
    fast: RateArm,
    slow: RateArm,
    *,
    criteria: CollapseCriteria | None = None,
    fit_criteria: TransitionCriteria | None = None,
    instrument: str = "RateDependenceTest",
    seed: int = 0,
) -> "RateDependence | Refusal":
    """The two-run rate test. Two arms in, one verdict out, or the reason there is none.

    Four ways this refuses, and each is a different fact about the design rather than about the
    system:

    `RECORD_INCOMPLETE` when either arm is too short for a band, or when either arm's schedule does
    not move. An arm at a constant `lambda` is not an arm of an anneal.

    `ENVELOPE_VIOLATED` when the two rates are closer together than `min_rate_ratio`. Two arms at
    nearly the same rate collapse by construction, and reporting that as a licence for the
    early-warning layer would be the confident wrong answer this test exists to prevent. **This is
    the refusal most likely to fire on a real pair of arms**, because rates are set by wall-clock
    budget and two runs sized to the same budget end up close together.

    `RECORD_INCOMPLETE` again when the two arms' `lambda` ranges overlap over less than
    `min_overlap_fraction` of either. Curves cannot be compared where only one was measured.

    `BELOW_LOD` when the bands are so wide that the test could not have separated: if the against-`t`
    baseline does not separate either, the arms carry too little signal for this comparison at any
    axis, and a collapse verdict from them would be an underpowered null.
    """
    criteria = criteria or CollapseCriteria()
    fit_criteria = fit_criteria or TransitionCriteria()

    for arm in (fast, slow):
        if arm.lambdas.size < criteria.min_points:
            return refuse_incomplete(
                instrument,
                field=f"at least {criteria.min_points} finite recorded steps",
                subject=f"arm {arm.label!r} ({arm.lambdas.size} recorded)",
                remedy=(
                    "log the schedule parameter and the order parameter on every step of both "
                    "arms. A band over fewer than a dozen points is the width of the smoother "
                    "rather than of the data."
                ),
                n=int(arm.lambdas.size),
            )
        if not math.isfinite(arm.rate) or arm.rate <= 0:
            return refuse_incomplete(
                instrument,
                field="a schedule that moves",
                subject=(f"arm {arm.label!r}, whose |d log lambda / dt| is {arm.rate!r}, and so"),
                remedy=(
                    "anneal the parameter on both arms and record its value each step. An arm at a "
                    "constant lambda has no rate, so there is no rate for the other arm to differ "
                    "from."
                ),
                rate=arm.rate,
            )

    ratio = max(fast.rate, slow.rate) / min(fast.rate, slow.rate)
    if ratio < criteria.min_rate_ratio:
        return Refusal(
            instrument=instrument,
            reason=RefusalReason.ENVELOPE_VIOLATED,
            detail=(
                f"the two arms' rates are {fast.rate:.4g} and {slow.rate:.4g}, a ratio of "
                f"{ratio:.3g}, below the {criteria.min_rate_ratio:.3g} this test needs"
            ),
            remedy=(
                f"re-run one arm at a rate at least {criteria.min_rate_ratio:.3g} times the other, "
                f"holding the seed, the total lambda range and everything else fixed. The fast arm "
                f"costs proportionally less wall-clock, so the cheap change is to speed one up "
                f"rather than to slow one down. Arms this close collapse whether or not the "
                f"transition is bifurcation-induced, so a collapse from them licenses nothing."
            ),
            statistics={
                "rate_fast": fast.rate,
                "rate_slow": slow.rate,
                "ratio": ratio,
                "min_ratio": criteria.min_rate_ratio,
            },
        )

    lo = max(float(np.min(fast.lambdas)), float(np.min(slow.lambdas)))
    hi = min(float(np.max(fast.lambdas)), float(np.max(slow.lambdas)))
    spans = [float(np.ptp(fast.lambdas)), float(np.ptp(slow.lambdas))]
    shared = max(hi - lo, 0.0)
    if min(spans) <= 0 or shared / min(spans) < criteria.min_overlap_fraction:
        return refuse_incomplete(
            instrument,
            field=(
                f"a shared lambda range covering at least "
                f"{criteria.min_overlap_fraction:.0%} of the shorter arm"
            ),
            subject=(
                f"arms {fast.label!r} and {slow.label!r}, which share "
                f"{shared:.4g} of lambda against arm spans of {spans[0]:.4g} and {spans[1]:.4g}, "
                f"and so"
            ),
            remedy=(
                "anneal both arms over the same lambda interval and change only how many steps "
                "they take to cross it. Two arms that swept different ranges are two experiments "
                "and their curves have nowhere to be compared."
            ),
            shared=shared,
            spans=spans,
        )

    frac, max_gap, n_shared, span = _separated_fraction(fast, slow, "lambda", criteria, seed)
    frac_t, _, _, _ = _separated_fraction(fast, slow, "t", criteria, seed + 100)
    if not math.isfinite(frac):
        return refuse_incomplete(
            instrument,
            field="a shared support the bands are both defined on",
            subject=f"arms {fast.label!r} and {slow.label!r}, and so",
            remedy=(
                "record both arms over the same lambda interval at a cadence fine enough that the "
                "kernel has points on both sides of every grid node."
            ),
            span=span,
        )
    if math.isfinite(frac_t) and frac_t <= criteria.min_separated_fraction and frac <= frac_t:
        return Refusal(
            instrument=instrument,
            reason=RefusalReason.BELOW_LOD,
            detail=(
                f"the bands do not separate on either axis: {frac:.1%} of the shared support "
                f"against lambda and {frac_t:.1%} against t, where the against-t comparison "
                f"separates whenever the rates differ at all. These arms carry too little signal "
                f"for the comparison rather than agreeing with each other"
            ),
            remedy=(
                "add seeds to each arm and average the order parameter within an arm before "
                "comparing arms, or record a sharper order parameter. The band is the noise of one "
                "arm and it is currently wider than the difference between two arms at a rate "
                "ratio this test already accepted."
            ),
            statistics={
                "separated_fraction_lambda": frac,
                "separated_fraction_t": frac_t,
                "rate_ratio": ratio,
            },
        )

    fast_fit = _fit_arm(fast, fit_criteria, instrument)
    slow_fit = _fit_arm(slow, fit_criteria, instrument)
    if fast_fit.identified and slow_fit.identified:
        pooled = 0.5 * (fast_fit.width + slow_fit.width)
        d_lambda = fast_fit.midpoint - slow_fit.midpoint
        shift = d_lambda / pooled if pooled > 0 else float("nan")
    else:
        d_lambda = float("nan")
        shift = float("nan")

    return RateDependence(
        separated_fraction=frac,
        max_gap=max_gap,
        rate_induced=bool(frac >= criteria.min_separated_fraction),
        shift_in_widths=shift,
        shift_in_lambda=d_lambda,
        fast=fast_fit,
        slow=slow_fit,
        rate_ratio=float(ratio),
        n_grid_shared=n_shared,
        lambda_span=span,
        against_t_separated_fraction=frac_t,
        criteria=criteria,
        series=fast.series,
    )


# ---------------------------------------------------------------------------
# The instrument
# ---------------------------------------------------------------------------

#: H2 cannot require `QUASI_STATIC`: it is the experiment that decides whether the run is
#: quasi-static, so requiring it would be circular. It requires `STATIONARY_GRADER`, because a
#: grader that moved during either arm makes the two arms two different experiments and the curves
#: separate for that reason instead. Refuse rather than downgrade: unlike a relaxation-time fit
#: there is no weaker reading left over, since the whole quantity is a comparison between arms.
COLLAPSE_ENVELOPE = EnvelopeSpec(
    requires=frozenset({RegimeCondition.STATIONARY_GRADER}),
    measured_by=MEASURED_BY,
    on_violation="refuse",
)

#: CONTROL on the optimizer, because the experiment is setting the schedule rather than reading it,
#: and RECORD because both arms have to be logged. `access_min` in the catalogue reads
#: "CONTROL, two small arms, shared seed" and this is that, in the type system.
_COLLAPSE_ACCESS: AccessMatrix = {
    Component.OPTIMIZER: Access.CONTROL,
    Component.RECORD: Access.RECORD,
}

#: The catalogue names one baseline, "a single-rate run", and the registered prediction names
#: "assume bifurcation". They are the same reflex from two directions and both are scored.
COLLAPSE_BASELINES = (
    "baseline.single_rate_run",
    "baseline.against_t",
)


class RateDependenceTest(BaseObservable):
    """H2. Two arms at different anneal rates, compared against `lambda`, with their bands.

    Reads two recorded arms. It does not run them: producing the arms is the compute this package
    is gated on, and the instrument is the part that can be written, tested and priced before any
    of it is bought.

    What it cannot do, three lines in as this library's convention has it. It answers only over the
    interval of rates the two arms span, so a collapse is a licence for that interval and not for
    the schedule someone else runs. It compares one order parameter, so an arm whose transition is
    in a channel nobody logged looks like an arm with no transition. And it cannot separate a
    rate-induced transition from a seed effect on its own: the arms share a seed by design, and if
    they do not, this instrument has no way to know and will report the seed difference as rate
    dependence.
    """

    name = "RateDependenceTest"
    version = "1.0"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    faithful_to: str | None = "H2"
    deviations: tuple[str, ...] = (
        "the criterion is 'if the curves collapse onto each other'. Collapse is made mechanical "
        "here as the fraction of the shared lambda support on which two block-bootstrap bands fail "
        "to overlap, with a threshold of 5 percent of the support. No threshold is fixed for it "
        "anywhere else, so this one is a default.",
        "the midpoint shift is reported in fitted transition widths, which is H4's unit rather than "
        "a unit stated for this instrument. A second unit for the same displacement "
        "is what the lead-time argument objects to.",
    )

    quantity = "run.rate_dependence"
    requires: AccessMatrix = _COLLAPSE_ACCESS
    substrates = frozenset(Substrate)
    #: Not PRE_RUN and not DEPLOYED: this is a comparison of two completed arms of a training loop.
    phases = frozenset({Phase.POST_RUN})
    envelope = COLLAPSE_ENVELOPE
    invariance = "units"
    invariance_relation = INVARIANT
    baselines = COLLAPSE_BASELINES
    rung = 0

    def __init__(
        self,
        fast: RateArm,
        slow: RateArm,
        *,
        criteria: CollapseCriteria | None = None,
        fit_criteria: TransitionCriteria | None = None,
        seed: int = 0,
    ) -> None:
        self.fast = fast
        self.slow = slow
        self.criteria = criteria or CollapseCriteria()
        self.fit_criteria = fit_criteria or TransitionCriteria()
        self.seed = seed
        self._computed: RateDependence | None = None

    def compute(self) -> "RateDependence | Refusal":
        return two_run_rate_test(
            self.fast,
            self.slow,
            criteria=self.criteria,
            fit_criteria=self.fit_criteria,
            instrument=self.name,
            seed=self.seed,
        )

    def estimate(self, ctx: Context) -> Reading:
        """Preflight, compute, refuse or emit. Never a bare number, never a silent zero."""
        pre = self.preflight(ctx)
        if not pre.ok and pre.refusal is not None:
            return pre.refusal
        out = self.compute()
        if isinstance(out, Refusal):
            return out
        self._computed = out
        try:
            return run(self, ctx)
        finally:
            self._computed = None

    def measure(self, ctx: Context) -> "Evidence":
        out = self._computed if self._computed is not None else self.compute()
        if isinstance(out, Refusal):
            raise ValueError(
                f"{self.name}.measure was called on a pair of arms that declines to produce "
                f"Evidence: {out.reason.name}. Call `estimate`, which returns the refusal as a "
                f"value with its remedy."
            )
        return ctx.emit(out, baselines=self.baseline_scores(out))

    def baseline_scores(self, reading: RateDependence) -> dict[str, float]:
        """What the two reflexes say, scored in the reading's own unit.

        `baseline.single_rate_run` is what one arm produces, which is exactly nothing: a single run
        has no second curve to collapse onto, so its separated fraction is zero by construction and
        it would report "bifurcation-induced" on every run ever trained. Scoring it as zero is not a
        formality, it is the statement that the standard practice cannot fail this test.

        `baseline.against_t` is the same band comparison on the step axis. It separates whenever the
        rates differ, so the distance between it and the measurement is what the reparametrisation
        bought, and a measurement that matches it has bought nothing.
        """
        return {
            "baseline.single_rate_run": 0.0,
            "baseline.against_t": float(reading.against_t_separated_fraction),
        }


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def _register() -> None:
    """One rung, which is all `spec/QUANTITIES.yaml` gives `run.rate_dependence`.

    The entry carries `run=None` like every other estimator in this package: nothing in this library
    calls an estimator through the registry yet, and a registry entry that claims a callable nobody
    invokes is worse than one that says what it costs. What the entry does carry is the cost, which
    is the point of registering a compute-gated estimator at all: `reward-lens capabilities` can
    then tell a reader that this quantity exists, needs CONTROL on the optimizer, and costs two arms.
    """
    register_estimator(
        EstimatorEntry(
            quantity="run.rate_dependence",
            impl="run.rate_dependence.two_arm_lambda_collapse",
            requires=_COLLAPSE_ACCESS,
            envelope=COLLAPSE_ENVELOPE,
            rung=0,
            bias=BiasStatement(
                direction="downward",
                why=(
                    "the verdict is a band overlap, and every source of extra within-arm noise "
                    "widens the bands and makes the arms more likely to be called collapsed. "
                    "Fewer seeds, a coarser log cadence and a noisier order parameter all push the "
                    "same way, toward licensing the early-warning layer. The band is a block "
                    "bootstrap rather than an independent-residual one for the same reason: the "
                    "independent version is narrower and would push the other way."
                ),
            ),
            cost=CostModel(
                note=(
                    "two training arms over the same lambda interval at rates a factor of four "
                    "apart, sharing a seed. The arms are the whole cost and the analysis is free; "
                    "studies/w6_rate prices the arms."
                )
            ),
            phases=frozenset({Phase.POST_RUN}),
            run=None,
        )
    )


_register()


__all__ = [
    "COLLAPSE_BASELINES",
    "COLLAPSE_ENVELOPE",
    "ArmFit",
    "CollapseCriteria",
    "RateArm",
    "RateDependence",
    "RateDependenceTest",
    "two_run_rate_test",
]

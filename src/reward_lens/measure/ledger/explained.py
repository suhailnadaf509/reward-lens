"""F2: `Λ`, the selection-explained fraction, and `η_eff`, the step size that fits.

`Λ` is the `R²` of `Δz` regressed on `η·Cov_group(A, f)` across steps, and the OLS slope is reported
as `η_eff`. It is the cheapest strong diagnostic in the design: it needs a record and a featuriser,
it costs one pass over rollouts already written to disk, and **it has no higher rung**. That is
unusual and it is the point. `Λ` is not a quantity somebody would spend more access to estimate
better; it is a **validity certificate for every other Level 1 claim**, because every one of them is
a first-order expansion and `Λ` is the measurement of whether the first order carried the movement.
`RegimeCondition.LINEAR_RESPONSE` names `selection.explained_fraction` as the quantity that measures
it, which is this one, so F1 cannot pass its own envelope until F2 has run.

The disambiguation from Frank's parameter-space Price partition is at the top of
`measure.ledger.price` and applies here without change: this is a measurement instrument on
behavioural traits in a live run, not a derivation of an update rule.

**The fit, precisely, because two conventions here are easy to get wrong.**

The regression is **through the origin**. The model is `Δz = η·Cov + ρ` and it has no intercept: to
first order, zero selection pressure predicts zero movement. An intercept would absorb any drift
common to every feature, and a common drift is a real part of `ρ` (entropy collapse shortens
everything; the KL term pulls everything back) that belongs in the residual rather than in the fit.
So `R²` here is the **uncentred** `R² = 1 − Σ(y − η̂x)² / Σy²`, which for a through-origin fit equals
`(Σxy)² / (Σx²·Σy²)`. That is a squared uncentred correlation, so `Λ ∈ [0, 1]` by Cauchy-Schwarz
rather than by hope, which matters: a "fraction explained" that can come out negative is a bug
waiting to be reported as a finding. Quoting a centred `R²` for this model would be a different
number and is the commoner mistake.

Each feature enters in units of its own rollout-level spread: both `Δz_i` and `Cov_i` are divided by
`sd(f_i)` over the window. Dividing both sides of one feature by the same positive constant leaves
that feature's own fit untouched and makes the **pooled** fit a statement about behaviour rather
than about the units a converter happened to record in. Without it, one feature recorded in
characters rather than in words would carry the whole regression. `x̃ = Cov(A, f)/sd(f)` is then
`corr_group(A, f)·sd(A)`, dimensionless, and `ỹ = Δz/sd(f)` is movement in units of one rollout's
spread.

**What a low `Λ` means, and the four things it does not distinguish.** It is a fraction of variance
across steps, so it is a window statistic and cannot license a single step inside a window where it
is low. And a low value has causes other than a large step: a feature basis missing the axis
selection actually acted on lowers it exactly as a second-order term does; prompt resampling between
consecutive steps puts task noise into `Δz`; and a grader that abstains often shrinks the group the
covariance is taken over. `LINEAR_RESPONSE` is therefore **necessary and not sufficient**, which is
the sentence `measure.rate.regime` already carries and which is repeated here because this is where
the number is produced. The kill condition in the catalogue is that there is none: a low `Λ` is
itself the finding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

from reward_lens.core.envelope import EnvelopeSpec, RegimeCondition
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.quantity import (
    BiasStatement,
    CostModel,
    EstimatorEntry,
    register_estimator,
)
from reward_lens.core.reading import Reading, Refusal, RefusalReason, refuse_incomplete
from reward_lens.core.types import (
    AccessMatrix,
    Capability,
    GaugeStatus,
    Phase,
    Substrate,
)
from reward_lens.measure.base import BaseObservable, Context, PreflightResult, run
from reward_lens.measure.ledger.features import TrajectoryFeaturiser
from reward_lens.measure.ledger.price import (
    LEDGER_ACCESS,
    StepLedger,
    StepSample,
    Window,
    _remedy_for,
    ledger_series,
    steps_from_run,
    whole_run,
)
from reward_lens.measure.rate.regime import MEASURED_BY
from reward_lens.record.schema import Run

if TYPE_CHECKING:
    from reward_lens.core.evidence import Evidence

#: F2's envelope. The catalogue prints `envelope_requires: OPEN` for this record, so the two
#: conditions here are a proposal rather than a transcription and are open to revision.
#:
#: `LINEAR_RESPONSE` is deliberately **not** among them: `MEASURED_BY` names
#: `selection.explained_fraction` as the quantity that measures that condition, and this instrument
#: is what produces it, so requiring it would be circular in exactly the way `REGIME_ENVELOPE`
#: avoids for H5. The other two are not circular and they matter: a window of degenerate groups has
#: a covariance of zero on every feature, which makes `Λ` an argument about noise divided by noise,
#: and a window of partial rollouts has no single generating policy per trajectory, which makes
#: `Δz` a difference between two mixtures rather than between two policies.
EXPLAINED_ENVELOPE = EnvelopeSpec(
    requires=frozenset({RegimeCondition.GROUP_NONDEGENERATE, RegimeCondition.NEAR_POLICY}),
    measured_by={
        c: MEASURED_BY[c]
        for c in (RegimeCondition.GROUP_NONDEGENERATE, RegimeCondition.NEAR_POLICY)
    },
    on_violation="refuse",
)


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LambdaFit:
    """`Λ`, `η_eff`, and everything needed to decide whether to believe either.

    ``by_feature`` holds each feature's own `Λ`, which is the diagnostic that separates "the
    first-order picture works here" from "one feature carries the whole fit". A pooled `Λ` of 0.6
    made of one feature at 0.95 and four at 0.05 is a different state of the world from five
    features at 0.6, and only the per-feature split can tell them apart.
    """

    lambda_: float
    eta_eff: float
    se_eta_eff: float
    ci_low: float
    ci_high: float
    ci_level: float
    n_points: int
    n_steps: int
    n_features: int
    by_feature: Mapping[str, float]
    eta_eff_by_feature: Mapping[str, float]
    scales: Mapping[str, float]
    dropped: tuple[str, ...] = ()
    method: str = "through-origin OLS, uncentred R^2, features scaled by sd(f)"

    @property
    def is_certificate(self) -> bool:
        """Whether this reading licenses a first-order claim at the regime module's own threshold.

        `RegimeThresholds.explained_fraction_min` is 0.5, chosen there on the argument that below a
        half the term you expanded explains less of what moved than everything you dropped. It is
        that module's default and is marked as needing ratification, which is why this property
        reads it from there rather than restating the number.
        """
        from reward_lens.measure.rate.regime import RegimeThresholds

        return bool(self.lambda_ >= RegimeThresholds().explained_fraction_min)

    def render(self) -> str:
        worst = min(self.by_feature.items(), key=lambda kv: kv[1], default=("", float("nan")))
        best = max(self.by_feature.items(), key=lambda kv: kv[1], default=("", float("nan")))
        return (
            f"Lambda = {self.lambda_:.4f} [{self.ci_low:.4f}, {self.ci_high:.4f}] at "
            f"{self.ci_level:.0%} over {self.n_steps} step pairs and {self.n_features} features; "
            f"eta_eff = {self.eta_eff:.6g} +/- {self.se_eta_eff:.3g}\n"
            f"    best feature {best[0]} at {best[1]:.4f}, worst {worst[0]} at {worst[1]:.4f}"
        )


def feature_scales(samples: Sequence[StepSample]) -> dict[str, float]:
    """The pooled standard deviation of each feature over every rollout in the window.

    Pooled over the whole window rather than per step, because a per-step scale would divide each
    step by a number that itself moves during the run, and a regression whose units change with `t`
    is not a regression. A feature with zero spread over the whole window is constant and is dropped
    by the fit, named, rather than dividing by zero.
    """
    if not samples:
        return {}
    names = samples[0].names
    stacked = np.vstack([s.features for s in samples if s.n]) if any(s.n for s in samples) else None
    if stacked is None or stacked.shape[0] < 2:
        return {n: 0.0 for n in names}
    sd = stacked.std(axis=0, ddof=1)
    return {n: float(sd[i]) for i, n in enumerate(names)}


def fit_lambda(
    ledgers: Sequence[StepLedger],
    scales: Mapping[str, float],
    *,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> LambdaFit | None:
    """The pooled through-origin fit of `Δz` on `Cov(A, f)`. None when nothing is left to fit.

    The interval on `Λ` comes from resampling **step pairs**, not points: the `k` features of one
    step share one batch of rollouts and one optimiser update, so they are one observation and
    resampling them independently would report an interval narrower than the data supports by
    roughly the square root of the feature count. The standard error on `η_eff` is clustered at the
    same level, by the sandwich form, for the same reason.
    """
    usable = [name for name in (ledgers[0].names if ledgers else ()) if scales.get(name, 0.0) > 0.0]
    dropped = tuple(n for n in (ledgers[0].names if ledgers else ()) if n not in usable)
    if not ledgers or not usable:
        return None

    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    for ledger in ledgers:
        xs, ys = [], []
        for name in usable:
            row = ledger.row(name)
            s = scales[name]
            xs.append(row.covariance / s)
            ys.append(row.delta_z / s)
        x_rows.append(np.asarray(xs, dtype=np.float64))
        y_rows.append(np.asarray(ys, dtype=np.float64))
    x = np.vstack(x_rows)
    y = np.vstack(y_rows)
    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        return None
    x = np.where(finite, x, 0.0)
    y = np.where(finite, y, 0.0)

    lam, eta = _through_origin(x.ravel(), y.ravel())
    se = _clustered_slope_se(x, y, eta)
    lo, hi = _bootstrap_lambda(x, y, n_bootstrap=n_bootstrap, ci=ci, seed=seed)

    by_feature: dict[str, float] = {}
    eta_by_feature: dict[str, float] = {}
    for j, name in enumerate(usable):
        lam_j, eta_j = _through_origin(x[:, j], y[:, j])
        by_feature[name] = lam_j
        eta_by_feature[name] = eta_j

    return LambdaFit(
        lambda_=lam,
        eta_eff=eta,
        se_eta_eff=se,
        ci_low=lo,
        ci_high=hi,
        ci_level=ci,
        n_points=int(np.count_nonzero(finite)),
        n_steps=len(ledgers),
        n_features=len(usable),
        by_feature=by_feature,
        eta_eff_by_feature=eta_by_feature,
        scales={n: float(scales[n]) for n in usable},
        dropped=dropped,
    )


def _through_origin(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """``(uncentred R^2, slope)`` for `y = b·x`. Both NaN when `x` or `y` is identically zero."""
    sxx = float(np.dot(x, x))
    syy = float(np.dot(y, y))
    sxy = float(np.dot(x, y))
    if sxx <= 0.0 or syy <= 0.0:
        return float("nan"), float("nan")
    return float(sxy * sxy / (sxx * syy)), float(sxy / sxx)


def _clustered_slope_se(x: np.ndarray, y: np.ndarray, slope: float) -> float:
    """Sandwich standard error on the through-origin slope, clustered by step pair (the rows)."""
    if not np.isfinite(slope) or x.shape[0] < 2:
        return float("nan")
    residual = y - slope * x
    scores = np.sum(x * residual, axis=1)
    bread = float(np.sum(x * x))
    if bread <= 0.0:
        return float("nan")
    meat = float(np.sum(scores**2))
    n = x.shape[0]
    correction = n / (n - 1) if n > 1 else 1.0
    return float(np.sqrt(correction * meat) / bread)


def _bootstrap_lambda(
    x: np.ndarray, y: np.ndarray, *, n_bootstrap: int, ci: float, seed: int
) -> tuple[float, float]:
    """Percentile interval on `Λ`, resampling whole step pairs.

    Declined rather than reported below five step pairs. A bootstrap over `K` clusters has
    `C(2K-1, K)` distinct resamples and resolving a tail of mass `(1-ci)/2` needs at least
    `2/(1-ci)` of them, which at 95% is 40 and puts the floor at `K = 5`. That is the same rule
    `stats.baselines.base` derives for its paired margin interval, and it is derived rather than
    picked for the same reason.
    """
    n = x.shape[0]
    if n_bootstrap <= 0 or n < 5:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(n_bootstrap, n))
    values = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = draws[i]
        values[i] = _through_origin(x[idx].ravel(), y[idx].ravel())[0]
    finite = values[np.isfinite(values)]
    if finite.size < 10:
        return float("nan"), float("nan")
    alpha = (1.0 - ci) / 2.0
    return float(np.quantile(finite, alpha)), float(np.quantile(finite, 1.0 - alpha))


def lambda_by_step(
    ledgers: Sequence[StepLedger],
    scales: Mapping[str, float],
    *,
    context: int = 5,
) -> list[tuple[int, float]]:
    """`Λ` per step, fitted over a sliding window of `2·context + 1` step pairs centred on each.

    The quantity is `Λ` **per step**, and `Λ` is a fraction of variance across steps,
    so a single pair cannot produce one: with one point the through-origin fit passes through it and
    the uncentred `R²` is 1 by construction. That is the vacuous answer, so a window is used and its
    width is reported. ``context=5`` gives eleven pairs per point, which is above the five-pair floor
    the bootstrap needs and short enough to resolve a transition tens of steps wide.

    Returned as `(step, Λ)` pairs indexed by the earlier step of the centre pair, so the series lines
    up with anything else indexed by optimiser step.
    """
    out: list[tuple[int, float]] = []
    width = 2 * context + 1
    if len(ledgers) < width:
        return out
    for centre in range(context, len(ledgers) - context):
        window = ledgers[centre - context : centre + context + 1]
        fit = fit_lambda(window, scales, n_bootstrap=0)
        out.append((window[context].step, float("nan") if fit is None else fit.lambda_))
    return out


# ---------------------------------------------------------------------------
# The instruments
# ---------------------------------------------------------------------------


class _ExplainedInstrument(BaseObservable):
    """Shared plumbing for F2's two quantities: one fit, two registered readings."""

    version = "1.0"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    faithful_to: str | None = "Price equation, first-order residual test"
    deviations: tuple[str, ...] = (
        "the regression is specified without its conventions. It is fitted through the "
        "origin, so the reported R-squared is the uncentred one, and each feature enters divided "
        "by its own pooled standard deviation over the window so that the pooled fit is not a "
        "statement about the units the converter recorded in.",
        "`Lambda` is a window statistic. `lambda_by_step` reports it per step over a sliding "
        "window and names the width; a single step pair has one point and an uncentred R-squared "
        "of exactly 1, which is vacuous rather than perfect.",
    )

    requires: AccessMatrix = LEDGER_ACCESS
    substrates = frozenset(Substrate)
    phases = frozenset({Phase.IN_RUN, Phase.POST_RUN})
    envelope = EXPLAINED_ENVELOPE
    #: `units` is the one invariance group whose assertion is a refusal rather than a numeric
    #: relation, so `check_invariance` routes it to `check_unit_refusal` and the generated test is
    #: about a comparison rather than about a value. Both quantities here are dimensionless, and
    #: the substantive checks that are not vacuous are property tests: `Lambda` and `eta_eff` are
    #: unchanged by a per-feature rescale of the features and by an affine rescale of the reward.
    invariance = "units"
    invariance_relation = INVARIANT
    baselines = ("baseline.permuted_step",)
    rung = 0

    def __init__(
        self,
        run_: Run,
        featuriser: TrajectoryFeaturiser,
        *,
        window: Window | None = None,
        n_bootstrap: int = 1000,
        seed: int = 0,
        basis: str = "all_rollouts",
    ) -> None:
        self.run = run_
        self.featuriser = featuriser
        self.window = window
        self.n_bootstrap = n_bootstrap
        self.seed = seed
        self.basis = basis
        self._computed: LambdaFit | None = None

    # -- preflight, with the remedy that fits the condition -----------------

    def preflight(self, ctx: Context) -> PreflightResult:
        """The base preflight, with the envelope remedy specialised to the condition that failed."""
        pre = super().preflight(ctx)
        if pre.ok or pre.refusal is None or self.envelope is None:
            return pre
        from dataclasses import replace as _replace

        return _replace(pre, refusal=_remedy_for(pre.refusal, self.envelope, ctx.regime_reading))

    # -- the computation ----------------------------------------------------

    def samples(self) -> list[StepSample]:
        lo, hi = self._window()
        return steps_from_run(self.run, self.featuriser, window=(lo, hi))

    def _window(self) -> Window:
        return self.window if self.window is not None else whole_run(self.run)

    def compute(self) -> LambdaFit | Refusal:
        """The fit, or the refusal that says what the window could not supply.

        `eta = 1.0` here on purpose: the fit reports the slope, so the step size is the answer and
        not an input. A `StepLedger` built at `eta = 1` carries a `selection` column equal to the raw
        covariance and a `residual` equal to `Δz − Cov`, neither of which F2 reads; it reads
        `delta_z` and `covariance`, which do not depend on the step size at all.
        """
        lo, hi = self._window()
        indices = sorted(self.run.steps.indices)
        inside = [i for i in indices if lo <= i < hi]
        if not inside:
            have = f"steps {min(indices)} to {max(indices)}" if indices else "no steps at all"
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.VOID,
                detail=(
                    f"the window [{lo}, {hi}) contains no recorded steps of run {self.run.id}, "
                    f"which holds {have}."
                ),
                remedy=(
                    "Ask for a window inside the recorded range. A run with no steps is void "
                    "rather than a run whose selection explained nothing."
                ),
                statistics={"window": [lo, hi], "recorded": len(indices)},
            )
        samples = self.samples()
        ledgers = ledger_series(samples, eta=1.0, basis=self.basis)
        if len(ledgers) < 2:
            return refuse_incomplete(
                self.name,
                field="a second step pair",
                subject=(
                    f"window [{lo}, {hi}) of run {self.run.id}, which yields "
                    f"{len(ledgers)} step pair(s)"
                ),
                remedy=(
                    "Widen the window. Lambda is a fraction of variance across steps: one pair "
                    "gives a through-origin fit that passes through its single point, so the "
                    "uncentred R-squared is 1 whatever the data says. Two pairs is the arithmetic "
                    "minimum and five is the floor below which the interval is declined."
                ),
                n_pairs=len(ledgers),
                window=[lo, hi],
            )
        scales = feature_scales(samples)
        fit = fit_lambda(ledgers, scales, n_bootstrap=self.n_bootstrap, seed=self.seed)
        if fit is None:
            return refuse_incomplete(
                self.name,
                field="a feature with non-zero spread across rollouts",
                subject=f"the {len(samples)} steps of run {self.run.id} in [{lo}, {hi})",
                remedy=(
                    "Supply a featuriser whose features vary between rollouts. A feature that is "
                    "constant over the window has no covariance with anything and no scale to be "
                    "expressed in, so it cannot enter the fit; the reading names which were "
                    "dropped when some of them vary."
                ),
                n_features=len(samples[0].names) if samples else 0,
            )
        return fit

    # -- the two methods ---------------------------------------------------

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
        return ctx.emit(self.payload(out))

    def payload(self, fit: LambdaFit) -> dict[str, Any]:  # pragma: no cover - base
        raise NotImplementedError


def _fit_payload(fit: LambdaFit) -> dict[str, Any]:
    return {
        "n_steps": fit.n_steps,
        "n_features": fit.n_features,
        "n_points": fit.n_points,
        "by_feature": dict(fit.by_feature),
        "scales": dict(fit.scales),
        "dropped_features": list(fit.dropped),
        "method": fit.method,
    }


class SelectionExplainedFraction(_ExplainedInstrument):
    """F2: `Λ`, the share of what moved that this step's selection pressure explains.

    Says: "`Λ` = 0.62. Sixty-two percent of the variance in what moved is explained by what this
    step selected for." It is the validity certificate F1 and every other Level 1 instrument is
    checked against, and `measure.rate.regime` reads it as the measurement of `LINEAR_RESPONSE`.

    There is no higher rung and there is no kill condition. A low `Λ` is not a failure of the
    instrument; it is the finding that the first-order picture is not carrying this run, and the
    honest response to it is to look at `SelectionResidual` and at F3's cost book rather than at a
    better estimator of `Λ`.
    """

    name = "SelectionExplainedFraction"
    quantity = "selection.explained_fraction"

    def payload(self, fit: LambdaFit) -> dict[str, Any]:
        return {
            **_fit_payload(fit),
            "lambda": fit.lambda_,
            "ci_low": fit.ci_low,
            "ci_high": fit.ci_high,
            "ci_level": fit.ci_level,
            "licenses_first_order": fit.is_certificate,
        }


class EffectiveStepSize(_ExplainedInstrument):
    """F2's slope: `η_eff`, the step size that best explains the movement it produced.

    This is the number a run cannot read off its own config. The optimiser logs a Euclidean learning
    rate and the identity is derived for a Fisher-preconditioned step, so the constant relating
    `Cov_group(A, f)` to `Δz` is a property of the run's curvature rather than of its configuration.
    Fitting it is the only way to get it from a record, and the ratio `η_eff / learning_rate` is the
    honest measure of how far the two are apart on this run.

    A negative `η_eff` is not a bug and is not clipped. It says the features moved against the
    selection pressure over this window, which happens when the residual is large and correlated
    with the selection term, and it is worth more as a reported number than as a floor at zero.
    """

    name = "EffectiveStepSize"
    quantity = "selection.eta_eff"

    def payload(self, fit: LambdaFit) -> dict[str, Any]:
        return {
            **_fit_payload(fit),
            "eta_eff": fit.eta_eff,
            "se_eta_eff": fit.se_eta_eff,
            "eta_eff_by_feature": dict(fit.eta_eff_by_feature),
            "lambda": fit.lambda_,
        }


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def _register() -> None:
    """One rung each. `selection.explained_fraction` has no higher rung by design, not by omission.

    A higher rung would have to estimate the same quantity better with more access, and there is
    nothing more to get: the two sides of the identity are both already exact functions of the
    record. What more access buys is a *different* quantity (`G`, the reachable covariance, at
    `POLICY: BACKWARD`), and that is C2 rather than a second rung of this.
    """
    register_estimator(
        EstimatorEntry(
            quantity="selection.explained_fraction",
            impl="selection.explained_fraction.record_series",
            requires=LEDGER_ACCESS,
            envelope=EXPLAINED_ENVELOPE,
            rung=0,
            bias=BiasStatement(
                direction="downward",
                why=(
                    "every source of noise in `Delta z` that the selection term cannot predict "
                    "enters the denominator of the uncentred R-squared and none of it enters the "
                    "numerator, so finite batches, prompt resampling between steps and grader "
                    "abstention all push Lambda toward zero. A measured Lambda is therefore a "
                    "lower bound on the first-order share, and the way to raise it honestly is "
                    "more rollouts per step rather than a different estimator."
                ),
            ),
            cost=CostModel(
                note="one pass over the window's rollouts plus the featuriser; no grader calls, "
                "no GPU"
            ),
            phases=frozenset({Phase.IN_RUN, Phase.POST_RUN}),
            run=None,
        )
    )
    register_estimator(
        EstimatorEntry(
            quantity="selection.eta_eff",
            impl="selection.eta_eff.record_series",
            requires=LEDGER_ACCESS,
            envelope=EXPLAINED_ENVELOPE,
            rung=0,
            bias=BiasStatement(
                direction="downward",
                why=(
                    "the regressor is a measured covariance and carries sampling error, so the "
                    "through-origin slope is attenuated toward zero by the classical "
                    "errors-in-variables factor. The attenuation is the ratio of true to observed "
                    "covariance variance across steps, which C1's within-prompt rollout variance "
                    "estimates, and correcting it is what reconciles the two."
                ),
            ),
            cost=CostModel(
                note="one pass over the window's rollouts plus the featuriser; no grader calls, "
                "no GPU"
            ),
            phases=frozenset({Phase.IN_RUN, Phase.POST_RUN}),
            run=None,
        )
    )


_register()


__all__ = [
    "EXPLAINED_ENVELOPE",
    "EffectiveStepSize",
    "LambdaFit",
    "SelectionExplainedFraction",
    "feature_scales",
    "fit_lambda",
    "lambda_by_step",
]

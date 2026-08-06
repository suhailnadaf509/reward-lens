"""The three registered instruments: F4's two quantities and F6's one.

`ReconciliationResidual` reports `ρ` per feature with its itemised budget. `BudgetClosure` reports
the verdict. `LandeSlope` reports the slope of observed response on predicted response. All three
read a record, a featuriser and a `MetricG`, and none of them computes `G`: capacity is
`measure.efficiency`'s book and an instrument that estimated its own would be a second one.

**The envelope, and why the two families differ.** F4 declares F1's three conditions with
`on_violation="downgrade"` rather than `"refuse"`. Outside `LINEAR_RESPONSE` the residual is still
computable and still means something (it is then most of the movement), so refusing would make the
one instrument that can say "the first order explains nothing here" unable to say it. The quantity
stays defined, trust caps at EXPLORATORY, and the violated condition lands on the reading.

F6 declares F2's two conditions and deliberately not `LINEAR_RESPONSE`. `Λ` is the `R²` of `Δz` on
`η·S` and the Lande slope is the slope of `Δz` on `η·G·β`; requiring the first before measuring the
second is close to requiring the answer before asking the question, which is the circularity
`measure.ledger.explained` avoids for the same reason and in the same words.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from reward_lens.core.budget import UncertaintyBudget
from reward_lens.core.envelope import EnvelopeSpec, RegimeCondition
from reward_lens.core.evidence import Uncertainty
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.quantity import (
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
from reward_lens.measure.ledger.explained import feature_scales
from reward_lens.measure.ledger.features import TrajectoryFeaturiser
from reward_lens.measure.ledger.price import (
    LEDGER_ENVELOPE,
    StepSample,
    Window,
    _remedy_for,
    learning_rates,
    ledger_series,
    steps_from_run,
    whole_run,
)
from reward_lens.measure.rate.regime import MEASURED_BY
from reward_lens.measure.reconcile.books import (
    BasisMismatch,
    MetricGLike,
    StepReconciliation,
    reconcile_series,
)
from reward_lens.measure.reconcile.closure import ClosureResult, closure_of
from reward_lens.measure.reconcile.facts import facts_from_run
from reward_lens.measure.reconcile.lande import CircularEstimator, LandeFit, fit_lande
from reward_lens.measure.reconcile.residual import FeatureBudget, itemise
from reward_lens.record.schema import Run

if TYPE_CHECKING:
    from reward_lens.core.evidence import Evidence

#: F4 and F6 read the record for `Δz`, `S` and `η`, and the policy for `G`. The `POLICY` entry is
#: what makes "all four books" of the catalogue's `access_min` a declaration rather than a phrase:
#: three of the four are free from a record and capacity is not.
RECONCILE_ACCESS: AccessMatrix = {
    Component.RECORD: Access.RECORD,
    Component.POLICY: Access.BACKWARD,
}

#: F4's envelope: F1's three conditions, downgraded rather than refused. The conditions are taken
#: from `LEDGER_ENVELOPE` rather than restated, so there is one place they are declared.
RECONCILE_ENVELOPE = EnvelopeSpec(
    requires=LEDGER_ENVELOPE.requires,
    measured_by={c: MEASURED_BY[c] for c in LEDGER_ENVELOPE.requires},
    on_violation="downgrade",
)

#: F6's envelope. `LINEAR_RESPONSE` is deliberately absent, for the reason in the module docstring.
LANDE_ENVELOPE = EnvelopeSpec(
    requires=frozenset({RegimeCondition.GROUP_NONDEGENERATE, RegimeCondition.NEAR_POLICY}),
    measured_by={
        c: MEASURED_BY[c]
        for c in (RegimeCondition.GROUP_NONDEGENERATE, RegimeCondition.NEAR_POLICY)
    },
    on_violation="downgrade",
)


@dataclass(frozen=True)
class Reconciliation:
    """Everything one window of one run produces: the books, the budgets, and the verdict."""

    run_id: str
    steps: tuple[StepReconciliation, ...]
    budgets: tuple[FeatureBudget, ...]
    closure: ClosureResult
    samples: tuple[StepSample, ...]

    def render(self) -> str:
        return "\n".join([self.closure.render(), *(b.render() for b in self.budgets)])


class _ReconcileBase(BaseObservable):
    """Shared plumbing: one reconciliation, read by three instruments at two quantities."""

    version = "1.0"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    requires: AccessMatrix = RECONCILE_ACCESS
    substrates = frozenset(Substrate)
    phases = frozenset({Phase.IN_RUN, Phase.POST_RUN})
    envelope = RECONCILE_ENVELOPE
    #: `units` is the one registered group whose assertion is a refusal rather than a numeric
    #: relation, so `check_invariance` routes it to `check_unit_refusal`. The substantive property
    #: this package asserts instead is that the Lande slope and the closure ratio are unchanged by a
    #: per-feature rescale of the features, which is not vacuous: `Δz`, `S`, `C` and `G` all rescale
    #: together and only a fit that divides by `sd(f)` is invariant under it.
    invariance = "units"
    invariance_relation = INVARIANT
    rung = 1

    def __init__(
        self,
        run_: Run,
        featuriser: TrajectoryFeaturiser,
        metric_g: MetricGLike,
        *,
        window: Window | None = None,
        eta: float | str = "schedule",
        ridge: float = 0.0,
        c_context: int | None = None,
        n_bootstrap: int = 1000,
        seed: int = 0,
    ) -> None:
        self.run = run_
        self.featuriser = featuriser
        self.metric_g = metric_g
        self.window = window
        self.eta = eta
        self.ridge = ridge
        self.c_context = c_context
        self.n_bootstrap = n_bootstrap
        self.seed = seed
        self._computed: Any = None

    # -- preflight, with F1's remedies rather than the generic one ----------

    def preflight(self, ctx: Context) -> PreflightResult:
        pre = super().preflight(ctx)
        if pre.ok or pre.refusal is None or self.envelope is None:
            return pre
        return replace(pre, refusal=_remedy_for(pre.refusal, self.envelope, ctx.regime_reading))

    # -- the computation ----------------------------------------------------

    def _window(self) -> Window:
        return self.window if self.window is not None else whole_run(self.run)

    def reconcile(self) -> Reconciliation | Refusal:
        """The books, the budgets and the verdict, or the refusal that says what was missing."""
        lo, hi = self._window()
        indices = sorted(self.run.steps.indices)
        inside = [i for i in indices if lo <= i < hi]
        if len(inside) < 2:
            have = f"steps {min(indices)} to {max(indices)}" if indices else "no steps at all"
            return refuse_incomplete(
                self.name,
                field="two recorded steps inside the window",
                subject=f"window [{lo}, {hi}) of run {self.run.id}, which holds {have}",
                remedy=(
                    "Widen the window. The reconciliation differences a feature mean between "
                    "consecutive steps, so a window holding fewer than two steps has nothing to "
                    "difference."
                ),
                window=[lo, hi],
                steps_in_window=len(inside),
            )
        samples = steps_from_run(self.run, self.featuriser, window=(lo, hi))
        if isinstance(self.eta, str) and self.eta == "schedule":
            rates = learning_rates(self.run, (lo, hi))
            if not rates:
                return refuse_incomplete(
                    self.name,
                    field="schedule['learning_rate']",
                    subject=f"every step of run {self.run.id} in [{lo}, {hi})",
                    remedy=(
                        "Pass `eta=` explicitly. Unlike F1, the reconciliation cannot fall back to "
                        "the raw covariance: `Δz_pred = eta * G * beta` has the step size inside "
                        "it, so there is no step-size-free half to report."
                    ),
                )
            ledgers = ledger_series(samples, eta_by_step=rates, basis="all_rollouts")
        else:
            ledgers = ledger_series(samples, eta=float(self.eta))  # type: ignore[arg-type]
        if not ledgers:
            return refuse_incomplete(
                self.name,
                field="a step pair with a step size",
                subject=f"run {self.run.id} over [{lo}, {hi})",
                remedy=(
                    "No consecutive pair in this window has both steps recorded and a learning "
                    "rate on the earlier one. Pass `eta=` explicitly."
                ),
            )
        try:
            steps = reconcile_series(
                samples, ledgers, self.metric_g, ridge=self.ridge, c_context=self.c_context
            )
        except BasisMismatch as exc:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.UNIT_MISMATCH,
                detail=str(exc),
                remedy=(
                    "Build `G` and the ledger from one `TrajectoryFeaturiser`. `metric_g` takes "
                    "the same `StepSample` list this instrument does, so passing "
                    "`steps_from_run(run, featuriser)` to both is enough to make the bases match."
                ),
                statistics={
                    "g_names": list(self.metric_g.names),
                    "ledger_names": list(samples[0].names) if samples else [],
                },
            )
        if not steps:
            return refuse_incomplete(
                self.name,
                field="a feature with non-zero spread across rollouts",
                subject=f"the {len(samples)} steps of run {self.run.id} in [{lo}, {hi})",
                remedy=(
                    "Supply a featuriser whose features vary between rollouts. A feature that is "
                    "constant over the window has a zero row in C and cannot enter the solve."
                ),
            )
        budgets = itemise(steps, samples, facts_from_run(self.run, (lo, hi)))
        verdict = closure_of(
            budgets,
            steps,
            run_id=self.run.id,
            n_bootstrap=self.n_bootstrap,
            seed=self.seed,
        )
        return Reconciliation(
            run_id=self.run.id,
            steps=tuple(steps),
            budgets=tuple(budgets),
            closure=verdict,
            samples=tuple(samples),
        )

    def compute(self) -> Any:  # pragma: no cover - overridden
        raise NotImplementedError

    # -- the two methods of the instrument protocol -------------------------

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
        return self.emit(ctx, out)

    def emit(self, ctx: Context, out: Any) -> "Evidence":  # pragma: no cover - overridden
        raise NotImplementedError


def _common(reconciliation: Reconciliation) -> dict[str, Any]:
    steps = reconciliation.steps
    head = steps[0]
    return {
        "run": reconciliation.run_id,
        "n_pairs": len(steps),
        "features": list(head.names),
        "steps": [[s.step, s.next_step] for s in steps],
        "eta": [s.eta for s in steps],
        "eta_source": head.eta_source,
        "operator": head.operator,
        "ridge": head.ridge,
        "c_pooled_over_steps": head.c_n_steps,
        "c_conditioning": head.c_conditioning,
        "g_rung": head.g_rung,
        "g_method": head.g_method,
        "g_damping": head.g_damping,
        "g_damping_stable": head.g_damping_stable,
        "task_overlap": [s.task_overlap for s in steps],
        "notes": sorted({n for s in steps for n in s.notes}),
    }


class ReconciliationResidual(_ReconcileBase):
    """F4's residual: `ρ = Δz_obs − η·G·C⁻¹·S`, per feature, with its itemised budget.

    Says: "`Var(ρ)` = 0.0041. The itemised budget accounts for 0.0038." The budget is the product
    rather than the residual: a residual reported as unexplained is the baseline this instrument
    has to beat, and it is named in `baselines` for that reason.

    What it cannot do: it cannot compute `G`, it cannot separate a residual from sampling noise at
    a single step, and every term whose input the record does not carry is named rather than
    estimated, which makes the combined uncertainty a lower bound whenever anything is missing.
    """

    name = "ReconciliationResidual"
    quantity = "books.reconciliation_residual"
    faithful_to: str | None = "the reconciliation identity"
    deviations: tuple[str, ...] = (
        "`C` is pooled across the window by default rather than estimated per step. Eight rollouts "
        "in two groups leave six within-group degrees of freedom at one step, which is too few to "
        "invert. `S` stays per step, so the gradient still moves with the step's own pressure.",
        "terms of the budget table whose inputs the record does not carry are named in "
        "`missing` rather than estimated, so the combined uncertainty is a lower bound and an "
        "excess of `Var(rho)` over it cannot be read as an unmodelled term.",
        "`u_basis` converts `1 - R^2` into feature units by assuming the unexplained advantage "
        "drives response in proportion to the explained part. The identity-based evaluation needs "
        "`J` and `F` and is a higher rung than this one.",
    )
    baselines = ("baseline.unbudgeted_residual", "baseline.permuted_step")

    def compute(self) -> Reconciliation | Refusal:
        return self.reconcile()

    def emit(self, ctx: Context, out: Reconciliation) -> "Evidence":
        worst = max(out.budgets, key=lambda b: b.var_residual, default=None)
        budget = worst.budget if worst is not None else UncertaintyBudget(terms=())
        return ctx.emit(
            {
                **_common(out),
                "residual": [[r.residual for r in s.rows] for s in out.steps],
                "delta_z_obs": [[r.delta_z_obs for r in s.rows] for s in out.steps],
                "delta_z_pred": [[r.delta_z_pred for r in s.rows] for s in out.steps],
                "heritability": [[r.heritability for r in s.rows] for s in out.steps],
                "ledger_residual": [[r.ledger_residual for r in s.rows] for s in out.steps],
                "var_residual": {b.feature: b.var_residual for b in out.budgets},
                "mean_residual": {b.feature: b.mean_residual for b in out.budgets},
                "accounted": {b.feature: b.accounted for b in out.budgets},
                "budget": {
                    b.feature: {t.name: t.contribution for t in b.budget.terms} for b in out.budgets
                },
                "missing_terms": {b.feature: [m.name for m in b.missing] for b in out.budgets},
            },
            uncertainty=Uncertainty.from_budget(budget, n=len(out.steps)),
        )


class BudgetClosure(_ReconcileBase):
    """F4's verdict: is `Var(ρ)` accounted for by `Σ u_i²`?

    Says: "the books close within 8%." Four verdicts rather than two, because an open budget with
    terms missing is a measurement gap and an open budget with every term computed is a discovery,
    and calling both "open" is how the second gets claimed for the first.

    What it cannot do: it cannot tell a budget that closes from one too coarse to fail.
    `FeatureClosure.detectable_u` is the smallest extra term the interval would have separated, and
    a verdict of `closed` with that floor far above the first-order prediction means the test had
    no power at the scale it was arbitrating. The reading carries both numbers.
    """

    name = "BudgetClosure"
    quantity = "books.budget_closure"
    rung = 2
    faithful_to: str | None = "the closure test"
    deviations: tuple[str, ...] = (
        "the interval on the ratio is a cluster bootstrap over step pairs and resamples the "
        "numerator only. `sum u^2` is composed from window-level statistics computed on the same "
        "steps, so resampling it as well would count the same sampling variation twice.",
        "`Var(rho)` is the variance across steps and does not see a systematic offset in the "
        "residual. The mean and its standard error travel on the reading for that reason: a "
        "budget can close on the scatter while the mean sits many standard errors from zero.",
    )
    baselines = ("baseline.unbudgeted_residual",)

    def compute(self) -> Reconciliation | Refusal:
        return self.reconcile()

    def emit(self, ctx: Context, out: Reconciliation) -> "Evidence":
        return ctx.emit(
            {
                **_common(out),
                "verdict": out.closure.verdict,
                "closed": out.closure.closed,
                "detail": out.closure.detail,
                "by_feature": {
                    f.feature: {
                        "verdict": f.verdict,
                        "ratio": f.ratio,
                        "ci_low": f.ci_low,
                        "ci_high": f.ci_high,
                        "var_residual": f.var_residual,
                        "accounted": f.accounted,
                        "n_missing": f.n_missing,
                        "dominant": f.dominant,
                        "detectable_u": f.detectable_u,
                        "predicted_rms": f.predicted_rms,
                        "feature_sd": f.feature_sd,
                        "powered_at_prediction": f.powered_at_prediction,
                        "effective_dof": f.effective_dof,
                        "coverage_k": f.coverage_k,
                    }
                    for f in out.closure.features
                },
            }
        )


class LandeSlope(_ReconcileBase):
    """F6: the slope of observed `Δz` on `η G β` across a window, with an interval.

    Says: "regressing observed `Δz` on `ηGβ` across 200 steps gives slope 0.83 [0.71, 0.95]. Lande
    holds to within 17%." A slope near zero retires Level 1, which is why this is the load-bearing
    assumption of the whole story rather than one more diagnostic.

    What it cannot do: at rung 0 with `G = C` the regressor collapses to `η·S` and the fit is F2's
    `η_eff` rescaled, so it is not evidence about Lande's equation. `LandeFit.is_degenerate` says
    so on every fit that has collapsed, and the independent test needs a rung-2 Fisher `G`.
    """

    name = "LandeSlope"
    quantity = "selection.lande_slope"
    envelope = LANDE_ENVELOPE
    rung = 1
    faithful_to: str | None = "Lande's equation"
    deviations: tuple[str, ...] = (
        "the fit is through the origin and each feature enters divided by its own pooled spread "
        "over the window, which are `measure.ledger.explained`'s conventions and are stated "
        "because the registered entry gives the regression and not its conventions.",
        "with `G` at the covariance bound the regressor reduces to `eta*S` exactly and the slope "
        "is F2's `eta_eff` rescaled. The fit reports `is_degenerate` rather than a caveat.",
        "a `G` fitted from `Delta z` makes the regression circular and is refused rather than "
        "reported with a warning.",
    )
    baselines = ("baseline.permuted_step", "baseline.random_feature")

    def compute(self) -> LandeFit | Refusal:
        out = self.reconcile()
        if isinstance(out, Refusal):
            return out
        scales = feature_scales(list(out.samples))
        try:
            fit = fit_lande(list(out.steps), scales, n_bootstrap=self.n_bootstrap, seed=self.seed)
        except CircularEstimator as exc:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ENVELOPE_VIOLATED,
                detail=str(exc),
                remedy=(
                    "Estimate `G` by `metric_g(..., method='fisher_kernel')` at POLICY: BACKWARD, "
                    "which does not read `Delta z`. The covariance bound also does not read it, "
                    "but it makes the regressor `eta*S` and the fit degenerate; the reading says "
                    "so rather than hiding it."
                ),
                statistics={"g_method": out.steps[0].g_method if out.steps else ""},
            )
        if fit is None:
            return refuse_incomplete(
                self.name,
                field="a feature with non-zero spread across rollouts",
                subject=f"run {self.run.id} over {self._window()}",
                remedy=(
                    "Supply a featuriser whose features vary between rollouts. A constant feature "
                    "has no scale to be expressed in and cannot enter the pooled fit."
                ),
            )
        self._reconciliation = out
        return fit

    def emit(self, ctx: Context, out: LandeFit) -> "Evidence":
        reconciliation = getattr(self, "_reconciliation", None)
        common = _common(reconciliation) if reconciliation is not None else {}
        return ctx.emit(
            {
                **common,
                "slope": out.slope,
                "se_slope": out.se_slope,
                "ci_low": out.ci_low,
                "ci_high": out.ci_high,
                "ci_level": out.ci_level,
                "r_squared": out.r_squared,
                "by_feature": dict(out.by_feature),
                "scales": dict(out.scales),
                "is_degenerate": out.is_degenerate,
                "excludes_zero": out.excludes_zero,
                "consistent_with_lande": out.consistent_with_lande,
                "method": out.method,
            },
            uncertainty=Uncertainty(
                ci_low=out.ci_low,
                ci_high=out.ci_high,
                ci_level=out.ci_level,
                n=out.n_steps,
                method="bootstrap-percentile, clustered on step pairs",
            ),
        )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_RECONCILE_COST = CostModel(
    note=(
        "one pass over the window's rollouts plus the featuriser, plus whatever `G` cost. The "
        "reconciliation itself is a k-by-k solve per step and is free; capacity is the expensive "
        "book and it is `measure.efficiency`'s to cost."
    )
)

_RESIDUAL_BIAS = BiasStatement(
    direction="unknown",
    why=(
        "`Delta z` is unbiased for the change in the feature mean and `S` is unbiased for the "
        "within-group covariance, so the residual inherits the bias of `G` and of the ridge on "
        "`C`. A ridge shrinks `beta` toward zero, which shrinks the predicted response and "
        "inflates the residual; `G` at the covariance bound is an upper bound on the reachable "
        "covariance, which does the opposite. Which dominates is a property of the run's "
        "curvature and of the conditioning of `C`, and neither is signable from the record."
    ),
)

_CLOSURE_BIAS = BiasStatement(
    direction="downward",
    why=(
        "every budget term whose input the record does not carry is omitted rather than "
        "estimated, so `sum u^2` is a lower bound on the uncertainty the apparatus really has and "
        "the ratio `Var(rho) / sum u^2` is biased upward. The verdict names the omitted terms and "
        "reports `incomplete` rather than `unmodelled` for exactly this reason."
    ),
)

_LANDE_BIAS = BiasStatement(
    direction="downward",
    why=(
        "the regressor is a measured covariance solved against a measured covariance, so the "
        "through-origin slope is attenuated toward zero by the classical errors-in-variables "
        "factor, and the attenuation is worse here than for F2's `eta_eff` because `C^-1` "
        "amplifies the error in `S` along the directions where `C` is smallest. The correction "
        "is `beta_corr = (C_obs - C_err)^-1 S` with `C_err` the within-prompt "
        "rollout variance."
    ),
)


def _register() -> None:
    """Two rungs for the residual, one for the verdict, one for the slope.

    The catalogue gives `books.reconciliation_residual` three rungs: the residual alone, the GUM
    itemisation, and the closure test. The third is registered against `books.budget_closure`
    rather than twice, because the closure test is where that rung's estimate is produced and
    registering an entry against a quantity this package does not compute at that rung would be a
    ladder rung with nothing on it.
    """
    for rung, impl, note in (
        (0, "books.reconciliation_residual.record_and_metric", "the residual alone"),
        (1, "books.reconciliation_residual.gum_itemised", "the GUM itemisation of the budget"),
    ):
        register_estimator(
            EstimatorEntry(
                quantity="books.reconciliation_residual",
                impl=impl,
                requires=RECONCILE_ACCESS,
                envelope=RECONCILE_ENVELOPE,
                rung=rung,
                bias=_RESIDUAL_BIAS,
                cost=_RECONCILE_COST,
                phases=frozenset({Phase.IN_RUN, Phase.POST_RUN}),
                run=None,
            )
        )
    register_estimator(
        EstimatorEntry(
            quantity="books.budget_closure",
            impl="books.budget_closure.bootstrap_ratio",
            requires=RECONCILE_ACCESS,
            envelope=RECONCILE_ENVELOPE,
            rung=2,
            bias=_CLOSURE_BIAS,
            cost=_RECONCILE_COST,
            phases=frozenset({Phase.IN_RUN, Phase.POST_RUN}),
            run=None,
        )
    )
    register_estimator(
        EstimatorEntry(
            quantity="selection.lande_slope",
            impl="selection.lande_slope.through_origin_window",
            requires=RECONCILE_ACCESS,
            envelope=LANDE_ENVELOPE,
            rung=1,
            bias=_LANDE_BIAS,
            cost=_RECONCILE_COST,
            phases=frozenset({Phase.IN_RUN, Phase.POST_RUN}),
            run=None,
        )
    )


_register()


__all__ = [
    "LANDE_ENVELOPE",
    "RECONCILE_ACCESS",
    "RECONCILE_ENVELOPE",
    "BudgetClosure",
    "LandeSlope",
    "Reconciliation",
    "ReconciliationResidual",
]

"""F4: the residual as a budget with named terms, and the test of whether it closes.

There are nine contributions to `ρ = Δz_obs − η·G·C⁻¹·S` and one question to ask of them:
**is `Var(ρ)` accounted for by `Σ u_i²`?** If yes the ledger is closed and the instrument is
characterised, which is a stronger statement than any individual measurement in the design. If no
there is an unmodelled term and finding it is a result. Either outcome is publishable, which is
what makes this a good experiment rather than a hopeful one.

Following the GUM, every contribution is identified, sized, and combined, with Type A evaluated
from statistics and Type B from judgement, and both treated identically once they are standard
uncertainties. The composition, the Welch-Satterthwaite effective degrees of freedom and the
coverage factor are `core.budget`'s and are not reimplemented here.

**What this cannot do, stated here rather than on a caveats page.** A term whose input the record
does not carry is not zero, and it is not estimated: it goes into ``missing`` by name, and the
combined uncertainty is then a **lower bound**, so an excess of `Var(ρ)` over it cannot be
attributed to an unmodelled term. The verdict distinguishes those two cases and refuses to call the
second one a discovery. On the two GRPO records this library ships, four of the nine terms are
missing, and the reason each is missing is a fact about what the tap wrote rather than about the
run.

`Var(ρ)` is the variance **across steps** for one feature, so the budget composes per-step standard
uncertainties into a predicted scatter. The mean of `ρ` is carried separately: a systematic offset
is a bias rather than an uncertainty, and a budget that closes on the scatter while the mean sits
ten standard errors from zero has not closed anything a reader should be told about in one word.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from reward_lens.core.budget import BudgetTerm, UncertaintyBudget
from reward_lens.measure.ledger.price import StepSample
from reward_lens.measure.reconcile.books import StepReconciliation
from reward_lens.measure.reconcile.facts import Absent, RunFacts

#: The nine terms of the budget, in the order the table gives them. Fixed as a constant so that
#: a budget missing a term is missing a *named* term rather than one nobody noticed was absent.
TERM_ORDER: tuple[str, ...] = (
    "u_stale",
    "u_KL",
    "u_entropy",
    "u_momentum",
    "u_batch",
    "u_curv",
    "u_clip",
    "u_MC",
    "u_basis",
)

#: Degrees of freedom assigned to a Type B term believed to about 10% in its own uncertainty.
#: GUM G.4.2 gives `ν_i ≈ ½(Δu_i/u_i)⁻²`, so a 10% relative uncertainty on the uncertainty is 50.
#: Stated rather than defaulted to infinity, because `core.budget.effective_dof` refuses to
#: substitute infinity and it is right to: substituting it silently narrows the interval.
TYPE_B_DOF: float = 50.0


@dataclass(frozen=True)
class MissingTerm:
    """A contribution that was not computed, with what it would take to compute it.

    ``needed_to_close`` is the standard uncertainty this term alone would have to carry for the
    budget to account for the observed variance. It is the honest form of "we do not know": rather
    than reporting the gap as a mystery, it reports the size of the thing that would fill it, which
    is a number a reader can compare against what they know about their own optimiser.
    """

    name: str
    why: str
    remedy: str
    needed_to_close: float = float("nan")

    def render(self) -> str:
        tail = (
            f"  (would need u = {self.needed_to_close:.4g} to close)"
            if np.isfinite(self.needed_to_close)
            else ""
        )
        return f"{self.name}: {self.why}{tail}"


@dataclass(frozen=True)
class FeatureBudget:
    """One feature's residual, its itemised budget, and the terms that are not in it."""

    feature: str
    n_steps: int
    mean_residual: float
    var_residual: float
    se_mean_residual: float
    budget: UncertaintyBudget
    missing: tuple[MissingTerm, ...]

    @property
    def combined(self) -> float:
        """`u_c`, the quadrature sum. A **lower bound** whenever ``missing`` is non-empty."""
        return self.budget.combined

    @property
    def accounted(self) -> float:
        """`Σ u_i²`, the variance the budget predicts, against `var_residual` observed."""
        return float(self.budget.combined**2)

    @property
    def ratio(self) -> float:
        """`Var(ρ) / Σ u_i²`. One is a closed budget; above one is an unexplained excess."""
        return float(self.var_residual / self.accounted) if self.accounted > 0 else float("inf")

    @property
    def is_complete(self) -> bool:
        """Whether all nine terms were computed. Only then can an excess be a discovery."""
        return not self.missing

    def render(self) -> str:
        lines = [
            f"{self.feature}: rho mean {self.mean_residual:+.5g} +/- {self.se_mean_residual:.4g}, "
            f"Var(rho) {self.var_residual:.5g} over {self.n_steps} steps",
            *(f"    {line}" for line in self.budget.render().splitlines()),
            f"    accounted {self.accounted:.5g}, ratio Var(rho) / sum u^2 = {self.ratio:.4g}",
        ]
        for term in self.missing:
            lines.append(f"    not computed: {term.render()}")
        return "\n".join(lines)


def _rms(values: Sequence[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.sqrt(np.mean(arr**2))) if arr.size else float("nan")


def advantage_r_squared(samples: Sequence[StepSample], columns: Sequence[int]) -> tuple[float, int]:
    """`R²` of the within-group regression of the advantage on the features, and its sample size.

    `u_basis`, the `ηJF⁻¹e` term, is bounded by `1 − R²` of the regression of `A` on `f`. Taken
    within group and through the origin after centring, because that is the decomposition performed:
    `A = Σ_i β_i (f_i − E f_i) + ε` under the policy's own sampling distribution, and `ε`
    is what the feature span does not reach. A pooled regression would attribute prompt-to-prompt
    heterogeneity to the features and report a higher `R²` than the basis earns.
    """
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for sample in samples:
        if sample.n == 0:
            continue
        f = sample.features[:, list(columns)]
        a = sample.advantages
        ok = np.isfinite(a) & np.all(np.isfinite(f), axis=1)
        for label in np.unique(sample.group_ids[ok]) if ok.any() else ():
            mask = (sample.group_ids == label) & ok
            if int(np.count_nonzero(mask)) < 2:
                continue
            xs.append(f[mask] - f[mask].mean(axis=0))
            ys.append(a[mask] - a[mask].mean())
    if not xs:
        return float("nan"), 0
    x = np.vstack(xs)
    y = np.concatenate(ys)
    total = float(np.dot(y, y))
    if total <= 0.0:
        return float("nan"), int(y.size)
    coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
    residual = float(np.sum((y - x @ coefficients) ** 2))
    return float(1.0 - residual / total), int(y.size)


def itemise(
    reconciliations: Sequence[StepReconciliation],
    samples: Sequence[StepSample],
    facts: RunFacts,
) -> list[FeatureBudget]:
    """The GUM table, per feature, over the window.

    Each term is a per-step standard uncertainty on `ρ`, root-mean-squared across the window, so
    that the quadrature sum is comparable against `Var(ρ)` across the same window. Terms whose
    natural size is a fraction rather than a quantity in feature units carry the fraction as
    ``value`` and the size it acts on as ``sensitivity``, which is the split `core.budget.BudgetTerm`
    was built for: it lets the table say that the clip fraction is small and the reading is very
    sensitive to it, which a pre-multiplied contribution cannot express.
    """
    if not reconciliations:
        return []
    names = reconciliations[0].names
    columns = [i for i, n in enumerate(samples[0].names) if n in set(names)] if samples else []
    r_squared, n_regression = advantage_r_squared(samples, columns)
    n_steps = len(reconciliations)
    out: list[FeatureBudget] = []

    for j, feature in enumerate(names):
        rows = [rec.row(feature) for rec in reconciliations]
        residuals = np.asarray([r.residual for r in rows], dtype=np.float64)
        finite = residuals[np.isfinite(residuals)]
        var_residual = float(np.var(finite, ddof=1)) if finite.size > 1 else float("nan")
        mean_residual = float(np.mean(finite)) if finite.size else float("nan")
        se_mean = (
            float(np.std(finite, ddof=1) / np.sqrt(finite.size))
            if finite.size > 1
            else float("nan")
        )
        predicted = _rms([r.delta_z_pred for r in rows])

        terms: list[BudgetTerm] = []
        missing: list[MissingTerm] = []
        _stale(terms, missing, facts, n_steps)
        _kl(terms, missing, facts, n_steps)
        _entropy(terms, missing, facts, predicted, n_steps)
        _momentum(terms, missing, facts, predicted, n_steps)
        _batch(terms, reconciliations, j, n_steps)
        _curvature(terms, missing, facts, reconciliations)
        _clip(terms, missing, facts, predicted, n_steps)
        _monte_carlo(terms, rows, reconciliations)
        _basis(terms, missing, predicted, r_squared, n_regression, len(names))

        budget = UncertaintyBudget(terms=tuple(terms))
        gap = var_residual - float(budget.combined**2)
        needed = float(np.sqrt(gap)) if np.isfinite(gap) and gap > 0 else float("nan")
        out.append(
            FeatureBudget(
                feature=feature,
                n_steps=n_steps,
                mean_residual=mean_residual,
                var_residual=var_residual,
                se_mean_residual=se_mean,
                budget=budget,
                missing=tuple(
                    MissingTerm(m.name, m.why, m.remedy, needed_to_close=needed) for m in missing
                ),
            )
        )
    return out


# ---------------------------------------------------------------------------
# The nine terms, one function each so that each carries its own argument
# ---------------------------------------------------------------------------


def _stale(
    terms: list[BudgetTerm], missing: list[MissingTerm], facts: RunFacts, n_steps: int
) -> None:
    """Off-policy staleness. Zero when every segment was generated by the policy it updated."""
    value = facts.max_staleness
    if isinstance(value, Absent):
        missing.append(MissingTerm("u_stale", value.why, value.remedy))
        return
    if int(value) == 0:
        terms.append(
            BudgetTerm(
                name="u_stale",
                value=0.0,
                kind="A",
                dof=max(n_steps - 1, 1),
                note=(
                    "every trajectory segment in the window declares staleness_steps = 0, so every "
                    "rollout was generated by the policy that was then updated on it. This term is "
                    "exactly zero rather than small."
                ),
            )
        )
        return
    missing.append(
        MissingTerm(
            "u_stale",
            f"rollouts are up to {int(value)} step(s) stale and the size of the resulting bias "
            f"cannot be read off the record",
            "rescore a held-out fraction of the stale rollouts under the policy that was updated "
            "on them, and pass the difference. That is the Type A evaluation and there is no "
            "record-only substitute for it.",
        )
    )


def _kl(terms: list[BudgetTerm], missing: list[MissingTerm], facts: RunFacts, n_steps: int) -> None:
    """`η·Cov(−β_KL ∇KL, f)`. Exactly zero when the trainer applied no KL penalty."""
    value = facts.kl_coefficient
    if isinstance(value, Absent):
        missing.append(MissingTerm("u_KL", value.why, value.remedy))
        return
    if float(value) == 0.0:
        terms.append(
            BudgetTerm(
                name="u_KL",
                value=0.0,
                kind="A",
                dof=max(n_steps - 1, 1),
                note=(
                    "the schedule carries beta = 0.0 at every step of the window, so there is no "
                    "KL penalty pulling the policy back toward a reference and this term is "
                    "identically zero. That is a property of the run and not a gap in the record."
                ),
            )
        )
        return
    missing.append(
        MissingTerm(
            "u_KL",
            f"the trainer applied a KL penalty at beta = {float(value):g} and the record does not "
            f"carry the per-rollout KL gradient the covariance is taken against",
            "compute `Cov_group(-beta_KL * grad KL, f)` at POLICY: BACKWARD and pass it, or "
            "re-run the window at beta = 0 to measure the same policy without the pull.",
        )
    )


def _entropy(
    terms: list[BudgetTerm],
    missing: list[MissingTerm],
    facts: RunFacts,
    predicted: float,
    n_steps: int,
) -> None:
    """The entropy bonus, by the same construction as the KL term."""
    value = facts.entropy_coefficient
    if isinstance(value, Absent):
        missing.append(MissingTerm("u_entropy", value.why, value.remedy))
        return
    if float(value) == 0.0:
        terms.append(
            BudgetTerm(
                name="u_entropy",
                value=0.0,
                kind="A",
                dof=max(n_steps - 1, 1),
                note="the trainer applied no entropy bonus, so this term is identically zero.",
            )
        )
        return
    missing.append(
        MissingTerm(
            "u_entropy",
            f"the trainer applied an entropy bonus at {float(value):g} and the record does not "
            f"carry the per-rollout entropy gradient",
            "compute `Cov_group(coefficient * grad H, f)` at POLICY: BACKWARD and pass it.",
        )
    )


def _momentum(
    terms: list[BudgetTerm],
    missing: list[MissingTerm],
    facts: RunFacts,
    predicted: float,
    n_steps: int,
) -> None:
    """Optimiser state. Adam is not SGD, and the gap is the applied step against the raw one."""
    value = facts.momentum_gap
    if isinstance(value, Absent):
        missing.append(MissingTerm("u_momentum", value.why, value.remedy))
        return
    terms.append(
        BudgetTerm(
            name="u_momentum",
            value=float(value),
            kind="B",
            distribution="normal",
            sensitivity=predicted,
            dof=TYPE_B_DOF,
            note=(
                "the fractional gap between the applied step and the raw gradient step, acting on "
                "the predicted response. Type B because the conversion assumes the optimiser "
                "changed the step's magnitude and not its direction, and Adam changes both, so "
                "this is a lower bound even where both norms are recorded."
            ),
        )
    )


def _batch(
    terms: list[BudgetTerm],
    reconciliations: Sequence[StepReconciliation],
    j: int,
    n_steps: int,
) -> None:
    """Gradient interference between prompts, propagated from the clustered error on `S`.

    `Δz_pred = η·G·C⁻¹·S`, so an uncertainty on the selection differential reaches the prediction
    through `η·G·C⁻¹`, which the reconciliation carries as ``response_jacobian``. The propagation
    treats the per-feature errors on `S` as independent, which they are not; the correlation would
    have to come from the full sampling covariance of the differential and the ledger reports only
    its diagonal. Stated rather than assumed away.
    """
    values: list[float] = []
    dof = 0.0
    for rec in reconciliations:
        jac = np.asarray(rec.response_jacobian, dtype=np.float64)
        errors = np.asarray([r.se_differential for r in rec.rows], dtype=np.float64)
        if jac.shape[0] <= j or not np.all(np.isfinite(errors)):
            continue
        values.append(float(np.sqrt(np.sum((jac[j] * errors) ** 2))))
        dof += max(rec.n_groups - 1, 1)
    terms.append(
        BudgetTerm(
            name="u_batch",
            value=_rms(values) if values else 0.0,
            kind="A",
            dof=max(dof, 1.0),
            note=(
                "the group-clustered standard error on the selection differential, propagated "
                "through eta * G * C^-1. Per-feature errors are composed as independent because "
                "the ledger reports the diagonal of the differential's sampling covariance."
            ),
        )
    )


def _curvature(
    terms: list[BudgetTerm],
    missing: list[MissingTerm],
    facts: RunFacts,
    reconciliations: Sequence[StepReconciliation],
) -> None:
    """`½η²‖∇²‖`, the term dropped in the first-order expansion."""
    value = facts.hessian_norm
    if isinstance(value, Absent):
        missing.append(MissingTerm("u_curv", value.why, value.remedy))
        return
    etas = np.asarray([rec.eta for rec in reconciliations], dtype=np.float64)
    terms.append(
        BudgetTerm(
            name="u_curv",
            value=_rms(0.5 * etas**2 * float(value)),
            kind="B",
            distribution="normal",
            sensitivity=1.0,
            dof=TYPE_B_DOF,
            note="half eta squared times the supplied curvature norm, root-mean-squared over steps",
        )
    )


def _clip(
    terms: list[BudgetTerm],
    missing: list[MissingTerm],
    facts: RunFacts,
    predicted: float,
    n_steps: int,
) -> None:
    """Ratio clipping and loss masking, as the fraction of the update they removed."""
    value = facts.clip_fraction
    if isinstance(value, Absent):
        missing.append(MissingTerm("u_clip", value.why, value.remedy))
        return
    terms.append(
        BudgetTerm(
            name="u_clip",
            value=float(value),
            kind="A",
            sensitivity=predicted,
            dof=max(n_steps - 1, 1),
            note=(
                "the clipped fraction acting on the predicted response. The exact evaluation "
                "recomputes the update unclipped and differences it; this bounds it by assuming a "
                "clipped token contributes nothing, which is the largest it can be."
            ),
        )
    )


def _monte_carlo(
    terms: list[BudgetTerm],
    rows: Sequence,
    reconciliations: Sequence[StepReconciliation],
) -> None:
    """`s/√K`: the sampling error in `Δz` itself, which the ledger already reports per row."""
    dof = sum(max(rec.n_before + rec.n_after - 2, 1) for rec in reconciliations)
    terms.append(
        BudgetTerm(
            name="u_MC",
            value=_rms([r.se_delta_z for r in rows]),
            kind="A",
            dof=max(float(dof), 1.0),
            note=(
                "the standard error of the difference of two feature means, root-mean-squared over "
                "the window. This is the noise in the measured side of the identity and it is the "
                "one term that is always available from a record."
            ),
        )
    )


def _basis(
    terms: list[BudgetTerm],
    missing: list[MissingTerm],
    predicted: float,
    r_squared: float,
    n_regression: int,
    k: int,
) -> None:
    """The `ηJF⁻¹e` term: the response driven by advantage the feature basis does not explain."""
    if not np.isfinite(r_squared) or r_squared <= 0.0:
        missing.append(
            MissingTerm(
                "u_basis",
                "the within-group regression of the advantage on the features has no usable "
                "R-squared, so the share of the advantage outside the feature span is unknown",
                "supply a featuriser whose features vary inside a prompt group. With no "
                "within-group spread there is nothing for the advantage to be regressed on.",
            )
        )
        return
    terms.append(
        BudgetTerm(
            name="u_basis",
            value=float(np.sqrt((1.0 - r_squared) / r_squared)),
            kind="A",
            sensitivity=predicted,
            dof=max(float(n_regression - k - 1), 1.0),
            note=(
                f"the within-group R-squared of the advantage on the features is {r_squared:.4f}, "
                f"so {(1.0 - r_squared):.4f} of the advantage variance lies outside the measured "
                f"basis. Converted to feature units by assuming the unexplained advantage drives "
                f"response in proportion to the explained part, which is a judgement about the "
                f"conversion rather than about the R-squared: the identity-based evaluation needs "
                f"J and F at POLICY: BACKWARD."
            ),
        )
    )


__all__ = [
    "TERM_ORDER",
    "TYPE_B_DOF",
    "FeatureBudget",
    "MissingTerm",
    "advantage_r_squared",
    "itemise",
]

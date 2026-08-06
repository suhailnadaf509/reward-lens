"""The join: four books, two independent predictions of `Δz`, and the residual between them.

The effect book measures `Δz_obs` directly. Cause and capacity together predict
`Δz_pred = η · G · C⁻¹ · S`. The difference is the reconciliation residual

    ρ = Δz_obs − Δz_pred

and it is not noise. `measure.reconcile.residual` itemises it as a budget with named terms and
tests whether that budget closes.

**Three things this cannot do, stated here rather than on a caveats page.** It cannot compute `G`:
the reachable covariance needs a Fisher solve against the policy that wrote the record, which is
`measure.efficiency` at `POLICY: BACKWARD`, and this module takes a `MetricG` as an argument and
refuses without one. It cannot tell a residual from the sampling noise in `Δz` at a single step,
because both are the same size on eight rollouts; the closure test is a statement about variance
across a window and it says so. And `β` is a direct effect conditional on the measured feature
basis, which is the Table 2 fallacy, so a residual attributed to "everything outside the basis" is
attributed to a set nobody has enumerated.

**The join key is the feature basis and it is `measure.ledger.price.StepSample.names`.** `G`, the
ledger's `Δz`, and the cost book's per-feature shares are vectors in one basis, in one order,
produced by one `TrajectoryFeaturiser`. A `G` whose names do not match the ledger's, element for
element, cannot be reconciled against it, and the reconciliation is the whole of F4. So a mismatch
refuses with `UNIT_MISMATCH` rather than aligning by name behind the caller's back: a silently
reordered basis produces numbers that look right and mean nothing, which is the failure this
library exists to make impossible.

**On the covariance operator.** `C` is the **within-group** covariance, both features centred inside
their own prompt group with the pooled `n - G` denominator, matching `Cov_group(A, f)` in
`measure.ledger.price`. That is the operator the identity means, and it is not the pooled
covariance: between the two sits prompt-to-prompt heterogeneity, which is a property of the task
distribution rather than of the policy. Mixing a within-group `S` with a pooled `C` does
not solve `S = Cβ` on either operator. Every reading names the operator it used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from reward_lens.measure.ledger.price import (
    Differential,
    StepLedger,
    StepSample,
    selection_differential,
)

# ---------------------------------------------------------------------------
# What this module reads from `measure/efficiency/`, as a structural protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class MetricGLike(Protocol):
    """`G = J F⁻¹ Jᵀ`, the covariance a parameter move can actually reach.

    This is `measure.efficiency.MetricG` written as a Protocol, field for field, so that the real
    dataclass satisfies it structurally with no adapter and no import. It exists because F4 and F3
    were written against an interface fixed in advance rather than one after the other, and a
    Protocol is how that is expressed in the type system rather than in a comment.

    ``damping`` is the `λ` in `(F + λI)⁻¹` and it is a field rather than a parameter, because a
    reading that hides its regularisation cannot be checked for the stability it also has to claim.
    ``rung`` is 0 for the realised estimator that needs only a record and 2 for the Fisher solve.
    """

    names: tuple[str, ...]
    matrix: np.ndarray
    damping: float
    damping_stable: bool
    conditioning: float
    rung: int
    method: str
    n_samples: int


@runtime_checkable
class StepCostLike(Protocol):
    """One step of F3's cost book, as `measure.efficiency.StepCost`.

    Read here only for the cost consistency check: `KL_min(Δz_obs) ≤ D_t` must hold, and a
    violation is an instrument bug rather than a finding. `kl_min` is not recomputed here. The
    cost book owns it, recomputing it would be a second implementation of the same quadratic form,
    and a check that recomputes the thing it is checking is not a check.
    """

    step: int
    next_step: int
    kl_spent: float
    kl_min: float
    efficiency: float
    shares: Mapping[str, float]
    residual_share: float


class BasisMismatch(ValueError):
    """`G` and the ledger disagree about the feature basis, so nothing can be reconciled."""


# ---------------------------------------------------------------------------
# Cause: the within-group covariance and the selection gradient
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureCovariance:
    """`C = Cov_group(f, f)` pooled over a window, with what decides whether to invert it.

    ``conditioning`` is `n_D = Σλ / λ_max`, the effective dimension that ships beside every `β`.
    It is not the condition number: `n_D` counts how many directions carry comparable variance
    and falls toward 1 when one direction dominates, which is the failure mode that makes `C⁻¹S`
    unstable. Both are reported because they answer different questions.

    ``n_used`` and ``n_groups`` give the denominator `n - G`. Eight rollouts in two groups per step
    is six degrees of freedom, so a `C` estimated at a single step is rank-deficient for more than
    six features and noisy well below that. Pooling across steps is the default and ``n_steps``
    records how many went in.
    """

    names: tuple[str, ...]
    matrix: np.ndarray
    n_used: int
    n_groups: int
    n_steps: int
    eigenvalues: np.ndarray
    operator: str = "within_group"

    @property
    def conditioning(self) -> float:
        """`n_D = Σλ / λ_max`. Between 1 and `k`; near 1 means one direction carries everything."""
        ev = self.eigenvalues
        top = float(ev.max()) if ev.size else 0.0
        return float(ev.sum() / top) if top > 0 else float("nan")

    @property
    def condition_number(self) -> float:
        """`λ_max / λ_min`. Infinite when the smallest eigenvalue is zero, which is honest."""
        ev = self.eigenvalues
        if ev.size == 0:
            return float("nan")
        low = float(ev.min())
        return float(ev.max() / low) if low > 0 else float("inf")

    @property
    def is_invertible(self) -> bool:
        """Whether an unregularised solve is defined. A zero eigenvalue means it is not."""
        return bool(self.eigenvalues.size and float(self.eigenvalues.min()) > 0.0)


def within_group_covariance(
    samples: Sequence[StepSample],
    *,
    columns: Sequence[int] | None = None,
) -> FeatureCovariance:
    """`C` pooled over the window, each feature centred inside its own prompt group.

    `C_ij = Σ_g Σ_r (f_gri − f̄_gi)(f_grj − f̄_gj) / (n − G)`, the same pooled unbiased estimator
    `selection_differential` uses for `S`, so `S = Cβ` is solved with both sides on one operator.
    Groups of one rollout have no within-group spread and contribute to neither sum.

    ``columns`` restricts to a subset of the feature basis, which is how constant features are kept
    out: a feature with zero spread over the window contributes a zero row and column, `C` is
    singular, and the solve fails for a reason that has nothing to do with the data. Dropping them
    by name is what `measure.ledger.explained.fit_lambda` already does and it is done the same way
    here.
    """
    if not samples:
        raise ValueError("a covariance over no steps has no sample; pass at least one StepSample")
    names_all = samples[0].names
    idx = list(range(len(names_all))) if columns is None else list(columns)
    names = tuple(names_all[i] for i in idx)
    k = len(idx)
    total = np.zeros((k, k), dtype=np.float64)
    n_used = 0
    n_groups = 0
    for sample in samples:
        if sample.names != names_all:
            raise BasisMismatch(
                f"step {sample.index} carries basis {list(sample.names)} against "
                f"{list(names_all)} at the head of the window. A covariance pooled across two "
                f"bases is a covariance of nothing."
            )
        if sample.n == 0:
            continue
        f = sample.features[:, idx]
        finite = np.all(np.isfinite(f), axis=1)
        for label in np.unique(sample.group_ids[finite]) if finite.any() else ():
            mask = (sample.group_ids == label) & finite
            size = int(np.count_nonzero(mask))
            if size < 2:
                continue
            centred = f[mask] - f[mask].mean(axis=0)
            total += centred.T @ centred
            n_used += size
            n_groups += 1
    denominator = n_used - n_groups
    matrix = total / denominator if denominator > 0 else np.full((k, k), np.nan)
    eigenvalues = (
        np.linalg.eigvalsh(matrix)
        if np.all(np.isfinite(matrix))
        else np.full(k, np.nan, dtype=np.float64)
    )
    return FeatureCovariance(
        names=names,
        matrix=matrix,
        n_used=n_used,
        n_groups=n_groups,
        n_steps=len(samples),
        eigenvalues=eigenvalues,
    )


@dataclass(frozen=True)
class SelectionGradient:
    """`β = C⁻¹S`, the direct push, with the regularisation it was solved under.

    ``ridge`` is `δ` in `(C + δ·(tr C / k)·I)⁻¹S`, scaled by the mean eigenvalue so that one value
    of `δ` means the same thing on features recorded in characters and on features recorded in
    words. Swept on real data, the answer is flat over `δ` from 0 to 1e-2 and by `δ = 1` the
    solution has collapsed onto `S / (δ·tr C / k)`, which is `S` rescaled and has stopped being a
    gradient. So the default is zero and any non-zero value is reported.

    **`S` and `β` can differ in sign, and both belong on the page.** `S` is the marginal
    association and `β` is the direct effect conditional on the measured basis. Reporting a
    marginal covariance and reading it as an influence is the most-warned-against error in
    quantitative genetics, and reporting a filter without its pattern is the most-warned-against
    error in explanation-correctness. The two fields carry both.
    """

    names: tuple[str, ...]
    value: np.ndarray
    differential: np.ndarray
    ridge: float
    ridge_scale: float
    conditioning: float
    operator: str = "within_group"

    def as_dict(self) -> dict[str, float]:
        return {n: float(v) for n, v in zip(self.names, self.value)}

    def render(self) -> str:
        pairs = "  ".join(f"{n} {b:+.4g}" for n, b in zip(self.names, self.value))
        return f"beta ({self.operator}, ridge {self.ridge:g}, n_D {self.conditioning:.3g}): {pairs}"


def selection_gradient(
    covariance: FeatureCovariance,
    differential: Differential | np.ndarray,
    *,
    ridge: float = 0.0,
) -> SelectionGradient:
    """Solve `S = Cβ` for `β` on the within-group operator.

    This is a local implementation and it is here only because `measure/selection/` was not on the
    branch when F4 was written. It is one function on purpose: when the sibling package lands, the
    swap is the one call in `reconcile_series` and nothing else in this module moves. Both take the
    within-group operator, so the two answers are the same estimator rather than two conventions.

    Solved rather than inverted. `np.linalg.solve` on a k-by-k system is the same arithmetic with
    better conditioning than forming `C⁻¹` and multiplying, and at `k` in the single digits the cost
    difference is nothing while the accuracy difference is real.
    """
    s = (
        np.asarray(differential.value, dtype=np.float64)
        if isinstance(differential, Differential)
        else np.asarray(differential, dtype=np.float64)
    )
    c = np.asarray(covariance.matrix, dtype=np.float64)
    k = c.shape[0]
    if s.shape[0] != k:
        raise BasisMismatch(
            f"the differential has {s.shape[0]} entries and the covariance is {k} by {k}. "
            f"They are supposed to be the same feature basis in the same order."
        )
    scale = float(np.trace(c) / k) if k else 0.0
    a = c + ridge * scale * np.eye(k) if ridge else c
    try:
        beta = np.linalg.solve(a, s)
    except np.linalg.LinAlgError:
        beta = np.full(k, np.nan, dtype=np.float64)
    return SelectionGradient(
        names=covariance.names,
        value=np.asarray(beta, dtype=np.float64),
        differential=s,
        ridge=float(ridge),
        ridge_scale=scale,
        conditioning=covariance.conditioning,
        operator=covariance.operator,
    )


# ---------------------------------------------------------------------------
# The reconciliation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BookRow:
    """One feature, one step, across all four books.

    ``residual`` is F4's `ρ = Δz_obs − η·G·β`. ``ledger_residual`` is F1's `Δz_obs − η·S`, carried
    beside it because the two are the same number exactly when `G = C`, and the gap between them is
    what capacity buys over cause alone. A reader who sees them equal has learned that heritability
    is doing no work on this feature.
    """

    feature: str
    delta_z_obs: float
    delta_z_pred: float
    residual: float
    eta: float
    differential: float
    gradient: float
    response: float
    heritability: float
    se_delta_z: float
    se_differential: float
    ledger_residual: float

    @property
    def predicted_share(self) -> float:
        """`Δz_pred / Δz_obs`, or NaN when nothing moved. 1.0 is a fully predicted step."""
        return float(self.delta_z_pred / self.delta_z_obs) if self.delta_z_obs else float("nan")

    def render(self) -> str:
        return (
            f"{self.feature:<20} Dz_obs {self.delta_z_obs:+.5g}  Dz_pred {self.delta_z_pred:+.5g}  "
            f"rho {self.residual:+.5g}   (S {self.differential:+.4g}, beta {self.gradient:+.4g}, "
            f"h2 {self.heritability:.3g})"
        )


@dataclass(frozen=True)
class StepReconciliation:
    """The four books for one step pair, per feature, and what the join was allowed to assume.

    ``notes`` carries every choice that changes a number: which covariance operator, how many steps
    `C` was pooled over, what regularisation `β` and `G` were solved under, and whether the two
    steps shared any prompt. A reconciliation read without them is a residual attributed to a model
    whose assumptions the reader cannot see.
    """

    step: int
    next_step: int
    rows: tuple[BookRow, ...]
    eta: float
    eta_source: str
    task_overlap: float
    n_scored: int
    n_groups: int
    n_before: int
    n_after: int
    #: `η·G·C⁻¹`, the sensitivity of `Δz_pred` to the selection differential. Carried rather than
    #: recomputed because the uncertainty budget propagates `u(S)` through exactly this matrix, and
    #: re-deriving it there would be a second place for the operator and the ridge to disagree.
    response_jacobian: np.ndarray
    c_n_used: int
    c_n_groups: int
    c_n_steps: int
    c_conditioning: float
    ridge: float
    g_rung: int
    g_method: str
    g_damping: float
    g_damping_stable: bool
    operator: str = "within_group"
    notes: tuple[str, ...] = ()

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(r.feature for r in self.rows)

    def row(self, feature: str) -> BookRow:
        for r in self.rows:
            if r.feature == feature:
                return r
        raise KeyError(f"no book row for {feature!r}; this reconciliation holds {list(self.names)}")

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(Δz_obs, Δz_pred, ρ)`` in feature order, which is what the Lande fit consumes."""
        return (
            np.asarray([r.delta_z_obs for r in self.rows], dtype=np.float64),
            np.asarray([r.delta_z_pred for r in self.rows], dtype=np.float64),
            np.asarray([r.residual for r in self.rows], dtype=np.float64),
        )

    def render(self) -> str:
        head = (
            f"books {self.step} -> {self.next_step}  eta = {self.eta:.4g} ({self.eta_source}), "
            f"G at rung {self.g_rung} ({self.g_method}), C pooled over {self.c_n_steps} steps "
            f"(n_D {self.c_conditioning:.3g}), task overlap {self.task_overlap:.2f}"
        )
        return "\n".join([head, *("    " + r.render() for r in self.rows)])


def _response_jacobian(
    g: np.ndarray, covariance: FeatureCovariance, *, ridge: float = 0.0
) -> np.ndarray:
    """`G C⁻¹`, the derivative of the predicted response with respect to the differential.

    `Δz_pred = η G C⁻¹ S`, so an uncertainty on `S` reaches `Δz_pred` through this matrix and not
    through `G` alone. Solved rather than inverted, under the same ridge `β` was solved under, so
    the budget cannot end up propagating through a different operator than the estimate it is a
    budget for. Returns NaN rather than a pseudo-inverse when the solve fails: a pseudo-inverse
    here is a silent change of estimator.
    """
    c = np.asarray(covariance.matrix, dtype=np.float64)
    k = c.shape[0]
    scale = float(np.trace(c) / k) if k else 0.0
    a = c + ridge * scale * np.eye(k) if ridge else c
    try:
        return np.asarray(np.linalg.solve(a.T, g.T).T, dtype=np.float64)
    except np.linalg.LinAlgError:
        return np.full((k, k), np.nan, dtype=np.float64)


def _aligned_columns(g_names: Sequence[str], sample_names: Sequence[str]) -> list[int]:
    """The join key check: same names, same order, element for element."""
    if tuple(g_names) != tuple(sample_names):
        raise BasisMismatch(
            f"G is in basis {list(g_names)} and the ledger is in basis {list(sample_names)}. "
            f"These are vectors in different spaces and the difference between them is not a "
            f"residual. Build both from one TrajectoryFeaturiser: G's names and "
            f"StepSample.names are the join key and they are compared element for element, not "
            f"as sets, because a reordered basis produces numbers that look right."
        )
    return list(range(len(sample_names)))


def reconcile_series(
    samples: Sequence[StepSample],
    ledgers: Sequence[StepLedger],
    metric_g: MetricGLike,
    *,
    ridge: float = 0.0,
    c_context: int | None = None,
    gradients: Mapping[int, np.ndarray] | None = None,
) -> list[StepReconciliation]:
    """One reconciliation per ledger row, on the features that vary over the window.

    ``c_context`` is the half-width of the window `C` is pooled over, in step pairs. The default is
    None, meaning the whole window, and that is a stated choice rather than a silent one: eight
    rollouts in two groups is six degrees of freedom at a single step, which is too few to invert a
    covariance on and expect the inverse to mean anything. `S` stays per step, so `β_t = C⁻¹S_t`
    still moves with the step's own selection pressure; only the operator it is solved against is
    held still.

    ``gradients`` lets a caller supply `β` per step from elsewhere, which is the seam for
    `measure/selection/` when it lands. Absent, `selection_gradient` computes it here.

    Constant features are dropped rather than carried: a feature with no spread over the window has
    a zero row in `C`, contributes nothing to `S`, and makes the solve singular for a reason that is
    about the featuriser rather than about the run. The dropped names appear in ``notes``.
    """
    if not ledgers or not samples:
        return []
    _aligned_columns(metric_g.names, samples[0].names)

    by_index = {s.index: s for s in samples}
    variance = np.vstack([s.features for s in samples if s.n])
    spread = variance.std(axis=0, ddof=1) if variance.shape[0] > 1 else np.zeros(variance.shape[1])
    keep = [i for i in range(len(samples[0].names)) if spread[i] > 0.0]
    dropped = tuple(n for i, n in enumerate(samples[0].names) if i not in keep)
    if not keep:
        return []
    names = tuple(samples[0].names[i] for i in keep)
    g_full = np.asarray(metric_g.matrix, dtype=np.float64)
    g = g_full[np.ix_(keep, keep)]

    out: list[StepReconciliation] = []
    order = [s.index for s in samples]
    for ledger in ledgers:
        sample = by_index.get(ledger.step)
        if sample is None:
            continue
        if c_context is None:
            window = list(samples)
        else:
            centre = order.index(ledger.step)
            lo = max(0, centre - c_context)
            window = list(samples[lo : centre + c_context + 1])
        covariance = within_group_covariance(window, columns=keep)
        differential = selection_differential(
            sample.features[:, keep], sample.advantages, sample.group_ids, names
        )
        if gradients is not None and ledger.step in gradients:
            beta_vec = np.asarray(gradients[ledger.step], dtype=np.float64)
            gradient = SelectionGradient(
                names=names,
                value=beta_vec,
                differential=differential.value,
                ridge=float("nan"),
                ridge_scale=float("nan"),
                conditioning=covariance.conditioning,
            )
        else:
            gradient = selection_gradient(covariance, differential, ridge=ridge)
        response = g @ gradient.value
        eta = float(ledger.eta)
        jacobian = _response_jacobian(g, covariance, ridge=ridge) * eta
        diag_c = np.diag(covariance.matrix)
        diag_g = np.diag(g)
        rows = tuple(
            BookRow(
                feature=name,
                delta_z_obs=float(ledger.row(name).delta_z),
                delta_z_pred=float(eta * response[j]),
                residual=float(ledger.row(name).delta_z - eta * response[j]),
                eta=eta,
                differential=float(differential.value[j]),
                gradient=float(gradient.value[j]),
                response=float(response[j]),
                heritability=(float(diag_g[j] / diag_c[j]) if diag_c[j] > 0 else float("nan")),
                se_delta_z=float(ledger.row(name).se_delta_z),
                se_differential=float(differential.standard_error[j]),
                ledger_residual=float(ledger.row(name).residual),
            )
            for j, name in enumerate(names)
        )
        notes = list(ledger.notes)
        if dropped:
            notes.append(
                f"{len(dropped)} feature(s) have no spread over this window and were dropped "
                f"before the solve: {', '.join(dropped)}. A constant feature has a zero row in C "
                f"and no covariance with anything, so it cannot enter a regression."
            )
        if ridge:
            notes.append(
                f"beta was solved under ridge delta = {ridge:g} toward (tr C / k) I. The answer "
                f"is flat to delta = 1e-2 and has collapsed onto a rescaled S by delta = 1, so "
                f"read the ridge as part of the estimate."
            )
        out.append(
            StepReconciliation(
                step=ledger.step,
                next_step=ledger.next_step,
                rows=rows,
                eta=eta,
                eta_source=ledger.eta_source,
                task_overlap=ledger.task_overlap,
                n_scored=ledger.n_scored,
                n_groups=ledger.n_groups,
                n_before=ledger.n_before,
                n_after=ledger.n_after,
                response_jacobian=jacobian,
                c_n_used=covariance.n_used,
                c_n_groups=covariance.n_groups,
                c_n_steps=covariance.n_steps,
                c_conditioning=covariance.conditioning,
                ridge=float(ridge),
                g_rung=int(metric_g.rung),
                g_method=str(metric_g.method),
                g_damping=float(metric_g.damping),
                g_damping_stable=bool(metric_g.damping_stable),
                operator=covariance.operator,
                notes=tuple(notes),
            )
        )
    return out


# ---------------------------------------------------------------------------
# The cost consistency check
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostConsistency:
    """`KL_min(Δz_obs) ≤ D_t` over a window. A violation is an instrument bug, not a finding.

    The reason for the inequality: `KL_min` is the minimum information cost of the behavioural
    change that was actually observed, and the optimiser spent `D_t` achieving it, so spending
    less than the minimum is arithmetically impossible. If it happens, either `G` is
    mis-estimated or the natural-gradient approximation has failed, and F3's own kill condition
    fires. That is F3's number to defend; this checks it at the join because the join is where the
    two books meet and a check that only one side runs is a check one side can drop.
    """

    n_steps: int
    n_violations: int
    worst_step: int | None
    worst_ratio: float
    max_efficiency: float
    checked: bool
    detail: str = ""

    @property
    def holds(self) -> bool:
        return self.checked and self.n_violations == 0

    def render(self) -> str:
        if not self.checked:
            return f"cost consistency: not checked. {self.detail}"
        if self.holds:
            return (
                f"cost consistency: KL_min <= D_t on all {self.n_steps} steps, worst efficiency "
                f"{self.max_efficiency:.4g}"
            )
        return (
            f"cost consistency: FAILED on {self.n_violations} of {self.n_steps} steps, worst at "
            f"step {self.worst_step} with KL_min / D_t = {self.worst_ratio:.4g}"
        )


def cost_consistency(costs: Sequence[StepCostLike] | None) -> CostConsistency:
    """Check `KL_min ≤ D_t` on a cost series, or record that no cost book was supplied.

    `kl_min` is read off the cost book rather than recomputed. Recomputing `½ Δzᵀ G⁻¹ Δz` here
    would be a second implementation of F3's central quadratic form, and a check that recomputes
    the quantity it checks tests the copy rather than the original.
    """
    if not costs:
        return CostConsistency(
            n_steps=0,
            n_violations=0,
            worst_step=None,
            worst_ratio=float("nan"),
            max_efficiency=float("nan"),
            checked=False,
            detail=(
                "no cost series was supplied, so the cost inequality was not tested. "
                "Pass F3's `cost_series` output to check it; the reconciliation residual does not "
                "depend on it and is reported without it."
            ),
        )
    ratios: list[tuple[int, float]] = []
    for c in costs:
        spent = float(c.kl_spent)
        ratios.append((int(c.step), float(c.kl_min / spent) if spent > 0 else float("inf")))
    violations = [(s, r) for s, r in ratios if not (r <= 1.0 + 1e-9)]
    worst = max(ratios, key=lambda sr: sr[1]) if ratios else (None, float("nan"))
    return CostConsistency(
        n_steps=len(ratios),
        n_violations=len(violations),
        worst_step=worst[0],
        worst_ratio=float(worst[1]),
        max_efficiency=max((float(c.efficiency) for c in costs), default=float("nan")),
        checked=True,
    )


__all__ = [
    "BasisMismatch",
    "BookRow",
    "CostConsistency",
    "FeatureCovariance",
    "MetricGLike",
    "SelectionGradient",
    "StepCostLike",
    "StepReconciliation",
    "cost_consistency",
    "reconcile_series",
    "selection_gradient",
    "within_group_covariance",
]

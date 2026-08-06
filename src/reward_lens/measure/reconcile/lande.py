"""F6: the Lande slope, which is the load-bearing assumption of Level 1 made checkable.

`Δz = η G β` is Lande's equation, derived from the natural-gradient step rather than
transplanted. Everything at Level 1 rests on it. Regressing observed `Δz` on `η G β` across a
window and reporting the slope with an interval is the test: **Lande holds at slope 1, and a slope
near zero retires Level 1.** If it fails, that failure is a publishable result about how policy
optimisation differs from natural selection.

**The degeneracy that has to be stated before any number is read.** With `G = C`, which is the
rung-0 `covariance_bound` estimator, `Gβ = C·C⁻¹S = S` exactly, so the regressor collapses to `η·S`
and this fit becomes F2's `η_eff` in different units. At rung 0 F6 therefore has **no content
independent of F2**, and a slope measured there is not evidence about Lande's equation: it is F2
restated. The independent test needs a `G` that is not `C`, which means the rung-2 Fisher solve at
`POLICY: BACKWARD`. `is_degenerate` carries this on every fit and the instrument says so in its
`deviations` rather than in a footnote.

The second thing that would invalidate the test is subtler and is checked the same way. The
`realised` estimator of `G` fits `Δz` against `η·β` across the window, so a slope measured against
it is a slope against a quantity derived from the response it is predicting, and it would come back
near 1 by construction. `metric_g` refuses that estimator on both GRPO records for an unrelated
reason (the fit is not positive semi-definite), but the circularity is a property of the estimator
rather than of these records, so it is refused here by name.

The fit follows `measure.ledger.explained`'s conventions exactly and reuses its arithmetic: through
the origin, because zero predicted response predicts zero movement to first order; each feature
divided by its own pooled spread, so the pooled slope is a statement about behaviour rather than
about the units a converter recorded in; and the interval clustered on step pairs, because the `k`
features of one step share one batch and one update and are one observation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

# `_through_origin` and `_clustered_slope_se` are F2's, imported rather than rewritten. The two
# fits differ only in their regressor, so a second copy of the arithmetic would be a second place
# for the through-origin convention and the clustering level to drift apart. Promoting them to
# public names in `measure.ledger.explained` is left open.
from reward_lens.measure.ledger.explained import _clustered_slope_se, _through_origin
from reward_lens.measure.reconcile.books import StepReconciliation

#: `G` estimators whose construction reads `Δz`, so a Lande slope against them is circular.
CIRCULAR_METHODS: frozenset[str] = frozenset({"realised"})


class CircularEstimator(ValueError):
    """`G` was fitted from the response this regression predicts, so the slope means nothing."""


@dataclass(frozen=True)
class LandeFit:
    """The slope of observed `Δz` on `η G β`, with everything needed to decide whether to read it.

    ``slope`` is 1 when Lande's equation holds exactly. ``r_squared`` is the uncentred `R²` of the
    same through-origin fit, and it is the number that says whether the slope describes anything: a
    slope of 0.83 at `R² = 0.001` is a line through a cloud, and reporting the first without the
    second is how a regression becomes a claim it cannot support.

    ``is_degenerate`` is True when `G` was the covariance bound, which makes `η G β = η S` and this
    fit F2's. It is a field rather than a caveat because the number is real, reproducible and
    uninformative about Lande, and only the flag distinguishes that from an informative one.
    """

    slope: float
    se_slope: float
    ci_low: float
    ci_high: float
    ci_level: float
    r_squared: float
    n_points: int
    n_steps: int
    n_features: int
    by_feature: Mapping[str, float]
    scales: Mapping[str, float]
    g_rung: int
    g_method: str
    is_degenerate: bool
    #: `(min, max)` of `h² = G_ii/C_ii` over the window, and the damping `G` was solved under. When
    #: `1 − min(h²)` is the damping, `G` is `C` plus a regulariser and the fit is not independent of
    #: F2's whatever rung it was computed at.
    heritability: tuple[float, float] = (float("nan"), float("nan"))
    g_damping: float = 0.0
    method: str = "through-origin OLS of dz on eta*G*beta, features scaled by sd(f)"

    @property
    def excludes_zero(self) -> bool:
        """Whether the interval clears zero, which is P10's registered resolution rule."""
        return bool(
            np.isfinite(self.ci_low)
            and np.isfinite(self.ci_high)
            and (self.ci_low > 0.0 or self.ci_high < 0.0)
        )

    @property
    def consistent_with_lande(self) -> bool:
        """Whether the interval contains 1, which is the equation holding rather than merely acting."""
        return bool(
            np.isfinite(self.ci_low)
            and np.isfinite(self.ci_high)
            and self.ci_low <= 1.0 <= self.ci_high
        )

    def render(self) -> str:
        head = (
            f"Lande slope {self.slope:.6g} [{self.ci_low:.6g}, {self.ci_high:.6g}] at "
            f"{self.ci_level:.0%} over {self.n_steps} step pairs and {self.n_features} features; "
            f"uncentred R^2 {self.r_squared:.5g}, se {self.se_slope:.4g}; G at rung {self.g_rung} "
            f"({self.g_method})"
        )
        if self.is_degenerate and self.g_rung == 0:
            head += (
                "\n    degenerate: G is the covariance bound, so eta*G*beta reduces to eta*S and "
                "this slope is F2's eta_eff rescaled. It is not evidence about Lande's equation. "
                "The independent test needs a rung-2 Fisher G at POLICY: BACKWARD."
            )
        elif self.is_degenerate:
            head += (
                f"\n    degenerate: h^2 is {self.heritability[0]:.5g} to "
                f"{self.heritability[1]:.5g} against a damping of {self.g_damping:.5g}, so G "
                f"differs from C only by its regulariser. The empirical Fisher has rank at most "
                f"the rollout count, so with fewer rollouts than parameters every feature lies in "
                f"the span of the scores and N = C - G is the damping rather than a measurement. "
                f"This fit is not independent of F2's."
            )
        return head


def _design(
    reconciliations: Sequence[StepReconciliation], scales: Mapping[str, float]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """``(x, y, names)`` with rows for step pairs and columns for features, each scaled by `sd(f)`.

    Dividing both sides of one feature by the same positive constant leaves that feature's own fit
    untouched and makes the pooled fit a statement about behaviour. Without it a feature recorded in
    characters rather than in words carries the whole regression, which is `measure.ledger`'s
    argument and applies here unchanged.
    """
    usable = [
        n for n in (reconciliations[0].names if reconciliations else ()) if scales.get(n, 0.0) > 0.0
    ]
    if not usable:
        return np.empty((0, 0)), np.empty((0, 0)), []
    x_rows, y_rows = [], []
    for rec in reconciliations:
        x_rows.append(np.asarray([rec.row(n).delta_z_pred / scales[n] for n in usable]))
        y_rows.append(np.asarray([rec.row(n).delta_z_obs / scales[n] for n in usable]))
    x = np.vstack(x_rows)
    y = np.vstack(y_rows)
    finite = np.isfinite(x) & np.isfinite(y)
    return np.where(finite, x, 0.0), np.where(finite, y, 0.0), usable


def fit_lande(
    reconciliations: Sequence[StepReconciliation],
    scales: Mapping[str, float],
    *,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
    allow_circular: bool = False,
) -> LandeFit | None:
    """The pooled through-origin fit of `Δz_obs` on `η G β`. None when nothing is left to fit.

    Raises `CircularEstimator` when `G` was fitted from `Δz`, because a slope near 1 against a
    regressor derived from the response is not a measurement and returning it with a caveat would
    put a number in front of a reader who has to read the caveat to know it means nothing.
    ``allow_circular`` exists so a test can assert the circular case reproduces 1, which is the
    demonstration that the guard is needed.
    """
    if not reconciliations:
        return None
    method = reconciliations[0].g_method
    if not allow_circular and any(token in method for token in CIRCULAR_METHODS):
        raise CircularEstimator(
            f"G was estimated by {method!r}, which fits the response `Δz` this regression is "
            f"predicting. The slope would come back near 1 by construction and would be a "
            f"property of the estimator rather than of the run. Estimate G by the covariance "
            f"bound (rung 0, and then read `is_degenerate`) or by the Fisher solve (rung 2)."
        )
    x, y, usable = _design(reconciliations, scales)
    if not usable or x.size == 0:
        return None

    r_squared, slope = _through_origin(x.ravel(), y.ravel())
    se = _clustered_slope_se(x, y, slope)
    low, high = _bootstrap_slope(x, y, n_bootstrap=n_bootstrap, ci=ci, seed=seed)
    by_feature = {name: _through_origin(x[:, j], y[:, j])[1] for j, name in enumerate(usable)}

    return LandeFit(
        slope=slope,
        se_slope=se,
        ci_low=low,
        ci_high=high,
        ci_level=ci,
        r_squared=r_squared,
        n_points=int(x.size),
        n_steps=len(reconciliations),
        n_features=len(usable),
        by_feature=by_feature,
        scales={n: float(scales[n]) for n in usable},
        g_rung=reconciliations[0].g_rung,
        g_method=method,
        is_degenerate=_is_degenerate(reconciliations),
        heritability=heritability_range(reconciliations),
        g_damping=reconciliations[0].g_damping,
    )


def heritability_range(reconciliations: Sequence[StepReconciliation]) -> tuple[float, float]:
    """`(min, max)` of `h² = G_ii / C_ii` over the window. `1` means `G` is `C` on the diagonal."""
    values = [
        row.heritability
        for rec in reconciliations
        for row in rec.rows
        if np.isfinite(row.heritability)
    ]
    return (min(values), max(values)) if values else (float("nan"), float("nan"))


def _is_degenerate(reconciliations: Sequence[StepReconciliation]) -> bool:
    """Whether `η G β` has collapsed onto `η S`, which makes this fit F2's rather than F6's.

    Two ways it happens and both are checked, because only the first is obvious.

    **Exactly**, when `G = C`: then `Gβ = C·C⁻¹S = S`. Tested on the numbers rather than on the
    method string, because the string is a label and this is a property of the arithmetic.

    **Numerically**, when the rung-2 Fisher solve has fewer rollouts than parameters. `F` is
    estimated from `n` score vectors so its rank is at most `n`; with `n` far below `|θ|` every
    feature lies exactly in the span of the scores, `N = C − G` collapses onto the damping, and the
    reported `h²` is `1 − λ` rather than a property of the policy. Measured on the 24-step window
    of the 200-step fixture: 192 rollouts against 2,453,368 parameters gives `h² = 0.9914` on all
    three varying features against a damping of 0.00874, which is the same number. A `G` that
    differs from `C` only by its regulariser cannot make this regression independent of F2's, and
    reporting the fit as an independent test because the two matrices are not bit-identical would
    be claiming a distinction the estimator could not have found.
    """
    for rec in reconciliations[: min(len(reconciliations), 8)]:
        for row in rec.rows:
            reference = abs(row.eta * row.differential)
            if reference <= 0.0:
                continue
            if abs(row.delta_z_pred - row.eta * row.differential) > 1e-9 * reference:
                break
        else:
            continue
        break
    else:
        return True

    low, _ = heritability_range(reconciliations)
    damping = reconciliations[0].g_damping if reconciliations else 0.0
    if np.isfinite(low) and damping > 0.0 and (1.0 - low) <= 2.0 * damping:
        return True
    return False


def _bootstrap_slope(
    x: np.ndarray, y: np.ndarray, *, n_bootstrap: int, ci: float, seed: int
) -> tuple[float, float]:
    """Percentile interval on the slope, resampling whole step pairs.

    Declined below five step pairs, on the same argument `measure.ledger.explained` derives for
    `Λ`: a bootstrap over `K` clusters resolving a tail of mass `(1-ci)/2` needs at least
    `2/(1-ci)` distinct resamples, which at 95% is 40 and puts the floor at `K = 5`.
    """
    n = x.shape[0]
    if n_bootstrap <= 0 or n < 5:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(n_bootstrap, n))
    values = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = draws[i]
        values[i] = _through_origin(x[idx].ravel(), y[idx].ravel())[1]
    finite = values[np.isfinite(values)]
    if finite.size < 10:
        return float("nan"), float("nan")
    alpha = (1.0 - ci) / 2.0
    return float(np.quantile(finite, alpha)), float(np.quantile(finite, 1.0 - alpha))


def permuted_lande_null(
    reconciliations: Sequence[StepReconciliation],
    scales: Mapping[str, float],
    *,
    n_draws: int = 1000,
    seed: int = 0,
) -> np.ndarray:
    """The slope under permutations pairing each step's `Δz` with another step's predicted response.

    Lande's equation claims a **within-step** correspondence between what selection could reach and
    what moved. Under the null that no such correspondence exists, the assignment of `Δz` vectors to
    `η G β` vectors across steps is exchangeable, so this permutation is the exact null for the
    claim. The exchangeable unit is a whole step rather than a scalar, which is why the `k`-vectors
    move together. A derangement is not forced: forcing every step to move would make the null
    harder than the hypothesis it tests.
    """
    x, y, usable = _design(reconciliations, scales)
    if not usable or x.shape[0] < 2:
        return np.asarray([], dtype=np.float64)
    rng = np.random.default_rng(seed)
    out = np.empty(n_draws, dtype=np.float64)
    for i in range(n_draws):
        order = rng.permutation(x.shape[0])
        out[i] = _through_origin(x[order].ravel(), y.ravel())[1]
    return out


__all__ = [
    "CIRCULAR_METHODS",
    "CircularEstimator",
    "LandeFit",
    "fit_lande",
    "heritability_range",
    "permuted_lande_null",
]

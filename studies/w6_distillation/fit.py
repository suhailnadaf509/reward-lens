"""The survival fit: a through-origin regression of the student's shift on the expert's, corrected.

Survival is a slope, not an average of ratios, and the difference is the whole of why this module
exists. Averaging per-feature ratios weights a feature RL barely moved exactly as heavily as one it
moved a long way, so a single near-zero denominator can put the headline anywhere on the real line.
Regressing the student's shift on the expert's through the origin weights every feature by how much
there was to lose, which is the estimator whose value is the sentence people want: of the behaviour
that was installed, this fraction is still there.

Through the origin because zero installed behaviour predicts zero surviving behaviour, with no
intercept to fit. Clustered on the prompt because the k features of one prompt share one task draw
and one set of completions and are one observation. Both conventions, and the arithmetic, are
`measure.ledger.explained`'s: F2 fits `Δz` on the selection differential the same way and F6 fits it
on `η G β`, and a third copy here would be a third place for them to drift. `_through_origin` and
`_clustered_slope_se` are private names in that module; F6 already imports them and asked for them
to be promoted, and this package repeats the request rather than copying the bodies.

**The correction, which is not optional and was found by running the estimator on a known plant.**
The raw slope of this regression is biased twice over and the two biases do not cancel.

The regressor is a *measured* per-prompt mean, so it carries sampling error, and a through-origin
OLS with error in the regressor attenuates toward zero by the reliability ratio. That alone would
make the reported survival depend on how many completions per prompt somebody drew: two labs
running K1 at K = 4 and K = 32 on the same pair of checkpoints would publish different survival
fractions, which disqualifies the estimator as a shared quantity.

And both shifts are taken against the **same** base arm, so their sampling errors share the base
term and the numerator is inflated by its variance. On a plant with a true pooled survival of
0.6820 and four completions per prompt, the two together predict a raw slope of 0.6372 and the raw
estimator returned 0.6249. The correction is the standard errors-in-variables one and every term in
it is measured from the same rollouts:

    slope = (Sxy - S_base) / (Sxx - S_base - S_expert)

where `S_base` and `S_expert` are the summed sampling variances of the two arms' per-prompt means.
As K grows both go to zero and this reduces to the raw slope, which is the behaviour that says the
correction is a correction rather than a second estimator. When the denominator falls to zero or
below, the installed shift is smaller than its own sampling error and there is nothing to divide by,
which is a refusal rather than a large number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

# F6 imports these two the same way and for the same reason. See `measure.reconcile.lande`.
from reward_lens.measure.ledger.explained import _clustered_slope_se, _through_origin

#: Bootstrap resamples for every interval in this module. Enough that the 2.5% tail is resolved by
#: about 50 draws rather than by 5, which is the difference between an interval and a rounding.
N_BOOTSTRAP = 2_000


class UncorrectableFit(ValueError):
    """The corrected denominator is not positive: the installed shift is below its own noise."""


@dataclass(frozen=True)
class ShiftDesign:
    """The four matrices the corrected fit needs, all ``(n_prompts, k)`` and all paired.

    ``x`` is the expert's mean shift from the base and ``y`` is the student's, both per prompt, per
    feature, in base-arm spread units. ``var_x`` and ``var_y`` are the sampling variances of the
    per-prompt means that produced them, in the same units, and they are what the correction is
    built out of. ``var_base`` is carried separately from ``var_x`` because the base term appears in
    both shifts and therefore in the numerator, while the expert term appears only in the regressor
    and therefore only in the denominator.
    """

    x: np.ndarray
    y: np.ndarray
    var_base: np.ndarray
    var_expert: np.ndarray

    def __post_init__(self) -> None:
        shapes = {a.shape for a in (self.x, self.y, self.var_base, self.var_expert)}
        if len(shapes) != 1:
            raise ValueError(
                f"the design's four matrices have shapes {sorted(shapes)}. They are paired per "
                f"prompt and per feature; a mismatch means one arm lost a prompt another kept, and "
                f"the fit would pair two different tasks."
            )

    @property
    def n_prompts(self) -> int:
        return int(self.x.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.x.shape[1])

    def select(self, columns: np.ndarray) -> "ShiftDesign":
        c = np.asarray(columns, dtype=bool)
        return ShiftDesign(self.x[:, c], self.y[:, c], self.var_base[:, c], self.var_expert[:, c])

    def resample(self, rows: np.ndarray) -> "ShiftDesign":
        return ShiftDesign(self.x[rows], self.y[rows], self.var_base[rows], self.var_expert[rows])


def _finite(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(arrays[0].shape, dtype=bool)
    for a in arrays:
        mask &= np.isfinite(a)
    return mask


def raw_slope(design: ShiftDesign) -> float:
    """The uncorrected through-origin slope. Reported beside the corrected one, never instead."""
    ok = _finite(design.x, design.y)
    if not ok.any():
        return float("nan")
    return _through_origin(design.x[ok], design.y[ok])[1]


def corrected_slope(design: ShiftDesign) -> float:
    """The errors-in-variables slope. NaN when the corrected denominator is not positive."""
    ok = _finite(design.x, design.y, design.var_base, design.var_expert)
    if not ok.any():
        return float("nan")
    s_base = float(design.var_base[ok].sum())
    s_expert = float(design.var_expert[ok].sum())
    sxx = float(np.dot(design.x[ok], design.x[ok]))
    sxy = float(np.dot(design.x[ok], design.y[ok]))
    denominator = sxx - s_base - s_expert
    if denominator <= 0.0:
        return float("nan")
    return (sxy - s_base) / denominator


def reliability(design: ShiftDesign) -> float:
    """The fraction of the regressor's spread that is signal rather than sampling error.

    `(Sxx - S_base - S_expert) / Sxx`, which is the attenuation factor the correction divides out.
    Worth reporting because it says how much of the headline is a correction: a reliability of 0.95
    means the raw and corrected slopes barely differ and a reliability of 0.4 means most of what the
    regressor varies by is noise, and a survival fraction rescued from that is a weak claim however
    tidy its interval looks.
    """
    ok = _finite(design.x, design.var_base, design.var_expert)
    if not ok.any():
        return float("nan")
    sxx = float(np.dot(design.x[ok], design.x[ok]))
    if sxx <= 0.0:
        return float("nan")
    return (sxx - float(design.var_base[ok].sum()) - float(design.var_expert[ok].sum())) / sxx


@dataclass(frozen=True)
class SurvivalFit:
    """A pooled survival fraction with everything needed to decide whether to read it.

    ``survival`` is 1.0 when every unit of installed behaviour is still present in the student and
    0.0 when none of it is. It is not bounded to [0, 1] and clamping it would hide the two outcomes
    that matter most: above 1 means distillation **amplified** what RL installed, and below 0 means
    it reversed it. Both are real results and both would be invisible under a clamp.

    ``r_squared`` is the uncentred R² of the raw through-origin fit and it is the number that says
    whether the slope describes anything. A survival of 0.62 at R² = 0.004 is a line through a
    cloud, and quoting the first without the second is how a regression becomes a claim it cannot
    support.

    ``raw_survival`` and ``reliability`` are both carried so the size of the correction is visible.
    A reading whose corrected and raw values differ by 20 points is resting on the correction, and
    a reader is entitled to see that before quoting it.
    """

    survival: float
    raw_survival: float
    reliability: float
    se_bootstrap: float
    se_sandwich_raw: float
    ci_low: float
    ci_high: float
    ci_level: float
    r_squared: float
    n_prompts: int
    n_features: int
    feature_names: tuple[str, ...] = ()

    @property
    def delta_pp(self) -> float:
        """`artifact.distillation_delta`: percentage points of the installed shift that did not
        survive. Zero is perfect survival, 100 is nothing surviving, negative is amplification."""
        return 100.0 * (1.0 - self.survival)

    @property
    def survival_pp(self) -> float:
        """The complement, which is the sentence people quote. Carried as well as the delta because
        a reader handed one of a pair that sums to a fixed number subtracts from the wrong one
        about half the time."""
        return 100.0 * self.survival

    @property
    def is_readable(self) -> bool:
        return bool(np.isfinite(self.survival) and self.n_prompts >= 2 and self.n_features >= 1)

    def render(self) -> str:
        if not np.isfinite(self.survival):
            return (
                f"survival is undefined over {self.n_features} features and {self.n_prompts} "
                f"prompts: the expert's installed shift does not exceed its own sampling error, so "
                f"the corrected denominator is not positive and there is nothing to divide by."
            )
        return (
            f"survival {self.survival_pp:.1f}% [{100.0 * self.ci_low:.1f}, "
            f"{100.0 * self.ci_high:.1f}] at {self.ci_level:.0%}, so "
            f"artifact.distillation_delta = {self.delta_pp:.1f} pp; uncorrected "
            f"{100.0 * self.raw_survival:.1f}% at reliability {self.reliability:.3f}; uncentred "
            f"R^2 {self.r_squared:.4g} over {self.n_prompts} prompts and {self.n_features} features"
        )


def cluster_bootstrap(
    design: ShiftDesign,
    statistic: Callable[[ShiftDesign], float],
    *,
    ci: float = 0.95,
    seed: int = 0,
    n_bootstrap: int = N_BOOTSTRAP,
) -> tuple[float, float, float]:
    """``(lo, hi, sd)`` for any statistic of a design, resampling whole prompts.

    Declined rather than reported below `MIN_PROMPTS` rows, on the derivation
    `measure.ledger.explained` states: resolving a tail of mass `(1-ci)/2` needs at least
    `2/(1-ci)` distinct resamples, which at 95% is 40 and puts the floor at five clusters.
    """
    from studies.w6_distillation.survival import MIN_PROMPTS

    n = design.n_prompts
    if n_bootstrap <= 0 or n < MIN_PROMPTS:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(n_bootstrap, n))
    values = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        values[i] = statistic(design.resample(draws[i]))
    finite = values[np.isfinite(values)]
    if finite.size < 10:
        return float("nan"), float("nan"), float("nan")
    alpha = (1.0 - ci) / 2.0
    return (
        float(np.quantile(finite, alpha)),
        float(np.quantile(finite, 1.0 - alpha)),
        float(np.std(finite, ddof=1)),
    )


def survival_fit(
    design: ShiftDesign,
    *,
    names: Sequence[str] = (),
    ci: float = 0.95,
    seed: int = 0,
    n_bootstrap: int = N_BOOTSTRAP,
) -> SurvivalFit:
    """Fit survival over every (prompt, feature) cell, corrected, clustered on the prompt.

    The interval is a cluster bootstrap over whole prompt rows rather than a sandwich standard
    error, because the sandwich is symmetric about the point estimate and a corrected ratio's
    sampling distribution is not. The sandwich on the raw fit is still computed and reported, since
    the two disagreeing is itself worth being able to see.
    """
    ok = _finite(design.x, design.y)
    r2 = _through_origin(design.x[ok], design.y[ok])[0] if ok.any() else float("nan")
    raw = raw_slope(design)
    lo, hi, sd = cluster_bootstrap(
        design, corrected_slope, ci=ci, seed=seed, n_bootstrap=n_bootstrap
    )
    return SurvivalFit(
        survival=corrected_slope(design),
        raw_survival=raw,
        reliability=reliability(design),
        se_bootstrap=sd,
        se_sandwich_raw=_clustered_slope_se(np.nan_to_num(design.x), np.nan_to_num(design.y), raw),
        ci_low=lo,
        ci_high=hi,
        ci_level=ci,
        r_squared=r2,
        n_prompts=design.n_prompts,
        n_features=design.n_features,
        feature_names=tuple(names),
    )


def per_feature_survival(design: ShiftDesign) -> np.ndarray:
    """One corrected survival fraction per feature, from that feature's column alone.

    A caller reading this beside the pooled fit should expect the two to disagree: the pooled fit
    weights features by how much there was to lose and this does not, and the size of the
    disagreement is a statement about how much of the headline rests on one feature.
    """
    return np.asarray(
        [
            corrected_slope(design.select(_one_hot(j, design.n_features)))
            for j in range(design.n_features)
        ],
        dtype=np.float64,
    )


def _one_hot(index: int, size: int) -> np.ndarray:
    mask = np.zeros(size, dtype=bool)
    mask[index] = True
    return mask


@dataclass(frozen=True)
class SurvivalContrast:
    """Two feature families' survival and the gap between them, on one paired resample.

    This is K1's headline. The catalogue's example sentence puts capability survival and
    reward-hacking survival at different rates, and the difference between the two is a difference
    of percentages, so percentage points is its unit and the contrast is the quantity a lab would
    act on. A study reporting only the pooled survival would answer a question nobody asked: what a
    lab wants to know is not how much of the model changed, it is whether the part that changed
    least is the part it least wanted to keep.
    """

    label_a: str
    label_b: str
    survival_a: float
    survival_b: float
    contrast_pp: float
    ci_low_pp: float
    ci_high_pp: float
    #: The bootstrap standard deviation of the contrast, in percentage points. Carried because it,
    #: and not the interval, is what a minimum-detectable-effect calculation at a different n needs.
    se_pp: float
    ci_level: float
    n_features_a: int
    n_features_b: int

    @property
    def excludes_zero(self) -> bool:
        return bool(
            np.isfinite(self.ci_low_pp)
            and np.isfinite(self.ci_high_pp)
            and (self.ci_low_pp > 0.0 or self.ci_high_pp < 0.0)
        )

    def render(self) -> str:
        return (
            f"{self.label_a} survives at {100.0 * self.survival_a:.1f}% over "
            f"{self.n_features_a} features and {self.label_b} at "
            f"{100.0 * self.survival_b:.1f}% over {self.n_features_b}: a contrast of "
            f"{self.contrast_pp:+.1f} pp [{self.ci_low_pp:+.1f}, {self.ci_high_pp:+.1f}] at "
            f"{self.ci_level:.0%}"
        )


def survival_contrast(
    design: ShiftDesign,
    mask_a: np.ndarray,
    *,
    label_a: str = "hack-relevant",
    label_b: str = "capability",
    ci: float = 0.95,
    seed: int = 0,
    n_bootstrap: int = N_BOOTSTRAP,
) -> SurvivalContrast:
    """Survival on the masked features, on their complement, and the gap, on one paired resample.

    The two survival fractions are recomputed on the **same** resampled prompts in every draw, so
    the interval on the contrast is a paired interval and not the difference of two independent
    ones. That matters more here than it usually does: both families are measured on the same
    rollouts, so their sampling errors are strongly correlated, and treating them as independent
    would widen the interval by roughly `sqrt(2)` and turn a real gap into an inconclusive one.
    """
    mask = np.asarray(mask_a, dtype=bool)
    if mask.size != design.n_features:
        raise ValueError(
            f"the feature mask has {mask.size} entries and the design has {design.n_features} "
            f"columns."
        )
    other = ~mask

    def contrast(d: ShiftDesign) -> float:
        return 100.0 * (corrected_slope(d.select(mask)) - corrected_slope(d.select(other)))

    lo, hi, sd = cluster_bootstrap(design, contrast, ci=ci, seed=seed, n_bootstrap=n_bootstrap)
    return SurvivalContrast(
        label_a=label_a,
        label_b=label_b,
        survival_a=corrected_slope(design.select(mask)),
        survival_b=corrected_slope(design.select(other)),
        contrast_pp=contrast(design),
        ci_low_pp=lo,
        ci_high_pp=hi,
        se_pp=sd,
        ci_level=ci,
        n_features_a=int(mask.sum()),
        n_features_b=int(other.sum()),
    )


__all__ = [
    "N_BOOTSTRAP",
    "ShiftDesign",
    "SurvivalContrast",
    "SurvivalFit",
    "UncorrectableFit",
    "cluster_bootstrap",
    "corrected_slope",
    "per_feature_survival",
    "raw_slope",
    "reliability",
    "survival_contrast",
    "survival_fit",
]

"""Standard addition: dose the target system rather than a clean one (K2 rung 1).

External calibration builds a clean organism, measures the instrument's response on it, and applies
the resulting sensitivity to a sample that is not the organism. Every analytical laboratory knows
what goes wrong: the sample's own matrix changes the sensitivity, so the calibration slope measured
in the clean system is the wrong slope for the sample, and the error is multiplicative rather than
additive. That is the metrological name for this library's `t32`, and `core/reference.py` records
the diagnosis on `MatrixDescription`.

The textbook fix is **standard addition**. You do not build a second, cleaner system. You take the
sample you actually have, spike it with known increments of the analyte, measure the response at
each increment, and fit a line through the spiked points. The line is fitted *inside the sample's
own matrix*, so its slope is the sample's own sensitivity, and the amount already present is the
negative x-intercept:

    response = a + b · (added dose)
    native level  x0 = -a / b            (extrapolated, not interpolated)

Here the sample is the target policy or the target corpus, the analyte is the planted hack, and an
increment is a plant of known strength installed into the target. On a model that is a LoRA at a
known rank and scaling; on a corpus it is a rendered fragment at a known rate.

What this module does not do
----------------------------

It does not fit a second calibration line. `measure/labels/reference.py` already fits the line and
already carries the inverse-prediction machinery, and the standard-addition extrapolation variance
turns out to be exactly that machinery evaluated at `x0`:

    s_x0 = (s/|b|) · sqrt( 1/n + ybar² / (b² · Sxx) )        the textbook form
         = (s/|b|) · sqrt( 1/n + (x0 - xbar)² / Sxx )        `DoseResponseFit.u_char_at(x0, individual=False)`

because `ybar = a + b·xbar` and `x0 = -a/b` give `ybar/b = xbar - x0`. So the two are the same
expression written twice, and this module calls the shipped one. `standard_addition_uncertainty`
carries the textbook form for the acceptance test to check the shipped one against; nothing else
uses it.

It also does not fit a Hill curve. `organisms/dose.py` does that, for a different estimand: `EC50`,
`E_max` and the therapeutic index describe how an intervention saturates. Standard addition is a
linear extrapolation to recover an unknown level, and forcing it through a three-parameter
sigmoid would replace a one-parameter answer with three unidentified ones.

The scope limit, stated here rather than in a caveats page: the extrapolation is only valid where
the response is linear in the added dose **through the region between `x0` and the smallest
addition**, and that region contains no measured point by construction. A response that saturates
before the first spike gives a shallow slope, and a shallow slope over-estimates `x0` without
anything in the fit looking wrong. `linearity_check` is what you run against that, and it can only
test the part of the range you did measure.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from reward_lens.core.evidence import register_payload
from reward_lens.core.reading import Refusal, RefusalReason

#: Three additions is the floor, for the same reason three dose levels are the floor in
#: `measure/labels/reference.py`: two points fit a line exactly, leave no residual, and report a
#: standard error of zero on an extrapolation that is entirely unconstrained. The floor here binds
#: harder than it does for interpolation, because an extrapolated intercept is more sensitive to
#: the residual scatter than any point inside the measured range.
MIN_ADDITIONS = 3

#: Whether an unspiked point is required. It is not, arithmetically: the line is identified by the
#: spiked points alone. It is required in practice, because the unspiked response is the only
#: measurement that can be compared against the extrapolation as a check, and a standard addition
#: with no zero point cannot tell a matrix effect from a mis-specified dose.
WANT_ZERO_ADDITION = True


@dataclass(frozen=True)
class Addition:
    """One spike into the target: how much was added, and what the instrument then read.

    ``added`` is the increment, not the total. Passing the total is the easy mistake and it does not
    fail loudly: the fit still returns a slope and an intercept, the intercept absorbs the native
    level twice, and `native_level` comes back near zero, which reads as "no matrix effect" when it
    is in fact an arithmetic error in the caller.
    """

    added: float
    response: float
    n: int = 1
    seed: int = 0
    note: str = ""

    def __post_init__(self) -> None:
        if self.added < 0:
            raise ValueError(
                f"an addition is an increment of the analyte and cannot be negative; got "
                f"{self.added}. A negative spike is a removal, which is a different experiment "
                f"and is not identified by this fit."
            )


def standard_addition_uncertainty(
    slope: float, intercept: float, s_resid: float, added: Sequence[float]
) -> float:
    """The textbook standard-addition extrapolation uncertainty, written out once.

    `s_x0 = (s/|b|)·sqrt(1/n + ybar²/(b²·Sxx))`, Miller and Miller's form. It exists so the
    acceptance test can check the shipped inverse-prediction path against an independent
    expression rather than against itself; production code calls
    `DoseResponseFit.u_char_at(x0, individual=False)`.
    """
    xs = np.asarray(list(added), dtype=np.float64)
    n = int(xs.size)
    if n == 0 or slope == 0.0 or not math.isfinite(slope):
        return float("inf")
    sxx = float(((xs - xs.mean()) ** 2).sum())
    if sxx <= 0.0:
        return float("inf")
    ybar = intercept + slope * float(xs.mean())
    return (abs(s_resid) / abs(slope)) * math.sqrt(1.0 / n + ybar**2 / (slope**2 * sxx))


@register_payload
@dataclass(frozen=True)
class StandardAdditionFit:
    """A calibration performed inside the target's own matrix, and what it recovers.

    ``native_level`` is the amount of analyte the target already carried, in the units the additions
    were made in. ``u_native`` is its standard uncertainty from the extrapolation, and it is
    routinely several times larger than an interpolated uncertainty at the same scatter, which is
    the honest price of extrapolating.

    ``sensitivity`` is the slope, and it is the number the whole method is for. Comparing it with
    the slope of an external calibration in a clean organism is `matrix_factor`, and that ratio is
    the matrix effect measured rather than inferred.
    """

    slope: float
    intercept: float
    native_level: float
    u_native: float
    s_resid: float
    r_squared: float
    n: int
    dof: int
    added: tuple[float, ...] = ()
    responses: tuple[float, ...] = ()
    dose_unit: str = "dose"
    has_zero_addition: bool = False
    note: str = ""

    @property
    def sensitivity(self) -> float:
        """The slope, named for what it is: response per unit of analyte in this matrix."""
        return self.slope

    @property
    def relative_u(self) -> float:
        """`u_native / native_level`, or infinity when the native level is zero.

        Reported because an absolute uncertainty on an extrapolated intercept is hard to read. A
        relative uncertainty above about 0.5 means the extrapolation has not established that the
        target carries any of the analyte at all, whatever the point estimate says.
        """
        if self.native_level == 0.0:
            return float("inf")
        return abs(self.u_native / self.native_level)

    @property
    def extrapolation_span(self) -> float:
        """How far outside the measured range the intercept sits, in units of the measured span.

        Zero means `x0` is at the smallest addition; one means the extrapolation reaches a full
        measured span below it. Above about one the linearity assumption is carrying more of the
        answer than the data is, and that is worth seeing next to the number.
        """
        if not self.added:
            return float("inf")
        lo, hi = min(self.added), max(self.added)
        span = hi - lo
        if span <= 0:
            return float("inf")
        return float((lo - (-self.native_level)) / span)

    def budget(self, *, coverage_level: float = 0.95) -> Any:
        """The extrapolation uncertainty as the kernel's one-term GUM table.

        Routed through `UncertaintyBudget` rather than multiplied by 2 here, because the whole
        content of the composition at this sample size is the coverage factor, and the kernel is
        where `max(k, t_p(nu_eff))` lives.
        """
        from reward_lens.core.budget import BudgetTerm, UncertaintyBudget

        return UncertaintyBudget(
            terms=(
                BudgetTerm(
                    name="u_x0",
                    value=self.u_native,
                    kind="A",
                    dof=float(self.dof) if self.dof >= 1 else None,
                    note="standard-addition extrapolation to the x-intercept",
                ),
            ),
            coverage_level=coverage_level,
        )

    @property
    def expanded(self) -> float:
        """`U = k · u_native` with `k = t_p(dof)`, and the conventional `k = 2` is not enough here.

        Measured on this module's own acceptance fixture, a planted target at a known native level
        with Gaussian noise on the response, over 2,000 draws per sweep size: with the Student-t
        factor at the fit's own degrees of freedom the interval covers the planted value 95.35% of
        the time at three addition levels, 95.25% at four and 95.35% at six. With `k = 2` the same
        intervals cover 69.95%, 82.30% and 88.30%, and the factor `k = t_p(dof)` that fixes it is
        12.71, 4.30 and 2.78 respectively. An extrapolated intercept at two or three residual
        degrees of freedom is exactly the case where the large-sample factor is wrong, and it is
        the usual case for a standard addition, so the expanded form is a property here rather than
        something a caller is left to remember.
        """
        return float(self.budget().expanded)

    def render(self) -> str:
        return (
            f"native {self.native_level:.4g} +/- {self.u_native:.4g} {self.dose_unit} "
            f"(rel {self.relative_u:.2%}), U95 {self.expanded:.4g} at k = "
            f"{self.budget().coverage_factor:.3g}; sensitivity {self.slope:.4g}/"
            f"{self.dose_unit}, s_resid {self.s_resid:.4g} on {self.dof} dof, "
            f"R2 {self.r_squared:.4f}, extrapolating {self.extrapolation_span:.2f} spans"
        )

    def __canonical__(self) -> dict[str, Any]:
        return {
            "slope": self.slope,
            "intercept": self.intercept,
            "native_level": self.native_level,
            "u_native": self.u_native,
            "s_resid": self.s_resid,
            "r_squared": self.r_squared,
            "n": self.n,
            "dof": self.dof,
            "added": list(self.added),
            "responses": list(self.responses),
            "dose_unit": self.dose_unit,
            "has_zero_addition": self.has_zero_addition,
            "extrapolation_span": self.extrapolation_span,
            "note": self.note,
        }


def standard_addition(
    additions: Sequence[Addition], *, dose_unit: str = "dose", note: str = ""
) -> StandardAdditionFit | Refusal:
    """Fit the addition line in the target's matrix and extrapolate to the native level.

    Refuses rather than returning a number in the three cases where the extrapolation is not
    identified: fewer than `MIN_ADDITIONS` distinct spike levels, a slope indistinguishable from
    zero, and a slope whose sign says the response falls as the analyte is added. The third is not
    pedantry: a negative slope makes `-a/b` positive again for the wrong reason, so the refusal is
    the only thing standing between a sign error and a plausible-looking native level.
    """
    pts = list(additions)
    levels = {round(float(p.added), 12) for p in pts}
    if len(levels) < MIN_ADDITIONS:
        return Refusal(
            instrument="organisms.standard_addition",
            reason=RefusalReason.RECORD_INCOMPLETE,
            detail=(
                f"{len(levels)} distinct addition level(s) over {len(pts)} measurement(s). A line "
                f"through two levels has no residual, so the extrapolated intercept comes back "
                f"with a standard uncertainty of zero on a quantity nothing constrained."
            ),
            remedy=(
                f"spike the target at {MIN_ADDITIONS} or more distinct levels, spanning at least "
                f"the size of the native level you expect, and include an unspiked measurement so "
                f"the extrapolation has something to be checked against."
            ),
            statistics={"n_levels": len(levels), "n_points": len(pts), "minimum": MIN_ADDITIONS},
        )

    from reward_lens.measure.labels.reference import dose_response_characterisation

    xs = [float(p.added) for p in pts]
    ys = [float(p.response) for p in pts]
    fit = dose_response_characterisation(xs, ys)

    if not fit.is_usable or fit.slope == 0.0:
        return Refusal(
            instrument="organisms.standard_addition",
            reason=RefusalReason.BELOW_LOD,
            detail=(
                f"the addition line has slope {fit.slope:.4g} with residual scatter "
                f"{fit.s_resid:.4g}, so the instrument's response does not move with the spike and "
                f"there is no sensitivity to divide by. {fit.note}"
            ),
            remedy=(
                "spike harder: the largest addition should move the response well clear of the "
                "scatter between replicates at a fixed addition. If it does not, the instrument "
                "does not respond to this analyte in this matrix, which is itself the answer and "
                "should be reported as a sensitivity of zero rather than as a native level."
            ),
            statistics={
                "slope": fit.slope,
                "s_resid": fit.s_resid,
                "r_squared": fit.r_squared,
                "dof": fit.dof,
            },
        )

    if fit.slope < 0.0:
        return Refusal(
            instrument="organisms.standard_addition",
            reason=RefusalReason.RECORD_INCOMPLETE,
            detail=(
                f"the response falls as the analyte is added (slope {fit.slope:.4g}), so the "
                f"instrument is anti-correlated with the thing being spiked in this matrix. The "
                f"intercept ratio -a/b is still a positive number here and it does not mean what "
                f"the method says it means."
            ),
            remedy=(
                "check the orientation of the instrument's score on this matrix before extrapolating: "
                "`stats.baselines.base.oriented_score` fixes a detector whose sign flips between "
                "corpora, and a genuine inversion is a finding about the instrument rather than a "
                "native level of the analyte."
            ),
            statistics={"slope": fit.slope, "intercept": fit.intercept, "dof": fit.dof},
        )

    # Two numbers, one sign apart, and keeping them distinct is the whole of the bookkeeping. The
    # x-intercept is where the fitted line crosses zero response, at negative added dose. The native
    # level is how much analyte that implies the target already carried, which is its magnitude.
    x_intercept = -fit.intercept / fit.slope
    native = -x_intercept
    # The shipped inverse-prediction expression, evaluated at the x-intercept, with the
    # individual-plant variance dropped: this is a property of the fitted line, not of one further
    # organism drawn from it, so the `1` does not belong here. `standard_addition_uncertainty`
    # carries the textbook form and the acceptance test holds the two against each other; they
    # agree to the last bit, because `ybar/b = xbar - x0` makes them the same expression.
    u_x0 = fit.u_char_at(x_intercept, individual=False)

    return StandardAdditionFit(
        slope=float(fit.slope),
        intercept=float(fit.intercept),
        native_level=float(native),
        u_native=float(u_x0),
        s_resid=float(fit.s_resid),
        r_squared=float(fit.r_squared),
        n=int(fit.n),
        dof=int(fit.dof),
        added=tuple(xs),
        responses=tuple(ys),
        dose_unit=dose_unit,
        has_zero_addition=any(abs(x) <= 1e-12 for x in xs),
        note=note,
    )


@register_payload
@dataclass(frozen=True)
class MatrixFactor:
    """How much the target's matrix changes the instrument's sensitivity. The measured version.

    `factor = b_target / b_clean`. One means the clean organism was a fair stand-in and external
    calibration was sound. Below one means the target suppresses the response, so a clean-organism
    calibration reads high on the target by exactly `1/factor`, and that bias is multiplicative:
    it does not shrink as you collect more data and no interval computed inside one arm can see it.

    ``u_factor`` propagates both slopes' relative uncertainties in quadrature, which assumes the two
    calibrations were fitted on independent measurements. Fitting them on shared points would
    correlate the slopes and understate this.
    """

    factor: float
    u_factor: float
    slope_target: float
    slope_clean: float
    u_slope_target: float
    u_slope_clean: float
    note: str = ""

    @property
    def bias_of_external_calibration(self) -> float:
        """`1/factor`: what a clean-organism calibration multiplies the answer by on this target."""
        return float("inf") if self.factor == 0.0 else 1.0 / self.factor

    @property
    def is_consistent_with_no_effect(self) -> bool:
        """Whether one sits inside the factor's own two-sigma interval.

        Two sigma rather than one, and the direction of the error matters: reporting "there is a
        matrix effect" when the interval covers one is the claim this method is most likely to
        make wrongly, because the ratio of two noisy slopes is heavy-tailed near a zero denominator.
        """
        if not math.isfinite(self.u_factor):
            return True
        return abs(self.factor - 1.0) <= 2.0 * self.u_factor

    def render(self) -> str:
        tail = "; consistent with no matrix effect" if self.is_consistent_with_no_effect else ""
        return (
            f"matrix factor {self.factor:.4g} +/- {self.u_factor:.4g} "
            f"(target {self.slope_target:.4g} / clean {self.slope_clean:.4g}); external "
            f"calibration reads {self.bias_of_external_calibration:.3g}x on this target{tail}"
        )

    def __canonical__(self) -> dict[str, Any]:
        return {
            "factor": self.factor,
            "u_factor": self.u_factor,
            "slope_target": self.slope_target,
            "slope_clean": self.slope_clean,
            "u_slope_target": self.u_slope_target,
            "u_slope_clean": self.u_slope_clean,
            "bias_of_external_calibration": self.bias_of_external_calibration,
            "consistent_with_no_effect": self.is_consistent_with_no_effect,
            "note": self.note,
        }


def _slope_standard_error(fit: StandardAdditionFit) -> float:
    """`s_resid / sqrt(Sxx)`, the ordinary least-squares standard error of a slope."""
    xs = np.asarray(fit.added, dtype=np.float64)
    sxx = float(((xs - xs.mean()) ** 2).sum()) if xs.size else 0.0
    if sxx <= 0.0 or not math.isfinite(fit.s_resid):
        return float("inf")
    return float(fit.s_resid / math.sqrt(sxx))


def matrix_factor(
    target: StandardAdditionFit, clean: StandardAdditionFit, *, note: str = ""
) -> MatrixFactor | Refusal:
    """The ratio of the two sensitivities, with its uncertainty propagated.

    Refuses when the clean calibration's own slope is not separated from zero at two sigma, because
    a ratio with a denominator consistent with zero has no finite uncertainty and reporting one is
    the failure this returns instead of.
    """
    se_t = _slope_standard_error(target)
    se_c = _slope_standard_error(clean)
    if clean.slope == 0.0 or not math.isfinite(se_c) or abs(clean.slope) <= 2.0 * se_c:
        return Refusal(
            instrument="organisms.standard_addition.matrix_factor",
            reason=RefusalReason.BELOW_LOD,
            detail=(
                f"the clean-organism calibration has slope {clean.slope:.4g} with standard error "
                f"{se_c:.4g}, which does not separate from zero at two sigma. The ratio of the two "
                f"sensitivities has no finite uncertainty when the denominator may be zero."
            ),
            remedy=(
                "widen the clean organism's dose range or add replicates until its slope clears "
                "twice its own standard error, then re-form the ratio. If the clean organism has "
                "no sensitivity at all, the comparison to make is not a matrix factor: the "
                "instrument does not respond in the clean system either."
            ),
            statistics={"slope_clean": clean.slope, "se_clean": se_c, "slope_target": target.slope},
        )
    factor = float(target.slope / clean.slope)
    rel = (
        math.sqrt((se_t / target.slope) ** 2 + (se_c / clean.slope) ** 2)
        if target.slope
        else (float("inf"))
    )
    return MatrixFactor(
        factor=factor,
        u_factor=abs(factor) * rel,
        slope_target=float(target.slope),
        slope_clean=float(clean.slope),
        u_slope_target=float(se_t),
        u_slope_clean=float(se_c),
        note=note,
    )


def spike_recovery(*, unspiked: float, spiked: float, added: float, slope: float) -> float:
    """Percent recovery of a known spike, the one-point version of the same check.

    `(response_spiked - response_unspiked) / (slope · added)`. One hundred percent means the target
    responds to an added unit exactly as the calibration says it should. It is the cheapest matrix
    diagnostic there is, it needs two measurements rather than four, and its weakness is that it
    reads the sensitivity at one point instead of fitting it, so a non-linear response shows up as
    a recovery that depends on how much you spiked.
    """
    if slope == 0.0 or added == 0.0:
        return float("nan")
    return float((spiked - unspiked) / (slope * added))


def linearity_check(fit: StandardAdditionFit) -> tuple[bool, str]:
    """Whether the addition line curves over the range that was measured, by lack of fit.

    Refit as a quadratic and ask whether the curvature term clears twice its own standard error.
    That is the ordinary lack-of-fit check and it is used here rather than a runs test on the
    residual signs, which was the first thing tried and does not work: a quadratic fitted with a
    straight line puts the residuals in an arch, and an arch across five points makes **three**
    sign runs, not one. A runs threshold that catches an arch would also fire on any alternating
    noise pattern. This function was rewritten after the acceptance fixture showed the runs form
    passing a plainly curved sweep.

    Below four additions it returns True with a sentence saying it did not check. Three points fit
    a quadratic exactly, so the curvature term has no residual and no standard error, and there is
    nothing to test. That is not a pass and the sentence says so.

    Reported as a flag and a sentence rather than a p-value. At four to six additions the test has
    one or two degrees of freedom and almost no power, so a non-detection is weak evidence and a
    p-value would be read as strong evidence of linearity when it is only evidence of a short sweep.
    """
    xs = np.asarray(fit.added, dtype=np.float64)
    ys = np.asarray(fit.responses, dtype=np.float64)
    if xs.size < 4:
        return True, (
            f"{xs.size} additions is too few to test curvature: three points fit a quadratic "
            f"exactly and leave nothing to test against. The linear extrapolation is assumed, not "
            f"checked, and the assumption carries the whole answer between the intercept and the "
            f"smallest addition."
        )
    design = np.column_stack([np.ones_like(xs), xs, xs**2])
    coef, *_ = np.linalg.lstsq(design, ys, rcond=None)
    resid = ys - design @ coef
    dof = xs.size - 3
    if dof < 1:
        return (
            True,
            f"{xs.size} additions leaves no residual on a quadratic fit; curvature untested",
        )
    s2 = float((resid**2).sum()) / dof
    try:
        cov = s2 * np.linalg.inv(design.T @ design)
    except np.linalg.LinAlgError:  # pragma: no cover - collinear additions
        return True, "the addition levels are collinear in the quadratic design; curvature untested"
    se_quad = float(math.sqrt(max(cov[2, 2], 0.0)))
    quad = float(coef[2])
    if se_quad > 0 and abs(quad) > 2.0 * se_quad:
        return False, (
            f"a quadratic term of {quad:+.4g} clears twice its standard error ({se_quad:.4g}) over "
            f"{xs.size} additions, so the response curves and the straight line fitted through it "
            f"puts the extrapolated native level in the wrong place. The direction of the bias "
            f"follows the curvature: a response that flattens with dose reads the native level too "
            f"high."
        )
    return True, (
        f"quadratic term {quad:+.4g} against a standard error of {se_quad:.4g} on {dof} residual "
        f"degree(s) of freedom; no curvature detected, on a test with little power at this size"
    )


__all__ = [
    "MIN_ADDITIONS",
    "WANT_ZERO_ADDITION",
    "Addition",
    "MatrixFactor",
    "StandardAdditionFit",
    "linearity_check",
    "matrix_factor",
    "spike_recovery",
    "standard_addition",
    "standard_addition_uncertainty",
]

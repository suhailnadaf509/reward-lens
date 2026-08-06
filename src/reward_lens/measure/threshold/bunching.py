"""I2: how hard the policy is pushing on the gate, as an elasticity.

I1 says whether the density jumps. It does not say how much effort produced the jump, and it cannot
be compared across two gates on different running variables or across two runs at different scales.
An excess mass of 400 rollouts means one thing at a cutoff of 512 tokens and another at 8192.

The bunching literature solved this for tax notches thirty years ago and the translation is exact.
Rung 0 is the excess mass `b = B / h0(c)`: observed mass in a window below the cutoff, over the
counterfactual density there, with the counterfactual fitted by a polynomial that **excludes** the
window. Expressed in units of the running variable that is `Delta z*`, the distance the marginal
buncher moved. Rung 1 divides it by the cutoff and by the fractional score drop at the gate, which
turns it into a Saez elasticity and makes adversarial effort comparable across gates and runs.

The two shapes are not interchangeable, and using the wrong one is a factor of `Delta z* / z*` in
the answer:

- a **notch**, which is what an `Override` is, drops the score discontinuously at the cutoff, and
  Kleven's reduced form is ``e = (Delta z*/z*)^2 / (2 * Delta rho)``
- a **kink**, which is what a `Piecewise` node with a slope change is, keeps the score continuous
  and changes its slope, and Saez's form is ``e = (Delta z*/z*) / Delta rho``

where ``Delta rho`` is the fractional score drop on crossing, the reward analogue of the change in
the net-of-tax rate.

**The estimator's own failure mode is that it measures the window rather than the behaviour.** A
polynomial fitted around a hole will report excess mass at any cutoff you point it at if the hole
is wide enough and the degree is high enough. Three things guard against that and all three are on
every reading: the placebo baseline, which runs the identical estimator at cutoffs where no gate
is, the window sensitivity sweep, which reports the estimate at every window width so a reader can
see whether the answer is the window, and `gate_response`, which moves the true gate and checks
that the estimate follows it. The third is the kill condition from the catalogue record: if the
elasticity does not respond when the gate is moved, this instrument is not measuring the gate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from reward_lens.core.evidence import register_payload
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.quantity import BaselineID
from reward_lens.core.reading import Refusal, RefusalReason, refuse_incomplete
from reward_lens.core.types import Capability, GaugeStatus
from reward_lens.measure.threshold._base import (
    ALL_SUBSTRATES,
    GATE_ACCESS,
    GATE_ENVELOPE,
    RECORD_PHASES,
    ThresholdInstrument,
)
from reward_lens.measure.threshold.density import NullBand, automatic_binsize, histogram
from reward_lens.measure.threshold.gates import DecodeLength, Gate, RunningVariable

#: The catalogue names one baseline for I2. Two more are added because the estimator's own failure
#: mode needs them, and all three run on every reading. The smooth-density null is the one that
#: bites when the cutoff sits near the mode of the running variable, where a polynomial fitted
#: around a hole under-fits the curvature and manufactures excess mass at any window width. A
#: placebo cutoff cannot catch that, because a placebo is by construction somewhere else.
BUNCHING_BASELINES: tuple[BaselineID, ...] = (
    "baseline.placebo_cutoff",
    "baseline.smooth_density_null",
    "baseline.window_sensitivity",
)

#: The polynomial degree the bunching literature uses by default. High enough to follow a skewed
#: density, low enough that it does not chase the hole it was fitted around.
DEFAULT_DEGREE = 7

#: How many bins on each side of the cutoff are excluded from the counterfactual fit when the
#: window is not chosen from the data. Reported on every reading and swept, because it is the one
#: free choice in the estimator and the reader has to be able to see what it bought.
DEFAULT_WINDOW_BINS = 3

#: The widest window `auto_window` will grow to, as a multiple of the binwidth. A pile that has not
#: ended by here is not a pile, it is the shape of the density.
MAX_WINDOW_BINS = 40

#: The exclusion widths `baseline.window_sensitivity` sweeps, in bins. Spaced out rather than
#: consecutive because the question the sweep answers is whether the excess mass survives a window
#: several times wider than the one that was used, and a run of adjacent widths answers a narrower
#: question at more cost. Named here rather than written into the signature twice, because the
#: function default and the instrument default drifting apart is exactly the defect being fixed.
DEFAULT_SWEEP_BINS: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16, 24)


# ---------------------------------------------------------------------------
# the counterfactual
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Counterfactual:
    """The density that would have been there without the gate, and what it leaves over."""

    midpoints: np.ndarray
    observed: np.ndarray
    counterfactual: np.ndarray
    in_window: np.ndarray
    binsize: float
    cutoff: float
    degree: int
    window_bins: int
    excess: float
    missing: float
    density_at_cutoff: float
    iterations: int
    converged: bool


def counterfactual_density(
    x: Sequence[float] | np.ndarray,
    cutoff: float,
    *,
    binsize: float | None = None,
    degree: int = DEFAULT_DEGREE,
    window_bins: int = DEFAULT_WINDOW_BINS,
    bunching_side: str = "below",
    integration: bool = True,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> Counterfactual:
    """Fit the counterfactual bin counts, excluding a window, under the integration constraint.

    The integration constraint is what stops the counterfactual absorbing the bunching. Mass that
    piled up below the cutoff came from somewhere, so a counterfactual fitted to the observed
    counts above the cutoff is fitted to a distribution that is already missing that mass, and the
    estimate is biased downward. The standard fix is to iterate: estimate the excess, inflate the
    observed counts above the window by the same total, refit, and repeat until the excess stops
    moving. Ten iterations is usually enough and fifty is the cap.
    """
    v = np.asarray(x, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    b = binsize if binsize is not None else automatic_binsize(v)
    hist = histogram(v, cutoff, b)
    mid = hist.midpoints
    counts = hist.counts.astype(np.float64).copy()
    lo = cutoff - window_bins * hist.binsize
    hi = cutoff + window_bins * hist.binsize
    in_window = (mid > lo - hist.binsize / 2.0) & (mid < hi + hist.binsize / 2.0)
    above_window = mid >= hi
    below = mid < cutoff
    scale = float(np.max(np.abs(mid - cutoff))) or 1.0

    def fit(y: np.ndarray) -> np.ndarray:
        keep = ~in_window
        if int(keep.sum()) <= degree + 1:
            return np.full_like(y, np.nan)
        design = np.vander((mid - cutoff) / scale, degree + 1, increasing=True)
        coef, *_ = np.linalg.lstsq(design[keep], y[keep], rcond=None)
        return np.asarray(design @ coef, dtype=np.float64)

    working = counts.copy()
    excess = 0.0
    iterations = 0
    converged = not integration
    fitted = fit(working)
    for iterations in range(1, int(max_iter) + 1):
        fitted = fit(working)
        side = in_window & (below if bunching_side == "below" else ~below)
        new_excess = float(np.sum(counts[side] - fitted[side]))
        if not integration:
            excess = new_excess
            break
        mass_above = float(np.sum(fitted[above_window]))
        if mass_above <= 0:
            excess = new_excess
            converged = False
            break
        if abs(new_excess - excess) <= tol * max(1.0, abs(excess)):
            excess = new_excess
            converged = True
            break
        excess = new_excess
        working = counts.copy()
        working[above_window] = counts[above_window] * (1.0 + excess / mass_above)

    side = in_window & (below if bunching_side == "below" else ~below)
    other = in_window & ~side
    missing = float(np.sum(fitted[other] - counts[other]))
    at_cutoff_idx = int(np.argmin(np.abs(mid - (cutoff - hist.binsize / 2.0))))
    density_at_cutoff = float(fitted[at_cutoff_idx]) if fitted.size else float("nan")
    return Counterfactual(
        midpoints=mid,
        observed=counts,
        counterfactual=fitted,
        in_window=in_window,
        binsize=float(hist.binsize),
        cutoff=float(cutoff),
        degree=int(degree),
        window_bins=int(window_bins),
        excess=float(excess),
        missing=missing,
        density_at_cutoff=density_at_cutoff,
        iterations=int(iterations),
        converged=bool(converged),
    )


def auto_window(
    x: Sequence[float] | np.ndarray,
    cutoff: float,
    *,
    binsize: float | None = None,
    degree: int = DEFAULT_DEGREE,
    bunching_side: str = "below",
    max_bins: int = MAX_WINDOW_BINS,
    passes: int = 3,
) -> int:
    """Grow the exclusion window until the pile ends, with no free constant to pick.

    The window is the estimator's one real choice and picking it by eye is how a bunching estimate
    becomes whatever the analyst wanted. The rule here has nothing to tune: starting from one bin,
    extend the window while the outermost bin on the bunching side is still above the
    counterfactual, and stop at the first bin that is not. That is "extend until the excess
    vanishes" stated so that it terminates.

    Because the counterfactual depends on the window, the rule is iterated a few times from its own
    answer. It converges in two or three passes on every density this was run on, and the returned
    width is on the reading so a reader can compare it against the sweep.
    """
    v = np.asarray(x, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    b = binsize if binsize is not None else automatic_binsize(v)
    width = 1
    for _ in range(int(passes)):
        cf = counterfactual_density(
            v,
            cutoff,
            binsize=b,
            degree=degree,
            window_bins=width,
            bunching_side=bunching_side,
            integration=False,
        )
        below = cf.midpoints < cutoff
        side = below if bunching_side == "below" else ~below
        order = np.argsort(np.abs(cf.midpoints - cutoff))
        grown = 0
        for idx in order:
            if not side[idx]:
                continue
            if cf.observed[idx] <= cf.counterfactual[idx]:
                break
            grown += 1
            if grown >= max_bins:
                break
        new_width = max(1, min(int(max_bins), grown))
        if new_width == width:
            return width
        width = new_width
    return width


# ---------------------------------------------------------------------------
# the elasticity
# ---------------------------------------------------------------------------


def saez_elasticity(dz_star: float, cutoff: float, penalty_fraction: float, kind: str) -> float:
    """Convert a bunching distance into an elasticity.

    ``dz_star`` is the marginal buncher's response in units of the running variable, ``cutoff`` is
    the gate, and ``penalty_fraction`` is the fractional score drop on crossing it. A notch and a
    kink of the same size imply elasticities that differ by a factor of ``dz_star / cutoff``, which
    is why `Gate.kind` is a declared field rather than an assumption.
    """
    if cutoff <= 0 or penalty_fraction <= 0 or not math.isfinite(dz_star):
        return float("nan")
    relative = dz_star / cutoff
    if kind == "kink":
        return float(relative / penalty_fraction)
    return float(relative * relative / (2.0 * penalty_fraction))


def smooth_null_excess(
    x: Sequence[float] | np.ndarray,
    cutoff: float,
    *,
    observed_excess: float,
    binsize: float,
    degree: int,
    window_bins: int,
    bunching_side: str,
    integration: bool,
    n_draws: int = 200,
    fit_degree: int = 7,
    seed: int = 0,
) -> NullBand:
    """How much excess mass this estimator manufactures at this cutoff on a smooth density.

    The construction is I1's: one polynomial fitted to the log of the binned counts across the
    whole support, which cannot be discontinuous anywhere, sampled at the same n. Running the
    bunching estimator on those samples **at the real cutoff** answers the question a placebo
    cannot, because a placebo cutoff is by construction somewhere the real one is not, and the
    estimator's bias depends on where the cutoff sits relative to the mode.

    This matters. On a Gaussian running variable with the cutoff at the mode, a degree-7
    counterfactual fitted around a hole under-fits the peak curvature and reports excess mass with
    a bootstrap z near 4 when nothing is there. The placebo baseline gave that an empirical p of
    zero. This null gives it a p near a half, which is the right answer.
    """
    v = np.asarray(x, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    n = v.size
    hist = histogram(v, cutoff, binsize)
    if n < 50 or hist.counts.size < fit_degree + 2:
        return NullBand(
            "smooth-density null",
            0,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            detail="too few observations or bins to fit a smooth null",
        )
    u = hist.midpoints - cutoff
    scale = float(np.max(np.abs(u))) or 1.0
    design = np.vander(u / scale, fit_degree + 1, increasing=True)
    coef, *_ = np.linalg.lstsq(design, np.log(hist.counts + 0.5), rcond=None)
    weights = np.exp(design @ coef)
    total = float(weights.sum())
    if not math.isfinite(total) or total <= 0:
        return NullBand(
            "smooth-density null",
            0,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            detail="the smooth fit did not produce a usable density",
        )
    pmf = weights / total
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(int(n_draws)):
        picked = rng.choice(hist.midpoints.size, size=n, replace=True, p=pmf)
        sample = hist.midpoints[picked] + rng.uniform(-binsize / 2.0, binsize / 2.0, size=n)
        e, _, _ = _excess_at(
            sample,
            cutoff,
            binsize=binsize,
            degree=degree,
            window_bins=window_bins,
            bunching_side=bunching_side,
            integration=integration,
        )
        if math.isfinite(e):
            draws.append(e)
    if len(draws) < 20:
        return NullBand(
            "smooth-density null",
            len(draws),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            detail="too few finite draws to form a null",
        )
    arr = np.asarray(draws, dtype=np.float64)
    p = (
        float(np.mean(np.abs(arr) >= abs(observed_excess)))
        if math.isfinite(observed_excess)
        else float("nan")
    )
    return NullBand(
        label="smooth-density null",
        n_draws=len(draws),
        mean=float(arr.mean()),
        sd=float(arr.std(ddof=1)),
        q95_abs=float(np.percentile(np.abs(arr), 95)),
        p_empirical=p,
        detail=(
            f"excess mass at the same cutoff on {len(draws)} samples of {n} drawn from a "
            f"degree-{fit_degree} log-density fitted across the whole support"
        ),
    )


@register_payload
@dataclass(frozen=True)
class WindowSweep:
    """The estimate at every window width. What separates behaviour from the window."""

    window_bins: tuple[int, ...] = ()
    excess_mass: tuple[float, ...] = ()
    dz_star: tuple[float, ...] = ()
    elasticity: tuple[float, ...] = ()

    @property
    def plateaued(self) -> bool:
        """Whether the excess mass stops growing before the widest window in the sweep.

        A pile that has ended shows up as an excess mass that flattens: widening the window past
        the pile adds bins whose observed and counterfactual counts agree, so the estimate stops
        moving. A sweep that is still climbing at the widest window has not covered the pile, and
        the estimate is a lower bound rather than an estimate. Both readings are useful and they
        are different, which is why this is reported rather than resolved.
        """
        vals = [b for b in self.excess_mass if math.isfinite(b)]
        if len(vals) < 3:
            return False
        peak = max(abs(v) for v in vals)
        if peak <= 0:
            return False
        return abs(vals[-1] - vals[-2]) < 0.05 * peak

    def render(self) -> str:
        pairs = ", ".join(
            f"{w}:{b:.1f}" for w, b in zip(self.window_bins, self.excess_mass) if math.isfinite(b)
        )
        verdict = "plateaued" if self.plateaued else "still climbing at the widest window"
        return f"window sensitivity ({verdict}) bins:B {pairs}"


@register_payload
@dataclass(frozen=True)
class BunchingReading:
    """Excess mass at the gate, converted to a comparable measure of adversarial effort."""

    gate: Gate
    running: str
    unit: str
    n: int
    cutoff: float
    binsize: float
    window_bins: int
    degree: int
    excess_mass: float
    missing_mass: float
    normalised_excess: float
    dz_star: float
    elasticity: float | None
    se_excess: float
    se_elasticity: float | None
    rung: int
    integration_converged: bool
    iterations: int
    placebo_excess: tuple[float, ...] = ()
    placebo_cutoffs: tuple[float, ...] = ()
    placebo_p: float = float("nan")
    smooth_null: NullBand | None = None
    sweep: WindowSweep | None = None
    window_chosen_from_data: bool = False
    decode: DecodeLength | None = None

    @property
    def says(self) -> str:
        head = (
            f"excess mass of {self.excess_mass:.1f} rollouts below the gate, "
            f"{self.normalised_excess:.2f} times the counterfactual density there, so the marginal "
            f"buncher moved {self.dz_star:.3g} {self.unit} "
            f"({100 * self.dz_star / self.cutoff:.2f}% of the cutoff)"
        )
        if self.elasticity is None or not math.isfinite(self.elasticity):
            return f"{head}. No elasticity: the gate does not state what crossing it costs"
        looser = 0.10
        return (
            f"{head}, which implies an elasticity of {self.elasticity:.3g}: a {looser:.0%} looser "
            f"gate buys {100 * self.elasticity * looser:.2f}% more {self.unit}"
        )

    def render(self) -> str:
        chosen = "chosen from the data" if self.window_chosen_from_data else "supplied"
        lines = [
            f"I2 bunching  {self.gate.render()}",
            f"  {self.says}",
            f"  B = {self.excess_mass:.2f} +/- {self.se_excess:.2f} rollouts, missing mass "
            f"{self.missing_mass:.2f}, b = B/h0(c) = {self.normalised_excess:.3f} bins, "
            f"dz* = {self.dz_star:.4g} {self.unit}",
            f"  n = {self.n}, binsize {self.binsize:.4g}, window {self.window_bins} bins each "
            f"side ({chosen}), degree {self.degree}, integration "
            f"{'converged' if self.integration_converged else 'did not converge'} in "
            f"{self.iterations} iterations, rung {self.rung}",
        ]
        if self.placebo_excess:
            lines.append(
                f"  baseline placebo cutoff: {len(self.placebo_excess)} cutoffs, "
                f"|B| median {np.median(np.abs(self.placebo_excess)):.2f}, "
                f"empirical p {self.placebo_p:.4g}"
            )
        if self.smooth_null is not None:
            lines.append(f"  baseline {self.smooth_null.render()}")
        if self.sweep is not None:
            lines.append(f"  baseline {self.sweep.render()}")
        if self.decode is not None:
            lines.append(f"  {self.decode.render()}")
        if self.gate.installed:
            lines.append(
                "  the gate is installed, so this is the estimator's reading of this density at "
                "this cutoff. The policy never optimised against this gate and cannot have bunched "
                "at it."
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# the estimator with its refusals
# ---------------------------------------------------------------------------


def _excess_at(
    v: np.ndarray,
    cutoff: float,
    *,
    binsize: float,
    degree: int,
    window_bins: int,
    bunching_side: str,
    integration: bool,
) -> tuple[float, float, float]:
    """Excess mass, normalised excess and dz*, at one cutoff. The inner loop of every baseline."""
    cf = counterfactual_density(
        v,
        cutoff,
        binsize=binsize,
        degree=degree,
        window_bins=window_bins,
        bunching_side=bunching_side,
        integration=integration,
    )
    if not math.isfinite(cf.density_at_cutoff) or cf.density_at_cutoff <= 0:
        return cf.excess, float("nan"), float("nan")
    normalised = cf.excess / cf.density_at_cutoff
    return cf.excess, normalised, normalised * cf.binsize


def bunching_elasticity(
    running: RunningVariable,
    gate: Gate,
    *,
    rung: int = 1,
    instrument: str = "BunchingElasticity",
    binsize: float | None = None,
    degree: int = DEFAULT_DEGREE,
    window_bins: int | None = None,
    integration: bool = True,
    n_boot: int = 300,
    n_placebos: int = 30,
    n_null: int = 200,
    sweep_bins: Sequence[int] = DEFAULT_SWEEP_BINS,
    decode: DecodeLength | None = None,
    seed: int = 0,
) -> BunchingReading | Refusal:
    """I2's reading, or the refusal that says why there is none.

    ``window_bins`` of None grows the exclusion window from the data with `auto_window`, which is
    the default because the window is the one choice that decides the answer and picking it by eye
    is how a bunching estimate becomes whatever the analyst wanted. Pass an integer to fix it, and
    the reading says which of the two happened.

    **The limitation worth knowing before reading a small excess mass as a finding.** The
    estimator's bias depends on where the cutoff sits on the density, and the smooth-density null
    measures that bias without fully absorbing it. Measured on a real 1,600-rollout length
    distribution with no gate anywhere in it, the four cutoffs 45, 55, 65 and 75 characters returned
    excess masses of -10.7, -10.5, -11.2 and -16.7 against nulls whose own means ran from -5.1 to
    +3.4 with spreads near 5 to 7; the worst of the four is a 2.2-sigma deficit with an empirical p
    of 0.007. So a single cutoff at two sigma is inside what this estimator does on a smooth
    density, and the number that separates a gate from that is the magnitude: the same estimator on
    the same density with a response planted at a known cutoff returned 155 to 346. Two orders of
    magnitude, not two sigma, is what a gate looks like here.
    """
    v = np.asarray(running.values, dtype=np.float64).ravel()
    finite = v[np.isfinite(v)]
    if finite.size < 50:
        return refuse_incomplete(
            instrument,
            field=f"{running.name} on more than {finite.size} of {v.size} rollouts",
            subject="the record",
            remedy=(
                f"record {running.name} per rollout. A counterfactual density fitted to fewer than "
                f"fifty observations is a polynomial through noise, and no window choice repairs "
                f"that."
            ),
            n=int(v.size),
            n_finite=int(finite.size),
        )

    b = binsize if binsize is not None else automatic_binsize(finite)
    chosen_from_data = window_bins is None
    if window_bins is None:
        window_bins = auto_window(
            finite,
            gate.cutoff,
            binsize=b,
            degree=degree,
            bunching_side=gate.bunching_side,
        )
    hist = histogram(finite, gate.cutoff, b)
    usable = int(
        (
            ~(
                (hist.midpoints > gate.cutoff - window_bins * hist.binsize)
                & (hist.midpoints < gate.cutoff + window_bins * hist.binsize)
            )
        ).sum()
    )
    if usable <= degree + 1:
        return refuse_incomplete(
            instrument,
            field="bins outside the exclusion window",
            subject=(
                f"{hist.counts.size} bins at binwidth {hist.binsize:.4g}, of which {usable} sit "
                f"outside a window of {window_bins} bins each side"
            ),
            remedy=(
                f"narrow the window with `window_bins=`, lower the polynomial degree from {degree}, "
                f"or collect more rollouts so the support carries more than {degree + 1} bins "
                f"outside the window. A degree-{degree} polynomial through {usable} points is not "
                f"a counterfactual."
            ),
            n_bins=int(hist.counts.size),
            n_usable=usable,
            degree=int(degree),
        )

    cf = counterfactual_density(
        finite,
        gate.cutoff,
        binsize=b,
        degree=degree,
        window_bins=window_bins,
        bunching_side=gate.bunching_side,
        integration=integration,
    )
    if not math.isfinite(cf.density_at_cutoff) or cf.density_at_cutoff <= 0:
        return Refusal(
            instrument=instrument,
            reason=RefusalReason.RECORD_INCOMPLETE,
            detail=(
                f"the counterfactual density at the cutoff is {cf.density_at_cutoff:.4g}. Excess "
                f"mass is normalised by it, and dividing by a non-positive counterfactual produces "
                f"a number with the wrong sign rather than an error."
            ),
            remedy=(
                "lower the polynomial degree, widen the binwidth, or test a cutoff inside the "
                "support of the running variable. A counterfactual that goes non-positive at the "
                "cutoff is a polynomial extrapolating past the data."
            ),
            statistics={
                "density_at_cutoff": cf.density_at_cutoff,
                "degree": degree,
                "binsize": cf.binsize,
                "cutoff": float(gate.cutoff),
            },
        )

    normalised = cf.excess / cf.density_at_cutoff
    dz_star = normalised * cf.binsize
    elasticity: float | None = None
    if rung >= 1 and gate.penalty_fraction is not None:
        elasticity = saez_elasticity(dz_star, gate.cutoff, gate.penalty_fraction, gate.kind)

    # Residual bootstrap, which is the standard one in this literature: resample the deviations of
    # the observed counts from the fitted counterfactual, add them back, refit, recompute.
    rng = np.random.default_rng(seed)
    resid = cf.observed - cf.counterfactual
    boot_excess: list[float] = []
    boot_elast: list[float] = []
    for _ in range(int(n_boot)):
        drawn = cf.counterfactual + rng.choice(resid, size=resid.size, replace=True)
        drawn = np.clip(drawn, 0.0, None)
        counts = np.rint(drawn).astype(np.int64)
        if counts.sum() <= 0:
            continue
        sample = np.repeat(cf.midpoints, counts) + rng.uniform(
            -cf.binsize / 2.0, cf.binsize / 2.0, size=int(counts.sum())
        )
        e, _, dz = _excess_at(
            sample,
            gate.cutoff,
            binsize=b,
            degree=degree,
            window_bins=window_bins,
            bunching_side=gate.bunching_side,
            integration=integration,
        )
        if math.isfinite(e):
            boot_excess.append(e)
        if elasticity is not None and math.isfinite(dz) and gate.penalty_fraction is not None:
            boot_elast.append(saez_elasticity(dz, gate.cutoff, gate.penalty_fraction, gate.kind))
    se_excess = float(np.std(boot_excess, ddof=1)) if len(boot_excess) > 2 else float("nan")
    se_elast = float(np.std(boot_elast, ddof=1)) if len(boot_elast) > 2 else None

    # The placebo baseline: the identical estimator where no gate is.
    lo, hi = np.percentile(finite, [10, 90])
    grid = np.linspace(float(lo), float(hi), int(n_placebos) + 2)[1:-1]
    keep_out = (window_bins + 1) * cf.binsize
    grid = grid[np.abs(grid - gate.cutoff) > keep_out]
    placebo: list[float] = []
    used: list[float] = []
    for c in grid:
        e, _, _ = _excess_at(
            finite,
            float(c),
            binsize=b,
            degree=degree,
            window_bins=window_bins,
            bunching_side=gate.bunching_side,
            integration=integration,
        )
        if math.isfinite(e):
            placebo.append(e)
            used.append(float(c))
    placebo_p = (
        float(np.mean(np.abs(placebo) >= abs(cf.excess))) if len(placebo) >= 5 else float("nan")
    )
    null = smooth_null_excess(
        finite,
        gate.cutoff,
        observed_excess=cf.excess,
        binsize=cf.binsize,
        degree=degree,
        window_bins=window_bins,
        bunching_side=gate.bunching_side,
        integration=integration,
        n_draws=n_null,
        seed=seed + 1,
    )

    sweep_w: list[int] = []
    sweep_b: list[float] = []
    sweep_dz: list[float] = []
    sweep_e: list[float] = []
    for w in sweep_bins:
        e, _, dz = _excess_at(
            finite,
            gate.cutoff,
            binsize=b,
            degree=degree,
            window_bins=int(w),
            bunching_side=gate.bunching_side,
            integration=integration,
        )
        sweep_w.append(int(w))
        sweep_b.append(float(e))
        sweep_dz.append(float(dz))
        sweep_e.append(
            saez_elasticity(dz, gate.cutoff, gate.penalty_fraction, gate.kind)
            if gate.penalty_fraction is not None
            else float("nan")
        )

    return BunchingReading(
        gate=gate,
        running=running.name,
        unit=running.unit,
        n=int(finite.size),
        cutoff=float(gate.cutoff),
        binsize=float(cf.binsize),
        window_bins=int(window_bins),
        degree=int(degree),
        excess_mass=float(cf.excess),
        missing_mass=float(cf.missing),
        normalised_excess=float(normalised),
        dz_star=float(dz_star),
        elasticity=elasticity,
        se_excess=se_excess,
        se_elasticity=se_elast,
        rung=int(rung if elasticity is not None else 0),
        integration_converged=bool(cf.converged),
        iterations=int(cf.iterations),
        placebo_excess=tuple(placebo),
        placebo_cutoffs=tuple(used),
        placebo_p=placebo_p,
        smooth_null=null,
        sweep=WindowSweep(
            window_bins=tuple(sweep_w),
            excess_mass=tuple(sweep_b),
            dz_star=tuple(sweep_dz),
            elasticity=tuple(sweep_e),
        ),
        window_chosen_from_data=chosen_from_data,
        decode=decode,
    )


# ---------------------------------------------------------------------------
# the kill condition: does it respond when the gate moves?
# ---------------------------------------------------------------------------


@register_payload
@dataclass(frozen=True)
class GateResponse:
    """What the estimator says as the true gate moves. I2's kill condition, run.

    ``tracks`` is the whole verdict: for every true cutoff, the estimate at that cutoff must exceed
    the estimate the same sample gives at the other cutoffs in the sweep. An estimator that reports
    the same excess mass wherever it is pointed is measuring its own exclusion window, and the
    catalogue kills the instrument on exactly that.
    """

    cutoffs: tuple[float, ...]
    at_true_cutoff: tuple[float, ...]
    at_other_cutoffs: tuple[tuple[float, ...], ...]
    elasticity: tuple[float, ...]
    tracks: bool
    margin: float
    detail: str = ""

    def render(self) -> str:
        lines = [
            f"gate-move response: {'TRACKS' if self.tracks else 'DOES NOT TRACK'} "
            f"(smallest margin {self.margin:.3g})"
        ]
        for c, own, others, e in zip(
            self.cutoffs, self.at_true_cutoff, self.at_other_cutoffs, self.elasticity
        ):
            best_other = max(others) if others else float("nan")
            lines.append(
                f"  gate at {c:g}: B = {own:.1f} here, {best_other:.1f} at the best other cutoff, "
                f"elasticity {e:.3g}"
            )
        if self.detail:
            lines.append(f"  {self.detail}")
        return "\n".join(lines)


def gate_response(
    samples: Mapping[float, Sequence[float] | np.ndarray],
    *,
    gate: Gate,
    instrument: str = "BunchingElasticity",
    binsize: float | None = None,
    degree: int = DEFAULT_DEGREE,
    window_bins: int = DEFAULT_WINDOW_BINS,
    integration: bool = True,
) -> GateResponse | Refusal:
    """Move the gate, and check the estimate follows it.

    ``samples`` maps a true cutoff to the running variable realised under a gate at that cutoff.
    For each one the estimator is run at its own cutoff and at every other cutoff in the sweep, and
    the reading tracks if each sample's largest excess mass is at its own gate.

    **Two identical samples mean the gate never applied**, which is void condition 8 rather than a
    failed kill test. A contrast that did not reach the trainer produces a tidy
    null with no anomaly in it anywhere, and a null is the result a reader is least likely to
    interrogate, so it is refused here rather than reported as a negative.
    """
    cutoffs = sorted(float(c) for c in samples)
    if len(cutoffs) < 2:
        return Refusal(
            instrument=instrument,
            reason=RefusalReason.NO_MATCHED_CONTROL,
            detail=f"a gate-move check needs at least two gate positions; {len(cutoffs)} supplied",
            remedy=(
                "supply the running variable realised under at least two different gate positions, "
                "keyed by the gate's cutoff. One position cannot show that an estimate moved."
            ),
        )
    arrays = {c: np.asarray(samples[c], dtype=np.float64).ravel() for c in cutoffs}
    for i, a in enumerate(cutoffs):
        for bb in cutoffs[i + 1 :]:
            left, right = arrays[a], arrays[bb]
            if left.shape == right.shape and np.array_equal(left, right):
                return Refusal(
                    instrument=instrument,
                    reason=RefusalReason.VOID,
                    detail=(
                        f"the running variable under a gate at {a:g} is identical to the running "
                        f"variable under a gate at {bb:g}, so the gate move did not reach the "
                        f"data. Any difference between the two readings would be zero by "
                        f"construction and any similarity would mean nothing."
                    ),
                    remedy=(
                        "check that moving the gate actually changed the rollouts, then re-run. "
                        "If the gate is installed counterfactually, the running variable is the "
                        "same by construction and this check has to be run on samples generated "
                        "under each gate rather than on one sample re-scored twice."
                    ),
                    statistics={"void_condition": "contrast_inert", "cutoffs": [a, bb]},
                )

    own: list[float] = []
    others: list[tuple[float, ...]] = []
    elast: list[float] = []
    margins: list[float] = []
    for c in cutoffs:
        v = arrays[c]
        b_here, _, dz = _excess_at(
            v,
            c,
            binsize=binsize if binsize is not None else automatic_binsize(v),
            degree=degree,
            window_bins=window_bins,
            bunching_side=gate.bunching_side,
            integration=integration,
        )
        row: list[float] = []
        for other in cutoffs:
            if other == c:
                continue
            b_other, _, _ = _excess_at(
                v,
                other,
                binsize=binsize if binsize is not None else automatic_binsize(v),
                degree=degree,
                window_bins=window_bins,
                bunching_side=gate.bunching_side,
                integration=integration,
            )
            row.append(float(b_other))
        own.append(float(b_here))
        others.append(tuple(row))
        elast.append(
            saez_elasticity(dz, c, gate.penalty_fraction, gate.kind)
            if gate.penalty_fraction is not None
            else float("nan")
        )
        margins.append(float(b_here - max(row)) if row else float("nan"))

    tracks = all(m > 0 for m in margins if math.isfinite(m)) and any(
        math.isfinite(m) for m in margins
    )
    return GateResponse(
        cutoffs=tuple(cutoffs),
        at_true_cutoff=tuple(own),
        at_other_cutoffs=tuple(others),
        elasticity=tuple(elast),
        tracks=bool(tracks),
        margin=float(min((m for m in margins if math.isfinite(m)), default=float("nan"))),
        detail=(
            "each sample's excess mass is largest at its own gate"
            if tracks
            else "at least one sample's excess mass is larger at somebody else's gate, which is "
            "what an estimator measuring its own window looks like"
        ),
    )


# ---------------------------------------------------------------------------
# the instrument
# ---------------------------------------------------------------------------


class BunchingElasticity(ThresholdInstrument):
    """I2. How hard the policy is pushing on the gate, in units that transfer.

    Kill condition, from the catalogue record: if the elasticity does not respond when the gate is
    synthetically moved. `gate_response` runs it.
    """

    name = "BunchingElasticity"
    version = "1.0"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "I2"
    deviations = (
        "the elasticity is Kleven's notch reduction of the iso-elastic quasi-linear model with "
        "the change in the net-of-tax rate replaced by the fractional score drop at the gate. That "
        "substitution is the translation from taxes to rewards and it carries the model's "
        "assumptions with it: no income effects, a single dimension of response, and no "
        "optimisation frictions",
        "optimisation frictions are not estimated. Kleven and Waseem recover them from the "
        "dominated region above a notch, where nobody should locate at all; that needs a second "
        "structural assumption and is not implemented, so the elasticity here is a lower bound on "
        "the frictionless one",
        "the standard error is a residual bootstrap over bins rather than an analytic delta "
        "method, which is the standard in this literature and is what the reported interval is",
    )

    quantity = "gate.bunching_elasticity"
    requires = GATE_ACCESS
    substrates = ALL_SUBSTRATES
    phases = RECORD_PHASES
    envelope = GATE_ENVELOPE
    #: `units` in the registry. The elasticity is dimensionless, and that is exactly why it must
    #: refuse a comparison against an excess mass, which is in units of the running variable.
    invariance = "units"
    invariance_relation = INVARIANT
    baselines = BUNCHING_BASELINES
    rung = 1

    def __init__(
        self,
        running: RunningVariable | None = None,
        gate: Gate | None = None,
        *,
        rung: int = 1,
        binsize: float | None = None,
        degree: int = DEFAULT_DEGREE,
        window_bins: int | None = None,
        integration: bool = True,
        n_boot: int = 300,
        n_placebos: int = 30,
        n_null: int = 200,
        sweep_bins: Sequence[int] = DEFAULT_SWEEP_BINS,
        decode: DecodeLength | None = None,
        seed: int = 0,
    ) -> None:
        self.running = running
        self.gate = gate
        self.rung = int(rung)
        self.binsize = binsize
        self.degree = int(degree)
        self.window_bins = None if window_bins is None else int(window_bins)
        self.integration = bool(integration)
        self.n_boot = int(n_boot)
        self.n_placebos = int(n_placebos)
        #: The draw count for `baseline.smooth_density_null` and the bin sweep for
        #: `baseline.window_sensitivity`. Both are mandatory baselines this instrument declares and
        #: neither could be set through the constructor, so a caller wanting a cheaper null or a
        #: different sweep had to bypass the instrument and call `bunching_elasticity` directly,
        #: which loses the quantity id and the envelope check on the reading.
        self.n_null = int(n_null)
        self.sweep_bins = tuple(int(s) for s in sweep_bins)
        self.decode = decode
        self.seed = int(seed)

    def compute(self) -> BunchingReading | Refusal:
        if self.running is None or self.gate is None:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail="no running variable or no gate was supplied, so there is nothing to fit",
                remedy=(
                    "pass `running=` a RunningVariable and `gate=` a Gate carrying "
                    "`penalty_fraction`, which is what turns excess mass into an elasticity. "
                    "Without it the instrument returns rung 0 and says so."
                ),
            )
        return bunching_elasticity(
            self.running,
            self.gate,
            rung=self.rung,
            instrument=self.name,
            binsize=self.binsize,
            degree=self.degree,
            window_bins=self.window_bins,
            integration=self.integration,
            n_boot=self.n_boot,
            n_placebos=self.n_placebos,
            n_null=self.n_null,
            sweep_bins=self.sweep_bins,
            decode=self.decode,
            seed=self.seed,
        )


__all__ = [
    "BUNCHING_BASELINES",
    "DEFAULT_DEGREE",
    "DEFAULT_SWEEP_BINS",
    "DEFAULT_WINDOW_BINS",
    "BunchingElasticity",
    "BunchingReading",
    "Counterfactual",
    "GateResponse",
    "WindowSweep",
    "bunching_elasticity",
    "counterfactual_density",
    "gate_response",
    "saez_elasticity",
]

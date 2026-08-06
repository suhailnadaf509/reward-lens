"""M1, the substrate noise floor and the two limits that follow from it.

Analytical chemistry has had `LOD = 3.3 sigma_blank / S` since the 1970s. Machine learning has
nothing of the kind and reports effect sizes below its own substrate noise routinely, which is the
single cheapest way to publish a confident wrong number: the effect is real arithmetic on real data
and it is smaller than the disagreement the apparatus has with itself.

Three things have to be measured before a limit exists, and the reason this instrument is worth its
own module is that two of them are usually skipped.

**The blank.** `sigma_blank` is the standard deviation of *this reading* over replicate
measurements of a sample in which the thing being looked for is absent by construction: a shuffled
label set, a semantically irrelevant direction, two arms of one dataset with no dose applied to
either. It is a measurement and never a guess, and `blank_n` travels with it so a floor from three
replicates is visibly weaker than one from two hundred.

**The calibration curve.** `S` is the slope of reading against dose. It is not 1, it is not assumed,
and it is not constant: the Hill slope varies with dose, so a sensitivity without a dose attached is
meaningless and `CalibrationCurve.slope` refuses to produce one. Fitting the curve is what turns a
kill into a specification, and the fitted `EC50` and cooperativity `n` are readings in their own
right. A sweep that will not fit is reported as a sweep that will not fit; the floor still holds and
the two limits do not exist yet.

**The three-outcome rule.** Below the LOD, refuse. Between the LOD and the LOQ, return a bound.
Above the LOQ, report with a budget. Two outcomes is what everyone has, and the missing third is
where most of the field's small effects live.

What this instrument does **not** claim. The catalogue gives `substrate.noise_floor` the access
level "two engines", and the highest rung of this quantity is the disagreement between two
nominally interchangeable engines reading the same input: vLLM against HuggingFace, eager against
compiled, bf16 against fp32. That rung needs a second engine and this build has no store containing
one, which is stated in `deviations` rather than papered over with a blank that costs nothing. The
rung built here reads blanks off a record, needs no second engine, and is biased **downward**: a
within-record blank cannot see any error term that is constant across the record, and every
engine-level term is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from reward_lens.core.budget import (
    LIMITS,
    CalibrationCurve,
    LimitOfDetection,
    SubstrateKey,
    Verdict,
)
from reward_lens.core.envelope import EnvelopeSpec, RegimeCondition
from reward_lens.core.evidence import Uncertainty, register_payload
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.quantity import (
    BaselineID,
    BiasStatement,
    CostModel,
    EstimatorEntry,
    register_estimator,
)
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.types import (
    Access,
    Capability,
    Component,
    GaugeStatus,
    Phase,
    Substrate,
)
from reward_lens.measure.meta._base import MetaInstrument
from reward_lens.measure.rate.regime import MEASURED_BY

#: A blank read off a record needs the record and nothing else. The engine-to-engine rung needs
#: `REPLICATE` on the grader and a second configuration to replicate against, which is a different
#: rung and is declared in the ladder rather than in this minimum.
NOISE_FLOOR_ACCESS: dict[Component, Access] = {Component.RECORD: Access.RECORD}

#: A floor is a claim that the apparatus disagrees with itself by this much, and the two numbers it
#: has to beat are the two ways of getting a smaller one for free: assume the apparatus is exact,
#: and quote the sampling standard error as if it were the whole uncertainty.
NOISE_FLOOR_BASELINES: tuple[BaselineID, ...] = (
    "baseline.assumed_noiseless",
    "baseline.sampling_error_only",
)

#: A noise floor measured while the substrate is drifting reports the drift. `STATIONARY_GRADER` is
#: the condition that separates the two, and the kernel already names what measures it.
NOISE_FLOOR_ENVELOPE = EnvelopeSpec(
    requires=frozenset({RegimeCondition.STATIONARY_GRADER}),
    measured_by={RegimeCondition.STATIONARY_GRADER: MEASURED_BY[RegimeCondition.STATIONARY_GRADER]},
    on_violation="refuse",
)

#: Fewer replicates than this and a standard deviation is a number with almost nothing in it. Three
#: gives two degrees of freedom, which is the minimum at which the word "standard deviation" is not
#: misleading; it is an argument rather than a constant so a caller can state a stricter floor.
MIN_BLANK_REPLICATES = 3

#: The minimum a Hill curve can be fitted from: three free parameters and a blank to anchor them.
MIN_SWEEP_POINTS = 4


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlankReplicates:
    """Replicate readings of a sample with no analyte in it, and how they were produced.

    ``method`` is not decoration. "Two disjoint arms of one record, no dose on either, 200 random
    splits" and "twenty repeat calls to one grader on one item" are both blanks and they see
    different error terms, so a floor that does not say which one it is cannot be compared with
    another floor or composed into a budget.
    """

    values: np.ndarray
    method: str
    #: What the reading's own full scale is, so the floor can be expressed relatively. `None` means
    #: no scale was declared and the relative floor is not reported rather than divided by a guess.
    scale: float | None = None
    key: SubstrateKey | None = None
    note: str = ""

    def __post_init__(self) -> None:
        arr = np.asarray(self.values, dtype=np.float64).ravel()
        object.__setattr__(self, "values", arr)
        if not self.method.strip():
            raise ValueError(
                "a blank has to say how it was produced. Two blanks made different ways see "
                "different error terms and are not the same measurement."
            )
        if self.scale is not None and self.scale <= 0:
            raise ValueError(f"the reading's full scale must be positive; got {self.scale}")

    @property
    def n(self) -> int:
        return int(self.values.size)

    @property
    def sigma(self) -> float:
        """The floor, in the reading's own units."""
        return float(np.std(self.values, ddof=1)) if self.n > 1 else float("nan")

    @property
    def mean(self) -> float:
        """The blank's own offset. Far from zero means the blank is not blank."""
        return float(np.mean(self.values)) if self.n else float("nan")

    @property
    def relative(self) -> float | None:
        """`sigma_blank` over the reading's declared full scale, which is the registered unit."""
        return None if self.scale is None else self.sigma / self.scale


@dataclass(frozen=True)
class DoseSweep:
    """A reading at each of several doses, including a blank, from which `S` is fitted."""

    doses: np.ndarray
    responses: np.ndarray
    dose_unit: str = "dose"
    n_per_point: int | None = None
    note: str = ""

    def __post_init__(self) -> None:
        d = np.asarray(self.doses, dtype=np.float64).ravel()
        r = np.asarray(self.responses, dtype=np.float64).ravel()
        if d.size != r.size:
            raise ValueError(f"{d.size} doses and {r.size} responses are not a sweep")
        order = np.argsort(d)
        object.__setattr__(self, "doses", d[order])
        object.__setattr__(self, "responses", r[order])

    @property
    def n_points(self) -> int:
        return int(self.doses.size)

    @property
    def has_blank(self) -> bool:
        return bool(self.doses.size and float(self.doses[0]) <= 0.0)

    @property
    def dynamic_range(self) -> float:
        """Largest response minus smallest. The scale a relative floor is relative to."""
        return float(self.responses.max() - self.responses.min()) if self.responses.size else 0.0


@dataclass(frozen=True)
class HillFit:
    """A fitted calibration curve, or the reason there is not one.

    ``curve`` is None when the fit did not converge or came back non-physical, and ``reason`` says
    which. A failed fit is a real outcome here: it means the substrate does not respond to this dose
    in this range, so the limits do not exist, and the floor is still measured.
    """

    curve: CalibrationCurve | None
    reason: str = ""
    residual_rms: float = float("nan")
    monotone: bool = True
    n_points: int = 0

    @property
    def converged(self) -> bool:
        return self.curve is not None


def fit_hill(sweep: DoseSweep) -> HillFit:
    """Fit `E(c) = E_max c^n / (EC50^n + c^n)` by least squares, or say why it did not fit.

    Only `RuntimeError` is caught, which is the single exception `scipy.optimize.curve_fit` raises
    for a fit that did not converge in its iteration budget. Anything else is a bug in the caller's
    data and is left to propagate, because the alternative is turning a shape mismatch into a
    reading that says "the substrate does not respond".
    """
    from scipy.optimize import curve_fit

    if sweep.n_points < MIN_SWEEP_POINTS:
        return HillFit(
            curve=None,
            reason=(
                f"{sweep.n_points} dose points; a three-parameter curve needs at least "
                f"{MIN_SWEEP_POINTS} and one of them should be a blank"
            ),
            n_points=sweep.n_points,
        )
    d, r = sweep.doses, sweep.responses
    monotone = bool(np.all(np.diff(r) >= -1e-12))

    def model(c: np.ndarray, e_max: float, ec50: float, hill_n: float) -> np.ndarray:
        safe = np.maximum(c, 0.0)
        num = e_max * np.power(safe, hill_n)
        return num / (np.power(ec50, hill_n) + np.power(safe, hill_n))

    span = float(r.max() - r.min())
    positive = d[d > 0]
    p0 = [span if span > 0 else 1.0, float(np.median(positive)) if positive.size else 1.0, 1.0]
    try:
        popt, _ = curve_fit(
            model,
            d,
            r,
            p0=p0,
            bounds=([-np.inf, 1e-12, 1e-3], [np.inf, np.inf, 20.0]),
            maxfev=20000,
        )
    except RuntimeError as exc:
        return HillFit(
            curve=None,
            reason=f"the least-squares fit did not converge: {exc}",
            monotone=monotone,
            n_points=sweep.n_points,
        )
    e_max, ec50, hill_n = (float(x) for x in popt)
    residual = float(np.sqrt(np.mean((model(d, e_max, ec50, hill_n) - r) ** 2)))
    if e_max <= 0.0:
        return HillFit(
            curve=None,
            reason=(
                f"the fitted maximum effect is {e_max:.4g}, so the reading does not increase with "
                f"dose over this range and no positive slope exists to divide by"
            ),
            residual_rms=residual,
            monotone=monotone,
            n_points=sweep.n_points,
        )
    return HillFit(
        curve=CalibrationCurve(e_max=e_max, ec50=ec50, hill_n=hill_n, at_dose=ec50),
        residual_rms=residual,
        monotone=monotone,
        n_points=sweep.n_points,
    )


def two_arm_blanks(
    outcomes: Sequence[float] | np.ndarray,
    *,
    n_replicates: int = 200,
    seed: int = 0,
    method: str = "",
    scale: float | None = None,
    key: SubstrateKey | None = None,
) -> BlankReplicates:
    """The standard blank for a two-arm reading: split one item set in half and read the difference.

    The reading whose floor this measures is "arm A minus arm B", so the blank is that same reading
    with no dose applied to either arm. Its expectation is exactly zero and its spread is what the
    apparatus and the item set contribute between them, which is the number a measured effect has to
    clear.

    A paired blank would be identically zero and would report a floor of zero, which is the mistake
    this helper exists to avoid: the two arms have to be disjoint, because the comparison the floor
    is for has disjoint arms.
    """
    y = np.asarray(outcomes, dtype=np.float64).ravel()
    n = int(y.size)
    if n < 4:
        raise ValueError(f"a split-half blank needs at least four items; got {n}")
    rng = np.random.default_rng(seed)
    half = n // 2
    values = np.empty(int(n_replicates), dtype=np.float64)
    for r in range(int(n_replicates)):
        perm = rng.permutation(n)
        values[r] = float(y[perm[:half]].mean() - y[perm[half:]].mean())
    return BlankReplicates(
        values=values,
        method=method
        or (
            f"two disjoint arms of one record, no dose on either, {n_replicates} random splits of "
            f"{n} items"
        ),
        scale=scale,
        key=key,
    )


# ---------------------------------------------------------------------------
# The reading
# ---------------------------------------------------------------------------


@register_payload
@dataclass
class NoiseFloor:
    """The floor, the two limits, and the calibration that produced them.

    ``lod`` and ``loq`` are `None` when no calibration slope exists, and that is the honest state
    rather than a large number. A limit is a dose and a dose needs a slope; without one the floor
    is still a measurement and the limits are simply not available yet.
    """

    sigma_blank: float
    blank_n: int
    blank_mean: float
    blank_method: str
    relative_floor: float | None
    scale: float | None
    sensitivity: float | None
    at_dose: float | None
    lod: float | None
    loq: float | None
    ec50: float | None
    e_max: float | None
    hill_n: float | None
    fit_reason: str
    fit_residual_rms: float
    dose_unit: str
    substrate: str
    baselines: Mapping[str, float] = field(default_factory=dict)

    @property
    def is_determinate(self) -> bool:
        return self.lod is not None

    def says(self) -> str:
        head = (
            f"The substrate disagrees with itself by {self.sigma_blank:.4g} "
            f"({self.blank_n} blank replicates, {self.blank_method})."
        )
        if self.relative_floor is not None:
            head += f" That is {self.relative_floor:.1%} of the reading's own full scale."
        if not self.is_determinate:
            return (
                head + f" No limit of detection exists on this configuration: {self.fit_reason} "
                f"Fit a dose sweep before quoting one."
            )
        return (
            head + f" With a calibration slope of {self.sensitivity:.4g} at dose "
            f"{self.at_dose:.4g}, the limit of detection is {self.lod:.4g} {self.dose_unit} and "
            f"the limit of quantitation is {self.loq:.4g} {self.dose_unit}. An effect below the "
            f"first is refused; between the two, a bound is returned."
        )


def limits_from(
    blanks: BlankReplicates,
    fit: HillFit,
    *,
    lod_k: float = 3.3,
    loq_k: float = 10.0,
) -> LimitOfDetection:
    """The kernel's `LimitOfDetection` from a measured blank and a fitted curve.

    A curve that did not fit gives a sensitivity of zero, which `LimitOfDetection` reports as an
    undefined limit rather than as an infinite one, and that is exactly the intended behaviour: an
    instrument whose reading does not respond to dose has no detection limit because it has no
    calibration.
    """
    slope = fit.curve.slope() if fit.curve is not None else 0.0
    return LimitOfDetection(
        sigma_blank=blanks.sigma,
        sensitivity=slope,
        key=blanks.key,
        blank_n=blanks.n,
        curve=fit.curve,
        note=blanks.method if blanks.key is None else "",
        lod_k=lod_k,
        loq_k=loq_k,
    )


def verdict_for(reading: float, limits: LimitOfDetection) -> Verdict:
    """Which of the three outcomes a reading falls into on this substrate."""
    return limits.verdict(reading)


# ---------------------------------------------------------------------------
# The instrument
# ---------------------------------------------------------------------------


class SubstrateNoiseFloor(MetaInstrument):
    """M1. What the apparatus disagrees with itself by, and the two limits that follow.

    Consulted by every preflight and cached per configuration, which is why a `SubstrateKey` is
    worth supplying: with one, the reading is put into `LIMITS` and every later instrument on that
    configuration can find it instead of assuming a floor of zero.
    """

    name = "SubstrateNoiseFloor"
    version = "1.0"
    quantity = "substrate.noise_floor"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    requires = NOISE_FLOOR_ACCESS
    substrates = frozenset(
        {
            Substrate.NEURAL_SCALAR,
            Substrate.NEURAL_GEN,
            Substrate.PROGRAM,
            Substrate.PROCEDURAL,
            Substrate.HUMAN,
            Substrate.COMPOSITE,
        }
    )
    phases = frozenset({Phase.PRE_RUN, Phase.POST_RUN})
    envelope = NOISE_FLOOR_ENVELOPE
    invariance = "repr.basis"
    invariance_relation = INVARIANT
    baselines = NOISE_FLOOR_BASELINES
    rung = 0
    faithful_to = "M1"
    deviations = (
        "the catalogue gives this quantity the access level `two engines` and the rung built here "
        "reads blanks off a single record, so it is one rung below the catalogued minimum. The "
        "direction of the resulting bias is stated on every reading: a within-record blank cannot "
        "see any error term that is constant across the record, and every engine-level term is, so "
        "this floor is a lower bound on the floor a two-engine comparison would report",
        "`substrate.lod` and `substrate.loq` are registered with unit `effect` and are computed "
        "here in dose units, because `LOD = 3.3 sigma_blank / S` divides a reading by a slope and "
        "the result is a dose. A limit in effect units would be `3.3 sigma_blank` with the "
        "calibration dropped, which is the number people quote when they have not fitted a curve",
    )

    def __init__(
        self,
        blanks: BlankReplicates | None = None,
        sweep: DoseSweep | None = None,
        *,
        substrate_key: SubstrateKey | None = None,
        lod_k: float = 3.3,
        loq_k: float = 10.0,
        min_replicates: int = MIN_BLANK_REPLICATES,
        cache: bool = False,
    ) -> None:
        self.blanks = blanks
        self.sweep = sweep
        self.substrate_key = substrate_key or (blanks.key if blanks is not None else None)
        self.lod_k = float(lod_k)
        self.loq_k = float(loq_k)
        self.min_replicates = int(min_replicates)
        self.cache = bool(cache)

    def compute(self) -> Any:
        blanks = self.blanks
        if blanks is None or blanks.n == 0:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail="no blank replicates were supplied, so there is no floor to measure",
                remedy=(
                    "read the same quantity several times on a sample with the analyte absent by "
                    "construction and pass the readings as "
                    "`BlankReplicates(values=..., method='...')`. On a record with per-item "
                    "outcomes, `two_arm_blanks(outcomes)` builds them from data you already hold "
                    "and costs nothing."
                ),
                statistics={"blank_n": 0},
            )
        if blanks.n < self.min_replicates:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail=(
                    f"{blanks.n} blank replicate(s), below the floor of {self.min_replicates}. A "
                    f"standard deviation on fewer than two degrees of freedom is a number with "
                    f"almost nothing in it, and a floor quoted from one would make every effect "
                    f"above it look established"
                ),
                remedy=(
                    f"produce at least {self.min_replicates} blank replicates. If the blank is a "
                    f"split of an existing record, more replicates cost only arithmetic: raise "
                    f"`n_replicates` on `two_arm_blanks`."
                ),
                statistics={"blank_n": blanks.n, "minimum": self.min_replicates},
            )

        fit = (
            fit_hill(self.sweep)
            if self.sweep is not None
            else HillFit(curve=None, reason="no dose sweep was supplied, so no slope was fitted.")
        )
        limits = limits_from(blanks, fit, lod_k=self.lod_k, loq_k=self.loq_k)
        if self.cache and self.substrate_key is not None:
            LIMITS.put(
                LimitOfDetection(
                    sigma_blank=limits.sigma_blank,
                    sensitivity=limits.sensitivity,
                    key=self.substrate_key,
                    blank_n=limits.blank_n,
                    curve=limits.curve,
                    note=limits.note,
                    lod_k=limits.lod_k,
                    loq_k=limits.loq_k,
                )
            )

        scale = blanks.scale
        if scale is None and self.sweep is not None and self.sweep.dynamic_range > 0:
            scale = self.sweep.dynamic_range
        relative = blanks.sigma / scale if scale else None
        curve = fit.curve
        return NoiseFloor(
            sigma_blank=blanks.sigma,
            blank_n=blanks.n,
            blank_mean=blanks.mean,
            blank_method=blanks.method,
            relative_floor=relative,
            scale=scale,
            sensitivity=limits.sensitivity if limits.is_determinate else None,
            at_dose=curve.at_dose if curve is not None else None,
            lod=limits.lod if limits.is_determinate else None,
            loq=limits.loq if limits.is_determinate else None,
            ec50=curve.ec50 if curve is not None else None,
            e_max=curve.e_max if curve is not None else None,
            hill_n=curve.hill_n if curve is not None else None,
            fit_reason=fit.reason,
            fit_residual_rms=fit.residual_rms,
            dose_unit=self.sweep.dose_unit if self.sweep is not None else "dose",
            substrate=str(self.substrate_key) if self.substrate_key is not None else "",
            baselines={
                # Assume the apparatus is exact and the floor is zero. This is what every reading
                # with no LOD beside it is implicitly claiming.
                "baseline.assumed_noiseless": 0.0,
                # Quote the standard error of the blank mean as if it were the whole uncertainty.
                # It shrinks with replicates and the floor does not, which is the whole point.
                "baseline.sampling_error_only": blanks.sigma / math.sqrt(blanks.n),
            },
        )

    def uncertainty(self, computed: NoiseFloor) -> Uncertainty | None:
        """The chi-squared interval on a standard deviation from `n - 1` degrees of freedom.

        Exact rather than bootstrapped, because the quantity is a standard deviation of replicates
        and the sampling distribution of one is known. A floor from three replicates comes back with
        an interval a factor of three wide, which is the correct and useful answer.
        """
        n = computed.blank_n
        if n < 2:
            return None
        from scipy.stats import chi2

        df = n - 1
        lo = computed.sigma_blank * math.sqrt(df / chi2.ppf(0.975, df))
        hi = computed.sigma_blank * math.sqrt(df / chi2.ppf(0.025, df))
        return Uncertainty(
            ci_low=lo,
            ci_high=hi,
            ci_level=0.95,
            n=n,
            method="chi-squared interval on a standard deviation",
        )

    def payload(self, computed: NoiseFloor) -> dict[str, Any]:
        return {
            "sigma_blank": computed.sigma_blank,
            "blank_n": computed.blank_n,
            "blank_mean": computed.blank_mean,
            "blank_method": computed.blank_method,
            "relative_floor": computed.relative_floor,
            "scale": computed.scale,
            "sensitivity": computed.sensitivity,
            "at_dose": computed.at_dose,
            "lod": computed.lod,
            "loq": computed.loq,
            "ec50": computed.ec50,
            "e_max": computed.e_max,
            "hill_n": computed.hill_n,
            "fit_reason": computed.fit_reason,
            "fit_residual_rms": computed.fit_residual_rms,
            "dose_unit": computed.dose_unit,
            "substrate": computed.substrate,
            "is_determinate": computed.is_determinate,
            "bias_direction": "downward",
            "baselines": dict(computed.baselines),
            "says": computed.says(),
        }


# ---------------------------------------------------------------------------
# The ladder
# ---------------------------------------------------------------------------


def register_ladder() -> list[str]:
    """Register M1's rungs for its four quantities. Not called at import, by design.

    Two rungs on the floor, and the second one is the catalogue's own access level rather than the
    one this build reaches. Registering a rung with no implementation is the documented way to say
    that a quantity has a better estimator nobody here has built: `EstimatorEntry.run` stays None
    and the capability report says so rather than hiding it.
    """
    envelope = NOISE_FLOOR_ENVELOPE
    entries = [
        EstimatorEntry(
            quantity="substrate.noise_floor",
            impl="m1.within_record_blank",
            requires=NOISE_FLOOR_ACCESS,
            envelope=envelope,
            rung=0,
            bias=BiasStatement(
                direction="downward",
                why=(
                    "a blank split out of one record cannot see any error term that is constant "
                    "across that record, and every engine-level, precision-level and revision-level "
                    "term is. What it does see is the item set and whatever varies within the run"
                ),
            ),
            cost=CostModel(note="arithmetic on a record already held"),
            run=None,
        ),
        EstimatorEntry(
            quantity="substrate.noise_floor",
            impl="m1.two_engine_residual",
            requires={Component.GRADER: Access.REPLICATE},
            envelope=envelope,
            rung=1,
            bias=BiasStatement(
                direction="approximately_unbiased",
                why=(
                    "two nominally interchangeable engines reading the same input disagree by "
                    "exactly the amount the configuration contributes, which is what the floor is "
                    "defined as. Specified and not built here: no store in this build carries a "
                    "second engine over the same items"
                ),
            ),
            cost=CostModel(note="one full re-scoring pass on a second engine"),
            run=None,
        ),
        EstimatorEntry(
            quantity="substrate.lod",
            impl="m1.lod_from_floor_and_slope",
            requires=NOISE_FLOOR_ACCESS,
            envelope=envelope,
            rung=0,
            bias=BiasStatement(
                direction="downward",
                why="it inherits the floor's bias divided by a slope that is measured, not assumed",
            ),
            cost=CostModel(note="a dose sweep, plus arithmetic"),
            run=None,
        ),
        EstimatorEntry(
            quantity="substrate.loq",
            impl="m1.loq_from_floor_and_slope",
            requires=NOISE_FLOOR_ACCESS,
            envelope=envelope,
            rung=0,
            bias=BiasStatement(
                direction="downward",
                why="the same construction as the LOD with the multiplier at 10 rather than 3.3",
            ),
            cost=CostModel(note="a dose sweep, plus arithmetic"),
            run=None,
        ),
        EstimatorEntry(
            quantity="intervention.ec50",
            impl="m1.hill_least_squares",
            requires=NOISE_FLOOR_ACCESS,
            envelope=envelope,
            rung=0,
            bias=BiasStatement(
                direction="unknown",
                why=(
                    "a least-squares Hill fit over a dose grid the caller chose. The grid decides "
                    "which part of the curve is resolved, and a sweep that stops before saturation "
                    "trades off E_max against EC50 with no way to tell from the fit alone"
                ),
            ),
            cost=CostModel(note="one reading per dose"),
            run=None,
        ),
    ]
    for e in entries:
        register_estimator(e)
    return [e.impl for e in entries]


__all__ = [
    "MIN_BLANK_REPLICATES",
    "MIN_SWEEP_POINTS",
    "NOISE_FLOOR_ACCESS",
    "NOISE_FLOOR_BASELINES",
    "NOISE_FLOOR_ENVELOPE",
    "BlankReplicates",
    "DoseSweep",
    "HillFit",
    "NoiseFloor",
    "SubstrateNoiseFloor",
    "fit_hill",
    "limits_from",
    "register_ladder",
    "two_arm_blanks",
    "verdict_for",
]

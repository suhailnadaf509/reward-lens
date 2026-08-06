"""L1 `reference.u_characterisation`, `reference.u_homogeneity`, `reference.u_stability`.

A planted organism is a reference material. It ships with an assigned value, the dose the plant was
made at, and every calibration in this library is performed against it. What it does not ship with,
here or anywhere else in the field, is an uncertainty on that assigned value.

An analytical laboratory would not accept it. A certified reference material carries
`u_CRM² = u_char² + u_bb² + u_stab²`, and all three terms have exact analogues here:

    u_char   how well the planted rule's strength is known. A LoRA plant at "dose 0.75" is a
             nominal dose. Nothing in the plant guarantees the organism came out at 0.75, and the
             way to find out is to measure the response at several doses and invert the fit.
    u_bb     whether two plants differing only in their seed give the same answer. The Model
             Organism Lottery says they do not, and that result *is* an uncharacterised
             between-bottle homogeneity term wearing an interpretability paper's clothes.
    u_stab   whether the assigned value drifts as the host is trained further. Nobody has measured
             this for any organism.

The three rungs of this instrument are those three terms, in that order, and the mandatory baseline
is a single-seed plant at a nominal dose, which is current practice everywhere and is what this
project's own campaign did.

**The rule that earns this module its place, and it is enforced rather than documented.**
`u_homogeneity is None` is not a missing field. It means nobody checked whether two plants with
different seeds give the same answer, it renders in every downstream reading as "reference
uncertainty not characterised", and `compute_trust` caps the trust ladder at `CALIBRATED`. Freezing
a prediction against a ruler of unknown length does not make the reading better; it makes the
prediction precise about something unmeasured. That cap fires here, on the campaign's own
cal-transfer organisms, and it is the whole point of the module.

**What this measures and what it does not.** A dose is planted in *data* and it governs a *trained
model*, and those are two different places to measure it. Everything here works at whichever level
the caller supplies a response for: hand it realised marker rates and it certifies the data, hand it
per-dose behavioural readouts from trained organisms and it certifies the organism. The data-level
number is a floor on the model-level one, never a substitute, because two trunks trained on
identical data with different seeds can express the planted rule at different strengths and that
difference is exactly what `u_bb` is for.

Kill condition, from the catalogue: **if homogeneity is negligible across seeds, in which case
single-seed plants are fine and this is one measurement, once.** Not settled here. The homogeneity
term this module has measured is data-level and it is small; the model-level term needs training
runs, and until somebody spends that compute the kill condition is open rather than answered.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np

from reward_lens.core.envelope import EnvelopeSpec
from reward_lens.core.evidence import Uncertainty, register_payload
from reward_lens.core.invariance import Relation
from reward_lens.core.quantity import (
    BiasStatement,
    CostModel,
    EstimatorEntry,
    Unit,
    register_estimator,
)
from reward_lens.core.reading import Reading, Refusal, RefusalReason
from reward_lens.core.reference import (
    CalibrationChain,
    MatrixDescription,
    ReferenceKind,
    ReferenceMaterial,
    Transfer,
    uncertified_refusal,
)
from reward_lens.core.types import Capability, Substrate
from reward_lens.measure.base import Context
from reward_lens.measure.labels._common import (
    ACCESS_PLANT,
    LabelsInstrument,
    emit_with_reference,
)

#: The catalogue's access line for L1 rung 1 reads "ORGANISM:MUTATE, at least 3 seeds", and three is
#: the operative number rather than a suggestion. A standard deviation on two observations has one
#: degree of freedom and a relative standard uncertainty of about 76%, so a two-seed homogeneity
#: term is a number whose own error bar is wider than most of the effects it would be composed with.
MIN_SEEDS_FOR_HOMOGENEITY = 3

#: Three dose levels is the floor for a calibration line: two points fit a line exactly and leave no
#: residual, so the fit reports zero scatter and `u_char` comes out as zero. A reference certificate
#: that says the assigned value is known perfectly because nobody measured twice is worse than none.
MIN_DOSES_FOR_CHARACTERISATION = 3

#: Three checkpoints for a stability slope, for the same reason: two give a slope with no residual
#: degrees of freedom and therefore no standard error on the slope.
MIN_CHECKPOINTS_FOR_STABILITY = 3


# ---------------------------------------------------------------------------
# The inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Plant:
    """One planted organism: what it was asked for, what came out, and what made it.

    ``pipeline`` is everything about the plant except the seed. Two plants are siblings, and their
    spread is a homogeneity term, only when their pipelines match. Two plants from different
    pipelines that happen to share a nominal dose have a spread that mixes homogeneity with whatever
    else changed, and reporting that as `u_bb` would understate the first and hide the second.
    """

    nominal_dose: float
    response: float
    seed: int
    family: str = ""
    pipeline: str = ""
    #: What the response is: `"dose"` when it is the realised dose in the same units as the nominal
    #: one, anything else when it is a behavioural readout that has to be inverted through the fit.
    response_kind: str = "dose"


@dataclass(frozen=True)
class StabilityPoint:
    """The reference re-measured after the host has been trained further."""

    steps: int
    response: float
    seed: int = 0


@register_payload
@dataclass(frozen=True)
class DoseResponseFit:
    """A calibration line through the dose sweep, and the uncertainty it puts on an assigned dose.

    The metrology is the ordinary one and the whole trick is which way the fit is read. A calibration
    line predicts a response from a dose; a certificate needs the uncertainty on a *dose* inferred
    from a response, so the residual scatter is divided by the sensitivity:

        s_resid / |slope|

    That ratio is in the units of the dose, which is what `reference.u_characterisation` is declared
    in. It is carried on ``u_char`` as the **common factor** of every term below, and it is not by
    itself the characterisation uncertainty. A sweep with a large response range and tight residuals
    characterises the dose well; a sweep whose readout barely moves across the whole range
    characterises nothing, and the ratio says so by going to infinity rather than by looking small.

    The inverse-prediction variance is that factor times a design term, and the design term is where
    two separate objections turn out to be one:

        (s/|b|)² · [ 1 + 1/n + (x0 - xbar)²/Sxx ]
                     ^   ^^^^^^^^^^^^^^^^^^^^^^^
                     |   how well the *line* is pinned down at x0
                     the departure of the one organism in hand from the line

    The first term is `u_bb` measured directly, because `between_seed_homogeneity` estimates exactly
    the plant-to-plant scatter about the line. So a certificate that reports `u_bb` must take
    ``u_char_at(..., individual=False)`` or the same variance is counted twice, and a certificate
    that does not report `u_bb` must take ``individual=True`` or nobody carries it at all. Shipping
    the flat ratio is wrong under both, which is why the two objections pointed in opposite
    directions and were both right.
    """

    slope: float
    intercept: float
    s_resid: float
    #: The rung-0 characterisation uncertainty read at the **centre of the sweep**, which is
    #: `(s/|b|)·sqrt(1 + 1/n)` because the leverage term is zero there. It is the best case this
    #: sweep supports and it is the number to quote when no assigned dose has been named yet. Use
    #: `u_char_at` for a specific dose and for the `individual=False` form.
    u_char: float
    r_squared: float
    n: int
    dof: int
    doses: tuple[float, ...] = ()
    responses: tuple[float, ...] = ()
    note: str = ""

    @property
    def is_usable(self) -> bool:
        return math.isfinite(self.u_char) and self.dof >= 1

    @property
    def sensitivity_scaled_residual(self) -> float:
        """`s_resid / |slope|`, the common factor of every inverse-prediction term.

        This is what `u_char` used to hold, and holding it under that name was the defect: it is a
        scatter, not a characterisation uncertainty, and it is smaller than every form of the
        latter. Kept as a named property because it is the quantity the two forms differ only by a
        design factor from, and a reader checking the arithmetic wants to see it.
        """
        if not math.isfinite(self.s_resid) or self.slope == 0.0:
            return float("inf")
        return self.s_resid / abs(self.slope)

    def u_char_at(self, dose: float, *, individual: bool) -> float:
        """The inverse-prediction standard uncertainty of a dose read back off this line.

        ``individual=True`` is the whole thing, `(s/|b|)·sqrt(1 + 1/n + (x0-xbar)²/Sxx)`, and is
        what `u_characterisation` must be when no `u_bb` is reported: the `1` is the departure of
        the one organism in hand from the fitted line and somebody has to carry it. ``False`` drops
        the `1`, which is what `u_characterisation` must be when `u_bb` is reported beside it,
        because `u_bb` is that same term measured directly and adding both counts it twice.

        Measured on the acceptance fixture, a three-dose single-seed sweep read at the sweep
        centre: the flat ratio the module used to ship is 15.47% below the ``individual=True``
        value, which is `sqrt(1 + 1/3)` exactly because the deviation term is zero at the centre.
        On the campaign's own cal-transfer organisms the same comparison is 15.47% on both arms.
        And on a three-seed family at `n = 9`, dropping the double count takes `u_CRM` down by
        1.354x.

        ``individual`` is required rather than defaulted. Which of the two is meant depends on what
        else the certificate reports, and a default here would be a decision about the uncertainty
        budget taken by whoever wrote the signature.
        """
        base = self.sensitivity_scaled_residual
        if not math.isfinite(base):
            return float("inf")
        xs = np.asarray(self.doses, dtype=np.float64)
        if xs.size == 0:
            return float("inf")
        sxx = float(((xs - xs.mean()) ** 2).sum())
        lever = (float(dose) - float(xs.mean())) ** 2 / sxx if sxx > 0 else float("inf")
        factor = (1.0 if individual else 0.0) + 1.0 / self.n + lever
        return base * math.sqrt(factor)

    def render(self) -> str:
        return (
            f"response = {self.intercept:.4g} + {self.slope:.4g}·dose, "
            f"s_resid {self.s_resid:.4g} on {self.dof} dof, R² {self.r_squared:.4f}; "
            f"u_char = {self.u_char:.4g} dose at the sweep centre, individual form"
        )

    def __canonical__(self) -> dict[str, Any]:
        return {
            "slope": self.slope,
            "intercept": self.intercept,
            "s_resid": self.s_resid,
            "u_char": self.u_char,
            "r_squared": self.r_squared,
            "n": self.n,
            "dof": self.dof,
            "doses": list(self.doses),
            "responses": list(self.responses),
            "note": self.note,
        }


def dose_response_characterisation(
    doses: Sequence[float], responses: Sequence[float]
) -> DoseResponseFit:
    """Least squares through the dose sweep, read backwards into a dose uncertainty.

    Refuses nothing and raises nothing: a fit that cannot support a characterisation comes back with
    `u_char` infinite and a note saying why, and the caller turns that into a refusal. That split
    keeps the arithmetic testable on inputs that have no business producing a certificate.
    """
    x = np.asarray(doses, dtype=np.float64).ravel()
    y = np.asarray(responses, dtype=np.float64).ravel()
    if x.size != y.size:
        raise ValueError(f"{x.size} doses and {y.size} responses; a fit needs them paired")
    n = int(x.size)
    dof = n - 2
    if n < MIN_DOSES_FOR_CHARACTERISATION:
        return DoseResponseFit(
            slope=float("nan"),
            intercept=float("nan"),
            s_resid=float("nan"),
            u_char=float("inf"),
            r_squared=float("nan"),
            n=n,
            dof=max(dof, 0),
            doses=tuple(float(v) for v in x),
            responses=tuple(float(v) for v in y),
            note=(
                f"{n} dose level(s); a calibration line through fewer than "
                f"{MIN_DOSES_FOR_CHARACTERISATION} leaves no residual and would report zero "
                f"scatter because nothing was measured twice."
            ),
        )
    sxx = float(((x - x.mean()) ** 2).sum())
    if sxx == 0.0:
        return DoseResponseFit(
            slope=float("nan"),
            intercept=float("nan"),
            s_resid=float("nan"),
            u_char=float("inf"),
            r_squared=float("nan"),
            n=n,
            dof=dof,
            doses=tuple(float(v) for v in x),
            responses=tuple(float(v) for v in y),
            note="every plant was made at the same nominal dose, so there is no sweep to fit.",
        )
    slope = float(((x - x.mean()) * (y - y.mean())).sum() / sxx)
    intercept = float(y.mean() - slope * x.mean())
    resid = y - (intercept + slope * x)
    sse = float((resid**2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    s_resid = math.sqrt(sse / dof) if dof > 0 else float("nan")
    # The inverse-prediction uncertainty at the centre of the sweep, individual form. This used to
    # be the bare ratio `s_resid/|slope|`, which is a scatter rather than a characterisation
    # uncertainty and is `sqrt(1 + 1/n)` too small: 13.40% low at the three-dose floor, read the
    # other way round the correct value is 15.47% above it.
    u_char = (
        (s_resid / abs(slope)) * math.sqrt(1.0 + 1.0 / n)
        if slope != 0.0 and math.isfinite(s_resid)
        else float("inf")
    )
    note = ""
    if slope == 0.0:
        note = (
            "the readout does not move across the dose range, so no response distinguishes one "
            "dose from another and the assigned value cannot be measured from this sweep at all."
        )
    elif s_resid == 0.0:
        note = (
            "the plants fall exactly on a line, so the fit estimates zero scatter and u_char comes "
            "out at zero. That is the arithmetic and it is not evidence of a perfectly "
            "characterised reference: a noiseless generator and luck produce the same collinear "
            "points, and a straight-line fit cannot tell them apart. Plant at more dose levels or "
            "replicate one of them before quoting this."
        )
    elif dof == 1:
        note = (
            "one residual degree of freedom. The scatter estimate is itself uncertain by roughly a "
            "factor of two, so this u_char is an order of magnitude rather than a figure."
        )
    return DoseResponseFit(
        slope=slope,
        intercept=intercept,
        s_resid=s_resid,
        u_char=u_char,
        r_squared=(1.0 - sse / sst) if sst > 0 else float("nan"),
        n=n,
        dof=dof,
        doses=tuple(float(v) for v in x),
        responses=tuple(float(v) for v in y),
        note=note,
    )


def between_seed_homogeneity(plants: Sequence[Plant]) -> tuple[float | None, dict[str, Any]]:
    """The pooled within-dose standard deviation across seeds, or None when nobody planted twice.

    Pooled rather than computed at one dose, because a homogeneity term is a property of the
    preparation and every dose level in the sweep is a preparation. Pooling by degrees of freedom is
    the standard combination and it is what makes a sweep of three doses at three seeds worth more
    than one dose at three seeds.

    Returns None, not zero, when no dose level reaches `MIN_SEEDS_FOR_HOMOGENEITY`. None is the
    reported state that caps the trust ladder, and the second element says exactly why so the
    capability report can print it.
    """
    by_dose: dict[tuple[float, str], list[float]] = {}
    for p in plants:
        by_dose.setdefault((round(float(p.nominal_dose), 12), p.pipeline), []).append(
            float(p.response)
        )
    seeds_per_dose = {k: len(v) for k, v in by_dose.items()}
    usable = {k: v for k, v in by_dose.items() if len(v) >= MIN_SEEDS_FOR_HOMOGENEITY}
    detail: dict[str, Any] = {
        "n_dose_levels": len(by_dose),
        "seeds_per_dose": {f"{k[0]:g}|{k[1]}": n for k, n in sorted(seeds_per_dose.items())},
        "min_seeds_required": MIN_SEEDS_FOR_HOMOGENEITY,
        "n_pipelines": len({p.pipeline for p in plants}),
    }
    if not usable:
        detail["why_none"] = (
            f"no nominal dose was planted at {MIN_SEEDS_FOR_HOMOGENEITY} or more seeds within one "
            f"pipeline, so there is no set of siblings to take a spread over."
        )
        return None, detail
    num = 0.0
    dof = 0
    for values in usable.values():
        arr = np.asarray(values, dtype=np.float64)
        num += float(((arr - arr.mean()) ** 2).sum())
        dof += arr.size - 1
    detail["pooled_dof"] = dof
    detail["n_dose_levels_used"] = len(usable)
    return math.sqrt(num / dof), detail


def stability_uncertainty(
    points: Sequence[StabilityPoint], *, horizon_steps: int | None = None
) -> tuple[float | None, dict[str, Any]]:
    """Drift over a continuation window, as the slope's standard error times the horizon.

    The form is ISO Guide 35's isochronous stability term, `u_stab = s(b) · t`: fit the re-measured
    response against training steps, take the standard error of the slope, and multiply by the
    horizon the certificate is being claimed over. A slope indistinguishable from zero still
    contributes, through its standard error, which is the point: "we looked and saw no drift" is a
    smaller uncertainty than "we did not look", and it is not zero.

    Returns None when fewer than `MIN_CHECKPOINTS_FOR_STABILITY` re-measurements exist. The response
    must be on the dose scale for the result to be in dose units; a caller working with a
    behavioural readout divides by the dose-response slope first.
    """
    if len(points) < MIN_CHECKPOINTS_FOR_STABILITY:
        return None, {
            "n_checkpoints": len(points),
            "min_required": MIN_CHECKPOINTS_FOR_STABILITY,
            "why_none": (
                "nobody re-measured the assigned value after continued training, or did so at too "
                "few checkpoints to fit a drift slope with a standard error."
            ),
        }
    t = np.asarray([p.steps for p in points], dtype=np.float64)
    y = np.asarray([p.response for p in points], dtype=np.float64)
    stt = float(((t - t.mean()) ** 2).sum())
    if stt == 0.0:
        return None, {
            "n_checkpoints": len(points),
            "why_none": "every re-measurement is at the same step, so there is no window to fit.",
        }
    slope = float(((t - t.mean()) * (y - y.mean())).sum() / stt)
    resid = y - (y.mean() + slope * (t - t.mean()))
    dof = len(points) - 2
    s_resid = math.sqrt(float((resid**2).sum()) / dof) if dof > 0 else float("nan")
    s_slope = s_resid / math.sqrt(stt)
    horizon = float(horizon_steps if horizon_steps is not None else (t.max() - t.min()))
    return abs(s_slope) * horizon, {
        "n_checkpoints": len(points),
        "drift_per_step": slope,
        "s_slope": s_slope,
        "horizon_steps": horizon,
        "window": [float(t.min()), float(t.max())],
    }


# ---------------------------------------------------------------------------
# The certificate
# ---------------------------------------------------------------------------


@register_payload
@dataclass(frozen=True)
class Certificate:
    """What a reference material ships with, or the named absence of it.

    Carries the three terms as plain fields rather than nesting a `ReferenceMaterial`, so the value
    codec has one registered payload to encode and `material()` rebuilds the kernel type on demand.
    The two `None`s are load-bearing: they are what `ReferenceMaterial.uncharacterised` names and
    what `compute_trust` reads to cap the ladder.
    """

    reference_id: str
    kind: str
    assigned_value: float
    u_characterisation: float
    u_homogeneity: float | None
    u_stability: float | None
    matrix_system: str
    matrix_scale: str = ""
    n_plants: int = 0
    n_seeds: int = 0
    n_dose_levels: int = 0
    fit: DoseResponseFit | None = None
    homogeneity_detail: Mapping[str, Any] = None  # type: ignore[assignment]
    stability_detail: Mapping[str, Any] = None  # type: ignore[assignment]
    #: What the dose is measured in. All three terms are in this unit, and two certificates whose
    #: dose units differ are not comparable: a spurious-correlation strength and a mislabel rate are
    #: both dimensionless fractions called "dose" and they are not the same quantity.
    dose_unit: str = "dose"
    note: str = ""

    def __post_init__(self) -> None:
        if self.homogeneity_detail is None:
            object.__setattr__(self, "homogeneity_detail", {})
        if self.stability_detail is None:
            object.__setattr__(self, "stability_detail", {})

    def unit(self) -> Unit:
        """The kernel `Unit` the three terms are expressed in."""
        return Unit(dimension=self.dose_unit, as_printed=self.dose_unit)

    def material(self) -> ReferenceMaterial:
        """The kernel's `ReferenceMaterial`, which is what carries the trust cap.

        The three degrees of freedom travel with the three terms, and they are not decoration.
        `core/budget.py` takes the coverage factor as `max(2, t_p(nu_eff))`, so a budget with no
        degrees of freedom falls back to the conventional `k = 2` and the whole correction is inert.
        This is the path where that matters most: a reference certified at the three-dose floor has
        one residual degree of freedom on the characterisation term, where `t_0.975(1) = 12.7`
        against a conventional 2, a factor of 6.35 on the expanded uncertainty. Reporting an
        interval six times too narrow on the ruler every calibration in the library is performed
        against is the worst place in the library to leave that number unstated.
        """
        return ReferenceMaterial(
            id=self.reference_id,
            kind=self.kind,  # type: ignore[arg-type]
            assigned_value=self.assigned_value,
            u_characterisation=self.u_characterisation,
            u_homogeneity=self.u_homogeneity,
            u_stability=self.u_stability,
            dof_characterisation=(
                float(self.fit.dof) if self.fit is not None and self.fit.dof > 0 else None
            ),
            dof_homogeneity=(
                float(self.homogeneity_detail["pooled_dof"])
                if self.homogeneity_detail.get("pooled_dof")
                else None
            ),
            dof_stability=(
                float(self.stability_detail["n_checkpoints"] - 2)
                if self.stability_detail.get("n_checkpoints", 0) > 2
                else None
            ),
            matrix=MatrixDescription(system=self.matrix_system, scale=self.matrix_scale),
            note=self.note,
        )

    @property
    def rung(self) -> int:
        """0 characterisation only, 1 plus homogeneity, 2 plus stability."""
        if self.u_stability is not None and self.u_homogeneity is not None:
            return 2
        return 1 if self.u_homogeneity is not None else 0

    def render(self) -> str:
        m = self.material()
        lines = [
            f"reference {self.reference_id} ({self.kind}) in {m.matrix.render()}",
            f"  assigned value      {self.assigned_value:.4g}",
            f"  u_characterisation  {self.u_characterisation:.4g}",
            "  u_homogeneity       "
            + (
                f"{self.u_homogeneity:.4g}"
                if self.u_homogeneity is not None
                else "NOT MEASURED  " + str(self.homogeneity_detail.get("why_none", ""))
            ),
            "  u_stability         "
            + (
                f"{self.u_stability:.4g}"
                if self.u_stability is not None
                else "NOT MEASURED  " + str(self.stability_detail.get("why_none", ""))
            ),
            f"  {m.status_line()}",
            f"  rung {self.rung}; {self.n_plants} plants, {self.n_seeds} seed(s), "
            f"{self.n_dose_levels} dose level(s)",
        ]
        if self.fit is not None:
            lines.append(f"  {self.fit.render()}")
        return "\n".join(lines)

    def __canonical__(self) -> dict[str, Any]:
        return {
            "reference_id": self.reference_id,
            "kind": self.kind,
            "assigned_value": self.assigned_value,
            "u_characterisation": self.u_characterisation,
            "u_homogeneity": self.u_homogeneity,
            "u_stability": self.u_stability,
            "matrix_system": self.matrix_system,
            "matrix_scale": self.matrix_scale,
            "n_plants": self.n_plants,
            "n_seeds": self.n_seeds,
            "n_dose_levels": self.n_dose_levels,
            "fit": self.fit.__canonical__() if self.fit is not None else None,
            "homogeneity_detail": dict(self.homogeneity_detail),
            "stability_detail": dict(self.stability_detail),
            "dose_unit": self.dose_unit,
            "note": self.note,
        }


def compare_u_char(a: Certificate, b: Certificate) -> float | Refusal:
    """The difference between two references' characterisation uncertainties, or a unit refusal.

    L1's quantities are declared under the `units` invariance group, whose assertion is not a
    numeric relation: a comparison across a unit boundary refuses rather than converting silently.
    This is the comparison that assertion is about, and it is the one a reader reaches for, because
    "which of these two organism families is better characterised" is the natural question and the
    two families need not be dosed in the same thing.

    The check goes through the kernel's `Unit.compatible_with` rather than a string equality, so an
    undecided unit is incomparable with everything including another undecided one, which is what
    that type already decided and what this must not relitigate.
    """
    if not a.unit().compatible_with(b.unit()):
        return Refusal(
            instrument="compare_u_char",
            reason=RefusalReason.UNIT_MISMATCH,
            detail=(
                f"reference {a.reference_id} is dosed in {a.dose_unit!r} and {b.reference_id} in "
                f"{b.dose_unit!r}. Both are dimensionless fractions and they are not the same "
                f"quantity, so the difference between their characterisation uncertainties is a "
                f"number with no meaning."
            ),
            remedy=(
                "compare references dosed in the same thing, or state the conversion explicitly "
                "and re-certify one of them on the other's scale. The conversion factor is a "
                "property of the two organism families rather than of the unit, so nothing here "
                "can supply it."
            ),
            statistics={"a": a.dose_unit, "b": b.dose_unit},
        )
    return a.u_characterisation - b.u_characterisation


def certify(
    plants: Sequence[Plant],
    *,
    reference_id: str,
    assigned_dose: float,
    matrix: MatrixDescription,
    kind: ReferenceKind = "planted_organism",
    stability: Sequence[StabilityPoint] = (),
    horizon_steps: int | None = None,
    dose_unit: str = "dose",
    note: str = "",
) -> Certificate | Refusal:
    """The three terms from a set of plants, or a refusal when characterisation itself fails.

    A refusal here is narrow and it is the right narrowness. Missing homogeneity and missing
    stability are *reported states*, not failures: the certificate comes back with `None` in those
    fields and the trust cap does the rest, which is what lets an honest rung-0 certificate exist at
    all. What cannot be reported as a state is a characterisation that could not be done, because
    then there is no assigned value with an uncertainty and there is nothing to certify.
    """
    if not plants:
        return Refusal(
            instrument="ReferenceCertificate",
            reason=RefusalReason.REFERENCE_UNCERTIFIED,
            detail=(
                f"reference {reference_id} has no plants at all, so its assigned dose "
                f"{assigned_dose:g} is a nominal figure with nothing measured behind it."
            ),
            remedy=(
                "plant the organism at three or more dose levels and measure the response at each, "
                "then pass the plants here. A single plant at a single dose supports no fit and "
                "therefore no uncertainty on the assigned value."
            ),
            statistics={"assigned_dose": assigned_dose},
        )

    doses = [p.nominal_dose for p in plants]
    responses = [p.response for p in plants]
    fit = dose_response_characterisation(doses, responses)
    u_bb_raw, hom_detail = between_seed_homogeneity(plants)
    u_stab_raw, stab_detail = stability_uncertainty(stability, horizon_steps=horizon_steps)

    # A response that is not already a dose has to be divided through the sensitivity to land in
    # dose units, which is the unit `reference.u_*` is declared in. Mixing the two is the
    # per-token-versus-per-sequence error wearing different clothes.
    kinds = {p.response_kind for p in plants}
    if len(kinds) > 1:
        return Refusal(
            instrument="ReferenceCertificate",
            reason=RefusalReason.REFERENCE_UNCERTIFIED,
            detail=(
                f"the plants report responses of mixed kinds {sorted(kinds)}, so the spread across "
                f"them is not in any one unit."
            ),
            remedy=(
                "certify one response kind at a time. A realised dose and a behavioural readout "
                "are different quantities and their spreads do not compose."
            ),
        )
    on_dose_scale = kinds == {"dose"}
    scale = 1.0 if on_dose_scale else (abs(fit.slope) if fit.slope else float("nan"))
    u_bb = None if u_bb_raw is None else u_bb_raw / scale
    u_stab = None if u_stab_raw is None else u_stab_raw / scale

    if not fit.is_usable:
        material = ReferenceMaterial(
            id=reference_id,
            kind=kind,
            assigned_value=assigned_dose,
            u_characterisation=float("inf"),
            u_homogeneity=u_bb,
            u_stability=u_stab,
            matrix=matrix,
        )
        refusal = uncertified_refusal("ReferenceCertificate", material)
        return replace(
            refusal,
            detail=(
                f"reference {reference_id} cannot be characterised from these plants: {fit.note} "
                f"Without a characterisation term there is no assigned value with an uncertainty, "
                f"so there is nothing to certify."
            ),
            statistics={**refusal.statistics, "fit": fit.__canonical__()},
        )

    # Which inverse-prediction term to take depends on whether `u_bb` is reported beside it. See
    # `DoseResponseFit.u_char_at`: the `1` in `1 + 1/n + dev²/Sxx` and `u_bb` are the same variance,
    # so exactly one of the two has to carry it.
    return Certificate(
        reference_id=reference_id,
        kind=kind,
        assigned_value=float(assigned_dose),
        u_characterisation=float(fit.u_char_at(assigned_dose, individual=u_bb is None)),
        u_homogeneity=u_bb,
        u_stability=u_stab,
        matrix_system=matrix.system,
        matrix_scale=matrix.scale,
        n_plants=len(plants),
        n_seeds=len({p.seed for p in plants}),
        n_dose_levels=len({round(float(p.nominal_dose), 12) for p in plants}),
        fit=fit,
        dose_unit=dose_unit,
        homogeneity_detail=hom_detail,
        stability_detail=stab_detail,
        note=note,
    )


def compose_chain(
    certificate: Certificate,
    *,
    transfers: Sequence[Transfer] = (),
    u_working: float = 0.0,
    u_instrument: float = 0.0,
    working_matrix: MatrixDescription | None = None,
) -> CalibrationChain:
    """`u_total² = u₁² + t₂₁² + t₃₂² + u_CRM² + u_instrument²`, with this certificate as `u_CRM`.

    A thin call on purpose. The kernel already composes the chain and already refuses to publish a
    total when the reference is uncertified; what this adds is that the reference going in came from
    a measurement rather than from a docstring.
    """
    return CalibrationChain(
        reference=certificate.material(),
        transfers=tuple(transfers),
        u_working=u_working,
        u_instrument=u_instrument,
        working_matrix=working_matrix,
    )


# ---------------------------------------------------------------------------
# The instrument
# ---------------------------------------------------------------------------

#: A certificate describes the reference material itself. The twelve envelope conditions are all
#: properties of a training run, and none of them can make a spread across seeds wrong. The one
#: precondition that does bite is that siblings differ only in the seed, and that is checked in
#: `between_seed_homogeneity` by partitioning on the pipeline rather than asserted here.
REFERENCE_ENVELOPE = EnvelopeSpec(
    unconditional=True,
    justification=(
        "a description of the reference material, computed from the plants themselves. It asserts "
        "nothing about the process the reference will later be used to measure, so no regime of "
        "that process can make it wrong. Sibling identity, the precondition that does bite, is "
        "enforced structurally by partitioning the spread on the pipeline."
    ),
)


class ReferenceCertificate(LabelsInstrument):
    """L1: the three uncertainty terms a planted organism should have shipped with and never does.

    Kill condition, from the catalogue: **if homogeneity is negligible across seeds, in which case
    single-seed plants are fine and this is one measurement, once.** The honest status is that the
    data-level homogeneity term is small and the model-level one is unmeasured, so the kill is open.
    Anyone with three trained plants and the compute to make them can close it.

    ``require_certified`` is the switch that separates the two uses. False, the default, is
    describing a reference: an uncertified one produces a certificate that says so, and the trust
    cap keeps it from being read as more. True is *calibrating against* one, which an uncertified
    reference cannot support, and the reading is a `REFERENCE_UNCERTIFIED` refusal carrying the
    lower bound on `u_CRM` as its partial.
    """

    name = "ReferenceCertificate"
    version = "1.0"
    quantity = "reference.u_characterisation"
    capabilities = Capability.NONE
    requires = ACCESS_PLANT
    substrates = frozenset(
        {Substrate.NEURAL_SCALAR, Substrate.NEURAL_GEN, Substrate.PROCEDURAL, Substrate.COMPOSITE}
    )
    envelope = REFERENCE_ENVELOPE
    invariance = "units"
    invariance_relation = Relation("invariant")
    baselines = (
        "a single-seed plant at a nominal dose with no stated uncertainty, which is current "
        "practice everywhere and is what this project's own campaign did",
    )
    rung = 0
    faithful_to = (
        "ISO Guide 35's u_CRM decomposition into characterisation, homogeneity and stability"
    )
    deviations = (
        "characterisation is the inverse-prediction uncertainty of an ordinary least squares line, "
        "and which of its two forms is taken depends on what else the certificate reports. With a "
        "homogeneity term beside it the individual-plant variance is dropped, because `u_bb` is "
        "that variance measured directly and quadrature would count it twice; without one the "
        "full `sqrt(1 + 1/n + dev^2/Sxx)` is taken, because otherwise nothing carries it. The "
        "convention is stated rather than defaulted, in `DoseResponseFit.u_char_at`.",
        "homogeneity pools the within-dose spread across dose levels by degrees of freedom, which "
        "assumes the spread does not itself depend on the dose. A plant whose seed sensitivity "
        "grows with the dose violates that and the pooled term understates it at the top.",
        "stability is ISO Guide 35's isochronous form, the slope's standard error times the "
        "horizon. It carries no term for the measurement uncertainty at each checkpoint.",
        "the three terms are combined in quadrature by the kernel, which assumes they are "
        "uncorrelated. A pipeline whose seed also drives the drift would break that.",
    )

    def __init__(
        self,
        plants: Sequence[Plant] = (),
        *,
        reference_id: str = "",
        assigned_dose: float = float("nan"),
        matrix: MatrixDescription | None = None,
        kind: ReferenceKind = "planted_organism",
        stability: Sequence[StabilityPoint] = (),
        horizon_steps: int | None = None,
        require_certified: bool = False,
        note: str = "",
    ) -> None:
        self.plants = tuple(plants)
        self.reference_id = reference_id or "unnamed-reference"
        self.assigned_dose = assigned_dose
        self.matrix = matrix or MatrixDescription(system="unstated")
        self.kind = kind
        self.stability = tuple(stability)
        self.horizon_steps = horizon_steps
        self.require_certified = require_certified
        self.note = note

    def certificate(self) -> Certificate | Refusal:
        assigned = self.assigned_dose
        if not math.isfinite(assigned) and self.plants:
            assigned = float(np.mean([p.nominal_dose for p in self.plants]))
        return certify(
            self.plants,
            reference_id=self.reference_id,
            assigned_dose=assigned,
            matrix=self.matrix,
            kind=self.kind,
            stability=self.stability,
            horizon_steps=self.horizon_steps,
            note=self.note,
        )

    def measure(self, ctx: Context) -> Any:
        cert = self.certificate()
        if isinstance(cert, Refusal):
            return cert
        material = cert.material()
        return emit_with_reference(
            ctx,
            cert,
            reference=material,
            quantity=self.quantity,
            uncertainty=Uncertainty(
                n=cert.n_plants,
                seed_spread=cert.u_homogeneity,
                method="ISO Guide 35 u_CRM terms; homogeneity None means never measured",
            ),
            subject_extra={
                "reference": cert.reference_id,
                "matrix": material.matrix.render(),
                "certified": str(material.is_certified),
            },
        )

    def estimate(self, ctx: Context | None = None) -> Reading:
        ctx = ctx or Context(readout="score")
        cert = self.certificate()
        if isinstance(cert, Refusal):
            return cert
        material = cert.material()
        if self.require_certified and not material.is_certified:
            bound = emit_with_reference(
                ctx,
                material.u_crm_lower_bound,
                reference=material,
                quantity=self.quantity,
                uncertainty=Uncertainty(
                    n=cert.n_plants, method="lower bound from the terms that were measured"
                ),
                subject_extra={"reference": cert.reference_id, "bound": "u_crm_lower"},
            )
            refusal = uncertified_refusal(self.name, material)
            return replace(
                refusal,
                partial=bound,
                statistics={
                    **refusal.statistics,
                    "u_crm_lower_bound": material.u_crm_lower_bound,
                    "n_seeds": cert.n_seeds,
                    "n_dose_levels": cert.n_dose_levels,
                    **{f"homogeneity.{k}": v for k, v in cert.homogeneity_detail.items()},
                },
            )
        return super().estimate(ctx)


# ---------------------------------------------------------------------------
# Reading the campaign's own organisms
# ---------------------------------------------------------------------------


def campaign_plants(
    store_dir: str,
    *,
    readout: str = "spurious_feature_tracking",
    observable: str = "campaign.organism.record",
    arm: str | None = None,
    sidecar_dirs: Sequence[str] = (),
) -> dict[str, list[Plant]]:
    """The cal-transfer organisms out of the campaign store, one plant list per matrix.

    Each stored organism record carries its arm, its nominal dose, and per-item scores with the
    answer-key labels beside them, so the behavioural response at each dose is recomputed here
    rather than read off a summary. The AUC of the readout against the planted label is the response:
    it is the direct measure of how strongly the planted rule governs behaviour, which is what a
    dose is supposed to control.

    The store is opened read-only through `record.convert.store.CampaignStore`.
    """
    from reward_lens.record.convert.store import CampaignStore
    from reward_lens.stats.roc import roc_pr

    store = CampaignStore(store_dir, sidecar_dirs=sidecar_dirs)
    store.assert_no_blind_payloads()
    out: dict[str, list[Plant]] = {}
    for row in store.by_observable(observable):
        value = store.value(row)
        fields_ = value if isinstance(value, dict) else value.__dict__
        meta = fields_.get("meta") or {}
        if "arm" not in meta or "rho" not in meta:
            continue
        this_arm = str(meta["arm"])
        if arm is not None and this_arm != arm:
            continue
        readouts = meta.get("readouts") or {}
        if readout not in readouts:
            continue
        block = readouts[readout]
        scores = np.asarray(block["scores"], dtype=np.float64)
        labels = np.asarray(block["labels"], dtype=np.float64)
        out.setdefault(this_arm, []).append(
            Plant(
                nominal_dose=float(meta["rho"]),
                response=float(roc_pr(scores, labels).auc),
                seed=0,
                family=str(fields_.get("name", "")),
                pipeline=f"{observable}:{this_arm}:{readout}",
                response_kind=f"auc:{readout}",
            )
        )
    return {k: sorted(v, key=lambda p: p.nominal_dose) for k, v in sorted(out.items())}


def foundry_plants(
    *,
    doses: Sequence[float] = (0.6, 0.7, 0.8, 0.9, 1.0),
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    n: int = 400,
    spurious_feature: str = "cites",
) -> list[Plant]:
    """Plants from the shipped foundry, at several doses and several seeds, measured not assumed.

    The realised dose is `measure_spurious_correlation`, which reads the rendered response text
    rather than the generator's own bookkeeping, so it is a measurement of the data that came out
    rather than a restatement of the argument that went in.

    **This is data-level homogeneity and it is a floor, not the term the Model Organism Lottery is
    about.** Two trunks trained on identical data with different training seeds can express the
    planted rule at different strengths, and that difference is invisible here because no trunk is
    trained. Torch-free by construction, which is what makes it free; the model-level term costs GPU
    and nobody in the field has paid it.
    """
    from reward_lens.organisms.foundry import (
        measure_spurious_correlation,
        spurious_correlation_organism,
    )

    plants: list[Plant] = []
    for rho in doses:
        for seed in seeds:
            view, key = spurious_correlation_organism(
                rho=float(rho), n=n, seed=int(seed), spurious_feature=spurious_feature
            )
            plants.append(
                Plant(
                    nominal_dose=float(rho),
                    response=float(measure_spurious_correlation(view, spurious_feature)),
                    seed=int(seed),
                    family=key.family,
                    pipeline=f"foundry.spurious:{spurious_feature}:n{n}",
                    response_kind="dose",
                )
            )
    return plants


# ---------------------------------------------------------------------------
# The ladder
# ---------------------------------------------------------------------------

_REGISTERED = False


def register() -> None:
    """Register L1's three rungs against the three quantities. Idempotent."""
    global _REGISTERED
    if _REGISTERED:
        return
    rungs = (
        (
            0,
            "reference.u_characterisation",
            "reference.certificate.r0_characterisation",
            BiasStatement(
                direction="downward",
                why=(
                    "the sensitivity-scaled residual omits the inverse-prediction term that grows "
                    "away from the sweep centre, so an assigned dose at the edge of the sweep is "
                    "characterised better on paper than it is in fact."
                ),
            ),
            CostModel(cpu_seconds=1.0, note="a least squares through the dose sweep"),
        ),
        (
            1,
            "reference.u_homogeneity",
            "reference.certificate.r1_homogeneity",
            BiasStatement(
                direction="downward",
                why=(
                    "the spread across seeds captures only the variation the pipeline actually "
                    "randomises. A plant whose data generator is seeded but whose trunk training is "
                    "not gives a homogeneity term that is a floor on the one that matters."
                ),
            ),
            CostModel(
                note="three or more plants, so three times the cost of the baseline practice"
            ),
        ),
        (
            2,
            "reference.u_stability",
            "reference.certificate.r2_stability",
            BiasStatement(
                direction="downward",
                why=(
                    "the slope's standard error over the observed window says nothing about drift "
                    "past the last checkpoint, so a certificate claimed over a longer horizon than "
                    "it was measured over understates it."
                ),
            ),
            CostModel(note="continued training plus a re-measurement at each checkpoint"),
        ),
    )
    for rung, quantity, impl, bias, cost in rungs:
        register_estimator(
            EstimatorEntry(
                quantity=quantity,
                impl=impl,
                requires=ACCESS_PLANT,
                envelope=REFERENCE_ENVELOPE,
                rung=rung,
                bias=bias,
                cost=cost,
                run=certify,
            )
        )
    _REGISTERED = True


__all__ = [
    "MIN_CHECKPOINTS_FOR_STABILITY",
    "MIN_DOSES_FOR_CHARACTERISATION",
    "MIN_SEEDS_FOR_HOMOGENEITY",
    "REFERENCE_ENVELOPE",
    "Certificate",
    "DoseResponseFit",
    "Plant",
    "ReferenceCertificate",
    "StabilityPoint",
    "between_seed_homogeneity",
    "campaign_plants",
    "certify",
    "compare_u_char",
    "compose_chain",
    "dose_response_characterisation",
    "foundry_plants",
    "register",
    "stability_uncertainty",
]

"""The planted dose sweep: intervals on the Hill parameters, and the therapeutic index (C4 rung 1).

`core/budget.py` holds the Hill curve and says, in as many words, that fitting its parameters over a
planted dose sweep is this module's job. `measure/meta/floor.py` already does the point fit,
because a noise floor needs the calibration slope and cannot get one without it. **This module does
not fit a second Hill curve.** `fit_hill` there is called for every point estimate here, including
every bootstrap refit, so the two cannot drift and a change to the bounds or the starting point
happens once.

What is added is the two things C4 rung 1 needs and a noise floor does not.

**Intervals on all three parameters.** The floor module returns `E_max`, `EC50` and `n` with a
residual and no uncertainty, which is right for its purpose: it divides by the slope and moves on. A
dose-response *claim* stands or falls on whether the fitted cooperativity is distinguishable from 1,
because `n` is the parameter that says switch rather than ramp, and a point estimate of 2.4 whose
interval runs from 0.6 to 9 is four points and luck. The intervals here are a paired bootstrap over
the sweep points rather than the curvature at the optimum, because a four-point sweep is exactly the
regime where the asymptotic standard error is estimated from the same four points it is describing.

**The second curve, and the ratio between them.** Sweep the dose against the effect you want *and*
against the capability you are damaging, fit both, and `TI = TD50 / ED50` is a specification rather
than a result. `TI = 40` says there is a wide dose window in which the intervention works and the
host survives. `TI = 1.2` says there is not, and the single-dose number that looked like a success
was a coin toss about where on two overlapping curves you happened to land. The library's own
`SURGERY` result is a therapeutic-index measurement at one dose, which is why sweeping it is C4's
rung 1 and not a refinement.

**What this cannot do.** The fit assumes the response is monotone in the dose and that the residual
scatter does not grow with it. A response that rises and then falls, which is what happens once the
intervention starts damaging the host rather than the concept, is not a Hill curve at all; the floor
module's `HillFit.monotone` is the flag that says so and it is carried through here rather than
swallowed. And nothing here can tell a fitted `n` of 6 from a step function: `MAX_HILL_N` is the
floor module's bound of 20 and a fit that hits it has stopped identifying the parameter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np

from reward_lens.core.budget import CalibrationCurve
from reward_lens.core.evidence import register_payload
from reward_lens.core.reading import Refusal, RefusalReason

if TYPE_CHECKING:
    from reward_lens.measure.meta.floor import DoseSweep, HillFit

#: The floor module's own minimum, restated here so a caller reading this module sees the number it
#: is bound by. Three free parameters plus one point to leave a residual.
MIN_DOSES_FOR_HILL = 4


def _floor() -> Any:
    """The `measure.meta.floor` fitting module, imported on use.

    Lazily, so `organisms` stays importable without pulling `measure` in behind it. The dependency
    runs organisms to measure and not the other way round, which is the direction `measure/base.py`
    documents when it explains why the calibration provider is a seam.
    """
    from reward_lens.measure.meta import floor

    return floor


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DosePoint:
    """One rung of a dose sweep: what was asked for, what it achieved, and what it cost.

    ``effect`` is what the intervention is for: drift removed, concept made undecodable, behaviour
    suppressed. ``cost`` is what it damages: benchmark accuracy lost, held-out loss gained.

    **Both are stated as magnitudes that rise with the dose.** Passing the cost as a signed delta is
    the easy mistake and it does not fail loudly: the fit comes back with a negative `E_max`, the
    floor module reports "the reading does not increase with dose over this range", and a caller who is not
    reading the reason sees a missing therapeutic index rather than a sign error.
    """

    dose: float
    effect: float
    cost: float | None = None
    n: int = 1
    note: str = ""

    def __post_init__(self) -> None:
        if self.dose < 0:
            raise ValueError(f"a dose is a magnitude and cannot be negative; got {self.dose}")


def sweep(
    doses: Sequence[float],
    run: Callable[[float], tuple[float, float | None]],
) -> list[DosePoint]:
    """Run one intervention at each dose and collect `(effect, cost)`.

    A thin loop on purpose. What it buys over writing the loop at the call site is that every point
    carries the dose it was actually taken at, so a sweep whose middle rung failed produces a
    shorter sweep rather than a curve fitted through a hole nobody can see.
    """
    out: list[DosePoint] = []
    for d in doses:
        effect, cost = run(float(d))
        out.append(DosePoint(dose=float(d), effect=float(effect), cost=cost))
    return out


def as_sweep(
    points: Sequence[DosePoint], *, which: str = "effect", dose_unit: str = "dose"
) -> "DoseSweep":
    """The points as a `DoseSweep`, selecting the effect column or the cost column."""
    floor = _floor()
    if which == "cost":
        used = [p for p in points if p.cost is not None]
        values = [float(p.cost) for p in used]  # type: ignore[arg-type]
    else:
        used = list(points)
        values = [float(p.effect) for p in used]
    return floor.DoseSweep(
        doses=np.asarray([p.dose for p in used], dtype=np.float64),
        responses=np.asarray(values, dtype=np.float64),
        dose_unit=dose_unit,
        note=f"{which} column of a {len(used)}-point planted sweep",
    )


# ---------------------------------------------------------------------------
# Intervals on the three parameters
# ---------------------------------------------------------------------------


@register_payload
@dataclass(frozen=True)
class DoseCurve:
    """The floor module's fitted Hill parameters with a bootstrap interval on each.

    ``monotone`` and ``residual_rms`` come straight off `HillFit` and are the two fields a
    reader checks before believing `hill_n`. A non-monotone sweep still fits, and the flag is the
    only thing in the record that says the curve is describing something that is not a curve.
    """

    e_max: float
    ec50: float
    hill_n: float
    e_max_low: float
    e_max_high: float
    ec50_low: float
    ec50_high: float
    hill_n_low: float
    hill_n_high: float
    residual_rms: float
    monotone: bool
    n_points: int
    ci_level: float
    n_boot_converged: int
    n_boot_attempted: int
    dose_unit: str = "dose"
    doses: tuple[float, ...] = ()
    responses: tuple[float, ...] = ()
    note: str = ""

    @property
    def is_switch_like(self) -> bool:
        """Whether the whole interval on `n` sits above 1: cooperative, threshold-shaped.

        The interval, not the point estimate. A large `n` is the Goodhart-hump shape worth finding
        and it is also what a three-point sweep with one outlier produces, so the claim is made from
        the lower endpoint or it is not made.
        """
        return math.isfinite(self.hill_n_low) and self.hill_n_low > 1.0

    @property
    def hit_the_bound(self) -> bool:
        """Whether the fit hit the upper bound on `n`, where the parameter stops meaning much."""
        return self.hill_n >= 20.0 - 1e-6

    def curve(self, at_dose: float | None = None) -> CalibrationCurve:
        """The kernel's `CalibrationCurve`, so the slope composes with the LOD machinery."""
        return CalibrationCurve(
            e_max=self.e_max,
            ec50=self.ec50,
            hill_n=self.hill_n,
            at_dose=at_dose if at_dose is not None else self.ec50,
        )

    def dose_for(self, fraction: float) -> float:
        """The dose reaching ``fraction`` of `E_max`, inverting the Hill curve analytically.

        `c = EC50 · (f / (1 - f))^(1/n)`. Infinite at `f >= 1`, which is the honest answer: the
        curve approaches `E_max` asymptotically and no finite dose arrives. `ED50` is
        `dose_for(0.5)` and equals `EC50` exactly, which is why this is a method rather than a
        second stored field that could disagree with the fit it came from.
        """
        f = float(fraction)
        if f <= 0.0:
            return 0.0
        if f >= 1.0:
            return math.inf
        return float(self.ec50 * (f / (1.0 - f)) ** (1.0 / self.hill_n))

    def render(self) -> str:
        flags = []
        if not self.monotone:
            flags.append("NOT MONOTONE")
        if self.hit_the_bound:
            flags.append("n at the fit bound")
        tail = f"  [{', '.join(flags)}]" if flags else ""
        return (
            f"E_max {self.e_max:.4g} [{self.e_max_low:.4g}, {self.e_max_high:.4g}], "
            f"EC50 {self.ec50:.4g} [{self.ec50_low:.4g}, {self.ec50_high:.4g}] {self.dose_unit}, "
            f"n {self.hill_n:.3g} [{self.hill_n_low:.3g}, {self.hill_n_high:.3g}]; "
            f"rms {self.residual_rms:.4g} on {self.n_points} points{tail}"
        )

    def __canonical__(self) -> dict[str, Any]:
        return {
            "e_max": self.e_max,
            "ec50": self.ec50,
            "hill_n": self.hill_n,
            "e_max_ci": [self.e_max_low, self.e_max_high],
            "ec50_ci": [self.ec50_low, self.ec50_high],
            "hill_n_ci": [self.hill_n_low, self.hill_n_high],
            "residual_rms": self.residual_rms,
            "monotone": self.monotone,
            "n_points": self.n_points,
            "ci_level": self.ci_level,
            "n_boot_converged": self.n_boot_converged,
            "n_boot_attempted": self.n_boot_attempted,
            "dose_unit": self.dose_unit,
            "doses": list(self.doses),
            "responses": list(self.responses),
            "note": self.note,
        }


def _refit(floor: Any, doses: np.ndarray, responses: np.ndarray, unit: str) -> "HillFit":
    return floor.fit_hill(floor.DoseSweep(doses=doses, responses=responses, dose_unit=unit))


def fit_curve(
    points: Sequence[DosePoint],
    *,
    which: str = "effect",
    n_boot: int = 400,
    seed: int = 0,
    level: float = 0.95,
    dose_unit: str = "dose",
) -> DoseCurve | Refusal:
    """The floor module's Hill fit over a sweep, plus a paired bootstrap interval on each parameter.

    Refuses rather than returning a curve when the floor module's fit does not converge or comes
    back with a non-positive `E_max`, carrying its own reason string. Both of those produce three numbers if
    you ask a least-squares routine nicely enough and all three are meaningless, which is the case
    a refusal exists for.
    """
    floor = _floor()
    swept = as_sweep(points, which=which, dose_unit=dose_unit)
    if swept.n_points < MIN_DOSES_FOR_HILL:
        return Refusal(
            instrument="organisms.dose.fit_curve",
            reason=RefusalReason.RECORD_INCOMPLETE,
            detail=(
                f"the {which} column has {swept.n_points} dose point(s) and a Hill curve has three "
                f"free parameters, so it would be fitted with no residual and report an interval of "
                f"zero width on the cooperativity"
            ),
            remedy=(
                f"dose at {MIN_DOSES_FOR_HILL} or more levels spanning the range you care about, "
                f"with at least one below the expected EC50 and one above it. A sweep entirely on "
                f"the plateau pins E_max and leaves EC50 and n unidentified. If the column is the "
                f"cost, measure the capability at every dose you measure the effect at: the "
                f"therapeutic index is the ratio of two curves and half of it is not half an index."
            ),
            statistics={"column": which, "n_points": swept.n_points, "minimum": MIN_DOSES_FOR_HILL},
        )

    # A response with no dynamic range at all fits a curve with a vanishing but *positive* `E_max`,
    # which slips past the floor module's `e_max <= 0` guard: on an identically-zero column the
    # optimiser returns `E_max = 1.4e-9` and `converged` is True. It divides by the slope and a
    # floor built on a 1e-9 curve is enormous rather than wrong, so the guard is adequate there;
    # here the curve is the reading, and reporting `EC50` and `n` for a response that never moved
    # is a confident wrong number. The check is scale-relative rather than absolute because the
    # columns are in whatever units the caller measured them in.
    scale = float(np.max(np.abs(swept.responses))) if swept.responses.size else 0.0
    if swept.dynamic_range <= 1e-9 * max(scale, 1.0):
        return Refusal(
            instrument="organisms.dose.fit_curve",
            reason=RefusalReason.BELOW_LOD,
            detail=(
                f"the {which} column spans {swept.dynamic_range:.4g} across the whole sweep, so the "
                f"response never moved and there is no curve to fit. A least-squares fit still "
                f"returns three parameters here, with a vanishing E_max, and they describe nothing"
            ),
            remedy=(
                "widen the dose range until the response moves, or check that the column is the "
                "one you meant: an effect measured after the intervention already saturated is "
                "flat for the same reason an unconnected readout is."
            ),
            statistics={
                "column": which,
                "dynamic_range": float(swept.dynamic_range),
                "n_points": int(swept.n_points),
            },
        )

    fit = floor.fit_hill(swept)
    if not fit.converged:
        return Refusal(
            instrument="organisms.dose.fit_curve",
            reason=RefusalReason.BELOW_LOD
            if "does not increase" in fit.reason
            else RefusalReason.RECORD_INCOMPLETE,
            detail=f"the {which} curve did not fit: {fit.reason}",
            remedy=(
                "check that the column rises with the dose. Both curves in a dose-response pair are "
                "stated as magnitudes that increase, so a cost passed as a signed delta reads as a "
                "response going the wrong way; and a response that rises then falls is the host "
                "being damaged past a threshold rather than a Hill curve, which wants the sweep "
                "truncated at the turning point."
            ),
            statistics={
                "column": which,
                "reason": fit.reason,
                "monotone": fit.monotone,
                "n_points": fit.n_points,
            },
        )

    d, r = swept.doses, swept.responses
    rng = np.random.default_rng(seed)
    n = int(d.size)
    draws: list[tuple[float, float, float]] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        got = _refit(floor, d[idx], r[idx], dose_unit)
        if got.converged and got.curve is not None:
            draws.append((got.curve.e_max, got.curve.ec50, got.curve.hill_n))
    alpha = (1.0 - level) / 2.0
    if draws:
        arr = np.asarray(draws, dtype=np.float64)
        lo, hi = np.quantile(arr, [alpha, 1.0 - alpha], axis=0)
    else:
        lo = hi = np.full(3, float("nan"))

    note = ""
    if len(draws) < n_boot // 2:
        note = (
            f"only {len(draws)} of {n_boot} bootstrap refits converged, so the intervals are taken "
            f"over the resamples that did and are optimistic by whatever the failures had in common."
        )
    elif swept.n_points == MIN_DOSES_FOR_HILL:
        note = (
            "four points and three parameters: one residual degree of freedom, so the residual is "
            "an order of magnitude rather than a figure and the bootstrap resamples a set most of "
            "whose draws omit a dose level entirely."
        )

    curve = fit.curve
    assert curve is not None  # `converged` is exactly `curve is not None`
    return DoseCurve(
        e_max=float(curve.e_max),
        ec50=float(curve.ec50),
        hill_n=float(curve.hill_n),
        e_max_low=float(lo[0]),
        e_max_high=float(hi[0]),
        ec50_low=float(lo[1]),
        ec50_high=float(hi[1]),
        hill_n_low=float(lo[2]),
        hill_n_high=float(hi[2]),
        residual_rms=float(fit.residual_rms),
        monotone=bool(fit.monotone),
        n_points=int(swept.n_points),
        ci_level=float(level),
        n_boot_converged=len(draws),
        n_boot_attempted=int(n_boot),
        dose_unit=dose_unit,
        doses=tuple(float(v) for v in d),
        responses=tuple(float(v) for v in r),
        note=note,
    )


# ---------------------------------------------------------------------------
# The therapeutic index
# ---------------------------------------------------------------------------


@register_payload
@dataclass(frozen=True)
class DoseResponse:
    """Both curves and the ratio between them: what an intervention buys and what it costs.

    ``therapeutic_index`` is `TD50 / ED50`, the dose costing half the maximum capability over the
    dose achieving half the maximum effect. It is None when the cost curve was not measured or did
    not fit, and None is the honest value: a missing second curve does not make the index 1, and
    almost nobody measures the second curve, which is half the reason the index is worth reporting.
    """

    effect: DoseCurve
    cost: DoseCurve | None
    ed50: float
    td50: float | None
    therapeutic_index: float | None
    effect_level: float
    cost_level: float
    cost_refusal: str = ""
    note: str = ""

    @property
    def has_a_window(self) -> bool:
        """Whether the effective dose is below the damaging one at all.

        `TI > 1` is the bare statement that the effect arrives first. It is not a safety margin: at
        `TI = 1.2` the two curves overlap across most of their range and any single-dose result is a
        coin toss about which one it landed on.
        """
        return self.therapeutic_index is not None and self.therapeutic_index > 1.0

    def render(self) -> str:
        lines = [f"effect  {self.effect.render()}"]
        if self.cost is not None:
            lines.append(f"cost    {self.cost.render()}")
        if self.therapeutic_index is None:
            lines.append(
                f"ED50 {self.ed50:.4g}; no TD50, so no therapeutic index. "
                + (self.cost_refusal or "The cost curve was not measured.")
            )
        else:
            lines.append(
                f"ED50 {self.ed50:.4g}, TD50 {self.td50:.4g}, TI = {self.therapeutic_index:.3g}"
                + ("" if self.has_a_window else "   (no usable dose window)")
            )
        if self.note:
            lines.append(self.note)
        return "\n".join(lines)

    def __canonical__(self) -> dict[str, Any]:
        return {
            "effect": self.effect.__canonical__(),
            "cost": self.cost.__canonical__() if self.cost is not None else None,
            "ed50": self.ed50,
            "td50": self.td50,
            "therapeutic_index": self.therapeutic_index,
            "effect_level": self.effect_level,
            "cost_level": self.cost_level,
            "cost_refusal": self.cost_refusal,
            "note": self.note,
        }


def dose_response(
    points: Sequence[DosePoint],
    *,
    n_boot: int = 400,
    seed: int = 0,
    level: float = 0.95,
    effect_level: float = 0.5,
    cost_level: float = 0.5,
    dose_unit: str = "dose",
) -> DoseResponse | Refusal:
    """Fit both curves over a sweep and report `EC50`, `E_max`, `n` and `TI = TD50/ED50`.

    The effect curve is required and the cost curve is not. A sweep that measured only the effect
    still gives three parameters and a calibration slope, and it gives no therapeutic index. That
    asymmetry is deliberate rather than lenient: it is exactly the state most published interventions
    are in, and the reading should be able to say so by name rather than by silence.

    ``effect_level`` and ``cost_level`` default to the half-maximal convention that gives `ED50` and
    `TD50` their names. Both are settable, because "the dose that removes 90% of the drift" is a
    legitimate question and a caller should not have to invert the fit by hand to ask it.
    """
    if not points:
        return Refusal(
            instrument="organisms.dose.dose_response",
            reason=RefusalReason.RECORD_INCOMPLETE,
            detail="the sweep is empty, so there is nothing to fit",
            remedy=(
                f"run the intervention at {MIN_DOSES_FOR_HILL} or more doses and pass the "
                f"`DosePoint`s. `organisms.dose.sweep` runs the loop given a callable that takes a "
                f"dose and returns `(effect, cost)`."
            ),
        )
    effect = fit_curve(
        points, which="effect", n_boot=n_boot, seed=seed, level=level, dose_unit=dose_unit
    )
    if isinstance(effect, Refusal):
        return effect

    costed = [p for p in points if p.cost is not None]
    cost: DoseCurve | None = None
    cost_refusal = ""
    note = ""
    if costed:
        got = fit_curve(
            points, which="cost", n_boot=n_boot, seed=seed + 1, level=level, dose_unit=dose_unit
        )
        if isinstance(got, Refusal):
            cost_refusal = f"{got.reason.name}: {got.detail}"
        else:
            cost = got
        if len(costed) < len(points):
            note = (
                f"{len(costed)} of {len(points)} sweep points carry a cost. The two curves are "
                f"fitted over different dose sets, so the index compares a TD50 and an ED50 that "
                f"were not measured on the same design."
            )
    else:
        cost_refusal = (
            "no sweep point carries a cost, so this is a single-curve sweep. It gives EC50, E_max, "
            "the cooperativity and a calibration slope, and it cannot give a therapeutic index."
        )

    ed50 = effect.dose_for(effect_level)
    td50 = cost.dose_for(cost_level) if cost is not None else None
    # At the half-maximal convention this is exactly `CalibrationCurve.therapeutic_index`, and it is
    # taken from there rather than recomputed so the ratio lives in one place. Away from the default
    # levels the kernel's method does not apply, because it divides by `EC50` specifically rather
    # than by whichever effective dose was asked for, so the general ratio is formed here and the
    # equality above is what keeps the two consistent where they overlap.
    if td50 is None or not math.isfinite(td50) or ed50 <= 0:
        ti = None
    elif effect_level == 0.5:
        ti = float(effect.curve().therapeutic_index(float(td50)))
    else:
        ti = float(td50 / ed50)
    return DoseResponse(
        effect=effect,
        cost=cost,
        ed50=float(ed50),
        td50=None if td50 is None else float(td50),
        therapeutic_index=ti,
        effect_level=float(effect_level),
        cost_level=float(cost_level),
        cost_refusal=cost_refusal,
        note=note,
    )


# ---------------------------------------------------------------------------
# A reference material with all three terms, made rather than assumed
# ---------------------------------------------------------------------------


def certified_micro_reference(
    *,
    doses: Sequence[float] = (0.5, 0.7, 0.85, 1.0),
    seeds: Sequence[int] = (0, 1, 2),
    stability_epochs: Sequence[int] = (6, 11, 16),
    n_pairs: int = 96,
    spurious_feature: str = "cites",
    reference_id: str = "micro.spurious",
    d_model: int = 32,
    device: str = "cpu",
) -> Any:
    """Train planted micro-organisms at several doses and seeds and certify the family (L1).

    This is the **model-level** certificate `measure/labels/reference.py` says nobody has paid for.
    Its own `foundry_plants` measures the realised dose in the *data*, and says in as many words
    that this is a floor rather than the term the Model Organism Lottery is about: two trunks
    trained on identical data with different seeds can express the planted rule at different
    strengths, and that difference is invisible without training. Here every plant is a trunk that
    was actually trained, the response is the trained model's own behaviour, and the stability arm
    re-measures after further training, which is the `u_stab` term.

    The response is the pairwise margin the trained reward model puts on the planted feature: the
    mean of `r(text with the marker) - r(the same text without it)` over held-out probes. It is a
    behavioural readout rather than a realised dose, so `certify` divides the seed spread and the
    drift through the calibration slope to land all three terms in dose units.

    Roughly eight seconds of CPU per plant on the 32-wide two-layer trunk, so the default four doses
    by three seeds plus three stability checkpoints is about two minutes. Everything is imported on
    use: this module stays torch-free until somebody calls this function.
    """
    import warnings

    from reward_lens.core.reference import MatrixDescription
    from reward_lens.measure.labels.reference import Plant, StabilityPoint, certify
    from reward_lens.organisms._features import render_response
    from reward_lens.organisms._tiny import make_micro_trunk
    from reward_lens.organisms.foundry import spurious_correlation_organism
    from reward_lens.organisms.train import TrainRecipe, train_organism

    def response_of(signal: Any, feature: str) -> float:
        """Mean `r(with the marker) - r(without)` on held-out probes: the behavioural readout."""
        import torch

        topics = ("safety", "coding", "history", "biology")
        tok = signal.tokenizer
        deltas = []
        for topic in topics:
            pair = [render_response(topic, {feature}), render_response(topic, set())]
            batch = tok(pair, return_tensors="pt", padding=True, truncation=True, max_length=64)
            with torch.no_grad():
                logits = signal.model(**batch).logits.reshape(-1).to(torch.float32)
            deltas.append(float(logits[0] - logits[1]))
        return float(np.mean(deltas))

    pipeline = f"micro.spurious:{spurious_feature}:d{d_model}:n{n_pairs}"
    plants: list[Plant] = []
    anchor: Any = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for rho in doses:
            for seed in seeds:
                view, key = spurious_correlation_organism(
                    rho=float(rho), n=n_pairs, seed=int(seed), spurious_feature=spurious_feature
                )
                trunk = make_micro_trunk(d_model=d_model, seed=int(seed), device=device)
                trained = train_organism(trunk, view, TrainRecipe.micro(seed=int(seed)), key)
                plants.append(
                    Plant(
                        nominal_dose=float(rho),
                        response=response_of(trained.signal, spurious_feature),
                        seed=int(seed),
                        family=key.family,
                        pipeline=pipeline,
                        response_kind=f"margin:{spurious_feature}",
                    )
                )
                if anchor is None:
                    anchor = (view, key)

        # Stability, ISO Guide 35's isochronous form: the assigned value re-measured at several
        # points along the continuation. The recipe is deterministic given its seed, so a trunk
        # trained for `e` epochs *is* the checkpoint at `e`, and re-running to each checkpoint costs
        # the same as saving and reloading one on a model this size.
        stability: list[StabilityPoint] = []
        view, key = anchor  # type: ignore[misc]
        base_recipe = TrainRecipe.micro(seed=0)
        for epochs in stability_epochs:
            trunk = make_micro_trunk(d_model=d_model, seed=0, device=device)
            trained = train_organism(
                trunk,
                view,
                TrainRecipe(**{**base_recipe.__dict__, "epochs": int(epochs)}),
                key,
            )
            stability.append(
                StabilityPoint(
                    steps=int(epochs),
                    response=response_of(trained.signal, spurious_feature),
                    seed=0,
                )
            )

    return certify(
        plants,
        reference_id=reference_id,
        assigned_dose=float(np.mean(list(doses))),
        matrix=MatrixDescription(
            system="micro trunk",
            scale=f"LlamaForSequenceClassification d_model={d_model}, 2 layers, CPU",
            note=(
                "the tiny planted organism. A real 8B reward model is a different matrix and this "
                "certificate does not transfer to it; that gap is what a Transfer term is for."
            ),
        ),
        kind="planted_organism",
        stability=stability,
        dose_unit="rho",
        note=(
            "model-level: every plant is a trunk that was trained, and the response is the trained "
            "model's own pairwise margin on the planted marker rather than a property of the data "
            "it was trained on."
        ),
    )


__all__ = [
    "MIN_DOSES_FOR_HILL",
    "DoseCurve",
    "DosePoint",
    "DoseResponse",
    "as_sweep",
    "certified_micro_reference",
    "dose_response",
    "fit_curve",
    "sweep",
]

"""The contract model's parameters, each carrying where it came from.

The formula this layer is built on needs five things: the marginal value of each task `B'`, the
curvature of the agent's effort cost `C''`, the grader noise covariance `Sigma`, the agent's risk
aversion `r`, and the sensitivity matrix `M` that says how much each signal moves per unit of effort.
Exactly one of those five is measurable from a reward record, and it is `Sigma`, which A2 produces.

That asymmetry is the reason this module exists. A recommendation assembled from one measured number
and four silent defaults looks exactly like a recommendation assembled from five measured numbers,
and it is the same failure the whole library is built against. So every parameter carries a
`ParameterSource`, no parameter has a plausible default, and an instrument asked for a reading that
depends on an `UNKNOWN` parameter returns a refusal naming which one and what would close it.

**The units trap, which is not in the source and is the most consequential thing here.** The
diagonal formula `alpha_i = B_i / (1 + r C_ii sigma_i^2)` is not scale-free. Rescale a component's
score by `a` and its variance goes to `a^2 sigma^2`, while the `1` in the denominator does not move,
so the recommended weight changes by something other than the `1/a` that a change of units demands.
The resolution is that the formula is stated at `M = I`, meaning one unit of effort moves the signal
by exactly one unit, and rescaling the signal breaks that normalisation. The general form

    alpha* = [M C''^-1 M' + r Sigma]^-1 M C''^-1 B'

is exactly covariant: send `(Sigma, M)` to `(a^2 Sigma, a M)` and `alpha*` goes to `alpha*/a`, which
is what a change of units is supposed to do. Reward model outputs have arbitrary scale, and the
eleven open models in this project's own campaign store have standard deviations spanning two orders
of magnitude on the same items, so applying the diagonal formula to raw scores is a units error
rather than an approximation. `M` is therefore a first-class parameter with no default, and
`assume_unit_sensitivity` is a named act that records itself.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

import numpy as np

from reward_lens.core.reading import Refusal, RefusalReason

# ---------------------------------------------------------------------------
# Where a parameter came from
# ---------------------------------------------------------------------------


class ParameterSource(enum.Enum):
    """How this library came to hold a value for a parameter. Five states, all of them different.

    `ASSUMED` is the one that earns its place. It is not `SUPPLIED`, because the caller did not
    choose it, and it is not `MEASURED`, because nothing measured it. It is a default this package
    picked, and the only defaults this package picks are ones it can name in a sentence on the
    reading. There is exactly one: the unit sensitivity matrix.
    """

    #: Produced by an instrument in this library, from data, with a named quantity behind it.
    MEASURED = enum.auto()
    #: The caller stated it. Recorded as stated, never promoted to measured.
    SUPPLIED = enum.auto()
    #: A default this package chose, named on the reading with what it costs.
    ASSUMED = enum.auto()
    #: Not fixed at all. The reading is a function of it over a stated range.
    SWEPT = enum.auto()
    #: Nobody said, nothing measured it, and no range was given. Instruments refuse.
    UNKNOWN = enum.auto()


#: The five parameters of the model, in the order they appear in the formula.
PARAMETER_NAMES: tuple[str, ...] = (
    "benefit",
    "cost_curvature",
    "noise",
    "risk_aversion",
    "sensitivity",
)

#: The sixth thing some readings need. Not part of the optimal-weight formula, which is why it is
#: listed apart: N7's information coefficient divides the effort-equivalent noise by the effort.
EFFORT_NAME = "effort"

#: What each parameter is, for a refusal that names something the reader can act on.
PARAMETER_MEANING: Mapping[str, str] = {
    "benefit": (
        "B', the principal's marginal value of one more unit of effort on each task, in whatever "
        "units the principal's objective is measured in. Only ratios between components matter"
    ),
    "cost_curvature": (
        "C'', the Hessian of the agent's effort cost. Its off-diagonal is the substitutability "
        "between tasks and it is the term that produces every interesting result in this layer"
    ),
    "noise": (
        "Sigma, the covariance of the grader noise across components. A2 measures this from a "
        "crossed replication design"
    ),
    "risk_aversion": (
        "r, the agent's coefficient of absolute risk aversion. It appears only in the product "
        "r * C'' * Sigma, so a sweep over r at fixed C'' is also a sweep over C'' at fixed r"
    ),
    "sensitivity": (
        "M, with M[i, j] the derivative of signal i's mean with respect to effort on task j. The "
        "diagonal is each component's dose-response slope, measurable by a weight sweep"
    ),
    EFFORT_NAME: (
        "t, the effort the agent spends on each task at the operating point. It enters only the "
        "information coefficient rho = n / t, so a common scaling of it leaves every ordering "
        "unchanged"
    ),
}

#: What to do about each one when it is unknown. A remedy is an instruction.
PARAMETER_REMEDY: Mapping[str, str] = {
    "benefit": (
        "state the relative value of the components. Even an ordinal statement helps: the "
        "recommendation is homogeneous of degree one in B', so only the ratios matter and equal "
        "values is a defensible starting point if you say you used it"
    ),
    "cost_curvature": (
        "state C''. Two numbers are enough for the two-task reading: a common diagonal scale and "
        "the substitutability. Or pass a sweep and read the recommendation as a function of it"
    ),
    "noise": (
        "run A2 on a crossed design: score the same items with two or more grader draws and pass "
        "`noise_from_gauge_studies`. Two graders on fifty shared items produces a Sigma and needs "
        "no training run"
    ),
    "risk_aversion": (
        "state r, or pass `Sweep.for_risk_aversion()` and read the recommendation as a function of "
        "it. The sweep is the honest answer when nobody knows r, and it usually shows the ordering "
        "of the components is stable even where the weights are not"
    ),
    "sensitivity": (
        "measure the dose-response slope by a weight sweep: perturb one component's weight, "
        "re-optimise, and record how far that component's score moved. Or call "
        "`assume_unit_sensitivity()`, which records the normalisation on the reading rather than "
        "applying it silently"
    ),
    EFFORT_NAME: (
        "state the effort allocation across components, or pass `assume_equal_effort=True`, which "
        "leaves every ordering by the information coefficient unchanged and says so on the reading"
    ),
}


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Sweep:
    """A parameter reported as a range rather than as a value.

    Sensitivity to an unmeasured parameter is part of the reading and not a footnote, so a sweep is
    a first-class input rather than something a caller assembles by looping. ``reason`` says why
    this range and not another, because a range with no justification is a default wearing a
    disguise.
    """

    parameter: str
    values: np.ndarray
    reason: str

    def __post_init__(self) -> None:
        v = np.asarray(self.values, dtype=np.float64).ravel()
        if v.size < 2:
            raise ValueError(
                f"a sweep over {self.parameter!r} needs at least two values; got {v.size}. One "
                f"value is a supplied parameter, and calling it a sweep hides that."
            )
        if not np.all(np.isfinite(v)):
            raise ValueError(f"sweep over {self.parameter!r} carries non-finite values")
        object.__setattr__(self, "values", v)

    @classmethod
    def for_risk_aversion(cls, low: float = 1e-3, high: float = 1e3, n: int = 61) -> "Sweep":
        """Six decades of `r`, log spaced.

        The range is wide on purpose. Nobody has measured an RL policy's coefficient of absolute
        risk aversion and the literature on human agents does not transfer, so a narrow range would
        be a claim. What six decades buys is the useful statement: over how much of it does the
        recommended *ordering* of the components hold. That question has an answer even where the
        weights do not.
        """
        return cls(
            parameter="risk_aversion",
            values=np.logspace(math.log10(low), math.log10(high), int(n)),
            reason=(
                f"r is not measurable from a reward record and no published estimate applies to a "
                f"policy, so it is swept over {low:g} to {high:g} rather than assumed. r enters "
                f"only through r * C'' * Sigma, so this is simultaneously a sweep over a common "
                f"scaling of the effort-cost curvature."
            ),
        )

    def render(self) -> str:
        return (
            f"{self.parameter}: swept over {self.values.size} values from "
            f"{self.values.min():.4g} to {self.values.max():.4g}. {self.reason}"
        )


# ---------------------------------------------------------------------------
# The parameter set
# ---------------------------------------------------------------------------


def _symmetric(a: np.ndarray, tol: float = 1e-9) -> bool:
    return bool(np.allclose(a, a.T, atol=tol, rtol=0.0))


@dataclass(frozen=True)
class ContractParameters:
    """The five parameters of the linear-contract model, plus where each one came from.

    Nothing here has a plausible default. ``sensitivity`` and ``effort`` are None until somebody
    says otherwise, and `source` records `UNKNOWN` for them, which is what an instrument consults
    before it computes anything.
    """

    components: tuple[str, ...]
    #: B', shape (m,). The principal's marginal value of effort on each task.
    benefit: np.ndarray
    #: C'', shape (m, m), symmetric positive semidefinite.
    cost_curvature: np.ndarray
    #: Sigma, shape (m, m), symmetric positive semidefinite.
    noise: np.ndarray
    #: r, the agent's absolute risk aversion.
    risk_aversion: float
    #: M, shape (m, m). M[i, j] = d mu_i / d t_j. None until stated or assumed.
    sensitivity: np.ndarray | None = None
    #: t, shape (m,). Effort at the operating point. None until stated or assumed.
    effort: np.ndarray | None = None
    #: Task indices with no signal at all. Their optimal weight is zero by construction, not by
    #: arithmetic, and every reading says which they are.
    unmeasurable: tuple[int, ...] = ()
    source: Mapping[str, ParameterSource] = field(default_factory=dict)
    note: Mapping[str, str] = field(default_factory=dict)

    # -- construction ------------------------------------------------------

    def __post_init__(self) -> None:
        b = np.asarray(self.benefit, dtype=np.float64).ravel()
        m = b.size
        if m < 1:
            raise ValueError("a contract needs at least one component")
        if len(self.components) != m:
            raise ValueError(
                f"{len(self.components)} component name(s) for {m} benefit entries. The names are "
                f"not decoration: every table this layer prints is keyed by them."
            )
        c = np.asarray(self.cost_curvature, dtype=np.float64)
        s = np.asarray(self.noise, dtype=np.float64)
        for name, arr in (("cost_curvature", c), ("noise", s)):
            if arr.shape != (m, m):
                raise ValueError(f"{name} must be ({m}, {m}); got {arr.shape}")
            if not _symmetric(arr):
                raise ValueError(
                    f"{name} is not symmetric. C'' is a Hessian and Sigma is a covariance, so an "
                    f"asymmetric one is a transcription error rather than an unusual case."
                )
        if not np.all(np.isfinite(b)) or not np.all(np.isfinite(c)) or not np.all(np.isfinite(s)):
            raise ValueError("benefit, cost_curvature and noise must all be finite")
        if float(self.risk_aversion) < 0.0:
            raise ValueError(
                f"risk aversion is {self.risk_aversion:g}. A negative r is a risk-loving agent, "
                f"which reverses the sign of the whole shrinkage term; if that is what you mean, "
                f"the model this layer implements is not the one you want."
            )
        eig_c = float(np.min(np.linalg.eigvalsh(c)))
        if eig_c < -1e-9 * max(1.0, float(np.max(np.abs(c)))):
            raise ValueError(
                f"cost_curvature has a negative eigenvalue ({eig_c:.6g}), so the agent's cost is "
                f"not convex and the first-order condition does not characterise its choice."
            )
        eig_s = float(np.min(np.linalg.eigvalsh(s)))
        if eig_s < -1e-9 * max(1.0, float(np.max(np.abs(s)))):
            raise ValueError(
                f"noise has a negative eigenvalue ({eig_s:.6g}), so it is not a covariance matrix."
            )
        object.__setattr__(self, "benefit", b)
        object.__setattr__(self, "cost_curvature", c)
        object.__setattr__(self, "noise", s)
        object.__setattr__(self, "risk_aversion", float(self.risk_aversion))

        if self.sensitivity is not None:
            sm = np.asarray(self.sensitivity, dtype=np.float64)
            if sm.shape != (m, m):
                raise ValueError(f"sensitivity must be ({m}, {m}); got {sm.shape}")
            object.__setattr__(self, "sensitivity", sm)
        if self.effort is not None:
            t = np.asarray(self.effort, dtype=np.float64).ravel()
            if t.size != m:
                raise ValueError(f"effort must have {m} entries; got {t.size}")
            object.__setattr__(self, "effort", t)
        bad = [i for i in self.unmeasurable if not 0 <= i < m]
        if bad:
            raise ValueError(f"unmeasurable task index out of range: {bad}")

        src = dict(self.source)
        for name in (*PARAMETER_NAMES, EFFORT_NAME):
            src.setdefault(name, ParameterSource.UNKNOWN)
        if self.sensitivity is None:
            src["sensitivity"] = ParameterSource.UNKNOWN
        if self.effort is None:
            src[EFFORT_NAME] = ParameterSource.UNKNOWN
        object.__setattr__(self, "source", src)
        object.__setattr__(self, "note", dict(self.note))

    @classmethod
    def supplied(
        cls,
        components: Sequence[str],
        *,
        benefit: Any,
        cost_curvature: Any,
        noise: Any,
        risk_aversion: float,
        sensitivity: Any = None,
        effort: Any = None,
        unmeasurable: Sequence[int] = (),
        note: str = "",
        source: Mapping[str, ParameterSource] | None = None,
        notes: Mapping[str, str] | None = None,
    ) -> "ContractParameters":
        """Everything stated by the caller, and recorded as stated.

        The default source for every field here is `SUPPLIED` rather than `MEASURED`, and the only
        way to get `MEASURED` on a field is to say so in ``source``, which is what makes the flag
        mean something. ``source`` is a mapping rather than keyword arguments because the parameter
        names collide with the value arguments, and a caller who wrote `risk_aversion=SUPPLIED`
        meaning the provenance would silently be setting the value instead.
        """
        names = tuple(components)
        src: dict[str, ParameterSource] = {
            n: ParameterSource.SUPPLIED for n in (*PARAMETER_NAMES, EFFORT_NAME)
        }
        if source:
            unknown_keys = sorted(set(source) - set(src))
            if unknown_keys:
                raise ValueError(
                    f"source names {', '.join(unknown_keys)}, which is not a parameter of this "
                    f"model. The parameters are {', '.join(src)}."
                )
            src.update(source)
        if sensitivity is None:
            src["sensitivity"] = ParameterSource.UNKNOWN
        if effort is None:
            src[EFFORT_NAME] = ParameterSource.UNKNOWN
        return cls(
            components=names,
            benefit=np.asarray(benefit, dtype=np.float64).ravel(),
            cost_curvature=np.asarray(cost_curvature, dtype=np.float64),
            noise=np.asarray(noise, dtype=np.float64),
            risk_aversion=float(risk_aversion),
            sensitivity=None if sensitivity is None else np.asarray(sensitivity, np.float64),
            effort=None if effort is None else np.asarray(effort, np.float64).ravel(),
            unmeasurable=tuple(int(i) for i in unmeasurable),
            source=src,
            note={**({"contract": note} if note else {}), **(dict(notes) if notes else {})},
        )

    # -- shape -------------------------------------------------------------

    @property
    def m(self) -> int:
        return int(self.benefit.size)

    @property
    def measurable(self) -> tuple[int, ...]:
        """Task indices that have a signal. The complement of `unmeasurable`."""
        return tuple(i for i in range(self.m) if i not in set(self.unmeasurable))

    @property
    def cost_condition_number(self) -> float:
        """How close C'' is to singular, which is how close the tasks are to perfect substitutes.

        Reported on every reading that uses `C''`, because the optimal weight stays finite as C''
        becomes singular while the effort it induces does not. A recommendation quoted without this
        number hides the fact that the agent's response to it is unbounded.
        """
        eig = np.linalg.eigvalsh(self.cost_curvature)
        lo, hi = float(np.min(np.abs(eig))), float(np.max(np.abs(eig)))
        return math.inf if lo <= 0.0 else hi / lo

    # -- provenance --------------------------------------------------------

    def unknown(self, names: Sequence[str] | None = None) -> tuple[str, ...]:
        """Which of the named parameters nobody has stated. Empty means the reading can proceed."""
        wanted = tuple(names) if names is not None else (*PARAMETER_NAMES, EFFORT_NAME)
        return tuple(n for n in wanted if self.source.get(n) is ParameterSource.UNKNOWN)

    def refuse_unstated(self, instrument: str, needs: Sequence[str]) -> Refusal | None:
        """The refusal a reading earns when it depends on a parameter nobody has stated.

        `ACCESS_INSUFFICIENT` rather than `RECORD_INCOMPLETE`, and the test that decides it is
        whether the remedy is answerable where the reader is standing. It is: stating r, or sweeping
        it, or running a crossed design for Sigma, are all things the reader does from here. Nothing
        upstream in whoever wrote the record has to change.
        """
        missing = self.unknown(needs)
        if not missing:
            return None
        detail = "; ".join(f"{n} is unstated: {PARAMETER_MEANING[n]}" for n in missing)
        remedy = " ".join(f"For {n}: {PARAMETER_REMEDY[n]}." for n in missing)
        return Refusal(
            instrument=instrument,
            reason=RefusalReason.ACCESS_INSUFFICIENT,
            detail=(
                f"this recommendation depends on {len(missing)} parameter(s) nobody has stated, "
                f"and a weight produced from a default would be indistinguishable from a weight "
                f"produced from a measurement. {detail}"
            ),
            remedy=remedy,
            statistics={
                "missing": list(missing),
                "stated": {n: s.name for n, s in sorted(self.source.items())},
            },
        )

    def with_source(self, **sources: ParameterSource) -> "ContractParameters":
        merged = dict(self.source)
        merged.update(sources)
        return replace(self, source=merged)

    def with_note(self, **notes: str) -> "ContractParameters":
        merged = dict(self.note)
        merged.update(notes)
        return replace(self, note=merged)

    def assume_unit_sensitivity(self) -> "ContractParameters":
        """Set `M = I`, and record that this package did it and what it costs.

        `M = I` is the normalisation the diagonal formula is stated in: one unit of effort moves
        each signal by exactly one unit of that signal's own scale. It is not innocuous. It ties the
        recommendation to the scale the component happens to be measured in, so the same reward
        model reported on a different scale gets a different recommended weight for no reason that
        is about the reward model. Gauge-fix the components first, or measure M.
        """
        return replace(
            self,
            sensitivity=np.eye(self.m, dtype=np.float64),
            source={**self.source, "sensitivity": ParameterSource.ASSUMED},
            note={
                **self.note,
                "sensitivity": (
                    "M = I, assumed. One unit of effort is defined as the amount that moves each "
                    "signal by one unit of its own scale, which is the normalisation the diagonal "
                    "formula is stated in. The recommendation is then tied to the scale each "
                    "component happens to be reported on: rescale a component by a and its "
                    "recommended weight changes by something other than 1/a. Gauge-fix the "
                    "components before reading this, or measure M by a weight sweep."
                ),
            },
        )

    def assume_equal_effort(self, level: float = 1.0) -> "ContractParameters":
        """Set every task's effort to a common value, and record it.

        The information coefficient is `rho_i = n_i / t_i`, so a common `t` cancels from every
        comparison between components. Every ordering, every partition and therefore the sorting
        cutoff's *membership* are unchanged by this assumption; only the cutoff's numeric value in
        `rho` units moves, and it moves by the same factor for every component.
        """
        return replace(
            self,
            effort=np.full(self.m, float(level), dtype=np.float64),
            source={**self.source, EFFORT_NAME: ParameterSource.ASSUMED},
            note={
                **self.note,
                EFFORT_NAME: (
                    f"t = {level:g} for every component, assumed. A common effort level cancels "
                    f"from every ratio between components, so the ordering by rho and the "
                    f"partition it induces are exactly what they would be under any other common "
                    f"level. Only the cutoff's numeric value depends on this."
                ),
            },
        )

    # -- the gauge action --------------------------------------------------

    def rescale(self, a: float) -> "ContractParameters":
        """The group action of `reward.affine` on this parameter set: `r -> a*r + b`.

        Rescaling every component's score by `a` sends `Sigma` to `a^2 Sigma` and `M` to `a M`, and
        leaves `B'`, `C''`, `r` and `t` alone, because those live in effort space and effort has not
        been rescaled. The additive part `b` acts on no parameter at all: a covariance and a
        derivative are both blind to a constant.

        Under this action `alpha*` goes to `alpha*/a` exactly, which is the whole content of the
        weight vector being covariant with weight -1. Every other reading in this package is
        invariant under it.
        """
        if a == 0.0:
            raise ValueError("a = 0 is not a group element of reward.affine; it is a projection")
        return replace(
            self,
            noise=(a * a) * self.noise,
            sensitivity=None if self.sensitivity is None else a * self.sensitivity,
        )

    # -- reporting ---------------------------------------------------------

    def provenance_rows(self) -> list[dict[str, str]]:
        rows = []
        for name in (*PARAMETER_NAMES, EFFORT_NAME):
            rows.append(
                {
                    "parameter": name,
                    "source": self.source.get(name, ParameterSource.UNKNOWN).name,
                    "note": self.note.get(name, ""),
                }
            )
        return rows

    def render_provenance(self, indent: str = "  ") -> str:
        lines = [f"{indent}Where each parameter came from:"]
        for row in self.provenance_rows():
            line = f"{indent}  {row['parameter']:<16} {row['source']}"
            lines.append(line)
            if row["note"]:
                lines.append(f"{indent}      {row['note']}")
        measured = sum(
            1 for r in self.provenance_rows() if r["source"] == ParameterSource.MEASURED.name
        )
        lines.append(
            f"{indent}  {measured} of {len(PARAMETER_NAMES) + 1} parameters were measured. The "
            f"rest were stated or assumed, and the recommendation is a claim about the model, not "
            f"about the grader, wherever they were not."
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Consuming A2
# ---------------------------------------------------------------------------


def noise_from_gauge_studies(
    studies: Sequence[tuple[str, Any]],
    correlation: Any,
) -> tuple[np.ndarray, str]:
    """`Sigma` from A2: the diagonal is each component's gauge variance, the rest is supplied.

    ``studies`` is `(component name, GaugeStudy)` pairs, one per component, in the order the
    components appear in the contract. Each study's `gauge.sigma_grr ** 2` is everything in that
    component's score variance that is not the object of measurement, which is exactly the noise the
    contract model's `Sigma` is the covariance of.

    ``correlation`` has no default and that is deliberate. Grader noise is correlated across
    components far more often than not (two judges built on the same base model share their
    mistakes), and an identity default would put a strong and usually false claim into every
    recommendation with nothing on the reading to show for it. Pass a measured correlation matrix,
    or pass `numpy.eye(m)` and own the assumption.

    Returns the matrix and a one-line note naming the rungs it came from, for the reading.
    """
    sd = np.array([float(s.gauge.sigma_grr) for _, s in studies], dtype=np.float64)
    rho = np.asarray(correlation, dtype=np.float64)
    m = sd.size
    if rho.shape != (m, m):
        raise ValueError(f"correlation must be ({m}, {m}) for {m} components; got {rho.shape}")
    if not _symmetric(rho):
        raise ValueError("the noise correlation matrix is not symmetric")
    if not np.allclose(np.diag(rho), 1.0, atol=1e-8):
        raise ValueError(
            "the noise correlation matrix does not have a unit diagonal. Pass a correlation, not a "
            "covariance: the diagonal comes from A2 and passing a covariance would square it."
        )
    sigma = rho * np.outer(sd, sd)
    rungs = ", ".join(f"{n} rung {s.rung}" for n, s in studies)
    note = (
        f"Sigma diagonal from A2 grader.variance_components: sigma_GRR^2 per component ({rungs}). "
        f"Off-diagonal from the supplied correlation matrix."
    )
    return sigma, note


def noise_correlation_from_residuals(residuals: Any) -> np.ndarray:
    """The correlation of grader noise across components, from residuals on shared items.

    ``residuals`` is `(n, m)`: one row per scored cell, one column per component, holding what is
    left of each component's score after its own object effect is removed. Two components that saw
    the same cell contribute a paired observation, which is what makes the off-diagonal estimable at
    all. A design where the components scored different items cannot produce this and should pass
    an explicit identity instead of a number computed from nothing.
    """
    x = np.asarray(residuals, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError(f"residuals must be (n, m) with n >= 2; got shape {x.shape}")
    sd = x.std(axis=0, ddof=1)
    if np.any(sd <= 0.0):
        flat = [int(i) for i in np.flatnonzero(sd <= 0.0)]
        raise ValueError(
            f"component(s) {flat} have zero residual spread, so their noise correlation with "
            f"anything is undefined rather than zero."
        )
    return np.corrcoef(x, rowvar=False)


__all__ = [
    "EFFORT_NAME",
    "PARAMETER_MEANING",
    "PARAMETER_NAMES",
    "PARAMETER_REMEDY",
    "ContractParameters",
    "ParameterSource",
    "Sweep",
    "noise_correlation_from_residuals",
    "noise_from_gauge_studies",
]

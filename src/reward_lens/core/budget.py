"""The uncertainty budget and the limits of detection.

Two things live here and they answer two different questions. The budget answers "how wrong is
this number, and which part of the apparatus is responsible?" The limits of detection answer "is
this number distinguishable from the substrate's disagreement with itself at all?"

**The budget is a table, not an interval.** A confidence interval is one number that has already
thrown away the thing worth knowing, which is *which term dominates*. The GUM (the Guide to the
Expression of Uncertainty in Measurement, ISO/IEC 98-3) formalises the alternative: enumerate every
contribution, state for each whether it was evaluated statistically (Type A) or by judgement
(Type B), give each a sensitivity coefficient, and compose in quadrature. The composition is
arithmetic and a property test asserts it. The payload is the last line of the table: **the largest
term is almost never sampling noise**, and a budget that cannot say so is not doing its job.

The Type A / Type B split is worth keeping even though both are treated identically once they are
standard uncertainties, because the split records *how you know*. "Type B, rectangular, half-width
from the vLLM-versus-HuggingFace residual comparison" is auditable. A pooled interval is not.

**The limits of detection turn the numerics floor into a decision rule.** Analytical chemistry has
had `LOD = 3.3 σ_blank / S` for decades; machine learning has nothing, and reports effect sizes
below its own substrate noise routinely. Three outcomes rather than two: refuse below the LOD,
return a bound between LOD and LOQ, report with a budget above the LOQ.

`S`, the sensitivity, is the slope of the calibration curve of reading against dose. It is not
assumed here: `CalibrationCurve` holds the Hill parameters and computes the slope, and fitting
those parameters over a planted dose sweep is `organisms/dose.py`'s job. The seam is
deliberate, so that an instrument with no dose sweep cannot silently invent a sensitivity of 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

#: `scipy.stats` costs about a fifth of a second to import and `core.budget` is pulled in by
#: `core.evidence`, which is pulled in by everything. The Student t quantile is needed only when a
#: budget actually has degrees of freedom, so the two distributions are fetched on first use and
#: cached here rather than imported at module scope.
_STUDENT_T: Any = None
_NORMAL: Any = None


def _distributions() -> tuple[Any, Any]:
    """`(t, norm)` from `scipy.stats`, imported once."""
    global _STUDENT_T, _NORMAL
    if _STUDENT_T is None:
        from scipy.stats import norm, t

        _STUDENT_T, _NORMAL = t, norm
    return _STUDENT_T, _NORMAL


#: Type A is evaluated by statistics on repeated observations. Type B is evaluated by judgement:
#: a manufacturer's specification, a published comparison, a bound the physics implies. The GUM is
#: explicit that both are standard uncertainties and compose identically; the letter records the
#: provenance of the number, not a difference in how it is treated.
UncertaintyType = Literal["A", "B"]

#: How a Type B half-width converts to a standard uncertainty. These divisors are the whole of the
#: Type B evaluation and getting one wrong scales a term by up to 1.7x.
DIVISORS: dict[str, float] = {
    "normal": 1.0,  # the value supplied is already a standard deviation
    "rectangular": math.sqrt(3.0),  # equally likely anywhere in +/- a
    "triangular": math.sqrt(6.0),  # peaked at centre, zero at +/- a
    "u_shaped": math.sqrt(2.0),  # arcsine; a cyclic effect sampled at an unknown phase
}


@dataclass(frozen=True)
class BudgetTerm:
    """One contribution to the combined uncertainty.

    ``value`` is the **standard** uncertainty of the input quantity, in that input's own units.
    ``sensitivity`` is the partial derivative of the reading with respect to that input, so
    ``sensitivity * value`` is the contribution in the reading's units. Keeping them separate is
    what lets a budget say "the grader replication variance is small but the reading is enormously
    sensitive to it", which a pre-multiplied contribution cannot express.

    ``dof`` is the degrees of freedom behind a Type A term, used for the Welch-Satterthwaite
    effective degrees of freedom. Absent is honest and common for Type B; the budget then reports
    no effective dof rather than assuming infinity.
    """

    name: str
    value: float
    kind: UncertaintyType = "A"
    distribution: str = "normal"
    sensitivity: float = 1.0
    dof: float | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError(
                f"term {self.name!r} has a negative standard uncertainty ({self.value}). An "
                f"uncertainty is a magnitude; a sign belongs on the sensitivity coefficient."
            )
        if self.distribution not in DIVISORS:
            raise ValueError(
                f"term {self.name!r} declares distribution {self.distribution!r}; known "
                f"distributions are {sorted(DIVISORS)}. Use `from_half_width` if you have a bound "
                f"rather than a standard deviation."
            )

    @classmethod
    def from_half_width(
        cls,
        name: str,
        half_width: float,
        distribution: str = "rectangular",
        *,
        sensitivity: float = 1.0,
        note: str = "",
    ) -> "BudgetTerm":
        """A Type B term stated as a bound, converted to a standard uncertainty by its divisor.

        This is the constructor most Type B terms should use, because most Type B knowledge
        arrives as "it is somewhere within +/- a" rather than as a standard deviation. Dividing by
        the right divisor is the entire Type B evaluation and doing it in the constructor is what
        stops it being skipped.
        """
        try:
            divisor = DIVISORS[distribution]
        except KeyError:
            raise ValueError(
                f"unknown distribution {distribution!r}; known: {sorted(DIVISORS)}"
            ) from None
        return cls(
            name=name,
            value=abs(half_width) / divisor,
            kind="B",
            distribution=distribution,
            sensitivity=sensitivity,
            note=note or f"half-width {half_width:g}, {distribution} (divisor {divisor:.4g})",
        )

    @property
    def contribution(self) -> float:
        """`c_i * u_i`, signed, in the reading's units."""
        return self.sensitivity * self.value

    @property
    def variance(self) -> float:
        """`(c_i * u_i)^2`, which is what composes."""
        return self.contribution**2

    def __canonical__(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "kind": self.kind,
            "distribution": self.distribution,
            "sensitivity": self.sensitivity,
            "dof": self.dof,
            "note": self.note,
        }


class BudgetLintError(Exception):
    """A budget that cannot be composed. Raised at construction."""


@dataclass(frozen=True)
class UncertaintyBudget:
    """A composing table of named contributions, with the dominant term named.

    The combined standard uncertainty is the quadrature sum of the contributions, which is the
    GUM's law of propagation with the correlation terms dropped. Dropping them is an assumption,
    so ``correlations`` exists to carry the ones that are not negligible, and the composition uses
    them when they are supplied. Silently assuming independence is the commonest way a budget
    understates itself.

    **The coverage factor is not a constant.** `k = 2` is the large-sample convention and it is
    right only when the combined uncertainty rests on enough degrees of freedom that the Student t
    quantile has settled onto the normal one. GUM 6.3.3 and Annex G are explicit: when the number
    of degrees of freedom is small, `k` is `t_p(nu_eff)`, and `nu_eff` is the Welch-Satterthwaite
    value this table already knows how to compute. At four effective degrees of freedom an interval
    built with `k = 2` covers 88.39%, not the 95% it is labelled with, and at the one residual
    degree of freedom a three-point calibration line leaves, `t = 12.706` and `k = 2` is a factor
    of 6.35 too small. So `expanded` raises `k` to the quantile whenever the table has degrees of
    freedom and the quantile is the larger, and falls back to `coverage_k` otherwise.
    `coverage_factor` says why it is a floor rather than a substitution.
    """

    terms: tuple[BudgetTerm, ...] = ()
    #: The coverage factor to use **when the table has no effective degrees of freedom**. k=2 is
    #: the large-sample convention and it delivers 95.45% for a normal, not 95%; that number is
    #: what `coverage_achieved` reports so a reader is never told 95% for a 2-sigma interval.
    coverage_k: float = 2.0
    #: `{(name_i, name_j): r_ij}` for pairs whose correlation is not negligible.
    correlations: Mapping[tuple[str, str], float] = field(default_factory=dict)
    #: The confidence level the expanded uncertainty targets when degrees of freedom exist and the
    #: factor is a Student t quantile. Stated rather than hard-coded, because a budget quoted at
    #: 99% is a legitimate thing to want and a silent 95% is how the wrong one gets shipped.
    coverage_level: float = 0.95

    def __post_init__(self) -> None:
        names = [t.name for t in self.terms]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise BudgetLintError(
                f"budget has duplicate term names {sorted(dupes)}. Two terms with one name compose "
                f"into a number nobody can attribute, which is the failure a budget exists to fix."
            )
        known = set(names)
        for (a, b), r in self.correlations.items():
            if a not in known or b not in known:
                raise BudgetLintError(
                    f"correlation declared between {a!r} and {b!r}, and the budget has no term "
                    f"named {a if a not in known else b!r}."
                )
            if not math.isfinite(r) or abs(r) > 1.0:
                raise BudgetLintError(
                    f"correlation r = {r} declared between {a!r} and {b!r}. A correlation "
                    f"coefficient lies in [-1, 1]; anything outside it is a caller error, and "
                    f"before this check it made the composed variance negative, which the "
                    f"square root then clamped to a combined uncertainty of exactly zero. "
                    f"Divide your covariance by both standard uncertainties, or pass the "
                    f"covariance through a term's sensitivity coefficient instead."
                )
        if not 0.0 < self.coverage_level < 1.0:
            raise BudgetLintError(
                f"coverage_level = {self.coverage_level} is not a confidence level. Pass a "
                f"probability strictly between 0 and 1, such as 0.95."
            )
        if self.coverage_k <= 0.0 or not math.isfinite(self.coverage_k):
            raise BudgetLintError(
                f"coverage_k = {self.coverage_k} is not a coverage factor. It multiplies the "
                f"combined standard uncertainty to give an interval half-width and must be a "
                f"positive finite number; 2 is the convention."
            )
        raw = self._raw_variance()
        independent = sum(t.variance for t in self.terms)
        if raw < -1e-12 * max(independent, 1.0):
            raise BudgetLintError(
                f"the declared correlations compose to a negative total variance ({raw:.6g}) from "
                f"an independent sum of {independent:.6g}. Every coefficient is inside [-1, 1] "
                f"individually, so the set of them is not a valid correlation matrix: it is not "
                f"positive semi-definite. Three terms cannot all be mutually at -1, because two "
                f"of them being opposite to a third makes those two agree. Re-derive the "
                f"coefficients from one covariance matrix rather than pair by pair. Before this "
                f"check the square root clamped the negative to zero and the budget reported a "
                f"combined uncertainty of exactly 0."
            )

    def _raw_variance(self) -> float:
        """The composed variance before any clamping, so the guard can see a negative one."""
        total = sum(t.variance for t in self.terms)
        if self.correlations:
            by_name = {t.name: t for t in self.terms}
            # Keyed on the unordered pair. `r_ij` and `r_ji` are the same correlation, and a caller
            # who declares both should not get it counted twice: two unit terms at r = 1 compose to
            # 2.0, and double-counting made them 2.449.
            seen: set[tuple[str, str]] = set()
            for (a, b), r in self.correlations.items():
                pair = (a, b) if a <= b else (b, a)
                if pair in seen:
                    continue
                seen.add(pair)
                total += 2.0 * r * by_name[a].contribution * by_name[b].contribution
        return total

    # -- composition ------------------------------------------------------

    @property
    def combined(self) -> float:
        """`u_c = sqrt( sum_i (c_i u_i)^2 + 2 sum_{i<j} r_ij c_i u_i c_j u_j )`."""
        return math.sqrt(max(self._raw_variance(), 0.0))

    @property
    def independent_combined(self) -> float:
        """`sqrt( sum_i (c_i u_i)^2 )`: the composition with every correlation set to zero.

        Kept separately because Welch-Satterthwaite is defined on it and not on `combined`. The
        correlation inflation belongs in `u_c` and only in `u_c`; letting it into the coverage
        factor as well applies it twice.
        """
        return math.sqrt(max(sum(t.variance for t in self.terms), 0.0))

    @property
    def coverage_factor(self) -> float:
        """The `k` actually applied: `max(coverage_k, t_p(nu_eff))` when the table has dof.

        GUM 6.3.3 and G.6.4. `k = 2` is a large-sample approximation to `t_{0.975}(inf) = 1.960`
        and it is wrong by 39% at four effective degrees of freedom and by a factor of 6.35 at
        one. The two cross at about `nu = 60`.

        It is a **floor** rather than a replacement, and that is deliberate. `t_{0.975}` is below 2
        for every `nu` above 60, so taking the quantile unconditionally would have quietly narrowed
        every well-replicated interval in the library by 1.85% in exchange for dropping a stated
        level from 95.45% to 95.00%. Nothing in the review asked for that and narrowing an interval
        is the direction a fix should not drift in. So the conventional factor stands where it is
        already adequate and the quantile takes over exactly where the small sample makes it too
        small. The guarantee this buys is worth stating: whenever the table has degrees of freedom,
        `coverage_achieved` is never below `coverage_level`.

        When no term carries a `dof` there is nothing to compute `nu_eff` from and this falls back
        to the declared `coverage_k`, which is honest but weak: `lint_budget` reports the missing
        degrees of freedom as a finding for exactly that reason.
        """
        nu = self.effective_dof()
        if nu is None or not math.isfinite(nu) or nu <= 0:
            return float(self.coverage_k)
        student_t, _ = _distributions()
        quantile = float(student_t.ppf(0.5 * (1.0 + self.coverage_level), nu))
        return max(float(self.coverage_k), quantile)

    @property
    def coverage_achieved(self) -> float:
        """The confidence level `expanded` really carries, which is what a record should be stamped
        with.

        With degrees of freedom this is at least `coverage_level`, and above it by up to half a
        point where the conventional factor is the binding one. Without them it is the
        normal-theory level of the declared factor, which for the conventional `k = 2` is 0.9545
        and not 0.95. The difference is small and the habit of writing 0.95 beside a 2-sigma
        interval is where the larger lie starts.
        """
        nu = self.effective_dof()
        k = self.coverage_factor
        student_t, normal = _distributions()
        if nu is None or not math.isfinite(nu) or nu <= 0:
            return 2.0 * float(normal.cdf(k)) - 1.0
        return 2.0 * float(student_t.cdf(k, nu)) - 1.0

    @property
    def expanded(self) -> float:
        """`U = k * u_c`, the interval half-width at the coverage factor the table supports."""
        return self.coverage_factor * self.combined

    @property
    def dominant(self) -> BudgetTerm | None:
        """The largest single contribution. Almost never sampling noise, which is the point."""
        return max(self.terms, key=lambda t: abs(t.contribution), default=None)

    def shares(self) -> dict[str, float]:
        """Each term's fraction of the combined *variance*, which is what sums to one.

        Variance shares rather than uncertainty shares, because uncertainties add in quadrature
        and so their shares do not sum to anything. A budget reporting "40% of the uncertainty" on
        a standard-deviation basis is reporting a number with no total.

        Each covariance term is split evenly between the two terms it belongs to, so the shares sum
        to one **against the combined variance the budget reports**. Dividing by the independent
        sum instead made the shares sum to one against a total that is not the one printed on the
        line above: two fully correlated unit terms compose to `u_c = 2`, and shares of 50% each
        against an independent total of 2 describe a variance of 2 where the table says 4.

        A negative share is a real answer and it is not clipped. A term with a strong negative
        correlation to the rest of the table reduces the combined variance, and a reader who sees
        `-18%` beside it has learned something a floor at zero would have hidden.
        """
        total = self._raw_variance()
        if total <= 0:
            return {t.name: 0.0 for t in self.terms}
        by_name = {t.name: t for t in self.terms}
        attributed = {t.name: t.variance for t in self.terms}
        seen: set[tuple[str, str]] = set()
        for (a, b), r in self.correlations.items():
            pair = (a, b) if a <= b else (b, a)
            if pair in seen:
                continue
            seen.add(pair)
            cov = r * by_name[a].contribution * by_name[b].contribution
            attributed[a] += cov
            attributed[b] += cov
        return {name: v / total for name, v in attributed.items()}

    def effective_dof(self) -> float | None:
        """Welch-Satterthwaite: `nu_eff = u^4 / sum_i (c_i u_i)^4 / nu_i`.

        The numerator is `independent_combined`, not `combined`. GUM G.4.1 states the formula
        against the uncorrelated propagation law of Equation (10) and it is derived from the
        variance of a sum of independent chi-square terms; there is no correlated version of it.
        Feeding the correlation-inflated `u_c` in is not a small approximation. At `r = -0.9`
        between two four-dof terms it returns `nu_eff = 0.08`, whose `t_{0.975}` is 1.2e15, so the
        expanded uncertainty comes back fifteen orders of magnitude larger than the combined one.
        That is the coupling worth stating plainly: the correlation belongs in `u_c` and the
        degrees of freedom belong to the independent components, and correcting the coverage factor
        without correcting this numerator would have shipped that 1.2e15.

        Returns None when any term lacks degrees of freedom, rather than substituting infinity.
        Substituting infinity is the common shortcut and it silently narrows the interval, which
        is the wrong direction for a shortcut to err in.
        """
        dofs: list[float] = []
        for t in self.terms:
            if t.dof is None or t.dof <= 0:
                return None
            dofs.append(float(t.dof))
        if not dofs:
            return None
        u = self.independent_combined
        if u <= 0:
            return None
        denom = sum(t.variance**2 / d for t, d in zip(self.terms, dofs, strict=True))
        return u**4 / denom if denom > 0 else None

    # -- presentation -----------------------------------------------------

    def with_term(self, term: BudgetTerm) -> "UncertaintyBudget":
        return UncertaintyBudget(
            terms=self.terms + (term,),
            coverage_k=self.coverage_k,
            correlations=self.correlations,
            coverage_level=self.coverage_level,
        )

    def render(self) -> str:
        """The table, as it appears on a card. The dominant term is the line that matters."""
        if not self.terms:
            return "uncertainty budget: no terms declared"
        shares = self.shares()
        width = max(len(t.name) for t in self.terms)
        lines = [f"{'term':<{width}}  type  {'u_i':>11}  {'c_i':>9}  {'c_i·u_i':>11}  share"]
        for t in sorted(self.terms, key=lambda t: -abs(t.contribution)):
            lines.append(
                f"{t.name:<{width}}  {t.kind:^4}  {t.value:>11.4g}  {t.sensitivity:>9.4g}  "
                f"{t.contribution:>11.4g}  {shares[t.name]:>5.1%}"
            )
        lines.append(f"{'combined':<{width}}        {'':>11}  {'':>9}  {self.combined:>11.4g}")
        dom = self.dominant
        if dom is not None:
            lines.append(f"dominant term: {dom.name} ({shares[dom.name]:.1%} of the variance)")
        nu = self.effective_dof()
        if nu is not None:
            lines.append(
                f"effective degrees of freedom: {nu:.1f} (Welch-Satterthwaite on the independent "
                f"components)"
            )
            k = self.coverage_factor
            source = (
                f"t_{{{0.5 * (1 + self.coverage_level):.4g}}}({nu:.1f})"
                if k > self.coverage_k
                else "convention"
            )
            lines.append(
                f"expanded (k={k:.4g}, {source}): {self.expanded:.4g} at "
                f"{self.coverage_achieved:.2%}"
            )
        else:
            lines.append(
                "effective degrees of freedom: none declared, so the coverage factor is the "
                "large-sample one and is too small wherever a term came from a handful of "
                "replicates"
            )
            lines.append(
                f"expanded (k={self.coverage_k:g}): {self.expanded:.4g} at "
                f"{self.coverage_achieved:.2%}"
            )
        return "\n".join(lines)

    def __canonical__(self) -> dict[str, Any]:
        return {
            "terms": [t.__canonical__() for t in self.terms],
            "coverage_k": self.coverage_k,
            "coverage_level": self.coverage_level,
            "correlations": {f"{a}|{b}": r for (a, b), r in sorted(self.correlations.items())},
        }


def budget_of(**terms: float) -> UncertaintyBudget:
    """A budget from plain named standard uncertainties, for tests and simple cases."""
    return UncertaintyBudget(terms=tuple(BudgetTerm(name=k, value=v) for k, v in terms.items()))


# ---------------------------------------------------------------------------
# The calibration curve, whose slope is the sensitivity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationCurve:
    """A Hill curve `E(c) = E_max · c^n / (EC50^n + c^n)`, and the slope that gives `S`.

    Reported as three numbers with intervals rather than one number at one dose, which is what
    interpretability interventions almost always report. The cooperativity `n` is the one that
    carries information nobody currently extracts: a large `n` is switch-like behaviour, which is
    a threshold, which is the shape worth finding.

    Fitting this is `organisms/dose.py`'s job. This type holds the fitted parameters and does the
    arithmetic, so that an instrument with no dose sweep has nothing to construct and therefore
    cannot invent a sensitivity.
    """

    e_max: float
    ec50: float
    hill_n: float = 1.0
    #: The dose the sensitivity is quoted at. The Hill slope varies with dose, so a single `S`
    #: without a dose is meaningless, and this field is what stops one being quoted.
    at_dose: float | None = None

    def __post_init__(self) -> None:
        if self.ec50 <= 0:
            raise ValueError(f"EC50 must be positive; got {self.ec50}")
        if self.hill_n <= 0:
            raise ValueError(f"the Hill coefficient must be positive; got {self.hill_n}")

    def response(self, dose: float) -> float:
        if dose <= 0:
            return 0.0
        num = self.e_max * float(dose**self.hill_n)
        return num / (float(self.ec50**self.hill_n) + float(dose**self.hill_n))

    def slope(self, dose: float | None = None) -> float:
        """`S = dE/dc`, analytic.

        `dE/dc = E_max · n · EC50^n · c^(n-1) / (EC50^n + c^n)^2`. Evaluated at ``at_dose`` when no
        dose is passed, and refusing when neither is available rather than defaulting to EC50,
        because the slope at EC50 is the maximum and quoting it silently is the optimistic error.
        """
        c = dose if dose is not None else self.at_dose
        if c is None:
            raise ValueError(
                "the Hill slope varies with dose, so a sensitivity has to be quoted at one. Pass a "
                "dose, or set at_dose on the curve."
            )
        if c <= 0:
            return 0.0
        k = float(self.ec50**self.hill_n)
        num = self.e_max * self.hill_n * k * float(c ** (self.hill_n - 1))
        return num / (k + float(c**self.hill_n)) ** 2

    def therapeutic_index(self, td50: float) -> float:
        """`TI = TD50 / ED50`: the dose that costs capability over the dose that achieves the effect.

        The library's own `SURGERY` result (exploit removed 0.886, benchmark accuracy fell 0.399)
        is a therapeutic-index measurement at a single dose. Sweeping the dose converts a kill into
        a specification, which is the whole argument for measuring a curve rather than a point.
        """
        return td50 / self.ec50


# ---------------------------------------------------------------------------
# Limits of detection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubstrateKey:
    """What the noise floor is a property of.

    Not the model. Two configurations of the same weights are two measurement instruments: vLLM
    compiled and vLLM eager disagree with each other about as much as either disagrees with
    HuggingFace, so "vLLM" is not one instrument, it is at least two. Every field here changes the
    floor, which is why the cache is keyed on all six.
    """

    model: str
    engine: str
    revision: str = ""
    dtype: str = ""
    attention_impl: str = ""
    layer: int | None = None

    def __canonical__(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "engine": self.engine,
            "revision": self.revision,
            "dtype": self.dtype,
            "attention_impl": self.attention_impl,
            "layer": self.layer,
        }

    def __str__(self) -> str:
        bits = [self.model, self.engine]
        bits += [b for b in (self.revision, self.dtype, self.attention_impl) if b]
        if self.layer is not None:
            bits.append(f"L{self.layer}")
        return "/".join(bits)


Verdict = Literal["below_lod", "above_lod_below_loq", "quantifiable"]


@dataclass(frozen=True)
class LimitOfDetection:
    """`LOD = 3.3 σ_blank / S` and `LOQ = 10 σ_blank / S`, with the three-outcome decision rule.

    ``sigma_blank`` is the standard deviation of the instrument's reading on a *blank*: a
    semantically irrelevant direction, a shuffled label set, a matched control with no planted
    signal. It is a measurement, not a guess, and `blank_n` records how many replicates it came
    from so a floor derived from three replicates is visibly weaker than one from thirty.

    ``sensitivity`` is `S`, the slope of the calibration curve of reading against dose.
    """

    sigma_blank: float
    sensitivity: float
    key: SubstrateKey | None = None
    blank_n: int | None = None
    curve: CalibrationCurve | None = None
    note: str = ""

    #: The two multipliers. 3.3 is roughly the 3-sigma one-sided false-positive rate carried
    #: through both the blank and the sample; 10 is the convention for a 10% relative standard
    #: deviation at the limit. Both are stated so an instrument can override with a justification.
    lod_k: float = 3.3
    loq_k: float = 10.0

    def __post_init__(self) -> None:
        if self.sigma_blank < 0:
            raise ValueError("sigma_blank is a standard deviation and cannot be negative")
        if not math.isfinite(self.lod_k) or self.lod_k <= 0:
            raise ValueError(
                f"lod_k = {self.lod_k} is not a detection multiplier. It scales the blank standard "
                f"deviation into a limit and must be positive and finite; 3.3 is the convention."
            )
        # The three-outcome rule is the whole point of this type, and it only has
        # three outcomes while LOQ sits above LOD. Overriding the multipliers is supported and
        # sometimes right, but overriding them into the wrong order deletes the middle outcome
        # without saying so: with lod_k = 10 and loq_k = 3.3 no reading anywhere on the real line
        # returns "above_lod_below_loq", so a detected-but-not-quantifiable effect at 0.25 comes
        # back as "below_lod" and is refused instead of bounded.
        if not math.isfinite(self.loq_k) or self.loq_k <= self.lod_k:
            raise ValueError(
                f"loq_k = {self.loq_k} is not above lod_k = {self.lod_k}. The decision rule has "
                f"three outcomes (refuse below LOD, return a bound between LOD and LOQ, report "
                f"above LOQ) and the middle one is unreachable unless LOQ > LOD. Raise loq_k above "
                f"lod_k, or if you meant to declare that detection and quantification coincide on "
                f"this substrate, say so in `note` and set loq_k just above lod_k so the bound "
                f"branch still exists."
            )

    @property
    def is_determinate(self) -> bool:
        """Whether a limit can be computed at all. A non-positive slope means it cannot."""
        return self.sensitivity > 0

    @property
    def lod(self) -> float:
        """Below this, refuse. Infinite when the sensitivity is non-positive, which is honest:
        an instrument whose reading does not respond to dose has no detection limit, it has no
        calibration."""
        return self.lod_k * self.sigma_blank / self.sensitivity if self.is_determinate else math.inf

    @property
    def loq(self) -> float:
        return self.loq_k * self.sigma_blank / self.sensitivity if self.is_determinate else math.inf

    def verdict(self, reading: float) -> Verdict:
        """Which of the three outcomes applies to a reading, on magnitude."""
        m = abs(reading)
        if m < self.lod:
            return "below_lod"
        if m < self.loq:
            return "above_lod_below_loq"
        return "quantifiable"

    def as_term(self, name: str = "substrate_noise") -> BudgetTerm:
        """The floor as a Type B budget term, so it composes rather than sitting beside the budget.

        The blank standard deviation *is* an uncertainty on the reading, and a budget that omits it
        while an LOD sits next to the number is double bookkeeping. Type B because it comes from a
        characterisation run rather than from replicates of this measurement.
        """
        return BudgetTerm(
            name=name,
            value=self.sigma_blank,
            kind="B",
            distribution="normal",
            sensitivity=1.0,
            dof=(self.blank_n - 1) if self.blank_n and self.blank_n > 1 else None,
            note=f"blank sigma on {self.key}" if self.key else self.note,
        )

    def __canonical__(self) -> dict[str, Any]:
        return {
            "sigma_blank": self.sigma_blank,
            "sensitivity": self.sensitivity,
            "key": self.key.__canonical__() if self.key else None,
            "blank_n": self.blank_n,
            "lod": self.lod if self.is_determinate else None,
            "loq": self.loq if self.is_determinate else None,
        }

    def render(self) -> str:
        if not self.is_determinate:
            return (
                f"limits of detection: undefined on {self.key}. The calibration slope is "
                f"{self.sensitivity:.4g}, so the reading does not respond to dose and no limit "
                f"exists. Fit a dose sweep before quoting a floor."
            )
        n = f", n={self.blank_n}" if self.blank_n else ""
        return (
            f"limits of detection on {self.key}: sigma_blank {self.sigma_blank:.4g}{n}, "
            f"S {self.sensitivity:.4g}  ->  LOD {self.lod:.4g}, LOQ {self.loq:.4g}"
        )


class LODCache:
    """Limits cached per `(model, engine, revision, dtype, attention_impl, layer)`.

    Characterising a floor costs a dose sweep, so it is measured once per configuration and
    consulted by every preflight. The cache is keyed on the configuration rather than the model
    for the reason `SubstrateKey` documents, and a miss returns None rather than a default: an
    instrument with no measured floor should refuse or say it is uncharacterised, not assume zero.
    """

    def __init__(self) -> None:
        self._by_key: dict[SubstrateKey, LimitOfDetection] = {}

    def put(self, lod: LimitOfDetection) -> LimitOfDetection:
        if lod.key is None:
            raise ValueError(
                "an LOD without a SubstrateKey cannot be cached; it is not a property of anything"
            )
        self._by_key[lod.key] = lod
        return lod

    def get(self, key: SubstrateKey) -> LimitOfDetection | None:
        return self._by_key.get(key)

    def __len__(self) -> int:
        return len(self._by_key)

    def __contains__(self, key: SubstrateKey) -> bool:
        return key in self._by_key

    def keys(self) -> Sequence[SubstrateKey]:
        return list(self._by_key)


#: The process-wide cache the preflight consults.
LIMITS = LODCache()


def refuse_below_lod(instrument: str, reading: float, lod: LimitOfDetection) -> Any:
    """The refusal an instrument returns below its detection limit, carrying both numbers.

    Imported lazily so this module stays free of a cycle with `reading.py`, which imports nothing
    from here.
    """
    from reward_lens.core.reading import Refusal, RefusalReason

    return Refusal(
        instrument=instrument,
        reason=RefusalReason.BELOW_LOD,
        detail=(
            f"reading {reading:.4g} is below the limit of detection {lod.lod:.4g} on {lod.key} "
            f"(sigma_blank {lod.sigma_blank:.4g}, S {lod.sensitivity:.4g})"
        ),
        remedy=(
            "increase the effect: raise the dose, or measure on a configuration with a lower noise "
            "floor. A reading below the floor is not attributable to the thing being measured, so "
            "averaging more of them will not help."
        ),
        statistics={
            "reading": reading,
            "lod": lod.lod,
            "loq": lod.loq,
            "sigma_blank": lod.sigma_blank,
            "sensitivity": lod.sensitivity,
        },
    )


__all__ = [
    "DIVISORS",
    "LIMITS",
    "BudgetLintError",
    "BudgetTerm",
    "CalibrationCurve",
    "IncrementalValidity",
    "LODCache",
    "LimitOfDetection",
    "SubstrateKey",
    "UncertaintyBudget",
    "UncertaintyType",
    "Verdict",
    "budget_of",
    "refuse_below_lod",
]


# ---------------------------------------------------------------------------
# Incremental validity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IncrementalValidity:
    """What a white-box reading adds over the best black-box method, required on every one.

    The bar is **decorrelation plus signal, not superiority.** A method ten points worse and
    uncorrelated with the baseline is more valuable than one two points better and redundant,
    because the ensemble of the first pair is better than either and the ensemble of the second is
    not. The library has to be able to say so, and a lone AUC cannot.

    `error_correlation` is the field that does the work and it is the one nobody reports.
    "Incremental validity" is the established psychometrics term, it returns eight arXiv hits of
    which seven are false friends, and it has zero uses in machine-learning interpretability. The
    substance is partly occupied: arXiv 2507.12691 makes the white-box-minus-black-box gap its
    benchmark and reports an ensemble lifting mean AUROC from 0.879 to 0.953, which is three of the
    four numbers here. What it does not report is the correlation between the two methods' errors,
    so complementarity is inferred from ensemble gain rather than measured. That statistic plus the
    psychometric framing is the narrow honest claim.
    """

    own_score: float
    baseline_score: float
    baseline_id: str
    #: Pearson or phi between the two methods' **errors**, not between their scores. Two methods
    #: that agree on which items are hard have correlated errors even when their scores differ.
    error_correlation: float
    ensemble_score: float

    @property
    def ensemble_gain(self) -> float:
        """How much the pair beats the better of the two alone. Negative is a real answer."""
        return self.ensemble_score - max(self.own_score, self.baseline_score)

    @property
    def is_redundant(self) -> bool:
        """Highly correlated errors and no ensemble gain: this method adds nothing."""
        return abs(self.error_correlation) > 0.8 and self.ensemble_gain <= 0.0

    def render(self) -> str:
        verdict = (
            "redundant with the baseline"
            if self.is_redundant
            else f"complementary (error correlation {self.error_correlation:+.2f})"
        )
        return (
            f"own {self.own_score:.3f}, baseline {self.baseline_id} {self.baseline_score:.3f}, "
            f"ensemble {self.ensemble_score:.3f} (gain {self.ensemble_gain:+.3f}): {verdict}"
        )

    def __canonical__(self) -> dict[str, Any]:
        return {
            "own_score": self.own_score,
            "baseline_score": self.baseline_score,
            "baseline_id": self.baseline_id,
            "error_correlation": self.error_correlation,
            "ensemble_score": self.ensemble_score,
        }

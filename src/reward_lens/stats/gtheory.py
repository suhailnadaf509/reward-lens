"""Generalizability theory: the crossed G-study, the D-study, and the finite-universe correction.

A grader is a measurement device and a score is a measurement, so the question "how much of this
number is the response and how much is which judge I drew on which day" has had a worked answer
since Cronbach, Gleser, Nanda and Rajaratnam published it in 1972. Generalizability theory
decomposes an observed score into the object being measured and every facet of the measurement
procedure, then tells you what a differently sized procedure would cost and buy. It is standard in
educational measurement and it has essentially never been pointed at a reward model.

There is no Python package for it. A full index scan of PyPI's 449,089 projects returns nothing,
and R's `gtheory` package was removed from CRAN on 2025-03-24 because email to its maintainer
bounced. So this module is the implementation, and it is small enough to read in one sitting.

**What is here.** Two fully crossed designs with one observation per cell: the two-facet `p x r`
and the three-facet `p x r x o`. `p` is the object of measurement, the thing whose differences you
want to resolve. `r` and `o` are facets of the measurement: which grader, which repeat call, which
rubric, which response style. The sums of squares are the textbook ones, the expected mean squares
are inverted in closed form, and the seven-component inversion for `p x r x o` is:

    sigma2(pro,e) = MS_pro
    sigma2(pr)    = (MS_pr - MS_pro) / n_o
    sigma2(po)    = (MS_po - MS_pro) / n_r
    sigma2(ro)    = (MS_ro - MS_pro) / n_p
    sigma2(p)     = (MS_p - MS_pr - MS_po + MS_pro) / (n_r * n_o)
    sigma2(r)     = (MS_r - MS_pr - MS_ro + MS_pro) / (n_p * n_o)
    sigma2(o)     = (MS_o - MS_po - MS_ro + MS_pro) / (n_p * n_r)

The last component is named `pro,e` rather than `pro` because with one observation per cell the
three-way interaction and the residual are the same term and nothing can separate them. Writing
`pro` alone would claim an interaction estimate the design does not contain.

**Two things this module is careful about.** Negative estimates are truncated at zero and the
truncation is recorded, because a component silently set to zero misrepresents which facet
dominates (see `stats.variance`). And an unbalanced design is refused rather than approximated: the
method of moments above assumes every cell is filled exactly once, and running it on a design with
holes gives a biased answer that looks identical to an unbiased one. `fit_unbalanced` names what to
install instead.

**The finite-universe correction is not a detail.** A facet has a universe of levels `N_i` and you
sample `n_i` of them. Declaring `N_i = n_i`, that is, declaring the facet fixed, moves the
object-by-facet interaction out of error and into universe-score variance. Brennan showed in 1992
that this raises reliability from .74 to .88 on the same data, while destroying any claim to
generalise to new levels of that facet. That is the mathematics of benchmark overfitting, published
34 years ago, and `GStudy.declare_fixed` computes both numbers so the trade is visible in one call.

The general form the D-study uses, for a term `a` over a set of facets, is

    error share    = [1 - prod_{i in a} (n_i / N_i)] * sigma2(a) / prod_{i in a} n_i
    universe share =      prod_{i in a} (n_i / N_i)  * sigma2(a) / prod_{i in a} n_i

which reduces to the fully random model when every `N_i` is infinite and to Brennan's mixed model
when a facet has `n_i = N_i`. The two shares sum to the term's total contribution, so declaring a
facet fixed moves variance between the numerator and the denominator of the coefficient and never
creates or destroys any.

References: Brennan, `Generalizability Theory` (Springer, 2001), chapters 3 and 5; Shavelson and
Webb, `Generalizability Theory: A Primer` (Sage, 1991), chapter 4.
"""

from __future__ import annotations

import importlib.util
import itertools
import math
from dataclasses import dataclass, field, replace
from typing import Mapping, Sequence

import numpy as np

from reward_lens.stats.variance import ComponentSet, truncate_at_zero

#: The object of measurement is always named `p`, following the literature's "person". In a reward
#: loop it is the rollout, the response or the prompt, whichever the caller declared.
OBJECT = "p"


class DesignError(ValueError):
    """The data does not have the shape the estimator assumes.

    A `ValueError` rather than a refusal, because this is the layer below the instruments: an
    instrument catches it and returns a `Refusal` with a remedy. Calling a balanced-design estimator
    on ragged data is a programming error at this level.
    """


# ---------------------------------------------------------------------------
# Mean squares
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MeanSquares:
    """The ANOVA table of a balanced crossed design: sums of squares, degrees of freedom, ratios.

    Carried whole rather than reduced to the components, because the components are a linear
    transform of it and a reader checking the arithmetic by hand needs the table it came from. It
    is also the only place the degrees of freedom survive, and a component estimated on three
    degrees of freedom is a different object from the same number estimated on three hundred.
    """

    ss: Mapping[str, float]
    df: Mapping[str, float]

    @property
    def ms(self) -> dict[str, float]:
        return {k: (self.ss[k] / self.df[k] if self.df[k] > 0 else math.nan) for k in self.ss}

    def render(self) -> str:
        ms = self.ms
        lines = [f"  {'source':<10} {'SS':>14} {'df':>8} {'MS':>14}"]
        for k in self.ss:
            lines.append(f"  {k:<10} {self.ss[k]:>14.6g} {self.df[k]:>8.0f} {ms[k]:>14.6g}")
        return "\n".join(lines)


def _check_finite(x: np.ndarray, where: str) -> None:
    if not np.all(np.isfinite(x)):
        n_bad = int(np.sum(~np.isfinite(x)))
        raise DesignError(
            f"{where} contains {n_bad} non-finite value(s). A NaN in a balanced design is a hole "
            f"in the design, not a number to propagate: drop the affected level or fill the cell."
        )


def mean_squares_pr(x: np.ndarray) -> MeanSquares:
    """The three-source ANOVA table of a crossed `p x r` design with one observation per cell.

    ``x`` is ``(n_p, n_r)``. Sources are `p`, `r` and the confounded `pr,e`.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise DesignError(f"a p x r design is a 2-D array of shape (n_p, n_r); got {x.shape}")
    n_p, n_r = x.shape
    if n_p < 2 or n_r < 2:
        raise DesignError(
            f"a crossed p x r design needs at least two objects and two levels of the facet to "
            f"have any degrees of freedom for the interaction; got n_p = {n_p}, n_r = {n_r}"
        )
    _check_finite(x, "the p x r score matrix")

    grand = float(x.mean())
    mp = x.mean(axis=1)
    mr = x.mean(axis=0)
    ss = {
        "p": float(n_r * np.sum((mp - grand) ** 2)),
        "r": float(n_p * np.sum((mr - grand) ** 2)),
        "pr,e": float(np.sum((x - mp[:, None] - mr[None, :] + grand) ** 2)),
    }
    df = {
        "p": float(n_p - 1),
        "r": float(n_r - 1),
        "pr,e": float((n_p - 1) * (n_r - 1)),
    }
    return MeanSquares(ss=ss, df=df)


def mean_squares_pro(x: np.ndarray) -> MeanSquares:
    """The seven-source ANOVA table of a crossed `p x r x o` design with one observation per cell.

    ``x`` is ``(n_p, n_r, n_o)``. Sources are `p`, `r`, `o`, `pr`, `po`, `ro` and the confounded
    `pro,e`. The degrees of freedom sum to `n_p*n_r*n_o - 1`, which is the check worth running on
    any implementation of this and is asserted in the tests.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3:
        raise DesignError(
            f"a p x r x o design is a 3-D array of shape (n_p, n_r, n_o); got {x.shape}"
        )
    n_p, n_r, n_o = x.shape
    if min(n_p, n_r, n_o) < 2:
        raise DesignError(
            f"a crossed p x r x o design needs at least two levels on every axis for the three-way "
            f"residual to have degrees of freedom; got n_p = {n_p}, n_r = {n_r}, n_o = {n_o}"
        )
    _check_finite(x, "the p x r x o score cube")

    g = float(x.mean())
    mp = x.mean(axis=(1, 2))
    mr = x.mean(axis=(0, 2))
    mo = x.mean(axis=(0, 1))
    mpr = x.mean(axis=2)
    mpo = x.mean(axis=1)
    mro = x.mean(axis=0)

    ss = {
        "p": float(n_r * n_o * np.sum((mp - g) ** 2)),
        "r": float(n_p * n_o * np.sum((mr - g) ** 2)),
        "o": float(n_p * n_r * np.sum((mo - g) ** 2)),
        "pr": float(n_o * np.sum((mpr - mp[:, None] - mr[None, :] + g) ** 2)),
        "po": float(n_r * np.sum((mpo - mp[:, None] - mo[None, :] + g) ** 2)),
        "ro": float(n_p * np.sum((mro - mr[:, None] - mo[None, :] + g) ** 2)),
    }
    resid = (
        x
        - mpr[:, :, None]
        - mpo[:, None, :]
        - mro[None, :, :]
        + mp[:, None, None]
        + mr[None, :, None]
        + mo[None, None, :]
        - g
    )
    ss["pro,e"] = float(np.sum(resid**2))
    df = {
        "p": float(n_p - 1),
        "r": float(n_r - 1),
        "o": float(n_o - 1),
        "pr": float((n_p - 1) * (n_r - 1)),
        "po": float((n_p - 1) * (n_o - 1)),
        "ro": float((n_r - 1) * (n_o - 1)),
        "pro,e": float((n_p - 1) * (n_r - 1) * (n_o - 1)),
    }
    return MeanSquares(ss=ss, df=df)


# ---------------------------------------------------------------------------
# The G-study
# ---------------------------------------------------------------------------

#: Which raw components each design produces, in reading order.
_TERMS: dict[str, tuple[str, ...]] = {
    "p x r": ("p", "r", "pr,e"),
    "p x r x o": ("p", "r", "o", "pr", "po", "ro", "pro,e"),
}

#: The facets each component name involves, so the D-study can compute divisors generically. The
#: object `p` is not a facet: you do not generalise over the thing you are measuring.
_FACETS_OF: dict[str, tuple[str, ...]] = {
    "p": (),
    "r": ("r",),
    "o": ("o",),
    "pr": ("r",),
    "po": ("o",),
    "ro": ("r", "o"),
    "pr,e": ("r",),
    "pro,e": ("r", "o"),
}

#: Whether a component involves the object of measurement, which decides whether it can contribute
#: to universe-score variance and to relative error.
_HAS_OBJECT: dict[str, bool] = {
    "p": True,
    "r": False,
    "o": False,
    "pr": True,
    "po": True,
    "ro": False,
    "pr,e": True,
    "pro,e": True,
}

#: Components that are an interaction and the residual added together, because the design has one
#: observation per cell and nothing can separate them. The `,e` in the name is the whole point: the
#: module docstring says these exist so nothing downstream quotes them as an interaction, and the
#: D-study's fixed-facet correction does exactly that unless it is told not to.
_CONFOUNDED: frozenset[str] = frozenset({"pr,e", "pro,e"})


@dataclass(frozen=True)
class GStudy:
    """A variance decomposition of one crossed design, plus what the facets' universes look like.

    ``levels`` is how many levels of each facet the study observed. ``universe`` is how many exist:
    `math.inf` for a facet you are willing to generalise over freely, a finite number for a facet
    with a bounded universe, and exactly `levels[facet]` for a facet you have declared fixed.

    The default universe is infinite for every facet, which is the conservative direction. Declaring
    a facet fixed always raises the reliability, so a library that defaulted to fixed would flatter
    every design it was handed.
    """

    design: str
    components: ComponentSet
    levels: Mapping[str, int]
    mean_squares: MeanSquares
    universe: Mapping[str, float] = field(default_factory=dict)
    #: What the caller called the object and the facets, for rendering. Purely cosmetic.
    labels: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.design not in _TERMS:
            raise DesignError(f"unknown design {self.design!r}; known are {sorted(_TERMS)}")
        missing = set(_TERMS[self.design]) - set(self.components.names)
        if missing:
            raise DesignError(
                f"design {self.design!r} needs components {sorted(missing)} and the set does not "
                f"carry them"
            )
        object.__setattr__(
            self, "universe", {f: float(self.universe.get(f, math.inf)) for f in self.facets}
        )

    @property
    def facets(self) -> tuple[str, ...]:
        """The facets of measurement, excluding the object. `('r',)` or `('r', 'o')`."""
        return tuple(f for f in ("r", "o") if f in self.levels)

    @property
    def fixed(self) -> tuple[str, ...]:
        """Facets declared to have a finite universe, so a D-study does not fully generalise over them.

        "Finite" rather than "exhausted", because the correction is continuous: a facet with a
        universe of 100 levels sampled at 20 gets four fifths of the random-model error and one
        fifth folded into universe score. Exhausting the universe is the endpoint of that scale, not
        a separate mode.
        """
        return tuple(f for f in self.facets if math.isfinite(self.universe[f]))

    def label(self, key: str) -> str:
        return self.labels.get(key, key)

    def declare_fixed(self, facet: str, *, universe_size: float | None = None) -> "GStudy":
        """Declare a facet's universe finite, defaulting to exactly the levels observed.

        This is Brennan's 1992 result made callable. Fixing a facet moves the object-by-facet
        interaction out of error and into universe-score variance, so the generalizability
        coefficient rises, sometimes a great deal. What it buys is a narrower claim: the reading now
        describes performance over *these* raters, *these* items, *this* rubric, and says nothing
        about a new draw from any of them.

        Both numbers are always available, because a `GStudy` is immutable and this returns a new
        one. Report them side by side; a reliability quoted without saying which universe it
        generalises over is not a reliability.
        """
        if facet not in self.facets:
            raise DesignError(
                f"{facet!r} is not a facet of a {self.design} design; its facets are "
                f"{self.facets}. The object of measurement cannot be fixed: fixing it would make "
                f"the universe-score variance zero and the coefficient undefined."
            )
        size = float(self.levels[facet]) if universe_size is None else float(universe_size)
        if size < 1:
            raise DesignError(f"a universe has at least one level; got {size} for facet {facet!r}")
        # Deliberately not required to be at least the G-study's own `n_facet`. Sampling eleven
        # graders to learn how much they disagree and then shipping with exactly one of them is the
        # commonest case there is, and it is `universe_size = 1` with `n'_r = 1`. The G-study's
        # level count says how well the components are estimated; the universe says what the
        # reading is a claim about. They are different statements.
        return replace(self, universe={**self.universe, facet: size})

    # -- the D-study --------------------------------------------------------

    def _sizes(self, **sizes: float | None) -> dict[str, float]:
        # A keyword that names no facet used to be discarded in silence, so `d_study(n_r=1)` and
        # `d_study(rater=1)` both returned the default-size answer with nothing raised. That is not
        # a hypothetical slip: `rater` and `occasion` are the labels `crossed_pro` assigns by
        # default and carries on `GStudy.labels`, so they are the natural mistaken call, and on the
        # worked example it moves a headline reliability from 0.531 to 0.843 while looking like it
        # worked.
        unknown = sorted(k for k in sizes if k not in self.facets)
        if unknown:
            known = ", ".join(sorted(self.facets))
            raise DesignError(
                f"d_study got {', '.join(repr(u) for u in unknown)}, which name no facet of this "
                f"design. The facets are {known}. A size for a facet that does not exist was "
                f"silently ignored and the default-size answer returned, which is a wrong "
                f"reliability that looks like it worked."
            )
        out: dict[str, float] = {}
        for f in self.facets:
            n = sizes.get(f)
            universe = float(self.universe.get(f, math.inf))
            if n is None:
                # The default D-study size is the G-study's own level count, capped at the universe.
                # Without the cap, `declare_fixed(f, universe_size=1)`, which the module's own
                # docstring calls "the commonest case there is", leaves `n' = levels[f]` against
                # `N = 1`, so the default path routinely runs with `n' > N`. The universe share is
                # then `min(1, n'/N) * sigma2(a)/n'` where the derivation gives `sigma2(a)/N`: the
                # clamp saves the ratio and the surviving `1/n'` understates the universe-score
                # variance anyway. Reproduced on a 40x11 design at tau understated 34.4%. Every
                # existing test passed matching sizes, so the path was untested.
                out[f] = min(float(self.levels[f]), universe)
            else:
                out[f] = float(n)
                if out[f] > universe:
                    raise DesignError(
                        f"a D-study asked for {out[f]:g} levels of facet {f!r} out of a universe of "
                        f"{universe:g}. Generalising to more levels than exist is not a smaller "
                        f"error, it is an undefined one. Raise the universe with declare_fixed, or "
                        f"ask for at most {universe:g}."
                    )
            if out[f] < 1:
                raise DesignError(f"a D-study needs at least one level of facet {f!r}; got {n}")
        return out

    def _shares(
        self, sizes: Mapping[str, float], *, credit_confounded: bool = True
    ) -> dict[str, tuple[float, float]]:
        """Per component: (universe share, error share) of its contribution at these sizes.

        ``credit_confounded`` is False for the lower end of the identified interval. A term named
        `pr,e` is an interaction plus the residual, and only the interaction half belongs in
        universe score when the facet is fixed. Setting it False charges the whole term to error,
        which is the other extreme of a split this design cannot make.
        """
        out: dict[str, tuple[float, float]] = {}
        for name in _TERMS[self.design]:
            facets = _FACETS_OF[name]
            if not facets:
                out[name] = (self.components.value(name), 0.0)
                continue
            divisor = 1.0
            fixed_fraction = 1.0
            for f in facets:
                divisor *= sizes[f]
                universe = self.universe[f]
                fixed_fraction *= 0.0 if math.isinf(universe) else min(1.0, sizes[f] / universe)
            if not credit_confounded and name in _CONFOUNDED:
                fixed_fraction = 0.0
            total = self.components.value(name) / divisor
            out[name] = (fixed_fraction * total, (1.0 - fixed_fraction) * total)
        return out

    def _universe(self, s: Mapping[str, float], *, credit_confounded: bool = True) -> float:
        shares = self._shares(s, credit_confounded=credit_confounded)
        return float(sum(u for name, (u, _) in shares.items() if _HAS_OBJECT[name]))

    def _relative(self, s: Mapping[str, float], *, credit_confounded: bool = True) -> float:
        shares = self._shares(s, credit_confounded=credit_confounded)
        return float(sum(e for name, (_, e) in shares.items() if _HAS_OBJECT[name]))

    def _absolute(self, s: Mapping[str, float], *, credit_confounded: bool = True) -> float:
        shares = self._shares(s, credit_confounded=credit_confounded)
        return float(sum(e for _, e in shares.values()))

    @staticmethod
    def _coefficient(tau: float, err: float) -> float:
        return 0.0 if tau + err <= 0 else tau / (tau + err)

    def universe_variance(self, **sizes: float | None) -> float:
        """`sigma2(tau)`: the variance of the thing you are trying to measure, at these sizes.

        Equal to `sigma2(p)` under a fully random model. Larger once a facet is fixed, because the
        object-by-facet interaction stops being error and starts being part of what you claim to
        measure.
        """
        return self._universe(self._sizes(**sizes))

    def relative_error(self, **sizes: float | None) -> float:
        """`sigma2(delta)`: error in a comparison between two objects measured the same way.

        Only terms involving the object contribute. A rater who marks everything two points high
        shifts every object equally and cancels out of a within-grader comparison, which is exactly
        why a group-relative estimator such as GRPO is insensitive to grader bias and sensitive to
        grader-by-item interaction.
        """
        return self._relative(self._sizes(**sizes))

    def absolute_error(self, **sizes: float | None) -> float:
        """`sigma2(Delta)`: error in an object's absolute level, not just its rank.

        Adds the facet main effects and their interactions, which do not cancel. This is the number
        that matters when two runs used different grader draws, and it is the one that does not
        shrink by buying more rollouts.
        """
        return self._absolute(self._sizes(**sizes))

    def generalizability(self, **sizes: float | None) -> float:
        """`E rho^2 = sigma2(tau) / (sigma2(tau) + sigma2(delta))`. Reliability for relative decisions.

        Read `d_study(...).identified` before quoting this with a facet fixed. In a design with one
        observation per cell, fixing every facet a confounded term carries credits that term's
        residual half to universe score as well as its interaction half, and the coefficient becomes
        an upper bound rather than an estimate. `generalizability_bounds` gives the interval.
        """
        s = self._sizes(**sizes)
        return self._coefficient(self._universe(s), self._relative(s))

    def dependability(self, **sizes: float | None) -> float:
        """`Phi = sigma2(tau) / (sigma2(tau) + sigma2(Delta))`. Reliability for absolute decisions."""
        s = self._sizes(**sizes)
        return self._coefficient(self._universe(s), self._absolute(s))

    # -- what the design cannot separate ------------------------------------

    @property
    def confounded_terms(self) -> tuple[str, ...]:
        """Components that are an interaction and the residual added together, in this design.

        A property of the design and not of the data: one observation per cell leaves no replication
        to estimate pure error from, so `pr,e` and `pro,e` carry both and nothing can split them.
        """
        return tuple(n for n in _TERMS[self.design] if n in _CONFOUNDED)

    def confounded_credited(self, **sizes: float | None) -> tuple[str, ...]:
        """Confounded terms that received universe-score credit at these sizes, if any.

        Non-empty is the condition under which the reliability coefficients stop being estimates.
        It fires when every facet a confounded term carries has a finite universe: the fixed-facet
        correction then moves the whole term into universe score, residual included. In a `p x r`
        design that is the only object-containing error term there is, so relative error goes to
        exactly zero and `E rho^2` is exactly 1.0000 whatever the residual was. Reproduced on 40
        objects by 8 raters with a true residual variance of 2.0: `sigma2(pr,e)` estimated at
        1.73819, `E rho^2` and `Phi` both 1.0000 after `declare_fixed("r")`.
        """
        s = self._sizes(**sizes)
        out: list[str] = []
        for name in self.confounded_terms:
            if self.components.value(name) <= 0.0:
                continue
            fraction = 1.0
            for f in _FACETS_OF[name]:
                universe = self.universe[f]
                fraction *= 0.0 if math.isinf(universe) else min(1.0, s[f] / universe)
            if fraction > 0.0:
                out.append(name)
        return tuple(out)

    def generalizability_bounds(self, **sizes: float | None) -> tuple[float, float]:
        """The interval `E rho^2` is identified to, given what this design cannot separate.

        The upper end credits the whole confounded term to universe score, which is what
        `generalizability` returns; the lower end charges the whole of it to error. The truth is
        between them and this design has no way to say where. The two ends coincide, and the
        interval collapses to a point, whenever no confounded term received universe credit, which
        is every fully random model and every design with a free facet left in the residual term.
        """
        s = self._sizes(**sizes)
        hi = self._coefficient(self._universe(s), self._relative(s))
        if not self.confounded_credited(**sizes):
            return (hi, hi)
        lo = self._coefficient(
            self._universe(s, credit_confounded=False),
            self._relative(s, credit_confounded=False),
        )
        return (lo, hi)

    def dependability_bounds(self, **sizes: float | None) -> tuple[float, float]:
        """The same interval for `Phi`. See `generalizability_bounds`."""
        s = self._sizes(**sizes)
        hi = self._coefficient(self._universe(s), self._absolute(s))
        if not self.confounded_credited(**sizes):
            return (hi, hi)
        lo = self._coefficient(
            self._universe(s, credit_confounded=False),
            self._absolute(s, credit_confounded=False),
        )
        return (lo, hi)

    def d_study(self, **sizes: float | None) -> "DStudy":
        """Every D-study number at one set of facet sizes, computed once."""
        s = self._sizes(**sizes)
        tau = self._universe(s)
        delta = self._relative(s)
        big_delta = self._absolute(s)
        return DStudy(
            sizes={k: int(v) for k, v in s.items()},
            universe_variance=tau,
            relative_error=delta,
            absolute_error=big_delta,
            generalizability=self._coefficient(tau, delta),
            dependability=self._coefficient(tau, big_delta),
            fixed=self.fixed,
            truncated=self.components.truncated_names,
            confounded_credited=self.confounded_credited(**sizes),
            generalizability_bounds=self.generalizability_bounds(**sizes),
            dependability_bounds=self.dependability_bounds(**sizes),
            determined=tau + delta > 0.0,
        )

    def render(self) -> str:
        head = f"G-study, {self.design}, " + ", ".join(
            f"n_{f} = {self.levels[f]}" for f in ("p", *self.facets) if f in self.levels
        )
        fixed = f"  fixed facets: {', '.join(self.fixed)}" if self.fixed else ""
        return "\n".join(x for x in (head, self.components.render(), fixed) if x)


@dataclass(frozen=True)
class DStudy:
    """What a differently sized measurement procedure would buy, at one set of sizes.

    Three flags travel with the numbers, and each of them exists because the number alone reads as
    a measurement when it is not one.

    ``truncated`` names components that came back negative and were set to zero. Negative estimates
    are ordinary in a small design, and a component silently zeroed is a lie about which facet
    dominates. ``confounded_credited`` names interaction-plus-residual terms that the fixed-facet
    correction moved into universe score whole; when it is non-empty the two coefficients are upper
    bounds and `generalizability_bounds` is the interval they are identified to. ``determined`` is
    False when the coefficient has no positive denominator to divide by, in which case the reported
    0.0 is a convention rather than a reading, which is the defect `GaugeRR.determined` fixed on
    the gauge side.
    """

    sizes: Mapping[str, int]
    universe_variance: float
    relative_error: float
    absolute_error: float
    generalizability: float
    dependability: float
    fixed: tuple[str, ...] = ()
    #: Components estimated below zero and truncated. Never silently empty.
    truncated: tuple[str, ...] = ()
    #: Confounded terms the fixed-facet correction credited to universe score, residual included.
    confounded_credited: tuple[str, ...] = ()
    #: `(lower, upper)` for `E rho^2`. A point, both ends equal, when nothing is confounded.
    generalizability_bounds: tuple[float, float] | None = None
    #: `(lower, upper)` for `Phi`, on the same rule.
    dependability_bounds: tuple[float, float] | None = None
    #: False when `sigma2(tau) + sigma2(delta)` is not positive, so neither coefficient is defined.
    determined: bool = True

    @property
    def identified(self) -> bool:
        """Whether the two coefficients are estimates rather than upper bounds."""
        return not self.confounded_credited

    def render(self) -> str:
        sizes = ", ".join(f"n'_{k} = {v}" for k, v in sorted(self.sizes.items()))
        fixed = f" [fixed: {', '.join(self.fixed)}]" if self.fixed else ""
        head = (
            f"D-study at {sizes}{fixed}: sigma2(tau) = {self.universe_variance:.6g}, "
            f"sigma2(delta) = {self.relative_error:.6g}, "
            f"sigma2(Delta) = {self.absolute_error:.6g}, "
            f"E rho^2 = {self.generalizability:.4f}, Phi = {self.dependability:.4f}"
        )
        lines = [head]
        if not self.determined:
            lines.append(
                "  Neither coefficient is determined: universe-score variance and relative error "
                "are both zero at these sizes, so the ratio has no denominator and the 0.0 above "
                "is a convention. Report the decomposition instead, and note that every component "
                "this design could estimate came back at or below zero."
            )
        if self.truncated:
            lines.append(
                f"  Truncated at zero: {', '.join(self.truncated)}. Their raw estimates were "
                f"negative, which this design could not resolve from zero. Do not read a truncated "
                f"component as an established zero, and do not read the share of a facet beside it "
                f"as that facet's true dominance."
            )
        if self.confounded_credited:
            lo, hi = self.generalizability_bounds or (
                self.generalizability,
                self.generalizability,
            )
            plo, phi = self.dependability_bounds or (self.dependability, self.dependability)
            lines.append(
                f"  {', '.join(self.confounded_credited)} is an interaction and the residual added "
                f"together, and fixing the facet(s) it carries moved all of it into universe score, "
                f"the residual with it. A rater panel you have exhausted still scores with error; "
                f"this design has no replication to estimate that error from, so it reports none. "
                f"Both coefficients above are upper bounds: E rho^2 is in [{lo:.4f}, {hi:.4f}] and "
                f"Phi in [{plo:.4f}, {phi:.4f}], and this design cannot say where. Score a second "
                f"pass per cell to identify it, or quote the random-universe numbers."
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# The two fitters
# ---------------------------------------------------------------------------


def crossed_pr(
    x: np.ndarray,
    *,
    object_label: str = "object",
    facet_label: str = "rater",
) -> GStudy:
    """Fit a fully crossed two-facet design, `n_p` objects by `n_r` raters, one score per cell.

    The interaction and the residual are confounded, which is a property of the design and not of
    this code: with one observation per cell there is no replication left over to estimate pure
    error from. The component is named `pr,e` so nothing downstream can quote it as an interaction.
    """
    ms = mean_squares_pr(x)
    n_p, n_r = np.asarray(x).shape
    m = ms.ms
    raw = {
        "p": (m["p"] - m["pr,e"]) / n_r,
        "r": (m["r"] - m["pr,e"]) / n_p,
        "pr,e": m["pr,e"],
    }
    components = truncate_at_zero(
        raw,
        df=ms.df,
        notes={
            "p": f"{object_label} to {object_label}: the signal",
            "r": f"{facet_label} main effect: which {facet_label} you drew",
            "pr,e": f"{object_label} by {facet_label} interaction, confounded with residual error",
        },
        design=f"crossed p x r, n_p = {n_p}, n_r = {n_r}",
    )
    return GStudy(
        design="p x r",
        components=components,
        levels={"p": int(n_p), "r": int(n_r)},
        mean_squares=ms,
        labels={"p": object_label, "r": facet_label},
    )


def crossed_pro(
    x: np.ndarray,
    *,
    object_label: str = "object",
    facet_labels: tuple[str, str] = ("rater", "occasion"),
) -> GStudy:
    """Fit a fully crossed three-facet design, `n_p` by `n_r` by `n_o`, one score per cell.

    This is the real instrument. The seven expected-mean-square equations are inverted in closed
    form; the arithmetic is in the module docstring and reproduced in the tests against a published
    worked example.

    Replications are the natural third facet: scoring each object with each grader `n_o` times makes
    `o` the repeat index, `sigma2(o)` the drift between passes, and `sigma2(pro,e)` the call-to-call
    noise. A response style, a rubric variant or a prompt ordering are equally valid third facets
    and the arithmetic does not care which.
    """
    ms = mean_squares_pro(x)
    n_p, n_r, n_o = np.asarray(x).shape
    m = ms.ms
    raw = {
        "p": (m["p"] - m["pr"] - m["po"] + m["pro,e"]) / (n_r * n_o),
        "r": (m["r"] - m["pr"] - m["ro"] + m["pro,e"]) / (n_p * n_o),
        "o": (m["o"] - m["po"] - m["ro"] + m["pro,e"]) / (n_p * n_r),
        "pr": (m["pr"] - m["pro,e"]) / n_o,
        "po": (m["po"] - m["pro,e"]) / n_r,
        "ro": (m["ro"] - m["pro,e"]) / n_p,
        "pro,e": m["pro,e"],
    }
    a, b = facet_labels
    components = truncate_at_zero(
        raw,
        df=ms.df,
        notes={
            "p": f"{object_label} to {object_label}: the signal",
            "r": f"{a} main effect: which {a} you drew",
            "o": f"{b} main effect: which {b} you drew",
            "pr": f"{object_label} by {a}: the raters disagree about which {object_label} is better",
            "po": f"{object_label} by {b}",
            "ro": f"{a} by {b}",
            "pro,e": "three-way interaction, confounded with residual error",
        },
        design=f"crossed p x r x o, n_p = {n_p}, n_r = {n_r}, n_o = {n_o}",
    )
    return GStudy(
        design="p x r x o",
        components=components,
        levels={"p": int(n_p), "r": int(n_r), "o": int(n_o)},
        mean_squares=ms,
        labels={"p": object_label, "r": a, "o": b},
    )


# ---------------------------------------------------------------------------
# Balance, and what to do when it is absent
# ---------------------------------------------------------------------------

#: The distribution name and version a caller needs for the unbalanced path. Named here so the
#: refusal, the probe and the docs cannot drift apart.
UNBALANCED_REQUIREMENT = "statsmodels>=0.14"


def statsmodels_available() -> bool:
    """Whether the unbalanced path can run at all.

    A probe rather than a try-import, so asking the question costs nothing and can be asked in a
    preflight before any data is touched.
    """
    try:
        return importlib.util.find_spec("statsmodels") is not None
    except Exception:
        # find_spec runs the import finders and a finder can raise on a half-removed distribution.
        # A probe that cannot answer is not a probe that says yes.
        return False


@dataclass(frozen=True)
class BalanceReport:
    """Whether a long-format design is crossed and balanced, with the numbers if it is not."""

    balanced: bool
    n_cells_expected: int
    n_cells_present: int
    n_observations: int
    #: Cells with no observation, up to a cap, so the message names examples rather than a count.
    missing_examples: tuple[tuple[object, ...], ...] = ()
    #: Cells with more than one observation, which is a replicated design rather than a hole.
    replicated_cells: int = 0

    @property
    def n_missing(self) -> int:
        return self.n_cells_expected - self.n_cells_present

    def render(self) -> str:
        if self.balanced:
            return f"balanced and fully crossed: {self.n_cells_present} cells, one observation each"
        return (
            f"not balanced: {self.n_cells_expected} cells expected, {self.n_cells_present} present, "
            f"{self.n_missing} empty, {self.replicated_cells} with more than one observation"
        )


def check_balance(
    factors: Sequence[Sequence[object]],
    *,
    max_examples: int = 5,
) -> BalanceReport:
    """Whether a long-format design has exactly one observation in every crossed cell.

    ``factors`` is one sequence of level labels per factor, all the same length: `[objects, raters]`
    or `[objects, raters, occasions]`. The check is the whole reason the balanced estimators can be
    closed form, and skipping it is how a biased answer comes out looking like an unbiased one.
    """
    if not factors:
        raise DesignError("check_balance needs at least one factor")
    lengths = {len(f) for f in factors}
    if len(lengths) != 1:
        raise DesignError(
            f"every factor must have one label per observation; got lengths {lengths}"
        )
    levels = [sorted(set(f), key=repr) for f in factors]
    expected = int(np.prod([len(lv) for lv in levels]))
    counts: dict[tuple[object, ...], int] = {}
    for cell in zip(*factors):
        counts[cell] = counts.get(cell, 0) + 1
    present = len(counts)
    replicated = sum(1 for v in counts.values() if v > 1)
    missing: list[tuple[object, ...]] = []
    if present < expected:
        for cell in itertools.product(*levels):
            if cell not in counts:
                missing.append(cell)
                if len(missing) >= max_examples:
                    break
    return BalanceReport(
        balanced=present == expected and replicated == 0,
        n_cells_expected=expected,
        n_cells_present=present,
        n_observations=len(factors[0]),
        missing_examples=tuple(missing),
        replicated_cells=replicated,
    )


def to_cube(
    values: Sequence[float] | np.ndarray,
    factors: Sequence[Sequence[object]],
) -> tuple[np.ndarray, list[list[object]]]:
    """Reshape a long-format balanced design into the dense array the estimators take.

    Returns the array and the level labels per axis, in the order the axes were built, so a caller
    can map a component back to the grader that produced it. Raises `DesignError` on an unbalanced
    design rather than filling holes, because there is no fill value that is not an invention.
    """
    report = check_balance(factors)
    if not report.balanced:
        raise DesignError(
            f"this design is {report.render()}. The closed-form estimators in this module assume "
            f"one observation in every crossed cell; running them here would return a biased number "
            f"indistinguishable from an unbiased one. Use `fit_unbalanced`, or drop the levels that "
            f"cause the holes and refit."
        )
    levels = [sorted(set(f), key=repr) for f in factors]
    index = [{lab: i for i, lab in enumerate(lv)} for lv in levels]
    shape = tuple(len(lv) for lv in levels)
    out = np.full(shape, np.nan, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size != len(factors[0]):
        raise DesignError(f"got {v.size} values for {len(factors[0])} factor rows")
    for row, cell in enumerate(zip(*factors)):
        out[tuple(index[axis][lab] for axis, lab in enumerate(cell))] = v[row]
    _check_finite(out, "the reshaped design")
    return out, levels


def fit_unbalanced(*_args: object, **_kwargs: object) -> "GStudy":
    """The unbalanced path, which needs a mixed model and is not implemented here.

    Method-of-moments on an unbalanced design is a different estimator with a different bias, not
    the same estimator applied to messier data, so this module will not pretend. Restricted maximum
    likelihood through `statsmodels.regression.mixed_linear_model.MixedLM` is the right tool and it
    is not a dependency of this package.

    Raises `DesignError` naming what to install. An instrument catches this and turns it into a
    `Refusal` carrying the same remedy, which is where a user meets it.
    """
    have = "installed" if statsmodels_available() else "not installed"
    raise DesignError(
        f"an unbalanced or incomplete crossed design needs a mixed model rather than the method of "
        f"moments, and this module implements only the balanced closed form. statsmodels is {have}. "
        f"Fit it with statsmodels.regression.mixed_linear_model.MixedLM using variance components "
        f"for each facet, or balance the design by dropping the levels that cause the holes and use "
        f"`crossed_pr`/`crossed_pro`. Install with: pip install '{UNBALANCED_REQUIREMENT}'. "
        f"It is not currently in any declared extra of this package, so it installs directly."
    )


__all__ = [
    "OBJECT",
    "UNBALANCED_REQUIREMENT",
    "BalanceReport",
    "DStudy",
    "DesignError",
    "GStudy",
    "MeanSquares",
    "check_balance",
    "crossed_pr",
    "crossed_pro",
    "fit_unbalanced",
    "mean_squares_pr",
    "mean_squares_pro",
    "statsmodels_available",
    "to_cube",
]

"""Variance components, the design effect, and the gauge R&R arithmetic built on top of them.

Three separate things live here because they are three different answers to one question, "how much
of this number is the thing and how much is the measurement", and they need each other:

`ComponentSet` is the container a variance decomposition comes back in. It carries the truncation
flag, and that flag is the reason the container exists at all. Negative variance estimates are
routine in small designs: the method of moments subtracts one mean square from another and the
difference goes below zero whenever the true component is near zero and the degrees of freedom are
few. Every implementation truncates at zero, because a negative variance is not a variance. Almost
none of them says it did. A silently zeroed component is a lie about which facet dominates, since
zeroing the residual and zeroing the rater effect produce the same clean-looking table and opposite
conclusions, so the raw estimate is kept beside the truncated one and `truncated_names` names the
casualties.

`kish_ess` and `design_effect_ess` are the two forms of "how many independent observations is this
worth". They are the same quantity approached from opposite directions: one counts how evenly the
weights are spread, the other divides by the design effect of a known correlation. Both answer
`sigma_single_squared / var(estimator)` and nothing here is allowed to mix them up silently.

`gauge_rr` is the automotive measurement-systems-analysis reading of a variance decomposition:
`%GRR = 100 * sigma_GRR / sigma_total` and `ndc = 1.41 * sigma_part / sigma_GRR`, with the AIAG
bands at 10% and 30% and the `ndc >= 5` rule. The arithmetic is fifty years old and no language
model benchmark reports it.

Everything here is numpy and pure arithmetic. No torch, no model, no I/O.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, Mapping, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# The container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VarianceComponent:
    """One estimated variance component, with the estimate that was truncated to produce it.

    ``value`` is what everything downstream uses and it is never negative. ``raw`` is what the
    method of moments actually produced, and when it is below zero the component was truncated.
    Keeping both is not bookkeeping: a reader who sees `sigma2(r) = 0.0` cannot tell whether the
    raters agreed perfectly or whether the design was too small to resolve them, and those two call
    for opposite decisions.
    """

    name: str
    value: float
    raw: float
    df: float | None = None
    #: What the component means in the design it came from, for the rendered table.
    note: str = ""

    @property
    def truncated(self) -> bool:
        return self.raw < 0.0

    @property
    def sd(self) -> float:
        return math.sqrt(self.value)

    def render(self) -> str:
        mark = f"  (raw {self.raw:.6g}, truncated at zero)" if self.truncated else ""
        return f"{self.name:<12} {self.value:.6g}{mark}"


@dataclass(frozen=True)
class ComponentSet:
    """A variance decomposition, in the order the design produced it.

    Ordered rather than a bare dict, because the order carries the design: object of measurement
    first, then main effects, then interactions, then the residual. A table that prints in that
    order can be read; one that prints in hash order cannot.
    """

    components: tuple[VarianceComponent, ...]
    #: Free-form note about the design, carried so a table can say what it is a table of.
    design: str = ""

    def __post_init__(self) -> None:
        names = [c.name for c in self.components]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate component names in {names}")

    def __getitem__(self, name: str) -> VarianceComponent:
        for c in self.components:
            if c.name == name:
                return c
        raise KeyError(
            f"no component named {name!r} in this decomposition; it has "
            f"{', '.join(c.name for c in self.components)}"
        )

    def __contains__(self, name: str) -> bool:
        return any(c.name == name for c in self.components)

    def __iter__(self) -> Iterator[VarianceComponent]:
        return iter(self.components)

    def __len__(self) -> int:
        return len(self.components)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.components)

    def value(self, name: str) -> float:
        return self[name].value

    @property
    def total(self) -> float:
        """The total variance of a single observation: every component summed."""
        return float(sum(c.value for c in self.components))

    def share(self, name: str) -> float:
        """One component as a fraction of the total. 0.0 when the total is zero."""
        total = self.total
        return 0.0 if total <= 0.0 else self[name].value / total

    @property
    def truncated_names(self) -> tuple[str, ...]:
        """Which components came back negative and were truncated. Never silently empty."""
        return tuple(c.name for c in self.components if c.truncated)

    @property
    def any_truncated(self) -> bool:
        return bool(self.truncated_names)

    def as_dict(self) -> dict[str, float]:
        return {c.name: c.value for c in self.components}

    def raw_dict(self) -> dict[str, float]:
        return {c.name: c.raw for c in self.components}

    def render(self) -> str:
        total = self.total
        lines = [self.design] if self.design else []
        for c in self.components:
            pct = 100.0 * (c.value / total) if total > 0 else 0.0
            mark = "  TRUNCATED" if c.truncated else ""
            lines.append(f"  {c.name:<12} {c.value:>12.6g}  {pct:>5.1f}%{mark}")
        lines.append(f"  {'total':<12} {total:>12.6g}")
        if self.truncated_names:
            lines.append(
                f"  {len(self.truncated_names)} component(s) estimated below zero and truncated: "
                f"{', '.join(self.truncated_names)}. Their true values are near zero and this "
                f"design could not resolve them; do not read a truncated component as an "
                f"established zero."
            )
        return "\n".join(lines)


def truncate_at_zero(
    raw: Mapping[str, float],
    *,
    df: Mapping[str, float] | None = None,
    notes: Mapping[str, str] | None = None,
    design: str = "",
) -> ComponentSet:
    """Build a `ComponentSet` from raw method-of-moments estimates, recording every truncation.

    This is the only constructor anything in this package should use, because it is the one that
    cannot forget the flag.
    """
    df = df or {}
    notes = notes or {}
    return ComponentSet(
        components=tuple(
            VarianceComponent(
                name=name,
                value=max(0.0, float(value)),
                raw=float(value),
                df=df.get(name),
                note=notes.get(name, ""),
            )
            for name, value in raw.items()
        ),
        design=design,
    )


# ---------------------------------------------------------------------------
# Effective sample size, in its two equivalent forms
# ---------------------------------------------------------------------------


def kish_ess(weights: Sequence[float] | np.ndarray) -> float:
    """Kish's effective sample size for a weighted mean: ``(sum w)^2 / sum w^2``.

    The number of equally-weighted observations whose mean would have the same sampling variance as
    the weighted mean these weights define. Equal weights give exactly `n`; one weight carrying
    everything gives 1; all weights zero has no weighted mean at all and returns 0.0.

    Negative weights are rejected rather than squared away. A negative sampling weight is not a
    thing, and taking `w**2` would turn a sign error into a plausible number.

    The weights are divided by their maximum before squaring. The formula is scale-free, so that
    changes no answer, and it is not tidiness: at a weight of 1e-160 the squares underflow to zero
    and the function returned 0.0 for a perfectly well-conditioned design, while at 1e200 they
    overflow to infinity and it returned 0.0 again. A property test found the first of those, on a
    subnormal, and rewards on a normalised scale get within a few orders of magnitude of it.
    """
    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.size == 0:
        return 0.0
    # Before the sign check, because `nan < 0.0` is False and a NaN would slip past it. A NaN
    # weight used to return 0.0, which is the same answer this function gives for a genuinely
    # degenerate design, so a corrupt input and a real result were indistinguishable. An infinite
    # weight returned NaN. Neither is a weight.
    if not np.all(np.isfinite(w)):
        bad = int(np.count_nonzero(~np.isfinite(w)))
        raise ValueError(
            f"kish_ess got {bad} non-finite weight(s) of {w.size}. A NaN weight used to return "
            f"0.0, which is also what a design with no spread returns, so a corrupt input was "
            f"indistinguishable from a real degenerate result. Drop them or fix them upstream."
        )
    if np.any(w < 0.0):
        raise ValueError(
            "kish_ess takes non-negative weights. A negative weight squares to a positive one, so "
            "passing signed values here returns a plausible number for a meaningless design. Pass "
            "magnitudes."
        )
    peak = float(np.max(w))
    if not (peak > 0.0):
        return 0.0
    w = w / peak
    denom = float(np.sum(w * w))
    if denom <= 0.0:
        return 0.0
    return float(np.sum(w) ** 2 / denom)


def design_effect_ess(k: int, icc: float) -> float:
    """``ESS = K / (1 + (K - 1) * rho)``, the design effect for a cluster of K correlated units.

    The other route to the same quantity as `kish_ess`. `rho` is the correlation between two
    distinct units in the same cluster: at 0 the K units are K independent observations, at 1 they
    are one observation repeated K times.

    Negative `rho` is allowed and is not a bug. A within-cluster correlation below zero happens when
    the units compete for a fixed total, and the design effect then falls below 1, meaning the
    cluster is worth *more* than K independent draws. Clipping it at zero would hide that. What is
    not allowed is a `rho` that drives the denominator to zero or below, which is outside the model.
    """
    k = int(k)
    if k < 1:
        raise ValueError(f"a cluster has at least one member; got K = {k}")
    if k == 1:
        return 1.0
    deff = 1.0 + (k - 1) * float(icc)
    if deff <= 0.0:
        raise ValueError(
            f"the design effect 1 + (K-1)*rho is {deff:.6g} at K = {k}, rho = {icc:.6g}, which is "
            f"not positive. A correlation below -1/(K-1) is outside the exchangeable model this "
            f"formula assumes."
        )
    return k / deff


def group_effective_size(scores: Sequence[float] | np.ndarray) -> float:
    """The Kish count of a group's own score spread: `kish_ess` on ``|score - group mean|``.

    This is what "effective group size" means at rung 0, where the only thing available is the
    recorded scores. A group of K rollouts drives learning through its centred scores, so a rollout
    sitting at the group mean contributes nothing and one sitting far from it contributes a lot.
    The Kish count of those magnitudes is the number of rollouts the group is actually spending its
    gradient on.

    **This returns K only for a two-point distribution, and the number most callers will see is
    about 0.64K.** The quantity is `(E|dev|)^2 / E[dev^2]` on the group's centred scores, so it is
    one exactly when every rollout sits the same distance from the mean and less than one
    otherwise. Gaussian scores converge to `2/pi = 0.6366`, uniform to `0.75`, and a lognormal
    group to about `0.34`. Even the textbook-tidy case of K equally spaced distinct scores tends to
    `0.75K`. An earlier version of this docstring claimed "K distinct scores spread symmetrically
    give K", which is false: the symmetric four-point set `(-2, -1, 1, 2)` gives 3.6. That claim
    was untested and it is the anchor a reader would have calibrated on.

    So a reading of 0.64K on a perfect grader with no measurement error at all is the expected
    result and is a statement about the **shape of the reward distribution**, not about the grader.
    It says the group spends its gradient unevenly, because rollouts near the mean contribute
    almost nothing. Anyone quoting this number as "your grader costs you a third of your rollouts"
    has misread it.

    Two anchors that are true and are tested. Fifteen rollouts tied at one value and one outlier
    give 3.75 at K = 16, not 2: the fifteen tied rollouts are all displaced from the mean by the
    same small amount and they do carry signal, just very little of it each. A group with no spread
    at all returns 0.0, and that is the degenerate case `GROUP_NONDEGENERATE` exists to catch
    rather than a number to report.
    """
    x = np.asarray(scores, dtype=np.float64).ravel()
    if x.size == 0:
        return 0.0
    return kish_ess(np.abs(x - float(np.mean(x))))


def icc_oneway(groups: Sequence[Sequence[float] | np.ndarray]) -> float:
    """The one-way random-effects intraclass correlation, ICC(1), over possibly unequal groups.

    ``ICC = (MSB - MSW) / (MSB + (n0 - 1) * MSW)`` with the variance-weighted average group size
    ``n0 = (N - sum(n_g^2)/N) / (G - 1)``. Returns 0.0 when there is nothing to estimate, which is
    fewer than two groups or a total mean square of zero.

    ``n0`` is not a harmonic mean and an earlier version of this line called it one, which invites
    a maintainer to "fix" a correct formula. On group sizes `(6, 2, 3, 2)` this `n0` is 2.9744 and
    the harmonic mean is 2.6667; substituting the plain mean group size instead gives an ICC of
    0.7966 where the correct value is 0.8106. The expression above is the one that makes `MSB` have
    the right expectation under unequal group sizes, and it collapses to `k` exactly when they are
    equal.

    Not truncated at zero, deliberately: a negative ICC(1) is informative, and `design_effect_ess`
    accepts it. Callers who need a non-negative correlation should say so at their own call site
    rather than have it done to them here.
    """
    arrays = [np.asarray(g, dtype=np.float64).ravel() for g in groups]
    arrays = [a for a in arrays if a.size > 0]
    n_groups = len(arrays)
    if n_groups < 2:
        return 0.0
    sizes = np.array([a.size for a in arrays], dtype=np.float64)
    n_total = float(sizes.sum())
    if n_total <= n_groups:
        return 0.0
    grand = float(np.concatenate(arrays).mean())
    means = np.array([float(a.mean()) for a in arrays])
    ss_between = float(np.sum(sizes * (means - grand) ** 2))
    ss_within = float(sum(float(np.sum((a - a.mean()) ** 2)) for a in arrays))
    ms_between = ss_between / (n_groups - 1)
    ms_within = ss_within / (n_total - n_groups)
    n0 = (n_total - float(np.sum(sizes**2)) / n_total) / (n_groups - 1)
    denom = ms_between + (n0 - 1.0) * ms_within
    if denom == 0.0:
        return 0.0
    return float((ms_between - ms_within) / denom)


# ---------------------------------------------------------------------------
# Gauge R&R
# ---------------------------------------------------------------------------

#: AIAG MSA 4th edition's acceptance bands on %GRR against total variation. Under 10% the gauge is
#: acceptable, 10 to 30 is conditional on the cost of the application, over 30 is not acceptable.
#: The catalogue states the pass threshold as `%GRR <= 30%`, which is the outer band.
GRR_ACCEPTABLE = 10.0
GRR_MARGINAL = 30.0

#: The number of distinct categories a gauge can resolve has to reach 5 for the gauge to be used
#: for process control. Below 5 it can sort into a handful of bins and nothing finer.
NDC_MINIMUM = 5


@dataclass(frozen=True)
class GaugeRR:
    """The measurement-systems-analysis reading of a variance decomposition.

    ``grr_percent`` is `100 * sigma_GRR / sigma_total`, a ratio of standard deviations rather than
    of variances, which is the automotive convention and is worth stating because the variance
    ratio is a different and smaller-looking number. ``ndc`` is `1.41 * sigma_part / sigma_GRR`,
    the number of distinct categories the gauge can separate, and `ndc_categories` is that value
    truncated to an integer as AIAG specifies.

    ``repeatability`` and ``reproducibility`` are `None` when the design cannot identify them
    separately, which is every design with one observation per cell. That is not a gap to fill with
    a plausible split: with one trial per rater there is no replication to estimate equipment
    variation from, and reporting a repeatability there would be reporting the interaction under a
    different name.

    **Those two are variances and every other scale on this object is a standard deviation.** AIAG
    writes EV and AV as standard deviations, so `repeatability` here is `EV^2` and not `EV`: on the
    worked case with EV = 2 and AV = 1 they read 4.0 and 1.0. The sibling fields carry a `sigma_`
    prefix and these do not, which is the only signal of the change and is a thin one. Take the
    square root before comparing against a published EV or AV.
    """

    sigma_part: float
    sigma_grr: float
    sigma_total: float
    grr_percent: float
    ndc: float
    ndc_categories: int
    part_share: float
    repeatability: float | None = None
    reproducibility: float | None = None
    components: ComponentSet | None = None
    #: Which components were counted as gauge rather than as part.
    gauge_terms: tuple[str, ...] = ()

    @property
    def determined(self) -> bool:
        """Whether there was a gauge study here at all.

        A decomposition whose total variance is zero has no parts to tell apart and no gauge to
        tell them apart with, so every quantity on this object is `0/0`. It is not an acceptable
        gauge; it is not a gauge. Reported separately from `acceptable` because "perfect" and
        "nothing was measured" have to be distinguishable, and before this existed they were not:
        an all-zero decomposition rendered as a gauge resolving 2,147,483,647 distinct levels.

        Zero *gauge* variance is a different case and is **not** undetermined: a deterministic
        program verifier replayed and agreeing every time really does have no measurement error, and
        `ndc` is then genuinely infinite. What was wrong there was the rendering, not the verdict.
        See `ndc_unbounded`.
        """
        return math.isfinite(self.sigma_total) and self.sigma_total > 0.0

    @property
    def ndc_unbounded(self) -> bool:
        """Whether `ndc` is infinite, which happens exactly when the gauge variance is zero.

        `ndc_categories` truncates that infinity to `2**31 - 1`, and `verdict()` used to render it
        as "resolves at least 2147483647 distinct levels of the thing being scored". The comment two
        lines above the truncation says "Infinity is the honest value"; the sentence converted it to
        a finite count. D7 hit this on a real deterministic verifier while assembling the grader
        card, which is the wedge's headline artifact and the first place anybody outside this
        project will look.
        """
        return self.determined and not math.isfinite(self.ndc)

    @property
    def acceptable(self) -> bool:
        """The binding automotive rule, which is `ndc >= 5`, with `%GRR <= 30` beside it.

        Both are checked, and it is worth saying that the conjunction is not doing what it looks
        like it is doing. Because `TV^2 = GRR^2 + PV^2` holds by construction here, `ndc >= 5` is
        equivalent to `%GRR <= 27.14%`, which already implies `%GRR <= 30`. So ndc decides every
        case: over twenty thousand random part-and-gauge splits there are cases where %GRR passes
        and ndc fails, and none the other way. Both are kept because they are both AIAG rules and a
        reader looking for one should find it, but nobody should believe the %GRR term is adding a
        constraint.

        Undetermined is not acceptable. A degenerate decomposition returns False rather than the
        True that `0.0 <= 30` and a sentinel ndc used to produce between them.
        """
        if not self.determined:
            return False
        return self.grr_percent <= GRR_MARGINAL and self.ndc_categories >= NDC_MINIMUM

    @property
    def band(self) -> str:
        if not self.determined:
            return "undetermined"
        if self.grr_percent <= GRR_ACCEPTABLE:
            return "acceptable"
        if self.grr_percent <= GRR_MARGINAL:
            return "marginal"
        return "unacceptable"

    def verdict(self) -> str:
        """The sentence a user acts on."""
        if self.ndc_unbounded:
            caveat = (
                ""
                if self.repeatability is not None
                else (
                    " This design has one observation per cell, so it carries no replication to "
                    "estimate repeatability from and a measured zero here is the weakest possible "
                    "evidence for it. Vary a facet and score again, or read the flakiness spread."
                )
            )
            return (
                f"%GRR = 0.0%, and ndc is unbounded because the measured gauge variance is exactly "
                f"zero. Every distinct value of the thing being scored is resolvable.{caveat}"
            )
        if not self.determined:
            return (
                "No gauge study here. Every variance component in this decomposition is zero, so "
                "there are no parts to tell apart and nothing to tell them apart with, and %GRR "
                "and ndc are both zero over zero. Supply a decomposition with a nonzero total "
                "variance, or read the degenerate-group fraction, which is the quantity that "
                "answers what went wrong."
            )
        head = (
            f"%GRR = {self.grr_percent:.1f}% ({self.band}), ndc = {self.ndc_categories} "
            f"(raw {self.ndc:.2f})"
        )
        if self.acceptable:
            return (
                f"{head}. This gauge resolves at least {self.ndc_categories} distinct levels of "
                f"the thing being scored."
            )
        if self.ndc_categories < 2:
            return (
                f"{head}. This gauge cannot resolve two adjacent items: the measurement spread is "
                f"as large as the spread it is measuring."
            )
        return (
            f"{head}. This gauge sorts into {self.ndc_categories} bins and nothing finer. A "
            f"difference smaller than that is inside the measurement system."
        )


def gauge_rr(
    components: ComponentSet,
    *,
    part: str = "p",
    repeatability: str | None = None,
    reproducibility_terms: Sequence[str] | None = None,
) -> GaugeRR:
    """Gauge repeatability and reproducibility from a variance decomposition.

    Everything that is not the part is gauge. That is the general rule and it reduces to the AIAG
    two-facet definition exactly: with parts, operators and trials, gauge is
    `sigma2(operator) + sigma2(part x operator) + sigma2(error)` and part is `sigma2(part)`.
    Stating it as "total minus part" rather than by listing the terms is what lets the same function
    read a three-facet crossed design, where there are four more terms and every one of them is
    still measurement rather than signal.

    ``repeatability`` names the pure-error component when the design has one; passing None leaves
    both the repeatability and the reproducibility as None rather than splitting a confounded term.
    """
    total = components.total
    part_var = components.value(part)
    grr_var = max(0.0, total - part_var)
    sigma_total = math.sqrt(total)
    sigma_part = math.sqrt(part_var)
    sigma_grr = math.sqrt(grr_var)

    grr_percent = 100.0 * (sigma_grr / sigma_total) if sigma_total > 0 else 0.0
    if sigma_grr > 0:
        ndc = 1.41 * sigma_part / sigma_grr
    else:
        # A gauge with no measurement variance resolves everything. Infinity is the honest value
        # and it only arises when every non-part component truncated to zero, which the component
        # set flags.
        ndc = math.inf
    ndc_categories = 2**31 - 1 if math.isinf(ndc) else int(ndc)

    rep = None
    rep2 = None
    if repeatability is not None and repeatability in components:
        rep = components.value(repeatability)
        if reproducibility_terms is None:
            rep2 = max(0.0, grr_var - rep)
        else:
            rep2 = float(sum(components.value(t) for t in reproducibility_terms))

    return GaugeRR(
        sigma_part=sigma_part,
        sigma_grr=sigma_grr,
        sigma_total=sigma_total,
        grr_percent=grr_percent,
        ndc=ndc,
        ndc_categories=ndc_categories,
        part_share=components.share(part),
        repeatability=rep,
        reproducibility=rep2,
        components=components,
        gauge_terms=tuple(n for n in components.names if n != part),
    )


__all__ = [
    "GRR_ACCEPTABLE",
    "GRR_MARGINAL",
    "NDC_MINIMUM",
    "ComponentSet",
    "GaugeRR",
    "VarianceComponent",
    "design_effect_ess",
    "gauge_rr",
    "group_effective_size",
    "icc_oneway",
    "kish_ess",
    "truncate_at_zero",
]

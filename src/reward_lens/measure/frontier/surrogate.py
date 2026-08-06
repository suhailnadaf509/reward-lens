"""N4. Falsifiable conditions on the proxy, and the concomitant of the n-th order statistic.

Two readings, and both are stronger than a predicted turning point because both are unoccupied
where the turning point is not.

**The checklist.** The surrogate-endpoint literature has spent forty years on exactly the question
"under what conditions does optimising a proxy fail to harm the true endpoint", and
`"surrogate endpoint" AND "reward"` returns zero on arXiv. Four criteria are implemented here as
explicit testable conditions on the joint distribution of ``(r, g, treatment)``, each returning
pass, fail or **untestable**:

1. Prentice (1989), as its own four operational conditions: treatment moves the surrogate,
   treatment moves the true endpoint, the surrogate moves the true endpoint, and the surrogate
   fully captures the treatment effect.
2. Freedman (1992), the proportion of treatment effect explained, with the interval Freedman's own
   paper is about: the point of that paper is that the interval is usually too wide to act on.
3. Buyse and Molenberghs (1998), individual-level against trial-level association. The trial-level
   half needs several units each carrying both arms, and where there is one sample there is no
   trial level, so it returns untestable and says what would make it testable.
4. VanderWeele (2013), the monotonicity conditions that exclude the surrogate paradox. These are
   statements about individual counterfactuals. They have testable *necessary* consequences, which
   are checked and can fail informatively, and the sufficient conditions cannot be established
   without an intervention that assigns treatment and observes both potential outcomes.

`untestable` is a first-class verdict rather than a soft failure, and criterion 4 in particular
usually comes back untestable. A report that says "this proxy fails Prentice's fourth condition and
here is the monotonicity condition it violates" is a more useful artifact than a predicted turning
point, and reporting honestly that a criterion could not be tested is the point rather than a
shortfall.

**The concomitant.** The gold reward evaluated at the proxy-maximising response *is* the concomitant
of the n-th order statistic. If ``(R_i, G_i)`` are n draws from a joint distribution and ``R_{n:n}``
is the largest proxy score, the gold paired with it is written ``G_{[n:n]}`` and has forty years of
exact finite-n distribution theory behind it (David and Nagaraja; Nagaraja and David 1994 solves the
maximum-of-selected-concomitants problem directly). The identity that makes it computable is one
line of the tower property:

    E[G_{[n:n]}] = E[ E[G | R = R_{n:n}] ] = n * integral_0^1 m(F^{-1}(u)) u^{n-1} du

and on the empirical joint distribution of the observed pairs that integral is a finite sum: the
probability that the maximum of n draws lands on the i-th smallest observed proxy score is
``(i/N)^n - ((i-1)/N)^n``, so

    E_hat[G_{[n:n]}] = sum_i g_[i] * [ (i/N)^n - ((i-1)/N)^n ]

with tied proxy scores collapsed into blocks and the block mean of the gold used, which is what a
uniform choice among tied maxima gives by exchangeability. That expression is **exact at the stated
n given the joint sample**. It is not a large-n approximation and it is not a simulation. Simulation
at the same n is the check on it and is one of this instrument's listed baselines.

Best-of-n Goodhart is a concomitant problem that nobody has recognised as one, and the exact answer
has been sitting in the order-statistics literature the whole time.

Kill condition, from the catalogue record: if every criterion returns untestable on every real
grader for want of an intervention, the checklist is a research note rather than an instrument and
only the concomitant reading ships.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np

from reward_lens.core.envelope import EnvelopeSpec, RegimeCondition
from reward_lens.core.evidence import register_payload
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.quantity import BaselineID
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.types import (
    Access,
    Capability,
    Component,
    GaugeStatus,
    Phase,
)
from reward_lens.measure.frontier._base import FrontierInstrument
from reward_lens.measure.frontier.horizon import ALL_SUBSTRATES
from reward_lens.measure.frontier.potential import (
    DEFAULT_ESS_FLOOR,
    bootstrap_indices,
    percentile_interval,
)

Verdict = Literal["pass", "fail", "untestable"]

N4_ACCESS: dict[Component, Access] = {
    Component.GRADER: Access.QUERY,
    Component.GOLD: Access.QUERY,
    Component.POLICY: Access.QUERY,
}

#: N4's envelope. The criteria are statements about the joint distribution of two channels measured
#: over a window, and a grader whose weights or rubric moved inside that window has produced two
#: samples from two graders rather than one sample from one.
N4_ENVELOPE = EnvelopeSpec(
    requires=frozenset({RegimeCondition.STATIONARY_GRADER}),
    measured_by={RegimeCondition.STATIONARY_GRADER: "monitor.check_standard_drift"},
    on_violation="refuse",
)

CHECKLIST_BASELINES: tuple[BaselineID, ...] = ("baseline.marginal_correlation",)

CONCOMITANT_BASELINES: tuple[BaselineID, ...] = (
    "baseline.marginal_correlation",
    "baseline.simulated_best_of_n",
)

#: Freedman's own suggested bar for the proportion of treatment effect explained. It is a
#: convention rather than a measurement and it is exposed as a parameter.
DEFAULT_PTE_TARGET = 0.75

#: The conventional bar for both of Buyse and Molenberghs' association coefficients.
DEFAULT_R2_TARGET = 0.8

#: How many units carrying both arms the trial-level association needs before it is worth
#: estimating. Below this the regression of effects on effects has fewer points than parameters
#: plus a handful, and its R^2 is a description of the units rather than an estimate.
DEFAULT_MIN_UNITS = 8


# ---------------------------------------------------------------------------
# The small statistical tools, written out because the assumptions matter here
# ---------------------------------------------------------------------------


def permutation_mean_difference(
    values: np.ndarray, arm: np.ndarray, *, permutations: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """`(difference in means, two-sided permutation p)` between arm 1 and arm 0.

    A permutation test rather than a t-test because the gold channel is often binary and the proxy
    is often skewed, and the exchangeability the permutation needs is exactly what the two arms are
    assumed to have under the null.
    """
    v = np.asarray(values, dtype=np.float64)
    a = np.asarray(arm).astype(bool)
    if a.all() or (~a).all():
        return float("nan"), float("nan")
    observed = float(v[a].mean() - v[~a].mean())
    rng = np.random.default_rng(seed)
    n1 = int(a.sum())
    total = v.sum()
    count = 0
    for _ in range(permutations):
        idx = rng.permutation(v.size)[:n1]
        s1 = v[idx].sum()
        diff = s1 / n1 - (total - s1) / (v.size - n1)
        if abs(diff) >= abs(observed) - 1e-15:
            count += 1
    return observed, (count + 1) / (permutations + 1)


def partial_correlation(x: np.ndarray, y: np.ndarray, given: np.ndarray | None) -> float:
    """Pearson correlation of `x` and `y` after removing a linear dependence on `given`.

    With `given` None this is the plain marginal correlation, which is the honest reading when
    there is one arm: there is nothing to adjust for.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if given is None:
        xr, yr = x - x.mean(), y - y.mean()
    else:
        z = np.asarray(given, dtype=np.float64)
        design = np.column_stack([np.ones_like(z), z])
        xr = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
        yr = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    denominator = float(np.sqrt((xr @ xr) * (yr @ yr)))
    return float(xr @ yr) / denominator if denominator > 0 else float("nan")


def ols_with_interval(
    y: np.ndarray, columns: Sequence[np.ndarray], level: float = 0.95
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(coefficients, standard errors, half-widths)` for `y ~ 1 + columns`, by least squares.

    The half-width uses the Student t quantile at the residual degrees of freedom rather than 1.96,
    because the mediation criterion is asked at small n often enough for the difference to change a
    verdict.
    """
    from scipy import stats

    design = np.column_stack([np.ones(len(y)), *[np.asarray(c, dtype=np.float64) for c in columns]])
    y = np.asarray(y, dtype=np.float64)
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ beta
    dof = max(1, design.shape[0] - design.shape[1])
    sigma2 = float(residual @ residual) / dof
    xtx_inv = np.linalg.pinv(design.T @ design)
    se = np.sqrt(np.maximum(0.0, sigma2 * np.diag(xtx_inv)))
    crit = float(stats.t.ppf(0.5 * (1.0 + level), dof))
    return beta, se, crit * se


@register_payload
@dataclass
class Criterion:
    """One surrogate-endpoint condition, its verdict, and the numbers behind it.

    ``testable_by`` is empty when the criterion was testable on the data supplied. When it is not,
    it names the design that would make it testable, which is the difference between a checklist
    that reports a gap and one that reports a failure.
    """

    number: int
    name: str
    source: str
    verdict: Verdict
    statistic: float = float("nan")
    threshold: float = float("nan")
    detail: str = ""
    testable_by: str = ""

    def render(self) -> str:
        head = f"  {self.number}. {self.name:<52} {self.verdict.upper()}"
        lines = [head, f"       {self.source}", f"       {self.detail}"]
        if self.testable_by:
            lines.append(f"       testable by: {self.testable_by}")
        return "\n".join(lines)


def _combine(sub: Sequence[Criterion]) -> Verdict:
    """Conjunction over sub-conditions: any fail is a fail, else any untestable is untestable."""
    if any(c.verdict == "fail" for c in sub):
        return "fail"
    if any(c.verdict == "untestable" for c in sub):
        return "untestable"
    return "pass"


# ---------------------------------------------------------------------------
# The four criteria
# ---------------------------------------------------------------------------


def _prentice(
    r: np.ndarray,
    g: np.ndarray,
    t: np.ndarray | None,
    *,
    alpha: float,
    permutations: int,
    equivalence_margin: float | None,
    seed: int,
) -> tuple[Criterion, tuple[Criterion, ...]]:
    """Prentice (1989)'s four operational conditions, then their conjunction."""
    no_arm = (
        "an arm. Score the same rollouts under two selection rules, or two prompts, or two "
        "checkpoints, and pass `treatment=` the 0/1 indicator"
    )
    sub: list[Criterion] = []

    if t is None:
        sub.append(
            Criterion(
                1,
                "treatment affects the surrogate",
                "Prentice (1989) condition 1",
                "untestable",
                detail="no treatment channel was supplied, so there is no effect to test",
                testable_by=no_arm,
            )
        )
        sub.append(
            Criterion(
                1,
                "treatment affects the true endpoint",
                "Prentice (1989) condition 2",
                "untestable",
                detail="no treatment channel was supplied",
                testable_by=no_arm,
            )
        )
    else:
        dr, pr = permutation_mean_difference(r, t, permutations=permutations, seed=seed)
        sub.append(
            Criterion(
                1,
                "treatment affects the surrogate",
                "Prentice (1989) condition 1",
                "pass" if pr < alpha else "fail",
                statistic=dr,
                threshold=alpha,
                detail=f"mean proxy difference {dr:+.4g}, permutation p = {pr:.4g}",
            )
        )
        dg, pg = permutation_mean_difference(g, t, permutations=permutations, seed=seed + 1)
        sub.append(
            Criterion(
                1,
                "treatment affects the true endpoint",
                "Prentice (1989) condition 2",
                "pass" if pg < alpha else "fail",
                statistic=dg,
                threshold=alpha,
                detail=f"mean gold difference {dg:+.4g}, permutation p = {pg:.4g}",
            )
        )

    rho = partial_correlation(r, g, t)
    beta, _, half_all = ols_with_interval(g, [r] if t is None else [t, r])
    slope_index = 1 if t is None else 2
    slope, slope_half = float(beta[slope_index]), float(half_all[slope_index])
    sub.append(
        Criterion(
            1,
            "the surrogate affects the true endpoint",
            "Prentice (1989) condition 3",
            "pass" if abs(slope) > slope_half else "fail",
            statistic=rho,
            threshold=0.0,
            detail=(
                f"partial correlation of proxy and gold given treatment = {rho:+.4g}; the "
                f"regression slope of gold on proxy is {slope:+.4g} +/- {slope_half:.4g}"
            ),
        )
    )

    if t is None:
        sub.append(
            Criterion(
                1,
                "the surrogate fully captures the treatment effect",
                "Prentice (1989) condition 4",
                "untestable",
                detail="no treatment channel, so there is no effect whose capture could be checked",
                testable_by=no_arm,
            )
        )
    else:
        beta, se, half = ols_with_interval(g, [t, r])
        adj, adj_half = float(beta[1]), float(half[1])
        lo, hi = adj - adj_half, adj + adj_half
        if lo > 0.0 or hi < 0.0:
            verdict: Verdict = "fail"
            detail = (
                f"the treatment coefficient adjusted for the proxy is {adj:+.4g} "
                f"[{lo:+.4g}, {hi:+.4g}], and the interval excludes zero. Treatment reaches gold by "
                f"a route the proxy does not carry, so the proxy does not fully mediate it"
            )
            testable_by = ""
        elif equivalence_margin is not None and max(abs(lo), abs(hi)) <= equivalence_margin:
            verdict = "pass"
            detail = (
                f"the adjusted treatment coefficient is {adj:+.4g} [{lo:+.4g}, {hi:+.4g}], entirely "
                f"inside the equivalence margin of +/-{equivalence_margin:.4g}"
            )
            testable_by = ""
        else:
            verdict = "untestable"
            needed = max(abs(lo), abs(hi))
            detail = (
                f"the adjusted treatment coefficient is {adj:+.4g} [{lo:+.4g}, {hi:+.4g}], which "
                f"contains zero. A non-significant coefficient is not evidence of full capture: it "
                f"is compatible with an unmediated effect of up to {needed:.4g}"
            )
            testable_by = (
                f"an equivalence test. State the largest unmediated effect on gold you would "
                f"tolerate and pass it as `equivalence_margin`; on this sample the interval only "
                f"rules out effects above {needed:.4g}, so a margin below that needs more rollouts"
            )
        sub.append(
            Criterion(
                1,
                "the surrogate fully captures the treatment effect",
                "Prentice (1989) condition 4",
                verdict,
                statistic=adj,
                threshold=equivalence_margin if equivalence_margin is not None else float("nan"),
                detail=detail,
                testable_by=testable_by,
            )
        )

    passed = sum(c.verdict == "pass" for c in sub)
    # The conjunction inherits its remedy from whichever sub-conditions could not be tested. An
    # aggregate verdict of untestable with no route to testing it is the shape of report that gets
    # read as "nothing to do here", which is the opposite of what it means.
    blocked = [c for c in sub if c.verdict == "untestable" and c.testable_by]
    return (
        Criterion(
            1,
            "Prentice's four operational conditions",
            "Prentice (1989)",
            _combine(sub),
            statistic=float(passed),
            threshold=4.0,
            detail=(
                f"{passed} of 4 conditions pass; "
                + ", ".join(f"c{i + 1} {c.verdict}" for i, c in enumerate(sub))
            ),
            testable_by="; ".join(
                dict.fromkeys(
                    f"condition {sub.index(c) + 1} needs {c.testable_by}" for c in blocked
                )
            ),
        ),
        tuple(sub),
    )


def _freedman(
    r: np.ndarray,
    g: np.ndarray,
    t: np.ndarray | None,
    *,
    pte_target: float,
    resamples: int,
    seed: int,
) -> Criterion:
    """Freedman (1992): the proportion of treatment effect explained, with its interval."""
    name = "the proxy explains the treatment effect on gold"
    source = "Freedman (1992), proportion of treatment effect explained"
    if t is None:
        return Criterion(
            2,
            name,
            source,
            "untestable",
            detail="the proportion explained is a ratio of two treatment effects, and there is no treatment",
            testable_by="a treatment arm. Without one the numerator and the denominator are both zero",
        )

    def pte(rr: np.ndarray, gg: np.ndarray, tt: np.ndarray) -> tuple[float, float]:
        unadj, _, _ = ols_with_interval(gg, [tt])
        adj, _, _ = ols_with_interval(gg, [tt, rr])
        if abs(unadj[1]) < 1e-12:
            return float("nan"), float(unadj[1])
        return 1.0 - float(adj[1]) / float(unadj[1]), float(unadj[1])

    point, total_effect = pte(r, g, np.asarray(t, dtype=np.float64))
    idx = bootstrap_indices(r.size, resamples, seed)
    tf = np.asarray(t, dtype=np.float64)
    draws = np.array([pte(r[i], g[i], tf[i])[0] for i in idx])
    lo, hi = percentile_interval(draws)

    _, _, half = ols_with_interval(g, [tf])
    if abs(total_effect) <= float(half[1]):
        return Criterion(
            2,
            name,
            source,
            "untestable",
            statistic=point,
            threshold=pte_target,
            detail=(
                f"the unadjusted treatment effect on gold is {total_effect:+.4g} +/- "
                f"{float(half[1]):.4g}, which contains zero. The proportion explained is a ratio "
                f"with a denominator indistinguishable from zero and it takes any value at all"
            ),
            testable_by=(
                "a treatment that moves gold. Freedman's ratio is only defined where there is an "
                "effect to divide up, which is Prentice's second condition"
            ),
        )

    if np.isfinite(lo) and lo >= pte_target:
        verdict: Verdict = "pass"
        detail = f"PTE = {point:.3g} [{lo:.3g}, {hi:.3g}], entirely above {pte_target:.3g}"
        testable_by = ""
    elif np.isfinite(hi) and hi < pte_target:
        verdict = "fail"
        detail = (
            f"PTE = {point:.3g} [{lo:.3g}, {hi:.3g}], entirely below {pte_target:.3g}. The proxy "
            f"leaves at least {100 * (1 - hi):.0f}% of the treatment effect on gold unexplained"
        )
        testable_by = ""
    else:
        verdict = "untestable"
        detail = (
            f"PTE = {point:.3g} [{lo:.3g}, {hi:.3g}], an interval of width {hi - lo:.3g} that "
            f"straddles {pte_target:.3g}. This is the finding Freedman's paper is about: the "
            f"proportion explained needs a sample several times larger than the one that "
            f"establishes the treatment effect itself"
        )
        testable_by = (
            f"more rollouts. The interval has to be narrower than the distance from the point "
            f"estimate to {pte_target:.3g}, and its width falls as one over the square root of n"
        )
    return Criterion(
        2,
        name,
        source,
        verdict,
        statistic=point,
        threshold=pte_target,
        detail=detail,
        testable_by=testable_by,
    )


def _buyse_molenberghs(
    r: np.ndarray,
    g: np.ndarray,
    t: np.ndarray | None,
    unit: np.ndarray | None,
    *,
    r2_target: float,
    min_units: int,
) -> Criterion:
    """Buyse and Molenberghs (1998): individual-level against trial-level association."""
    name = "association holds at the individual and the trial level"
    source = "Buyse and Molenberghs (1998)"
    rho = partial_correlation(r, g, t)
    r2_indiv = float(rho * rho)

    if unit is None or t is None:
        return Criterion(
            3,
            name,
            source,
            "untestable",
            statistic=r2_indiv,
            threshold=r2_target,
            detail=(
                f"individual-level R^2 = {r2_indiv:.3g}. The trial-level coefficient is the one "
                f"that decides this criterion and it is not estimable from a single undivided "
                f"sample: it is the R^2 of the regression of per-unit gold effects on per-unit "
                f"proxy effects, and there are no units here"
            ),
            testable_by=(
                f"at least {min_units} units each carrying both arms. A unit is anything the "
                f"treatment can be applied within: a prompt cluster, a task family, a grader "
                f"version, a checkpoint. Pass `unit=` the label per rollout"
            ),
        )

    labels = np.asarray(unit)
    tt = np.asarray(t).astype(bool)
    dr: list[float] = []
    dg: list[float] = []
    for u in np.unique(labels):
        m = labels == u
        if not (m & tt).any() or not (m & ~tt).any():
            continue
        dr.append(float(r[m & tt].mean() - r[m & ~tt].mean()))
        dg.append(float(g[m & tt].mean() - g[m & ~tt].mean()))
    used = len(dr)
    if used < min_units:
        return Criterion(
            3,
            name,
            source,
            "untestable",
            statistic=r2_indiv,
            threshold=r2_target,
            detail=(
                f"individual-level R^2 = {r2_indiv:.3g}; only {used} of "
                f"{len(np.unique(labels))} units carry both arms, against the {min_units} the "
                f"trial-level regression needs"
            ),
            testable_by=f"{min_units - used} more units with both arms present",
        )

    rho_trial = partial_correlation(np.array(dr), np.array(dg), None)
    r2_trial = float(rho_trial * rho_trial)
    if r2_indiv >= r2_target and r2_trial >= r2_target:
        verdict: Verdict = "pass"
        detail = (
            f"individual-level R^2 = {r2_indiv:.3g} and trial-level R^2 = {r2_trial:.3g} over "
            f"{used} units, both above {r2_target:.3g}"
        )
    else:
        verdict = "fail"
        weak = "trial" if r2_trial < r2_indiv else "individual"
        detail = (
            f"individual-level R^2 = {r2_indiv:.3g}, trial-level R^2 = {r2_trial:.3g} over {used} "
            f"units, against a bar of {r2_target:.3g}. The {weak} level is the weaker one: a proxy "
            f"that tracks gold within a unit but whose *effects* do not track gold's effects "
            f"across units is the classic invalid surrogate"
        )
    return Criterion(
        3, name, source, verdict, statistic=r2_trial, threshold=r2_target, detail=detail
    )


def _vanderweele(
    r: np.ndarray,
    g: np.ndarray,
    t: np.ndarray | None,
    *,
    alpha: float,
) -> Criterion:
    """VanderWeele (2013): the monotonicity conditions that exclude the surrogate paradox.

    The sufficient conditions are about individual counterfactuals and no observational sample can
    establish them. What a sample can do is falsify their necessary consequences, and it is worth
    being precise about which is which: a failure here is a real finding and a pass is not
    available.
    """
    from scipy import stats

    name = "monotonicity excludes the surrogate paradox"
    source = "VanderWeele (2013)"
    intervention = (
        "an intervention. The sufficient conditions are that treatment moves the proxy the same "
        "way for every individual and the proxy moves gold the same way for every individual, with "
        "no unmeasured confounding between proxy and gold. Those are statements about both "
        "potential outcomes for the same rollout and observing one of them destroys the other"
    )

    spearman = stats.spearmanr(r, g)
    rho_s, p_s = float(spearman.statistic), float(spearman.pvalue)
    if rho_s < 0.0 and p_s < alpha:
        return Criterion(
            4,
            name,
            source,
            "fail",
            statistic=rho_s,
            threshold=0.0,
            detail=(
                f"Spearman rank correlation of proxy and gold is {rho_s:+.4g} (p = {p_s:.3g}), "
                f"significantly negative. Monotonicity of gold in the proxy is a necessary "
                f"consequence of VanderWeele's conditions and it is falsified here, so the "
                f"surrogate paradox is not excluded: pushing the proxy up pushes gold down"
            ),
        )

    if t is not None:
        arm = np.asarray(t).astype(bool)
        if arm.any() and (~arm).any():
            ks = stats.ks_2samp(r[arm], r[~arm], alternative="greater")
            if float(ks.pvalue) < alpha:
                return Criterion(
                    4,
                    name,
                    source,
                    "fail",
                    statistic=float(ks.statistic),
                    threshold=alpha,
                    detail=(
                        f"the treated arm's proxy distribution is not stochastically at least as "
                        f"large as the untreated arm's (one-sided two-sample KS D = "
                        f"{float(ks.statistic):.4g}, p = {float(ks.pvalue):.3g}). Stochastic "
                        f"dominance is a necessary consequence of treatment moving the proxy the "
                        f"same way for every individual, and it is falsified"
                    ),
                )

    return Criterion(
        4,
        name,
        source,
        "untestable",
        statistic=rho_s,
        threshold=0.0,
        detail=(
            f"the testable necessary conditions hold: Spearman rank correlation of proxy and gold "
            f"is {rho_s:+.4g} (p = {p_s:.3g}) and nothing contradicts stochastic monotonicity. That "
            f"is consistency, not verification. The sufficient conditions are about individual "
            f"counterfactuals and this sample cannot reach them"
        ),
        testable_by=intervention,
    )


@register_payload
@dataclass
class ChecklistReading:
    """The four criteria, Prentice's four sub-conditions, and the sentence they produce."""

    n: int
    has_treatment: bool
    has_units: bool
    alpha: float
    criteria: tuple[Criterion, ...]
    prentice_conditions: tuple[Criterion, ...]
    n_pass: int
    n_fail: int
    n_untestable: int
    baselines: dict[str, float] = field(default_factory=dict)
    says: str = ""

    def verdict_of(self, number: int) -> Verdict:
        for c in self.criteria:
            if c.number == number:
                return c.verdict
        raise KeyError(f"no criterion numbered {number}")

    def render(self) -> str:
        return "\n".join([self.says, *(c.render() for c in self.criteria)])


def measure_checklist(
    reward: Sequence[float] | np.ndarray,
    gold: Sequence[float] | np.ndarray,
    *,
    treatment: Sequence[float] | np.ndarray | None = None,
    unit: Sequence[Any] | np.ndarray | None = None,
    alpha: float = 0.05,
    permutations: int = 2000,
    resamples: int = 400,
    equivalence_margin: float | None = None,
    pte_target: float = DEFAULT_PTE_TARGET,
    r2_target: float = DEFAULT_R2_TARGET,
    min_units: int = DEFAULT_MIN_UNITS,
    seed: int = 0,
    instrument: str = "SurrogateChecklist",
) -> ChecklistReading | Refusal:
    """The four criteria, each with a pass, fail or untestable verdict."""
    r = np.asarray(reward, dtype=np.float64).ravel()
    g = np.asarray(gold, dtype=np.float64).ravel()
    if r.size != g.size:
        raise ValueError(
            f"the proxy has {r.size} scores and the gold channel {g.size}; the criteria are "
            f"conditions on a joint distribution and need both on the same rollouts"
        )
    t = None if treatment is None else np.asarray(treatment).ravel()
    if t is not None and t.size != r.size:
        raise ValueError(f"the treatment indicator has {t.size} entries for {r.size} rollouts")
    u = None if unit is None else np.asarray(unit).ravel()
    if u is not None and u.size != r.size:
        raise ValueError(f"the unit label has {u.size} entries for {r.size} rollouts")

    if r.size < 4:
        return Refusal(
            instrument=instrument,
            reason=RefusalReason.BELOW_LOD,
            detail=f"{r.size} rollouts is fewer than any of the four criteria can be asked on",
            remedy="score more rollouts. Every criterion here is a test and a test needs a sample.",
            statistics={"n": int(r.size)},
        )

    prentice, sub = _prentice(
        r,
        g,
        t,
        alpha=alpha,
        permutations=permutations,
        equivalence_margin=equivalence_margin,
        seed=seed,
    )
    criteria = (
        prentice,
        _freedman(r, g, t, pte_target=pte_target, resamples=resamples, seed=seed + 10),
        _buyse_molenberghs(r, g, t, u, r2_target=r2_target, min_units=min_units),
        _vanderweele(r, g, t, alpha=alpha),
    )
    counts = {v: sum(c.verdict == v for c in criteria) for v in ("pass", "fail", "untestable")}

    failed = [c for c in criteria if c.verdict == "fail"]
    if failed:
        first = failed[0]
        says = (
            f"Fails criterion {first.number}: {first.name}. "
            f"{counts['pass']} pass, {counts['fail']} fail, {counts['untestable']} untestable."
        )
    else:
        says = (
            f"No criterion is falsified on this sample. {counts['pass']} pass and "
            f"{counts['untestable']} of 4 could not be tested on the data supplied."
        )

    reading = ChecklistReading(
        n=int(r.size),
        has_treatment=t is not None,
        has_units=u is not None,
        alpha=float(alpha),
        criteria=criteria,
        prentice_conditions=sub,
        n_pass=counts["pass"],
        n_fail=counts["fail"],
        n_untestable=counts["untestable"],
        baselines={"baseline.marginal_correlation": partial_correlation(r, g, None)},
        says=says,
    )
    return reading


# ---------------------------------------------------------------------------
# The concomitant of the n-th order statistic
# ---------------------------------------------------------------------------


def _blocks(r: np.ndarray, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ascending distinct proxy values as `(cumulative counts, block mean of gold)`.

    Ties are collapsed rather than broken. With tied maxima the response the optimiser picks is
    ambiguous, and the natural rule is a uniform choice among them, which by exchangeability gives
    the mean gold over the block. This matters far more than it sounds: a binary verifier has
    exactly two blocks and every best-of-n reading against one is a tied-maximum reading.
    """
    order = np.argsort(r, kind="stable")
    rs, gs = r[order], g[order]
    boundaries = np.flatnonzero(np.diff(rs)) + 1
    edges = np.concatenate([boundaries, [rs.size]])
    starts = np.concatenate([[0], boundaries])
    means = np.array([gs[a:b].mean() for a, b in zip(starts, edges)])
    return edges.astype(np.float64), means


def concomitant_weights(counts: np.ndarray, n: int, total: int) -> np.ndarray:
    """`P(max of n draws lands in block j) = (c_j/N)^n - (c_{j-1}/N)^n`."""
    upper = (counts / total) ** n
    lower = np.concatenate([[0.0], upper[:-1]])
    return upper - lower


def concomitant_expectation(
    reward: np.ndarray, gold: np.ndarray, n: int
) -> tuple[float, float, float]:
    """`(E[G_[n:n]], E[R_{n:n}], effective blocks)` for the empirical joint, exactly.

    Exact at the stated ``n`` given the joint sample: the only approximation anywhere in it is the
    empirical distribution standing in for the population one, and no simulation error enters at
    all. The third return value is ``1 / sum_j p_j^2``, the effective number of distinct proxy
    values carrying the answer, which is the same Kish construction the visibility horizon uses and
    is what tells you when a best-of-n reading has collapsed onto the single top observation.
    """
    counts, gold_means = _blocks(reward, gold)
    _, reward_means = _blocks(reward, reward)
    p = concomitant_weights(counts, int(n), int(reward.size))
    ess = 1.0 / float(p @ p) if float(p @ p) > 0 else float("nan")
    return float(p @ gold_means), float(p @ reward_means), ess


def simulate_concomitant(
    reward: np.ndarray, gold: np.ndarray, n: int, *, replicates: int = 20_000, seed: int = 0
) -> tuple[float, float]:
    """`(mean, Monte Carlo standard error)` of the gold at the proxy-argmax over `replicates` draws.

    The baseline the catalogue names, and the check on the exact expression rather than a substitute
    for it. Ties are broken uniformly, which is the rule the exact expression assumes, so the two
    are estimating the same object and a disagreement is a bug in one of them.
    """
    rng = np.random.default_rng(seed)
    total = reward.size
    out = np.empty(replicates, dtype=np.float64)
    for b in range(replicates):
        idx = rng.integers(0, total, int(n))
        vals = reward[idx]
        best = np.flatnonzero(vals == vals.max())
        out[b] = gold[idx[best[rng.integers(0, best.size)]]]
    return float(out.mean()), float(out.std(ddof=1) / np.sqrt(replicates))


@register_payload
@dataclass
class ConcomitantReading:
    """Best-of-n as a concomitant problem: the exact curve, its peak, and the simulation check."""

    n_pairs: int
    best_of: int
    expected_gold: float
    expected_gold_ci: tuple[float, float]
    expected_proxy: float
    gold_at_one: float
    effective_blocks: float
    #: The exact curve over n, which is where best-of-n stops buying gold.
    ns: np.ndarray
    gold_curve: np.ndarray
    proxy_curve: np.ndarray
    ess_curve: np.ndarray
    peak_n: int
    peak_gold: float
    peak_is_interior: bool
    baselines: dict[str, float] = field(default_factory=dict)
    says: str = ""

    def render(self) -> str:
        return self.says


def measure_concomitant(
    reward: Sequence[float] | np.ndarray,
    gold: Sequence[float] | np.ndarray,
    *,
    best_of: int = 64,
    n_grid: Sequence[int] | None = None,
    floor: float = DEFAULT_ESS_FLOOR,
    resamples: int = 400,
    simulate: int = 0,
    seed: int = 0,
    instrument: str = "ConcomitantBestOfN",
) -> ConcomitantReading | Refusal:
    """The exact finite-n expected gold at the proxy-argmax, with the curve over n.

    Refuses with `ESS_BELOW_FLOOR` when the requested ``n`` puts more than ``1 - floor`` of the
    probability onto too few distinct observed proxy values. That is the same horizon logic the tilt
    family has, in the best-of-n parametrisation: at large ``n`` relative to the bank, the exact
    answer is the gold of the single best observation, and reporting it with an interval would be
    reporting one rollout with an interval.
    """
    r = np.asarray(reward, dtype=np.float64).ravel()
    g = np.asarray(gold, dtype=np.float64).ravel()
    if r.size != g.size:
        raise ValueError(f"the proxy has {r.size} scores and the gold channel {g.size}")
    if r.size < 2:
        raise ValueError("the concomitant of a maximum needs at least two pairs")
    if best_of < 1:
        raise ValueError(f"best-of-n needs n >= 1; got {best_of}")

    total = int(r.size)
    point, proxy, ess = concomitant_expectation(r, g, best_of)

    if ess < floor * total:
        return Refusal(
            instrument=instrument,
            reason=RefusalReason.ESS_BELOW_FLOOR,
            detail=(
                f"at n = {best_of} on a bank of {total} pairs, the exact best-of-n expectation is "
                f"carried by {ess:.2f} effective distinct proxy values, which is below the "
                f"{floor:.0%} floor of {floor * total:.1f}. The expression is still exact and what "
                f"it is exact about is the gold of the handful of top rollouts in this bank"
            ),
            remedy=(
                f"score more rollouts, or ask for a smaller n. The effective count is "
                f"1 / sum_j p_j^2 with p_j the chance the maximum lands on the j-th distinct proxy "
                f"value, and it falls roughly as the bank over n, so n = {max(1, int(floor * total)):,} "
                f"is where this bank stops being about the population and starts being about its "
                f"own top rows."
            ),
            statistics={
                "best_of": int(best_of),
                "n_pairs": total,
                "effective_blocks": ess,
                "floor": float(floor),
                "expected_gold": point,
            },
        )

    ns = (
        np.array(sorted({int(v) for v in n_grid}), dtype=int)
        if n_grid is not None
        else np.unique(np.clip(np.round(np.geomspace(1, max(2, best_of * 2), 24)), 1, None)).astype(
            int
        )
    )
    curve = [concomitant_expectation(r, g, int(k)) for k in ns]
    gold_curve = np.array([c[0] for c in curve])
    proxy_curve = np.array([c[1] for c in curve])
    ess_curve = np.array([c[2] for c in curve])
    visible = ess_curve >= floor * total
    j = int(np.argmax(np.where(visible, gold_curve, -np.inf)))
    interior = bool(visible[j] and 0 < j < gold_curve.size - 1 and visible[j + 1])

    idx = bootstrap_indices(total, resamples, seed)
    draws = np.array([concomitant_expectation(r[i], g[i], best_of)[0] for i in idx])
    lo, hi = percentile_interval(draws)

    baselines = {"baseline.marginal_correlation": partial_correlation(r, g, None)}
    if simulate > 0:
        sim, sim_se = simulate_concomitant(r, g, best_of, replicates=simulate, seed=seed + 1)
        baselines["baseline.simulated_best_of_n"] = sim
        baselines["baseline.simulated_best_of_n_se"] = sim_se

    reading = ConcomitantReading(
        n_pairs=total,
        best_of=int(best_of),
        expected_gold=point,
        expected_gold_ci=(lo, hi),
        expected_proxy=proxy,
        gold_at_one=float(g.mean()),
        effective_blocks=ess,
        ns=ns,
        gold_curve=gold_curve,
        proxy_curve=proxy_curve,
        ess_curve=ess_curve,
        peak_n=int(ns[j]),
        peak_gold=float(gold_curve[j]),
        peak_is_interior=interior,
        baselines=baselines,
    )
    tail = (
        f" Gold at the proxy-argmax peaks at n = {int(ns[j])} and falls after it."
        if interior
        else " Gold at the proxy-argmax is still rising at the largest n this bank can see."
    )
    reading.says = (
        f"At n = {best_of}, expected gold at the proxy-argmax is {point:.3g} "
        f"[{lo:.3g}, {hi:.3g}], exactly, against {float(g.mean()):.3g} for a single draw."
    ) + tail
    return reading


# ---------------------------------------------------------------------------
# The two instruments
# ---------------------------------------------------------------------------


class SurrogateChecklist(FrontierInstrument):
    """N4, first reading. Four surrogate-endpoint criteria, each pass, fail or untestable.

    Kill condition: if every criterion returns untestable on every real grader for want of an
    intervention, the checklist is a research note rather than an instrument and only the
    concomitant reading ships.
    """

    name = "SurrogateChecklist"
    version = "1.0"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "N4"
    deviations = (
        "Prentice's fourth condition can return pass only against an equivalence margin the caller "
        "states. A non-significant adjusted treatment coefficient is not evidence of full "
        "capture, and reading it as one is the standard way this criterion is got wrong",
        "VanderWeele's conditions can fail and cannot pass. What is checked is their testable "
        "necessary consequences; the sufficient conditions are about individual counterfactuals",
        "the trial-level coefficient of Buyse and Molenberghs needs units carrying both arms. "
        "Where the sample is undivided it returns untestable rather than substituting the "
        "individual-level coefficient, which is the substitution the 1998 paper exists to warn "
        "against",
    )

    quantity = "frontier.prentice_checklist"
    requires: dict[Component, Access] = N4_ACCESS
    substrates = ALL_SUBSTRATES
    phases = frozenset({Phase.PRE_RUN})
    envelope = N4_ENVELOPE
    invariance = "reward.affine"
    invariance_relation = INVARIANT
    baselines = CHECKLIST_BASELINES
    rung = 0

    def __init__(
        self,
        reward: Sequence[float] | np.ndarray | None = None,
        gold: Sequence[float] | np.ndarray | None = None,
        *,
        treatment: Sequence[float] | np.ndarray | None = None,
        unit: Sequence[Any] | np.ndarray | None = None,
        alpha: float = 0.05,
        permutations: int = 2000,
        resamples: int = 400,
        equivalence_margin: float | None = None,
        pte_target: float = DEFAULT_PTE_TARGET,
        r2_target: float = DEFAULT_R2_TARGET,
        min_units: int = DEFAULT_MIN_UNITS,
        seed: int = 0,
    ) -> None:
        self.reward = reward
        self.gold = gold
        self.treatment = treatment
        self.unit = unit
        self.alpha = float(alpha)
        self.permutations = int(permutations)
        self.resamples = int(resamples)
        self.equivalence_margin = equivalence_margin
        self.pte_target = float(pte_target)
        self.r2_target = float(r2_target)
        self.min_units = int(min_units)
        self.seed = int(seed)

    def compute(self) -> Any:
        if self.reward is None or self.gold is None:
            return _no_gold_refusal(self.name, self.reward is None, self.gold is None)
        return measure_checklist(
            self.reward,
            self.gold,
            treatment=self.treatment,
            unit=self.unit,
            alpha=self.alpha,
            permutations=self.permutations,
            resamples=self.resamples,
            equivalence_margin=self.equivalence_margin,
            pte_target=self.pte_target,
            r2_target=self.r2_target,
            min_units=self.min_units,
            seed=self.seed,
            instrument=self.name,
        )


class ConcomitantBestOfN(FrontierInstrument):
    """N4, second reading. The exact finite-n expected gold at the proxy-maximising response.

    Kill condition: if every criterion of the checklist returns untestable on every real grader for
    want of an intervention, the checklist is a research note rather than an instrument and only
    this reading ships.
    """

    name = "ConcomitantBestOfN"
    version = "1.0"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "N4"
    deviations = (
        "the expectation is exact for the *empirical* joint distribution of the observed pairs, "
        "which is the plug-in of the exact finite-n theory rather than the theory applied to a "
        "fitted parametric joint. No simulation error enters, and the sampling error in the "
        "empirical distribution does, which is what the bootstrap interval reports",
        "draws are iid from the pooled bank. Realised best-of-n usually selects within a prompt, "
        "and the two differ whenever the gold rate varies across prompts. The per-prompt reading "
        "is the one to compare against and it is reported as a baseline where the caller has the "
        "grouping",
    )

    quantity = "frontier.concomitant_bon"
    requires: dict[Component, Access] = N4_ACCESS
    substrates = ALL_SUBSTRATES
    phases = frozenset({Phase.PRE_RUN})
    envelope = N4_ENVELOPE
    invariance = "reward.affine"
    invariance_relation = INVARIANT
    baselines = CONCOMITANT_BASELINES
    rung = 0

    def __init__(
        self,
        reward: Sequence[float] | np.ndarray | None = None,
        gold: Sequence[float] | np.ndarray | None = None,
        *,
        best_of: int = 64,
        n_grid: Sequence[int] | None = None,
        floor: float = DEFAULT_ESS_FLOOR,
        resamples: int = 400,
        simulate: int = 0,
        seed: int = 0,
    ) -> None:
        self.reward = reward
        self.gold = gold
        self.best_of = int(best_of)
        self.n_grid = n_grid
        self.floor = float(floor)
        self.resamples = int(resamples)
        self.simulate = int(simulate)
        self.seed = int(seed)

    def compute(self) -> Any:
        if self.reward is None or self.gold is None:
            return _no_gold_refusal(self.name, self.reward is None, self.gold is None)
        return measure_concomitant(
            self.reward,
            self.gold,
            best_of=self.best_of,
            n_grid=self.n_grid,
            floor=self.floor,
            resamples=self.resamples,
            simulate=self.simulate,
            seed=self.seed,
            instrument=self.name,
        )


def _no_gold_refusal(instrument: str, no_reward: bool, no_gold: bool) -> Refusal:
    missing = {}
    if no_reward:
        missing["GRADER"] = "QUERY"
    if no_gold:
        missing["GOLD"] = "QUERY"
    return Refusal(
        instrument=instrument,
        reason=RefusalReason.ACCESS_INSUFFICIENT,
        detail=(
            f"this reading is a joint statement about the proxy and a gold channel and needs both "
            f"on the same n rollouts; missing "
            f"{', '.join(f'{k}:{v}' for k, v in sorted(missing.items()))}"
        ),
        remedy=(
            "pass `reward=` and `gold=`, both scored on the same n base-policy rollouts in the "
            "same order. A gold channel here is any measurement you would rather have than the "
            "proxy: a human label, a verifier, a stronger grader, a downstream outcome."
        ),
        statistics={"missing": missing},
    )


__all__ = [
    "CHECKLIST_BASELINES",
    "CONCOMITANT_BASELINES",
    "DEFAULT_MIN_UNITS",
    "DEFAULT_PTE_TARGET",
    "DEFAULT_R2_TARGET",
    "N4_ACCESS",
    "N4_ENVELOPE",
    "ChecklistReading",
    "ConcomitantBestOfN",
    "ConcomitantReading",
    "Criterion",
    "SurrogateChecklist",
    "Verdict",
    "concomitant_expectation",
    "concomitant_weights",
    "measure_checklist",
    "measure_concomitant",
    "ols_with_interval",
    "partial_correlation",
    "permutation_mean_difference",
    "simulate_concomitant",
]

"""D4, `verifier.sobol_ST`: which inputs actually move the score.

A rubric with nine criteria looks like nine criteria. Decompose the variance of its total and you
often find that one of them carries most of it and the rest are decoration: they are written down,
they are argued over, and they do not move the number. Variance-based sensitivity analysis answers
that in one pass, it is forty years old, and `SALib` has shipped it since 2014.

Three rungs, and the first exists to be disbelieved.

**Rung 0, one-at-a-time.** Fix every input at its midpoint, sweep one from low to high, record how
far the score moved. This is what people actually do when they ask which criterion matters, and it
is biased: it explores only the axes through one point, so it cannot see interaction at all. It is
computed here so the bias can be *shown* rather than asserted. The interaction mass,
`sum(S_Ti) - sum(S1_i)`, is exactly the fraction of output variance one-at-a-time is structurally
blind to, and it is reported next to the rank order the two methods disagree about.

**Rung 1, first-order `S1`.** The fraction of output variance removed on average by learning one
input's value. Main effects only.

**Rung 2, total-effect `S_Ti` with bootstrap intervals.** The fraction of output variance that
*remains* when every input except this one is fixed, so it counts the input's main effect plus
every interaction it takes part in. `S_Ti = 0` is the strong statement: that input cannot move the
score through any path. It is the one to quote.

**The connection to the contract layer, which is why this instrument is worth more than it looks.**
Holmström and Milgrom's equal-compensation principle says the raw weight on a reward component
must be proportional to that component's sensitivity `μ'_i`, and the component with the lowest
`α_i / μ'_i` is the one the policy starves. `μ'_i` is a dose-response slope and nobody measures it.
The Sobol design already evaluates the grader on a space-filling sample of its inputs, so the slope
comes out of the same evaluations for free. `SensitivityProfile.contract_inputs()` is that handoff,
and it is deliberately a data structure rather than a check: the check is N6's.

**On sample counts.** `SALib.sample.sobol.sample(problem, N)` draws `N·(2D+2)` points with second
order and `N·(D+2)` without, where `N` is the *base* sample size and not the evaluation budget.
Passing a budget as `N` overshoots by a factor of eight on a three-input problem. `N` must also be
a power of two: SALib emits a `UserWarning` when it is not and then returns numbers anyway, which
is easy to lose under a pytest filter, so this module refuses a non-power-of-two `N` at
construction and names the next one up.

**Kill condition:** D4 has none, and that is right. A sensitivity profile is a description of a
function; there is no result that would make the decomposition wrong.
What can be wrong is the reading of it, so the profile carries the interaction mass and the
one-at-a-time comparison rather than a single headline number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from reward_lens.core.budget import BudgetTerm, UncertaintyBudget
from reward_lens.core.envelope import EnvelopeSpec, RegimeCondition
from reward_lens.core.evidence import Uncertainty, register_payload
from reward_lens.core.extras import require_extra
from reward_lens.core.invariance import Relation
from reward_lens.core.quantity import CostModel
from reward_lens.core.reading import Reading, Refusal, RefusalReason
from reward_lens.core.types import (
    Access,
    Capability,
    Component,
    GaugeStatus,
    Phase,
    Substrate,
)
from reward_lens.measure.base import BaseObservable, Context, PreflightResult, run
from reward_lens.verifier.metamorphic import QuerySubject

# ---------------------------------------------------------------------------
# The rubric under analysis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RubricInput:
    """One input the grader's score is a function of, with the range it is swept over.

    For a rubric this is a criterion and its scale (0 to 1, 1 to 5). For a threshold-style verifier
    it can be any continuous knob: a numeric tolerance, a partial-credit weight, a timeout. What it
    cannot be is a categorical choice, because Sobol' sampling is over a box and a categorical
    input encoded as a number produces indices about an arbitrary ordering.
    """

    name: str
    low: float = 0.0
    high: float = 1.0
    description: str = ""

    def __post_init__(self) -> None:
        if not self.high > self.low:
            raise ValueError(
                f"input {self.name!r} has bounds [{self.low}, {self.high}], which is empty or "
                f"inverted. A degenerate range makes every index for it exactly zero for the "
                f"wrong reason."
            )


#: A grader whose score is a function of named numeric inputs. This is `GRADER:QUERY` plus the
#: ability to choose the inputs, which is what makes factorial sampling possible.
RubricScorer = Callable[[Mapping[str, float]], float]


def _is_power_of_two(n: int) -> bool:
    return n > 0 and n & (n - 1) == 0


def _next_power_of_two(n: int) -> int:
    return 1 if n < 1 else 1 << (n - 1).bit_length()


# ---------------------------------------------------------------------------
# What the reading carries
# ---------------------------------------------------------------------------


@register_payload
@dataclass(frozen=True)
class ContractSensitivity:
    """One component's sensitivity, in the form the equal-compensation check consumes.

    The equal-compensation `μ'_i` is the derivative of signal `i`'s mean with respect to effort
    spent on task `i`. The Sobol' design already evaluates the grader on a space-filling sample of
    its inputs, so the slope is an ordinary least-squares coefficient over evaluations that have
    already been paid for. Because a Sobol' design draws inputs independently, the partial slope
    computed here and the marginal slope coincide in expectation, which is stated rather than
    assumed because it stops holding the moment somebody supplies a correlated design.

    **What this is not.** `n(k) = σ²(k) / μ'(k)²` needs a `σ²` that is the *grader's noise* on
    component `k`, which is a replication quantity from series A (`grader.variance_components`) and
    is not measurable from a Sobol' sweep at all: every evaluation here is a single call, so the
    spread in `Y` is the spread caused by moving the inputs, not the spread of repeated calls at
    fixed inputs. `output_variance_share` below is the second of those and must not be substituted
    for the first. Getting that wrong inverts the sorting theorem's ranking.

    **And `μ'` alone is not enough to rank a component.** A least-squares slope is a linear
    summary, so a criterion whose entire influence is an interaction with another has `μ' ≈ 0`
    while its total effect is large. That is measured rather than hypothetical: the acceptance
    fixture for this instrument has a criterion with `μ' = 0.00` and `S_Ti = 0.16`. A check that
    reads `μ'` and ignores `total_effect` will conclude such a component does nothing, and it does
    a sixth of the work.
    """

    name: str
    #: `μ'_i`: the OLS slope of the score on this input over the Sobol' sample.
    mu_prime: float
    mu_prime_stderr: float
    #: `S_Ti`. Zero means this component cannot move the score through any path.
    total_effect: float
    total_effect_conf: float
    #: The fraction of *input-driven* output variance this component accounts for. NOT `σ²(k)`.
    output_variance_share: float
    input_range: tuple[float, float]


@register_payload
@dataclass(frozen=True)
class SensitivityIndex:
    """All three rungs for one input, side by side, so the bias in rung 0 is visible."""

    name: str
    s1: float
    s1_conf: float
    st: float
    st_conf: float
    #: Rung 0. The score's swing when this input alone moves from low to high with the rest held
    #: at their midpoints. Kept for contrast and never for a conclusion.
    oat_effect: float
    oat_share: float
    mu_prime: float
    mu_prime_stderr: float
    input_low: float
    input_high: float

    @property
    def interaction(self) -> float:
        """`S_Ti - S1_i`: the share of variance this input moves only in company."""
        return self.st - self.s1


@register_payload
@dataclass(frozen=True)
class SensitivityProfile:
    """The reading: a sensitivity profile over the grader's inputs, with rung 0 kept for contrast.

    ``interaction_mass`` is `sum(S_Ti) - sum(S1_i)` and is the number that indicts one-at-a-time
    analysis: it is the fraction of output variance that lives in interactions, which sweeping one
    input at a time through a single base point cannot see by construction.
    """

    grader: str
    indices: tuple[SensitivityIndex, ...]
    n_base: int
    evaluations: int
    calc_second_order: bool
    num_resamples: int
    conf_level: float
    output_mean: float
    output_variance: float
    interaction_mass: float
    second_order: Mapping[str, float]
    oat_rank: tuple[str, ...]
    sobol_rank: tuple[str, ...]
    rank_inversions: int
    rung: int

    # -- what the headline is ---------------------------------------------

    @property
    def dominant(self) -> SensitivityIndex | None:
        return max(self.indices, key=lambda i: i.st, default=None)

    @property
    def inert(self) -> tuple[str, ...]:
        """Inputs whose total effect is exactly zero: they cannot move the score at all."""
        return tuple(i.name for i in self.indices if i.st == 0.0)

    # -- the handoff to the contract layer --------------------------------

    def contract_inputs(self) -> tuple[ContractSensitivity, ...]:
        """`μ'_i` per component, for the equal-compensation check. N6 consumes this."""
        total = sum(max(i.st, 0.0) for i in self.indices)
        return tuple(
            ContractSensitivity(
                name=i.name,
                mu_prime=i.mu_prime,
                mu_prime_stderr=i.mu_prime_stderr,
                total_effect=i.st,
                total_effect_conf=i.st_conf,
                output_variance_share=(max(i.st, 0.0) / total if total > 0 else float("nan")),
                input_range=(i.input_low, i.input_high),
            )
            for i in self.indices
        )

    def mu_prime(self) -> dict[str, float]:
        """`{component: μ'_i}`, the one mapping the equal-compensation check needs."""
        return {i.name: i.mu_prime for i in self.indices}

    # -- uncertainty -------------------------------------------------------

    def budget(self) -> UncertaintyBudget:
        """The GUM table for the headline `S_Ti`, composed from what was actually computed.

        Two terms, and the second only when it is real. SALib reports `ST_conf = Z · s`, where `s`
        is the standard deviation over `num_resamples` bootstrap replicates and
        `Z = Phi^-1(0.5 + conf/2)`, so dividing back out by `Z` recovers the standard uncertainty
        the GUM wants and the degrees of freedom are `num_resamples - 1`.

        The second term uses a property of the estimator rather than an assumption about it. A
        total-effect index cannot be negative, so any negative `S_Ti` in the profile is estimator
        noise measured directly at this sample size; the most negative one is used as a
        rectangular half-width. When every index is non-negative the term is omitted rather than
        set to zero, because a term of zero and an unestimated term look the same on a table and
        are not the same thing.
        """
        dom = self.dominant
        if dom is None:
            return UncertaintyBudget()
        z = _z_for(self.conf_level)
        terms = [
            BudgetTerm(
                name="sobol_bootstrap",
                value=dom.st_conf / z if z else dom.st_conf,
                kind="A",
                dof=float(self.num_resamples - 1),
                note=(
                    f"SALib bootstrap over {self.num_resamples} resamples; ST_conf divided by "
                    f"Z={z:.4g} for the {self.conf_level:.0%} level"
                ),
            )
        ]
        floor = -min((i.st for i in self.indices), default=0.0)
        if floor > 0:
            terms.append(
                BudgetTerm.from_half_width(
                    "estimator_noise_floor",
                    floor,
                    "rectangular",
                    note=(
                        f"the most negative S_Ti in this profile is {-floor:.4g}. A total-effect "
                        f"index cannot be negative, so that magnitude is the estimator's own noise "
                        f"at N={self.n_base}, measured rather than assumed"
                    ),
                )
            )
        return UncertaintyBudget(terms=tuple(terms))

    # -- presentation ------------------------------------------------------

    def render(self) -> str:
        dom = self.dominant
        lines = [
            f"sensitivity profile for {self.grader}",
            f"    {self.evaluations:,} evaluations (N={self.n_base}, D={len(self.indices)}, "
            f"second order {'on' if self.calc_second_order else 'off'})",
        ]
        if dom is not None:
            lines.append(
                f"    {dom.st:.1%} of the score variance is driven by {dom.name!r} "
                f"out of {len(self.indices)}"
            )
        width = max((len(i.name) for i in self.indices), default=4)
        lines.append(
            f"    {'input':<{width}}  {'S1':>9}  {'S_Ti':>9}  {'+/-':>9}  "
            f"{'interact':>9}  {'OAT (r0)':>9}  {'mu_prime':>10}"
        )
        for i in sorted(self.indices, key=lambda i: -i.st):
            lines.append(
                f"    {i.name:<{width}}  {i.s1:>9.4f}  {i.st:>9.4f}  {i.st_conf:>9.4f}  "
                f"{i.interaction:>9.4f}  {i.oat_share:>9.4f}  {i.mu_prime:>10.4g}"
            )
        lines.append(
            f"    interaction mass sum(S_Ti) - sum(S1) = {self.interaction_mass:.4f}: the share of "
            f"variance one-at-a-time cannot see"
        )
        if self.rank_inversions:
            lines.append(
                f"    rung 0 and rung 2 disagree on the order of {self.rank_inversions} pairs. "
                f"OAT ranks {' > '.join(self.oat_rank)}; Sobol' ranks {' > '.join(self.sobol_rank)}"
            )
        if self.inert:
            lines.append(
                "    total effect exactly zero (cannot move the score through any path): "
                + ", ".join(self.inert)
            )
        return "\n".join(lines)


def _z_for(conf_level: float) -> float:
    """`Phi^-1(0.5 + conf/2)`, the factor SALib multiplies the bootstrap std by."""
    from scipy.stats import norm

    return float(norm.ppf(0.5 + conf_level / 2.0))


# ---------------------------------------------------------------------------
# The estimators
# ---------------------------------------------------------------------------


def sobol_problem(inputs: Sequence[RubricInput]) -> dict[str, Any]:
    """The SALib problem dict for a set of rubric inputs."""
    return {
        "num_vars": len(inputs),
        "names": [i.name for i in inputs],
        "bounds": [[i.low, i.high] for i in inputs],
    }


def sobol_sample(
    inputs: Sequence[RubricInput],
    n_base: int,
    *,
    calc_second_order: bool = True,
    seed: int = 0,
) -> np.ndarray:
    """The `N·(2D+2)` design matrix. Raises on a non-power-of-two `N` rather than warning."""
    require_extra("verifier", subsystem="D4 (verifier.sobol_ST)")
    from SALib.sample import sobol as sobol_sampler

    if not _is_power_of_two(n_base):
        raise ValueError(
            f"N={n_base} is not a power of two. The balance property of a Sobol' sequence needs "
            f"one, and SALib warns and then returns numbers anyway, which is easy to lose under a "
            f"warning filter. Use N={_next_power_of_two(n_base)}."
        )
    return np.asarray(
        sobol_sampler.sample(
            sobol_problem(inputs), n_base, calc_second_order=calc_second_order, seed=seed
        )
    )


def sobol_indices(
    inputs: Sequence[RubricInput],
    y: np.ndarray,
    *,
    calc_second_order: bool = True,
    num_resamples: int = 100,
    conf_level: float = 0.95,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """`S1`, `ST` and their bootstrap intervals from a design matrix's outputs."""
    require_extra("verifier", subsystem="D4 (verifier.sobol_ST)")
    from SALib.analyze import sobol as sobol_analyzer

    return sobol_analyzer.analyze(
        sobol_problem(inputs),
        np.asarray(y, dtype=np.float64),
        calc_second_order=calc_second_order,
        num_resamples=num_resamples,
        conf_level=conf_level,
        seed=seed,
    )


def total_effect(
    inputs: Sequence[RubricInput], y: np.ndarray, *, calc_second_order: bool = True
) -> np.ndarray:
    """`S_Ti` alone, for the generated invariance test, which asserts on a scalar."""
    return np.asarray(
        sobol_indices(inputs, y, calc_second_order=calc_second_order, num_resamples=2)["ST"]
    )


def one_at_a_time(
    scorer: RubricScorer, inputs: Sequence[RubricInput]
) -> tuple[np.ndarray, np.ndarray]:
    """Rung 0. `(effect, share)`: the score's swing per input, and it normalised to sum to one.

    Biased by construction and computed for exactly that reason. It moves one input at a time
    through a single base point, so an input whose whole influence is an interaction with another
    registers as zero here and as its full total effect at rung 2. `2^D` corners would fix that and
    would also be the factorial design Sobol' already replaces, so this stays as the cheap thing
    people actually do, kept in the report so the difference is visible.
    """
    base = {i.name: 0.5 * (i.low + i.high) for i in inputs}
    effects = np.zeros(len(inputs), dtype=np.float64)
    for k, i in enumerate(inputs):
        lo = {**base, i.name: i.low}
        hi = {**base, i.name: i.high}
        effects[k] = abs(float(scorer(hi)) - float(scorer(lo)))
    total = effects.sum()
    shares = effects / total if total > 0 else np.zeros_like(effects)
    return effects, shares


def dose_response_slopes(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`(slope, stderr)` per input: ordinary least squares of the score on the inputs.

    This is the equal-compensation `μ'_i`. Fitted with an intercept and all inputs at once, so each
    coefficient is a partial derivative holding the others fixed, which is what the contract-theory
    definition asks for. Under a Sobol' design the inputs are independent, so the partial and the
    marginal slope agree; under a correlated design supplied by a caller they would not, and the
    partial one is still the right answer.
    """
    n, d = x.shape
    design = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coef
    dof = n - d - 1
    if dof <= 0:
        return coef[1:], np.full(d, float("nan"))
    sigma2 = float(residual @ residual) / dof
    xtx_inv = np.linalg.pinv(design.T @ design)
    stderr = np.sqrt(np.maximum(sigma2 * np.diag(xtx_inv), 0.0))
    return coef[1:], stderr[1:]


# ---------------------------------------------------------------------------
# The instrument
# ---------------------------------------------------------------------------

#: D4's envelope. A Sobol' design spends `N·(2D+2)` calls, and a variance decomposition over a
#: window in which the grader changed attributes the change to whichever inputs happened to be
#: moving. That is the one precondition access cannot see.
D4_ENVELOPE = EnvelopeSpec(
    requires=frozenset({RegimeCondition.STATIONARY_GRADER}),
    measured_by={RegimeCondition.STATIONARY_GRADER: "env.replay_fidelity"},
)


class SobolSensitivity(BaseObservable):
    """D4. Decompose the grader's output variance over its inputs, with rung 0 kept for contrast.

    **Kill condition: none, and that is the correct answer.** A variance decomposition is a
    description of a function and there is no measurement that would make it wrong. What can go
    wrong is the reading, so the profile carries the interaction mass and the one-at-a-time
    comparison rather than a single headline share.
    """

    name = "verifier.sobol_ST"
    version = "1.0"
    quantity = "verifier.sobol_ST"
    capabilities = Capability.SCORES
    requires = {Component.GRADER: Access.QUERY}
    substrates = frozenset({Substrate.PROGRAM, Substrate.PROCEDURAL, Substrate.COMPOSITE})
    phases = frozenset({Phase.PRE_RUN, Phase.POST_RUN})
    envelope = D4_ENVELOPE
    invariance = "reward.affine"
    invariance_relation = Relation("invariant")
    baselines = ("equal_weighting", "one_at_a_time")
    rung = 2
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "Sobol' (2001) variance decomposition; Saltelli et al. (2010) estimators"
    deviations = (
        "the model under analysis is a grader rather than a simulator, so the inputs are rubric "
        "criteria and the output is a score",
        "rung 0 is one-at-a-time, computed and reported only so its bias against rung 2 is a "
        "measured quantity rather than a claim",
        "SALib centres and scales the output internally, which is what makes the indices exactly "
        "invariant under an affine rescaling of the reward",
    )

    def __init__(
        self,
        scorer: RubricScorer,
        inputs: Sequence[RubricInput],
        *,
        n_base: int = 1024,
        calc_second_order: bool = True,
        num_resamples: int = 100,
        conf_level: float = 0.95,
        seed: int = 0,
        name: str = "",
    ) -> None:
        if not inputs:
            raise ValueError("a sensitivity analysis needs at least one input to be sensitive to")
        if not _is_power_of_two(n_base):
            raise ValueError(
                f"N={n_base} is not a power of two. The balance property of a Sobol' sequence "
                f"needs one; SALib warns and returns numbers anyway, which is easy to lose under "
                f"a warning filter. Use N={_next_power_of_two(n_base)}."
            )
        self.scorer = scorer
        self.inputs = tuple(inputs)
        self.n_base = n_base
        self.calc_second_order = calc_second_order
        self.num_resamples = num_resamples
        self.conf_level = conf_level
        self.seed = seed
        self.grader_name = name or getattr(scorer, "__name__", "rubric")
        self.subject = QuerySubject(name=self.grader_name, fn=scorer)  # type: ignore[arg-type]

    @property
    def evaluations(self) -> int:
        d = len(self.inputs)
        return self.n_base * (2 * d + 2 if self.calc_second_order else d + 2)

    def sample_matrix(self) -> np.ndarray:
        """The design this instrument evaluates, so a refusal about it can be acted on.

        Named on the non-finite refusal's remedy: "here is the design, the bad rows are your
        reproducers" is an instruction, and "the grader returned NaN" is not.
        """
        return sobol_sample(
            self.inputs, self.n_base, calc_second_order=self.calc_second_order, seed=self.seed
        )

    # -- preflight ---------------------------------------------------------

    def preflight(self, ctx: Context) -> PreflightResult:
        base = super().preflight(ctx)
        if not base.ok:
            return base
        calls = self.evaluations + 2 * len(self.inputs)
        return PreflightResult(
            instrument=self.name,
            ok=True,
            rung=self.rung,
            cost=CostModel(
                calls=calls,
                note=(
                    f"{self.evaluations:,} Sobol' evaluations at N={self.n_base}, D="
                    f"{len(self.inputs)} plus {2 * len(self.inputs)} one-at-a-time calls. "
                    f"No GPU, no model, no record."
                ),
            ),
            regime=base.regime,
            unchecked=base.unchecked,
            notes=(f"N·(2D+2) = {self.evaluations:,}; N is the base sample size, not the budget",),
        )

    # -- the measurement ---------------------------------------------------

    def estimate(self, ctx: Context | None = None) -> Reading:
        ctx = ctx if ctx is not None else Context(signal=self.subject, readout="score")
        pre = self.preflight(ctx)
        if not pre.ok and pre.refusal is not None:
            return pre.refusal
        return run(self, ctx)

    def measure(self, ctx: Context) -> Any:
        names = [i.name for i in self.inputs]
        x = self.sample_matrix()
        y = np.array([float(self.scorer(dict(zip(names, row)))) for row in x], dtype=np.float64)

        if not np.all(np.isfinite(y)):
            bad = int((~np.isfinite(y)).sum())
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.VOID,
                detail=(
                    f"the grader returned a non-finite score on {bad} of {y.size:,} points of the "
                    f"design. A variance decomposition over a sample containing NaN or infinity "
                    f"produces NaN indices, which read as 'no effect'."
                ),
                remedy=(
                    "find the inputs that produce it. `SobolSensitivity.sample_matrix` gives the "
                    "design, and the non-finite rows are the reproducers. A grader that returns "
                    "NaN on part of its own declared input range is `grader.silent_zero_rate`'s "
                    "question before it is this one's."
                ),
                statistics={"non_finite": bad, "evaluations": int(y.size)},
            )

        # `ptp` rather than `std`, and the difference is not pedantry. A constant array of 0.7 has
        # a standard deviation of 1.1e-16 rather than 0, because 0.7 is not exactly representable
        # and summing 256 copies of it does not divide back exactly. SALib then divides by that
        # 1e-16 in its internal centring step and the estimators return arrays where scalars are
        # expected. Peak-to-peak is exact, and it is the test SALib itself uses.
        if float(np.ptp(y)) == 0.0:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.BELOW_LOD,
                detail=(
                    f"the grader returned the same score, {y[0]:g}, on all {y.size:,} points of "
                    f"the design. There is no output variance to decompose, so every index would "
                    f"be 0/0."
                ),
                remedy=(
                    "widen the input bounds until the score moves, or check that the scorer is "
                    "reading the inputs at all. A rubric whose total does not respond to any "
                    "criterion over its own declared ranges is a finding in itself, and it is "
                    "`grader.silent_zero_rate`'s question rather than this one's."
                ),
                statistics={"constant_score": float(y[0]), "evaluations": int(y.size)},
            )

        si = sobol_indices(
            self.inputs,
            y,
            calc_second_order=self.calc_second_order,
            num_resamples=self.num_resamples,
            conf_level=self.conf_level,
            seed=self.seed,
        )
        s1 = np.asarray(si["S1"], dtype=np.float64)
        s1_conf = np.asarray(si["S1_conf"], dtype=np.float64)
        st = np.asarray(si["ST"], dtype=np.float64)
        st_conf = np.asarray(si["ST_conf"], dtype=np.float64)

        oat_effect, oat_share = one_at_a_time(self.scorer, self.inputs)
        slopes, slope_err = dose_response_slopes(x, y)

        indices = tuple(
            SensitivityIndex(
                name=inp.name,
                s1=float(s1[k]),
                s1_conf=float(s1_conf[k]),
                st=float(st[k]),
                st_conf=float(st_conf[k]),
                oat_effect=float(oat_effect[k]),
                oat_share=float(oat_share[k]),
                mu_prime=float(slopes[k]),
                mu_prime_stderr=float(slope_err[k]),
                input_low=inp.low,
                input_high=inp.high,
            )
            for k, inp in enumerate(self.inputs)
        )

        second: dict[str, float] = {}
        if self.calc_second_order and "S2" in si:
            s2 = np.asarray(si["S2"], dtype=np.float64)
            for a in range(len(names)):
                for b in range(a + 1, len(names)):
                    value = float(s2[a, b])
                    if math.isfinite(value):
                        second[f"{names[a]}|{names[b]}"] = value

        oat_rank = tuple(sorted(names, key=lambda n: -oat_share[names.index(n)]))
        sobol_rank = tuple(sorted(names, key=lambda n: -st[names.index(n)]))
        profile = SensitivityProfile(
            grader=self.grader_name,
            indices=indices,
            n_base=self.n_base,
            evaluations=int(y.size),
            calc_second_order=self.calc_second_order,
            num_resamples=self.num_resamples,
            conf_level=self.conf_level,
            output_mean=float(y.mean()),
            output_variance=float(y.var()),
            interaction_mass=float(st.sum() - s1.sum()),
            second_order=second,
            oat_rank=oat_rank,
            sobol_rank=sobol_rank,
            rank_inversions=_rank_inversions(oat_rank, sobol_rank),
            rung=self.rung,
        )

        budget = profile.budget()
        dom = profile.dominant
        return ctx.emit(
            profile,
            uncertainty=Uncertainty(
                ci_low=(dom.st - dom.st_conf) if dom else None,
                ci_high=(dom.st + dom.st_conf) if dom else None,
                ci_level=self.conf_level,
                n=int(y.size),
                method=(
                    f"SALib bootstrap, {self.num_resamples} resamples; u_c={budget.combined:.4g}"
                ),
            ),
            cost=None,
        )


def _rank_inversions(a: Sequence[str], b: Sequence[str]) -> int:
    """How many pairs the two rankings order differently. Zero means they agree completely."""
    pos_a = {name: k for k, name in enumerate(a)}
    pos_b = {name: k for k, name in enumerate(b)}
    names = list(pos_a)
    return sum(
        1
        for i, x in enumerate(names)
        for y in names[i + 1 :]
        if (pos_a[x] < pos_a[y]) != (pos_b[x] < pos_b[y])
    )


def sobol_sensitivity(
    scorer: RubricScorer,
    inputs: Sequence[RubricInput],
    *,
    ctx: Context | None = None,
    **kwargs: Any,
) -> Reading:
    """Run D4 and return the Reading. The one-call form, for a card renderer."""
    return SobolSensitivity(scorer, inputs, **kwargs).estimate(ctx)


__all__ = [
    "D4_ENVELOPE",
    "ContractSensitivity",
    "RubricInput",
    "RubricScorer",
    "SensitivityIndex",
    "SensitivityProfile",
    "SobolSensitivity",
    "dose_response_slopes",
    "one_at_a_time",
    "sobol_indices",
    "sobol_problem",
    "sobol_sample",
    "sobol_sensitivity",
    "total_effect",
]

"""A5, the decision study: where the next dollar goes.

Everybody running a reinforcement-learning loop picks a group size, picks a number of prompts, and
grades each rollout exactly once. Nobody derives those numbers. Generalizability theory has had the
machinery for this since 1972 and calls it a decision study: you have the variance components, you
have a cost per unit of each thing you could buy, and you minimise the error variance of the
decision you are actually going to make, subject to the budget.

**The trade this instrument prices.** A rollout costs a forward pass and a grader call. A grader
replication costs a grader call and no forward pass. Buying more rollouts shrinks the sampling error
over objects and does nothing at all to the grader's disagreement with itself. Buying more grader
draws per rollout shrinks that disagreement and buys no new objects. At a fixed budget there is an
interior optimum, and where it sits depends entirely on the variance components, which is why this
cannot be guessed and has to be measured first.

**Three objectives, because they are genuinely different decisions and the optimum moves.**

    batch_relative    (sigma2(p) + sigma2(delta)) / (n * K)
    batch_absolute    the same, plus sigma2(Delta) - sigma2(delta), which does not shrink with n
    advantage         sigma2(delta) * (1 - 1/K) / (n * K), the measurement error in the batch's
                      mean group-centred advantage

The middle one is the one people quietly assume they are optimising and almost never are. Its extra
term is the facet main effects: which grader you drew, which day you ran it. That term is shared by
every object in the batch, so it does not average away no matter how many rollouts you buy, and it
is the floor under any comparison across two runs that used different grader draws. Within one run
with one grader draw it cancels exactly, which is the formal reason a group-relative estimator such
as GRPO is insensitive to grader bias and remains sensitive to grader-by-item interaction.

The third is the only one in which the group size matters on its own, through the `(1 - 1/K)` factor
that mean-centring introduces. It is monotone increasing in K, so at a fixed budget it will sit on
whatever lower bound the caller sets for K. That is a corner solution and the plan says so by name:
a corner means the objective is missing whatever else makes a larger group worth having, and the
bound is doing the work rather than the arithmetic.

**What the optimiser does.** The continuous relaxation is minimised with `scipy.optimize.minimize`
under the budget constraint, the result is rounded, and every one of the integer neighbours is
evaluated and the best feasible one kept, because the rounded point is often not the integer
optimum. On a small search space the whole integer grid is enumerated as a cross-check and the
reading says whether that check ran.

Kill condition: if the optimum is always the current practice.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, replace
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

from reward_lens.core.envelope import EnvelopeSpec
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.quantity import (
    BaselineID,
    BiasStatement,
    CostModel,
    EstimatorEntry,
    Quantity,
    Unit,
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
from reward_lens.measure.metrology.gstudy import MetrologyInstrument
from reward_lens.stats.gtheory import GStudy

Objective = Literal["batch_relative", "batch_absolute", "advantage"]

#: What each objective is the error variance *of*, in a sentence, so a plan says which decision it
#: was optimised for rather than leaving the reader to guess.
OBJECTIVE_MEANING: Mapping[str, str] = {
    "batch_relative": (
        "the batch's estimate of a contrast measured with the same grader draw on both arms: "
        "comparing two checkpoints scored by one grader"
    ),
    "batch_absolute": (
        "the batch's estimate of an absolute level, or a comparison whose two arms used different "
        "grader draws. Carries the facet main effects, which do not shrink with more rollouts"
    ),
    "advantage": (
        "the batch's mean group-centred advantage, which is what a group-relative policy-gradient "
        "estimator actually consumes. The only objective in which the group size matters on its "
        "own, and it is monotone in the group size, so its optimum sits on a bound"
    ),
}

#: `spec/QUANTITIES.yaml` registers `grader.allocation` with `definition: OPEN` and a unit token of
#: `plan` whose dimension, per and scale are all OPEN, with the stated reason that the token does
#: not decompose. The unit is left exactly as registered; only the definition is written out.
ALLOCATION = Quantity(
    id="grader.allocation",
    definition=(
        "The four-tuple (n prompts, K rollouts per prompt, s grader draws per rollout, m repeat "
        "calls per draw) that minimises a stated error variance subject to a stated cost "
        "constraint, together with the error variance it achieves and the error variance the "
        "current allocation achieves. The error variance is one of three: the batch relative error "
        "(sigma2(p) + sigma2(delta; s, m)) / (n*K), the batch absolute error which adds "
        "sigma2(Delta) - sigma2(delta), or the per-advantage error sigma2(delta; s, m) * (1 - 1/K). "
        "sigma2(delta) and sigma2(Delta) are the D-study relative and absolute error variances of "
        "the crossed design that produced the components. The plan is a decision and not a "
        "measurement: it is exact arithmetic on estimated components and inherits their "
        "uncertainty."
    ),
    unit=Unit(dimension="OPEN", per="OPEN", scale="OPEN", as_printed="plan"),
    invariance="units",
    interpretation=(
        "Read the ratio, not the tuple. A plan that moves 30% of the spend from rollouts to grader "
        "replications and cuts the error variance by 22% is a 22% cut for free; the same plan on a "
        "grader whose sigma2(delta) is already small will move nothing, and that is the answer too."
    ),
    support=None,
    wedge=False,
)

#: A5's access is derived: it consumes A2's output and touches no grader itself. Declared empty
#: rather than omitted, which is the same distinction the power instrument draws.
ALLOCATION_ACCESS: dict[Component, Access] = {}

#: Catalogue A5's baseline, verbatim: the current allocation. Rung 0's equal split is reported
#: beside it, because the catalogue's rung 0 estimator is the equal split and a rung is not a
#: baseline.
ALLOCATION_BASELINES: tuple[BaselineID, ...] = ("baseline.current_allocation",)

#: A5's envelope. The catalogue prints OPEN for `envelope_requires`, which is a genuine absence
#: rather than a blank, so this is declared unconditional with the justification `EnvelopeSpec`
#: demands. The honest reading is that a decision study is arithmetic on components that were
#: already measured under their own envelope: it adds no regime assumption of its own, and it
#: inherits every assumption the G-study made. That inheritance is stated on the plan.
ALLOCATION_ENVELOPE = EnvelopeSpec(
    unconditional=True,
    justification=(
        "a decision study is a cost minimisation over sample sizes, performed before anything runs "
        "and using only components another instrument already measured. No regime of a run can "
        "make the arithmetic wrong. What can be wrong is the components, and they carry their own "
        "envelope, which travels with the plan rather than being re-declared here."
    ),
)

BIAS: Mapping[int, BiasStatement] = {
    0: BiasStatement(
        direction="unknown",
        why=(
            "an equal split of the budget is a convention rather than an estimate. It is above the "
            "optimum whenever one facet dominates the error and below it whenever none does, and "
            "which of those holds is exactly what the components decide"
        ),
    ),
    1: BiasStatement(
        direction="approximately_unbiased",
        why=(
            "the minimisation is exact arithmetic on the components, so the only error is theirs. "
            "The plan is a smooth function of the components near the optimum, which is the useful "
            "half of that: a 10% error in sigma2(pr) moves the achieved variance by much less than "
            "10% because the objective is flat there"
        ),
    ),
}


# ---------------------------------------------------------------------------
# The plan and its costs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AllocationCosts:
    """What each unit of the design costs, in whatever currency the caller is counting.

    Dollars, GPU-seconds and calls all work, as long as one currency is used throughout. The
    budget is in that same currency.

    ``rater_setup`` is the fixed cost of adding a grader to the panel: loading a second reward model
    into memory, or a second judge's prompt-engineering. It is zero when the second draw is another
    sample from the same model, which is the common case, and it is what makes a panel of eight
    graders expensive when it is not.
    """

    rollout: float
    grader_call: float
    rater_setup: float = 0.0
    budget: float = 0.0

    def __post_init__(self) -> None:
        for name in ("rollout", "grader_call", "rater_setup", "budget"):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.rollout <= 0 and self.grader_call <= 0:
            raise ValueError(
                "a rollout and a grader call cannot both be free: with no cost there is no "
                "constraint and the optimum is unbounded."
            )

    def of(self, plan: "Allocation") -> float:
        """The cost of one allocation."""
        return (
            plan.n * plan.k * (self.rollout + plan.s * plan.m * self.grader_call)
            + plan.s * self.rater_setup
        )

    def feasible(self, plan: "Allocation") -> bool:
        return self.budget <= 0.0 or self.of(plan) <= self.budget + 1e-9


@dataclass(frozen=True)
class Allocation:
    """One allocation: prompts, rollouts per prompt, grader draws per rollout, repeats per draw."""

    n: int
    k: int
    s: int = 1
    m: int = 1

    def __post_init__(self) -> None:
        for name in ("n", "k", "s", "m"):
            v = int(getattr(self, name))
            object.__setattr__(self, name, v)
            if v < 1:
                raise ValueError(f"{name} must be at least 1; got {v}")

    @property
    def rollouts(self) -> int:
        return self.n * self.k

    @property
    def grader_calls(self) -> int:
        return self.n * self.k * self.s * self.m

    def render(self) -> str:
        return f"n={self.n}, K={self.k}, s={self.s}, m={self.m}"

    def as_dict(self) -> dict[str, int]:
        return {"n": self.n, "K": self.k, "s": self.s, "m": self.m}


# ---------------------------------------------------------------------------
# The objective
# ---------------------------------------------------------------------------


def _facet_sizes(g: GStudy, s: float, m: float) -> dict[str, float]:
    """Map the allocation's `s` and `m` onto whichever facets this design actually has."""
    out: dict[str, float] = {}
    if "r" in g.facets:
        out["r"] = max(1.0, float(s))
    if "o" in g.facets:
        out["o"] = max(1.0, float(m))
    return out


def error_variance(
    g: GStudy,
    n: float,
    k: float,
    s: float,
    m: float,
    *,
    objective: Objective = "batch_relative",
) -> float:
    """`sigma2(delta; n, K, s, m)` for one of the three declared objectives.

    Accepts non-integer sizes so the continuous relaxation can be minimised. The G-study's own
    `relative_error` and `absolute_error` are linear in the reciprocals of the facet sizes, so the
    relaxation is exact rather than an approximation of a discrete function.
    """
    sizes = _facet_sizes(g, s, m)
    delta = g.relative_error(**sizes)
    objects = max(1.0, float(n) * float(k))
    if objective == "advantage":
        return delta * (1.0 - 1.0 / max(1.0, float(k))) / objects
    base = (g.components.value("p") + delta) / objects
    if objective == "batch_relative":
        return base
    if objective == "batch_absolute":
        return base + (g.absolute_error(**sizes) - delta)
    raise ValueError(f"unknown objective {objective!r}; known are {sorted(OBJECTIVE_MEANING)}")


@dataclass(frozen=True)
class AllocationPlan:
    """A5's reading: the optimum, what it costs, and what it beats."""

    optimum: Allocation
    current: Allocation
    equal_split: Allocation
    objective: Objective
    variance_optimum: float
    variance_current: float
    variance_equal_split: float
    cost_optimum: float
    cost_current: float
    budget: float
    #: Whether the whole integer grid was enumerated as a cross-check on the continuous solve.
    verified_by_grid: bool
    n_neighbours_checked: int
    #: What the components this plan rests on came from, carried so the plan is not read alone.
    source_design: str = ""
    notes: tuple[str, ...] = ()

    @property
    def improvement(self) -> float:
        """Fractional reduction in error variance against the current allocation. 0.0 when none."""
        if self.variance_current <= 0:
            return 0.0
        return max(0.0, 1.0 - self.variance_optimum / self.variance_current)

    @property
    def unchanged(self) -> bool:
        """Whether the optimum is the current practice, which is this instrument's kill condition."""
        return self.optimum == self.current

    @property
    def grader_spend_shift(self) -> float:
        """Change in the fraction of the budget going to grader calls, optimum minus current."""

        def share(plan: Allocation, cost: float) -> float:
            if cost <= 0:
                return 0.0
            return (plan.grader_calls * self._grader_unit) / cost

        return share(self.optimum, self.cost_optimum) - share(self.current, self.cost_current)

    #: Set by `optimise`, so `grader_spend_shift` can be a property rather than another field the
    #: caller has to thread through.
    _grader_unit: float = 0.0

    def says(self) -> str:
        if self.unchanged:
            return (
                f"At your budget the optimum is your current allocation ({self.current.render()}). "
                f"Nothing to move."
            )
        shift = self.grader_spend_shift
        direction = "to grader replications" if shift > 0 else "to rollouts"
        return (
            f"At your budget, move {abs(shift):.0%} of the spend {direction}: "
            f"{self.current.render()} becomes {self.optimum.render()}. Expected error variance "
            f"falls {self.improvement:.0%}."
        )

    def render(self) -> str:
        lines = [
            self.says(),
            f"    objective: {self.objective} ({OBJECTIVE_MEANING[self.objective]})",
            f"    current       {self.current.render():<28} var {self.variance_current:.6g}  "
            f"cost {self.cost_current:.6g}",
            f"    equal split   {self.equal_split.render():<28} var "
            f"{self.variance_equal_split:.6g}",
            f"    optimum       {self.optimum.render():<28} var {self.variance_optimum:.6g}  "
            f"cost {self.cost_optimum:.6g} of {self.budget:.6g}",
            f"    {self.n_neighbours_checked} integer neighbours checked"
            + ("; whole grid enumerated" if self.verified_by_grid else ""),
        ]
        lines.extend(f"    {n}" for n in self.notes)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# The optimiser
# ---------------------------------------------------------------------------


def _round_candidates(
    x: Sequence[float] | np.ndarray, bounds: Sequence[tuple[int, int]]
) -> list[tuple[int, ...]]:
    """Every integer neighbour of a continuous point, clipped into bounds.

    Three candidates per axis, floor, ceil and the nearest, deduplicated. Eighty-one points at most
    for four axes, which is nothing, and it matters: the rounded continuous optimum is frequently
    not the integer optimum when the objective is flat.
    """
    per_axis = []
    for value, (lo, hi) in zip(x, bounds):
        opts = {int(math.floor(value)), int(math.ceil(value)), int(round(value))}
        per_axis.append(sorted({min(hi, max(lo, o)) for o in opts}))
    return [tuple(c) for c in itertools.product(*per_axis)]


def equal_split_allocation(
    costs: AllocationCosts,
    *,
    k: int,
    bounds: Mapping[str, tuple[int, int]],
) -> Allocation:
    """Rung 0: split the budget equally between rollout spend and grader spend.

    The naive plan, and it is not a straw man: it is what a team does when it decides to "grade
    everything twice" without looking at the components. Grader draws are set so that the grader
    half of the budget is spent, with the rest going to prompts.
    """
    s_lo, s_hi = bounds.get("s", (1, 8))
    m_lo, m_hi = bounds.get("m", (1, 8))
    half = costs.budget / 2.0 if costs.budget > 0 else 0.0
    # Grader spend equals rollout spend when s*m*c_grader == c_rollout.
    if costs.grader_call > 0:
        target = costs.rollout / costs.grader_call
    else:
        target = float(s_hi * m_hi)
    s = int(min(s_hi, max(s_lo, round(math.sqrt(max(1.0, target))))))
    m = int(min(m_hi, max(m_lo, round(max(1.0, target) / max(1, s)))))
    per_prompt = k * (costs.rollout + s * m * costs.grader_call)
    n_lo, n_hi = bounds.get("n", (1, 1_000_000))
    if half > 0 and per_prompt > 0:
        n = int(
            min(n_hi, max(n_lo, math.floor((costs.budget - s * costs.rater_setup) / per_prompt)))
        )
    else:
        n = n_lo
    return Allocation(n=max(1, n), k=k, s=s, m=m)


def optimise(
    g: GStudy,
    costs: AllocationCosts,
    current: Allocation,
    *,
    objective: Objective = "batch_relative",
    bounds: Mapping[str, tuple[int, int]] | None = None,
    grid_limit: int = 20_000,
) -> AllocationPlan:
    """Minimise the stated error variance subject to the budget, over integer allocations.

    The budget defaults to the cost of the current allocation, which is the question a user
    normally means: "at what I am already spending, what should I be spending it on?"
    """
    if costs.budget <= 0:
        costs = replace(costs, budget=costs.of(current))

    b = {
        "n": (1, 1_000_000),
        "k": (2, 512),
        "s": (1, 32),
        "m": (1, 32),
        **{k: (int(v[0]), int(v[1])) for k, v in (bounds or {}).items()},
    }
    if "r" not in g.facets:
        b["s"] = (1, 1)
    if "o" not in g.facets:
        b["m"] = (1, 1)
    if objective != "advantage" and (bounds or {}).get("k") is None:
        # These two objectives depend on n and K only through their product, so the surface is
        # exactly flat along n*K = constant and any K on it is "optimal". Returning whichever point
        # a solver happened to stop at would read as a recommendation to change the group size,
        # which the arithmetic does not support. K is pinned and n absorbs the budget.
        b["k"] = (current.k, current.k)

    order = ("n", "k", "s", "m")
    lo = np.array([b[k][0] for k in order], dtype=np.float64)
    hi = np.array([b[k][1] for k in order], dtype=np.float64)

    def unpack(v: np.ndarray) -> tuple[float, float, float, float]:
        return float(v[0]), float(v[1]), float(v[2]), float(v[3])

    def f(v: np.ndarray) -> float:
        n, k, s, m = unpack(v)
        return error_variance(g, n, k, s, m, objective=objective)

    def budget_slack(v: np.ndarray) -> float:
        n, k, s, m = unpack(v)
        spend = n * k * (costs.rollout + s * m * costs.grader_call) + s * costs.rater_setup
        return costs.budget - spend

    x0 = np.clip(np.array([current.n, current.k, current.s, current.m], dtype=np.float64), lo, hi)
    res = minimize(
        f,
        x0,
        method="SLSQP",
        bounds=list(zip(lo, hi)),
        constraints=[{"type": "ineq", "fun": budget_slack}],
        options={"maxiter": 400, "ftol": 1e-14},
    )
    seed_points = [res.x if res.x is not None else x0, x0]

    best: Allocation | None = None
    best_v = math.inf
    checked = 0
    for point in seed_points:
        for cand in _round_candidates(point, [b[k] for k in order]):
            checked += 1
            plan = Allocation(n=cand[0], k=cand[1], s=cand[2], m=cand[3])
            if not costs.feasible(plan):
                continue
            v = error_variance(g, plan.n, plan.k, plan.s, plan.m, objective=objective)
            if v < best_v:
                best_v, best = v, plan

    # The continuous solve fixes n and K only through their product for two of the three
    # objectives, so the rounded neighbourhood can miss a better feasible n at the same K. Sweep the
    # budget-saturating n at every (s, m) and at a short list of K, which is what turns a local
    # answer into one that can be quoted.
    k_candidates = sorted(
        {
            v
            for v in (
                b["k"][0],
                b["k"][1],
                current.k,
                (best.k if best else current.k),
                2,
                4,
                8,
                16,
                32,
                64,
            )
            if b["k"][0] <= v <= b["k"][1]
        }
    )
    if objective != "advantage":
        # K is not in these objectives except through n*K, so sweeping it changes nothing and the
        # plan says so instead of manufacturing a difference from rounding.
        k_candidates = [current.k if best is None else best.k]
    grid_size = (
        (b["s"][1] - b["s"][0] + 1) * (b["m"][1] - b["m"][0] + 1) * max(1, len(k_candidates))
    )
    verified = grid_size <= grid_limit
    if verified:
        for k_star in k_candidates:
            for s in range(b["s"][0], b["s"][1] + 1):
                for m in range(b["m"][0], b["m"][1] + 1):
                    per_prompt = k_star * (costs.rollout + s * m * costs.grader_call)
                    if per_prompt <= 0:
                        continue
                    n = int(math.floor((costs.budget - s * costs.rater_setup) / per_prompt))
                    n = min(b["n"][1], max(b["n"][0], n))
                    plan = Allocation(n=n, k=k_star, s=s, m=m)
                    checked += 1
                    if not costs.feasible(plan):
                        continue
                    v = error_variance(g, plan.n, plan.k, plan.s, plan.m, objective=objective)
                    if v < best_v:
                        best_v, best = v, plan

    if best is None:
        best, best_v = (
            current,
            error_variance(g, current.n, current.k, current.s, current.m, objective=objective),
        )

    eq = equal_split_allocation(costs, k=current.k, bounds=b)
    notes = []
    if not res.success:
        notes.append(
            f"the continuous solve reported {res.message!r}; the integer answer comes from the "
            f"neighbourhood and grid sweep, which do not depend on it converging."
        )
    if objective in ("batch_relative", "batch_absolute"):
        notes.append(
            "n and K enter this objective only through their product, so the split between "
            "prompts and rollouts per prompt is not determined here. K is held at its current "
            "value; use the `advantage` objective to price K on its own."
        )
    if not costs.feasible(current):
        notes.append(
            f"the current allocation costs {costs.of(current):.6g}, above the stated budget of "
            f"{costs.budget:.6g}. The comparison is still reported and the optimum respects the "
            f"budget."
        )
    corners = [
        label
        for label, key, value in (("K", "k", best.k), ("s", "s", best.s), ("m", "m", best.m))
        if b[key][0] < b[key][1] and value in b[key]
    ]
    if corners:
        notes.append(
            f"the optimum sits on a bound in {', '.join(corners)}. The objective is monotone in "
            f"that direction over the range allowed, so the bound is deciding the answer rather "
            f"than the components. Widen the bound, or read it as a statement that this objective "
            f"does not contain whatever else makes the other end worth having."
        )

    return AllocationPlan(
        optimum=best,
        current=current,
        equal_split=eq,
        objective=objective,
        variance_optimum=best_v,
        variance_current=error_variance(
            g, current.n, current.k, current.s, current.m, objective=objective
        ),
        variance_equal_split=error_variance(g, eq.n, eq.k, eq.s, eq.m, objective=objective),
        cost_optimum=costs.of(best),
        cost_current=costs.of(current),
        budget=costs.budget,
        verified_by_grid=verified,
        n_neighbours_checked=checked,
        source_design=g.design,
        notes=tuple(notes),
        _grader_unit=costs.grader_call,
    )


# ---------------------------------------------------------------------------
# The instrument
# ---------------------------------------------------------------------------


class OptimalAllocation(MetrologyInstrument):
    """A5. Where the next dollar goes, given A2's variance components and a cost model.

    Kill condition: if the optimum is always the current practice.
    """

    name = "OptimalAllocation"
    version = "1.0"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "A5"
    deviations = (
        "the catalogue writes the objective as `sigma2(delta; n, K, m, s)` and does not say what "
        "it is the error variance of. G-theory's sigma2(delta) is per object and does not depend "
        "on how many objects you sample, so minimising it alone would spend the whole budget on "
        "replication. Three objectives are offered instead, each named for the decision it is the "
        "error of, and the default is the batch relative error",
        "for the two batch objectives, n and K enter only through their product, so the split "
        "between prompts and rollouts per prompt is not determined by the arithmetic. K is held "
        "fixed and the `advantage` objective is the one that prices K",
        "the cost model is linear in calls and carries one fixed per-rater term. Real costs are "
        "not linear: batching, KV-cache reuse and rate limits all bend the curve, and a caller "
        "whose costs bend should pass a measured cost per unit at their operating point rather "
        "than a list price",
    )

    quantity = "grader.allocation"
    requires = ALLOCATION_ACCESS
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
    phases = frozenset({Phase.PRE_RUN})
    envelope = ALLOCATION_ENVELOPE
    invariance = "units"
    invariance_relation = INVARIANT
    baselines = ALLOCATION_BASELINES
    rung = 1

    def __init__(
        self,
        gstudy: GStudy | None = None,
        costs: AllocationCosts | None = None,
        current: Allocation | None = None,
        *,
        objective: Objective = "batch_relative",
        bounds: Mapping[str, tuple[int, int]] | None = None,
    ) -> None:
        self.gstudy = gstudy
        self.costs = costs
        self.current = current
        self.objective = objective
        self.bounds = dict(bounds or {})

    def compute(self) -> Any:
        if self.gstudy is None:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail=(
                    "no variance components were supplied, so there is nothing to allocate "
                    "against. A decision study is arithmetic on a G-study and cannot be run before "
                    "one"
                ),
                remedy=(
                    "run A2 first: `VarianceComponents(design=...)`, then pass its `gstudy` here. "
                    "Two graders on fifty shared items is enough to produce the components, and it "
                    "needs no training run."
                ),
            )
        if self.costs is None or self.current is None:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail=(
                    "a decision study needs a cost per unit and the allocation you are currently "
                    "running; without them there is no constraint and no baseline to beat"
                ),
                remedy=(
                    "pass `costs=AllocationCosts(rollout=..., grader_call=..., budget=...)` and "
                    "`current=Allocation(n=..., k=..., s=..., m=...)`. Any consistent currency "
                    "works: dollars, GPU-seconds, or calls. Leave the budget at zero to price the "
                    "optimum at what you are already spending."
                ),
            )
        if self.objective not in OBJECTIVE_MEANING:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.UNIT_MISMATCH,
                detail=(
                    f"objective {self.objective!r} is not one of "
                    f"{sorted(OBJECTIVE_MEANING)}, and the three are error variances of different "
                    f"quantities rather than three ways of writing one"
                ),
                remedy=(
                    "pick one of batch_relative, batch_absolute or advantage. They are not "
                    "interchangeable: the optimum moves between them, which is the point of naming "
                    "them separately."
                ),
            )
        if self.gstudy.components.value("p") <= 0.0:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.BELOW_LOD,
                detail=(
                    "the object-of-measurement variance component is zero, so there is no signal "
                    "to allocate budget toward measuring. Every plan has the same ratio of signal "
                    "to error, which is zero"
                ),
                remedy=(
                    "check whether the objects in the G-study genuinely differ. A component of "
                    "zero here usually means the design scored near-identical items, or that the "
                    "component estimate was negative and truncated, which the component set flags."
                ),
                statistics={
                    "sigma2_p": self.gstudy.components.value("p"),
                    "truncated": list(self.gstudy.components.truncated_names),
                },
            )
        return optimise(
            self.gstudy,
            self.costs,
            self.current,
            objective=self.objective,
            bounds=self.bounds,
        )

    def payload(self, computed: AllocationPlan) -> dict[str, Any]:
        return {
            "optimum": computed.optimum.as_dict(),
            "current": computed.current.as_dict(),
            "equal_split": computed.equal_split.as_dict(),
            "objective": computed.objective,
            "objective_meaning": OBJECTIVE_MEANING[computed.objective],
            "variance_optimum": computed.variance_optimum,
            "variance_current": computed.variance_current,
            "variance_equal_split": computed.variance_equal_split,
            "improvement": computed.improvement,
            "grader_spend_shift": computed.grader_spend_shift,
            "cost_optimum": computed.cost_optimum,
            "cost_current": computed.cost_current,
            "budget": computed.budget,
            "unchanged": computed.unchanged,
            "verified_by_grid": computed.verified_by_grid,
            "n_neighbours_checked": computed.n_neighbours_checked,
            "source_design": computed.source_design,
            "baselines": {"baseline.current_allocation": computed.variance_current},
            "says": computed.says(),
        }


def register_ladder() -> list[str]:
    """Register A5's two rungs as `EstimatorEntry` rows. Not called at import, by design."""
    entries = [
        EstimatorEntry(
            quantity=ALLOCATION.id,
            impl="a5.equal_split",
            requires={},
            envelope=ALLOCATION_ENVELOPE,
            rung=0,
            bias=BIAS[0],
            cost=CostModel(note="free: arithmetic on the cost model alone"),
        ),
        EstimatorEntry(
            quantity=ALLOCATION.id,
            impl="a5.constrained_minimisation",
            requires={},
            envelope=ALLOCATION_ENVELOPE,
            rung=1,
            bias=BIAS[1],
            cost=CostModel(note="free: arithmetic on A2's components"),
        ),
    ]
    for e in entries:
        register_estimator(e)
    return [e.impl for e in entries]


__all__ = [
    "ALLOCATION",
    "ALLOCATION_ACCESS",
    "ALLOCATION_BASELINES",
    "ALLOCATION_ENVELOPE",
    "BIAS",
    "OBJECTIVE_MEANING",
    "Allocation",
    "AllocationCosts",
    "AllocationPlan",
    "Objective",
    "OptimalAllocation",
    "equal_split_allocation",
    "error_variance",
    "optimise",
    "register_ladder",
]

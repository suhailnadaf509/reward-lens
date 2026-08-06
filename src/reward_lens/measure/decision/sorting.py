"""N7, the sorting cutoff: which components belong in the same weighted sum, and which do not.

Holmstrom and Milgrom's task-assignment result says that when effort is substitutable, tasks sort by
an information coefficient and there is a cutoff: tasks above it should not share an incentive
contract with tasks below it. In reinforcement learning terms, **do not put a noisy judge and a crisp
unit test in the same weighted sum.** Split them across separate optimisation stages, separate models
or separate gradient paths. The optimum is provably asymmetric, so treating components symmetrically
is suboptimal whenever their noise differs at all.

**The two quantities.** A component's effort-equivalent noise is `n_i = sigma_i^2 / mu'_i^2`: the
score variance divided by the squared dose-response slope, which converts a variance in whatever
units the grader emits into a variance in units of effort and is what makes components on different
scales comparable at all. Its information coefficient is `rho_i = n_i / t_i`, the same quantity per
unit of effort actually spent on it.

**The mechanism, derived rather than asserted.** Components in one contract are substitutes for the
agent's capacity, so equal compensation forces a single shared commission `kappa` across them:
`alpha_i = kappa / mu'_i`. The risk premium is then `(r/2) kappa^2 sum_i n_i`, the agent's total
effort solves `c'(T) = kappa` with `c(T) = C T^2 / 2`, and the principal's surplus is

    S(kappa) = Bbar kappa / C - kappa^2 / (2C) - (r/2) kappa^2 N,   N = sum_{i in g} n_i

maximised at `kappa* = Bbar / (1 + r C N)` for a value of

    V(g) = Bbar_g^2 / [2 C (1 + r C N_g)]

That single shared intensity is the whole story. **One component's noise is a tax on every other
component in the same contract**, because they all ride on the same `kappa` and `N` adds up while the
value does not. Adding a component with noise `n` to a contract multiplies every other member's
incentive intensity by `(1 + r C N) / (1 + r C (N + n))`, and that factor is what the instrument
reports as the cost of keeping a noisy component in the sum.

`V` is not an ad hoc value function. It is N5's own surplus `g' H^-1 g / 2` specialised to a single
fungible group, and the one-component case is checked against N5 directly in the tests.

**What is held fixed, said plainly, because getting this wrong makes splitting look free.** A
contract is a capacity pool, so comparing one contract against two would compare one agent against
two unless the capacity is held fixed. It is, by the correction `C -> k C`: splitting one pool into
`k` gives each stage a `k`-times steeper cost curve, which is exactly the condition that spending a
given total effort costs the same whether it is spent in one stage or spread over `k`. With that
correction the comparison across `k` is honest, and it produces the source's own claim as a
theorem rather than as a slogan: **splitting is exactly neutral when the components are identical and
strictly pays as soon as their noise differs at all.** For one noiseless component and one with
`x = r C n`, the two-stage value beats the one-stage value by a factor of `(1 + x)^2 / (1 + 2x)`,
which is 1 at `x = 0` and above 1 everywhere else.

`Bbar_g` is the value per unit of capacity a contract delivers, taken as `max_{i in g} B_i`, because
equal compensation leaves the agent indifferent about the split within a contract and the principal
directs it.

**The sorting theorem is checked, not assumed.** The optimum is found by enumerating every partition
into the stated number of groups, and the best interval partition in `rho` order is computed beside
it. When the two coincide the theorem's claim holds on this instance; when they do not, the reading
says so and reports both. Above a component count where enumeration stops being free, only the
interval scan runs and the reading records that the check did not happen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from reward_lens.core.invariance import INVARIANT
from reward_lens.core.quantity import BaselineID
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.types import Capability, GaugeStatus
from reward_lens.measure.decision._base import NOISE_ACCESS, DecisionInstrument
from reward_lens.measure.decision.assumptions import render_assumptions
from reward_lens.measure.decision.parameters import ContractParameters

N7_BASELINES: tuple[BaselineID, ...] = (
    "baseline.single_contract",
    "baseline.one_contract_per_component",
)

#: How many partitions the exhaustive search will enumerate before giving up and reporting the
#: interval scan alone. Not a catalogue threshold: the catalogue carries no N7 record. At 200,000 a
#: full enumeration takes under a second, which keeps the sorting theorem's own check free for every
#: composite anybody actually builds.
DEFAULT_ENUMERATION_LIMIT = 200_000


# ---------------------------------------------------------------------------
# The two quantities
# ---------------------------------------------------------------------------


def noise_to_signal(
    noise_variance: Sequence[float] | np.ndarray,
    sensitivity: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """`n_i = sigma_i^2 / mu'_i^2`, the effort-equivalent noise.

    Dividing by the squared sensitivity is what makes two components comparable. Two graders
    reporting on scales two orders of magnitude apart have variances four orders apart, and this
    removes exactly that, which is why a raw variance is the wrong thing to sort on.
    """
    s2 = np.asarray(noise_variance, dtype=np.float64).ravel()
    mu = np.asarray(sensitivity, dtype=np.float64).ravel()
    if s2.size != mu.size:
        raise ValueError(f"{s2.size} variance(s) against {mu.size} sensitivity value(s)")
    if np.any(mu == 0.0):
        bad = [int(i) for i in np.flatnonzero(mu == 0.0)]
        raise ValueError(
            f"component(s) {bad} have a dose-response slope of exactly zero, so their "
            f"effort-equivalent noise is infinite rather than large. A component the policy cannot "
            f"move is a finding rather than a row in this table."
        )
    return s2 / (mu * mu)


def information_coefficient(
    noise_to_signal_values: Sequence[float] | np.ndarray,
    effort: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """`rho_i = n_i / t_i`. A common scaling of the effort cancels from every ratio between them."""
    n = np.asarray(noise_to_signal_values, dtype=np.float64).ravel()
    t = np.asarray(effort, dtype=np.float64).ravel()
    if n.size != t.size:
        raise ValueError(f"{n.size} noise value(s) against {t.size} effort value(s)")
    if np.any(t <= 0.0):
        bad = [int(i) for i in np.flatnonzero(t <= 0.0)]
        raise ValueError(
            f"component(s) {bad} carry non-positive effort. rho = n/t is not defined there, and a "
            f"component nobody works on has no information coefficient rather than an infinite one."
        )
    return n / t


def contract_value(
    benefit: float, noise_total: float, risk_aversion: float, cost_curvature: float
) -> float:
    """`V = Bbar^2 / [2 C (1 + r C N)]`, one contract's surplus at its own optimal intensity."""
    denom = 2.0 * cost_curvature * (1.0 + risk_aversion * cost_curvature * noise_total)
    if denom <= 0.0:
        raise ValueError(
            f"the contract-value denominator is {denom:.6g}, which is not positive. That needs a "
            f"non-convex cost or a negative risk aversion, and neither is inside this model."
        )
    return (benefit * benefit) / denom


# ---------------------------------------------------------------------------
# Partitions
# ---------------------------------------------------------------------------


def _partitions_into(m: int, k: int, limit: int) -> Iterator[tuple[int, ...]] | None:
    """Every assignment of `m` items into exactly `k` non-empty labelled-by-first-use groups.

    Returns None when the count would exceed ``limit``, so a caller can fall back rather than
    discover the cost by waiting. The count is the Stirling number of the second kind and it is
    computed before anything is enumerated.
    """
    if k < 1 or k > m:
        return iter(())
    stirling = [[0] * (k + 1) for _ in range(m + 1)]
    stirling[0][0] = 1
    for i in range(1, m + 1):
        for j in range(1, min(i, k) + 1):
            stirling[i][j] = j * stirling[i - 1][j] + stirling[i - 1][j - 1]
            if stirling[i][j] > limit:
                return None

    def walk(i: int, used: int, prefix: list[int]) -> Iterator[tuple[int, ...]]:
        if i == m:
            if used == k:
                yield tuple(prefix)
            return
        remaining = m - i
        if used + remaining < k:
            return
        for g in range(used):
            prefix.append(g)
            yield from walk(i + 1, used, prefix)
            prefix.pop()
        if used < k:
            prefix.append(used)
            yield from walk(i + 1, used + 1, prefix)
            prefix.pop()

    return walk(0, 0, [])


def _assignment_value(
    assignment: Sequence[int],
    benefit: np.ndarray,
    noise: np.ndarray,
    risk_aversion: float,
    cost_curvature: float,
) -> float:
    total = 0.0
    for g in range(max(assignment) + 1):
        members = [i for i, a in enumerate(assignment) if a == g]
        if not members:
            continue
        total += contract_value(
            float(np.max(benefit[members])),
            float(np.sum(noise[members])),
            risk_aversion,
            cost_curvature,
        )
    return total


def _best_interval_partition(
    order: Sequence[int],
    benefit: np.ndarray,
    noise: np.ndarray,
    risk_aversion: float,
    cost_curvature: float,
    k: int,
) -> tuple[float, tuple[int, ...]]:
    """The best partition of the `rho`-sorted components into `k` contiguous blocks, exactly.

    A dynamic program over cut positions rather than an enumeration, so it stays exact at any
    component count and is what runs when the full enumeration is declined.
    """
    m = len(order)
    neg = -math.inf
    best = [[neg] * (k + 1) for _ in range(m + 1)]
    cut = [[0] * (k + 1) for _ in range(m + 1)]
    best[0][0] = 0.0
    for i in range(1, m + 1):
        for j in range(1, min(i, k) + 1):
            for split in range(j - 1, i):
                if best[split][j - 1] == neg:
                    continue
                members = [order[x] for x in range(split, i)]
                value = best[split][j - 1] + contract_value(
                    float(np.max(benefit[members])),
                    float(np.sum(noise[members])),
                    risk_aversion,
                    cost_curvature,
                )
                if value > best[i][j]:
                    best[i][j] = value
                    cut[i][j] = split
    assignment = [0] * m
    i, j = m, k
    while j > 0:
        split = cut[i][j]
        for x in range(split, i):
            assignment[order[x]] = j - 1
        i, j = split, j - 1
    return best[m][k], tuple(assignment)


def _canonical(assignment: Sequence[int]) -> tuple[frozenset[int], ...]:
    """A partition as an order-free object, so two labellings of one partition compare equal."""
    groups: dict[int, set[int]] = {}
    for i, g in enumerate(assignment):
        groups.setdefault(g, set()).add(i)
    return tuple(sorted((frozenset(s) for s in groups.values()), key=lambda s: min(s)))


# ---------------------------------------------------------------------------
# The reading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Contract:
    """One group of components that share an incentive contract, with what it is worth."""

    members: tuple[str, ...]
    noise_total: float
    benefit: float
    value: float
    rho_low: float
    rho_high: float

    def render(self) -> str:
        return (
            f"    {', '.join(self.members):<40} N = {self.noise_total:.4g}, "
            f"rho in [{self.rho_low:.4g}, {self.rho_high:.4g}], value {self.value:.6g}"
        )


@dataclass(frozen=True)
class SortingReading:
    """N7's reading: the assignment, the cutoff, and whether the sorting theorem held here."""

    components: tuple[str, ...]
    noise: np.ndarray
    rho: np.ndarray
    contracts: tuple[Contract, ...]
    value: float
    n_contracts: int
    #: The best assignment found by enumerating every partition, when that ran.
    exhaustive_ran: bool
    exhaustive_value: float
    interval_value: float
    #: Whether an interval partition in `rho` order attains the optimum over all partitions. This
    #: is the sorting theorem's own claim, checked on this instance rather than assumed. `None`
    #: means the exhaustive search was declined and the check did not run.
    interval_optimal: bool | None
    #: The `rho` interval the cutoff falls in, when the assignment is an interval partition.
    cutoff: float | None
    cutoff_bracket: tuple[float, float] | None
    #: What every other component's incentive intensity is multiplied by, per component, if it stays
    #: in the same contract as the rest. The tax a noisy component levies on the sum.
    dilution: np.ndarray
    single_contract_value: float
    singleton_value: float
    effort_assumed_equal: bool
    #: Whether the two searches returned the same partition rather than merely the same value.
    #: `False` alongside `interval_optimal` true means the optimum is attained by more than one
    #: partition, which is a tie rather than a failure of the theorem.
    searches_agree: bool | None = None
    #: The best value at each number of contracts from 1 to m, from the interval scan, all carrying
    #: the capacity correction so they are comparable with each other.
    value_by_contract_count: tuple[float, ...] = ()
    #: The number of contracts that maximises that profile. It is what the reading recommends, and
    #: it is not always the number the caller asked for.
    best_contract_count: int = 1
    baselines: Mapping[str, float] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    @property
    def worst_dilution(self) -> tuple[str, float]:
        idx = int(np.argmin(self.dilution))
        return self.components[idx], float(self.dilution[idx])

    @property
    def split_gain(self) -> float:
        """What the split is worth against keeping everything in one weighted sum.

        A fraction of the single-contract value, and comparable with it, because both carry the
        capacity correction that makes `k` stages cost the same total effort as one.
        """
        if self.single_contract_value <= 0.0:
            return math.nan
        return self.value / self.single_contract_value - 1.0

    def says(self) -> str:
        name, factor = self.worst_dilution
        if self.n_contracts == 1 or len(self.contracts) == 1:
            return (
                f"One contract. Keeping {name} in the sum multiplies every other component's "
                f"incentive intensity by {factor:.2f}, which is the tax its noise levies on the "
                f"rest."
            )
        cut = f"{self.cutoff:.4g}" if self.cutoff is not None else "no single value"
        groups = " | ".join(", ".join(c.members) for c in self.contracts)
        return (
            f"Split at rho = {cut}: {groups}. Keeping {name} in one sum with the rest multiplies "
            f"every other component's incentive intensity by {factor:.2f}, and the split is worth "
            f"{self.split_gain:+.1%} against the single weighted sum at the same total capacity. "
            f"{self.best_contract_count} contract(s) is the best number over the whole profile."
        )

    def render(self) -> str:
        lines = [self.says(), "", "  component            n          rho     dilution"]
        for i, name in enumerate(self.components):
            lines.append(
                f"  {name:<16} {self.noise[i]:>10.4g} {self.rho[i]:>12.4g} "
                f"{self.dilution[i]:>10.3f}"
            )
        lines.append(f"  contracts ({self.n_contracts}):")
        lines.extend(c.render() for c in self.contracts)
        if self.interval_optimal is None:
            lines.append(
                "  the exhaustive search over partitions was declined at this component count, so "
                "the assignment above is the best interval partition in rho order and whether a "
                "non-interval partition beats it is not known."
            )
        elif self.interval_optimal:
            lines.append(
                f"  the sorting theorem holds here: the best partition over all "
                f"{self.n_contracts}-way splits is an interval partition in rho order, worth "
                f"{self.exhaustive_value:.6g}, and the interval scan found the same one."
            )
        else:
            lines.append(
                f"  the sorting theorem does not hold on this instance. The best partition over "
                f"all splits is worth {self.exhaustive_value:.6g} and is not an interval in rho "
                f"order; the best interval partition is worth {self.interval_value:.6g}. That is a "
                f"finding about this composite rather than an error, and the reported assignment "
                f"is the better of the two."
            )
        lines.append(
            f"  one contract holding everything is worth {self.single_contract_value:.6g}, one "
            f"contract per component {self.singleton_value:.6g}, and the profile over 1 to "
            f"{len(self.value_by_contract_count)} contracts is "
            + ", ".join(f"{v:.4g}" for v in self.value_by_contract_count)
            + ". All of these carry the capacity correction C -> k C, so a given total effort "
            "costs the same in every one of them and they are comparable."
        )
        if self.effort_assumed_equal:
            lines.append(
                "  effort was assumed equal across components, so rho is proportional to n. Every "
                "ordering, every partition and the assignment above are exactly what they would be "
                "under any other common effort level; only the cutoff's numeric value moves."
            )
        lines.extend(f"  {n}" for n in self.notes)
        lines.append("")
        lines.append(render_assumptions())
        return "\n".join(lines)


def sorting_cutoff(
    params: ContractParameters,
    *,
    n_contracts: int = 2,
    cost_scale: float | None = None,
    enumeration_limit: int = DEFAULT_ENUMERATION_LIMIT,
) -> SortingReading:
    """The assignment, the cutoff and the sorting theorem's own check, from a settled parameter set.

    ``cost_scale`` is the scalar `C` of the fungible-pool cost `C T^2 / 2`. It defaults to the mean
    diagonal of `C''`, which is the same object read off the matrix the rest of the layer uses, so
    the two cannot silently disagree about how hard effort is.
    """
    if params.sensitivity is None or params.effort is None:
        raise ValueError(
            "sorting_cutoff needs the sensitivity matrix and the effort allocation. The instrument "
            "refuses before reaching here; this path is for a direct caller who skipped it."
        )
    m = params.m
    if n_contracts < 1 or n_contracts > m:
        raise ValueError(
            f"{n_contracts} contract(s) for {m} component(s). A partition has between one group "
            f"and one group per component."
        )
    sigma2 = np.diag(params.noise).astype(np.float64)
    mu = np.diag(params.sensitivity).astype(np.float64)
    n = noise_to_signal(sigma2, mu)
    rho = information_coefficient(n, params.effort)
    benefit = params.benefit
    cost = (
        float(np.mean(np.diag(params.cost_curvature))) if cost_scale is None else float(cost_scale)
    )
    r = params.risk_aversion
    # The capacity correction. Splitting one pool into k stages gives each a k-times steeper cost
    # curve, which is the condition that a given total effort costs the same however it is spread.
    # Without it every comparison across k is a comparison of one agent against k.
    cost_k = n_contracts * cost

    order = [int(i) for i in np.argsort(rho, kind="stable")]
    interval_value, interval_assignment = _best_interval_partition(
        order, benefit, n, r, cost_k, n_contracts
    )

    walker = _partitions_into(m, n_contracts, enumeration_limit)
    exhaustive_value = -math.inf
    exhaustive_assignment: tuple[int, ...] | None = None
    if walker is not None:
        for assignment in walker:
            value = _assignment_value(assignment, benefit, n, r, cost_k)
            if value > exhaustive_value:
                exhaustive_value = value
                exhaustive_assignment = assignment

    # The theorem's claim is that *an* interval partition attains the optimum, not that the search
    # happens to return the same labelling, so the comparison is on value rather than on identity.
    # Comparing partitions instead would report a failure every time two partitions tie.
    interval_optimal: bool | None
    if exhaustive_assignment is None:
        chosen, value = interval_assignment, interval_value
        interval_optimal = None
        exhaustive_value = math.nan
    elif interval_value >= exhaustive_value * (1.0 - 1e-12):
        chosen, value = interval_assignment, interval_value
        interval_optimal = True
    else:
        chosen, value = exhaustive_assignment, exhaustive_value
        interval_optimal = False
    searches_agree = (
        None
        if exhaustive_assignment is None
        else _canonical(exhaustive_assignment) == _canonical(interval_assignment)
    )

    contracts = []
    for g in sorted(set(chosen), key=lambda g: min(rho[i] for i, a in enumerate(chosen) if a == g)):
        members = [i for i, a in enumerate(chosen) if a == g]
        contracts.append(
            Contract(
                members=tuple(params.components[i] for i in members),
                noise_total=float(np.sum(n[members])),
                benefit=float(np.max(benefit[members])),
                value=contract_value(
                    float(np.max(benefit[members])), float(np.sum(n[members])), r, cost_k
                ),
                rho_low=float(np.min(rho[members])),
                rho_high=float(np.max(rho[members])),
            )
        )

    cutoff: float | None = None
    bracket: tuple[float, float] | None = None
    contiguous = all(
        contracts[i].rho_high <= contracts[i + 1].rho_low for i in range(len(contracts) - 1)
    )
    if contiguous and len(contracts) > 1:
        lo, hi = contracts[0].rho_high, contracts[1].rho_low
        bracket = (float(lo), float(hi))
        cutoff = float(math.sqrt(lo * hi)) if lo > 0 and hi > 0 else float(0.5 * (lo + hi))

    total_n = float(np.sum(n))
    dilution = np.array(
        [(1.0 + r * cost * (total_n - n[i])) / (1.0 + r * cost * total_n) for i in range(m)],
        dtype=np.float64,
    )

    single = contract_value(float(np.max(benefit)), total_n, r, cost)
    singleton = float(
        sum(contract_value(float(benefit[i]), float(n[i]), r, m * cost) for i in range(m))
    )
    # The best value at every number of contracts, from the interval scan, so the reading can say
    # how many stages to split into rather than only how to fill the number it was handed. All of
    # these carry the capacity correction, so they are comparable with each other and with `single`.
    profile = tuple(
        _best_interval_partition(order, benefit, n, r, j * cost, j)[0] for j in range(1, m + 1)
    )
    best_k = int(np.argmax(profile)) + 1

    notes: list[str] = []
    if exhaustive_assignment is None:
        notes.append(
            f"the number of {n_contracts}-way partitions of {m} components exceeds the "
            f"enumeration limit of {enumeration_limit:,}, so only the interval scan ran."
        )
    if not bool(np.allclose(benefit, benefit[0])):
        notes.append(
            "the components are not equally valuable, and the interval property the sort rests on "
            "is not guaranteed there. A contract's value depends on the largest B among its members "
            "as well as on its summed noise, and those two do not order together, so a partition "
            "that is not an interval in rho can win. Measured over random instances, an interval "
            "partition attains the optimum in every case with equal benefits and in about three of "
            "five with unequal ones, which is why the exhaustive search runs rather than the sort "
            "being trusted."
        )
    return SortingReading(
        components=params.components,
        noise=n,
        rho=rho,
        contracts=tuple(contracts),
        value=value,
        n_contracts=n_contracts,
        exhaustive_ran=exhaustive_assignment is not None,
        exhaustive_value=exhaustive_value,
        interval_value=interval_value,
        interval_optimal=interval_optimal,
        searches_agree=searches_agree,
        cutoff=cutoff,
        cutoff_bracket=bracket,
        dilution=dilution,
        single_contract_value=single,
        singleton_value=singleton,
        effort_assumed_equal=bool(len(set(params.effort.tolist())) == 1),
        value_by_contract_count=profile,
        best_contract_count=best_k,
        baselines={
            "baseline.single_contract": single,
            "baseline.one_contract_per_component": singleton,
        },
        notes=tuple(notes),
    )


class SortingCutoff(DecisionInstrument):
    """N7. Which components belong in the same weighted sum, and where the boundary falls.

    Kill condition: if the single-contract baseline is optimal on every real composite tested,
    splitting never pays and the instrument reports a cutoff nobody should act on.
    """

    name = "SortingCutoff"
    version = "1.0"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "N7"
    deviations = (
        "the source names a cutoff and does not say what is being maximised over what. The "
        "objective here is the summed contract value with the capacity correction C -> k C, which "
        "makes a given total effort cost the same however many stages it is spread over. Without "
        "that correction every comparison across the number of contracts compares one agent "
        "against several and splitting looks free",
        "the value per unit of capacity a contract delivers is taken as the largest B_i among its "
        "members, on the argument that equal compensation leaves the agent indifferent about the "
        "split within a contract so the principal directs it. Taking the sum instead would say the "
        "tasks are complements, which is the opposite of the premise the sorting result rests on",
        "the sorting theorem's claim that the optimum is an interval partition in rho is checked "
        "rather than assumed, by enumerating every partition at the stated number of contracts and "
        "comparing. Above the enumeration limit only the interval scan runs and the reading records "
        "that the check did not happen",
        "the catalogue carries no N7 record and no registered quantity rows. "
        "`quantities.as_catalogue_rows()` emits the proposed record",
    )

    quantity = "reward.sorting_cutoff"
    #: The other quantity this instrument reports. `Instrument.quantity` is singular and N7
    #: produces two, so the second is declared here and the payload carries both.
    also_reports: tuple[str, ...] = ("reward.information_coefficient",)
    requires = NOISE_ACCESS
    invariance = "reward.affine"
    invariance_relation = INVARIANT
    baselines = N7_BASELINES
    rung = 1

    def __init__(
        self,
        parameters: ContractParameters | None = None,
        *,
        n_contracts: int = 2,
        cost_scale: float | None = None,
        enumeration_limit: int = DEFAULT_ENUMERATION_LIMIT,
    ) -> None:
        self.parameters = parameters
        self.n_contracts = int(n_contracts)
        self.cost_scale = cost_scale
        self.enumeration_limit = int(enumeration_limit)

    def compute(self) -> Any:
        if self.parameters is None:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail="no contract parameters were supplied, so there are no components to sort",
                remedy=(
                    "build a `ContractParameters` for the composite. The sort needs the noise "
                    "covariance, which A2 measures, the sensitivity matrix, which a weight sweep "
                    "measures, and the effort allocation, which `assume_equal_effort()` will stand "
                    "in for while recording that it did."
                ),
            )
        bad = self.parameters.refuse_unstated(
            self.name,
            ["benefit", "cost_curvature", "noise", "risk_aversion", "sensitivity", "effort"],
        )
        if bad is not None:
            return bad
        if self.parameters.m < 2:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail=(
                    f"the composite has {self.parameters.m} component(s), so there is nothing to "
                    f"sort and no boundary to place"
                ),
                remedy=(
                    "supply a composite with at least two components. A single-component reward "
                    "has no assignment problem; what it has is a weight, which N5 answers."
                ),
                statistics={"m": self.parameters.m},
            )
        return sorting_cutoff(
            self.parameters,
            n_contracts=self.n_contracts,
            cost_scale=self.cost_scale,
            enumeration_limit=self.enumeration_limit,
        )

    def payload(self, computed: SortingReading) -> dict[str, Any]:
        return {
            "components": list(computed.components),
            "noise": [float(v) for v in computed.noise],
            "information_coefficient": [float(v) for v in computed.rho],
            "dilution": [float(v) for v in computed.dilution],
            "contracts": [
                {
                    "members": list(c.members),
                    "noise_total": c.noise_total,
                    "benefit": c.benefit,
                    "value": c.value,
                    "rho_low": c.rho_low,
                    "rho_high": c.rho_high,
                }
                for c in computed.contracts
            ],
            "cutoff": computed.cutoff,
            "cutoff_bracket": list(computed.cutoff_bracket) if computed.cutoff_bracket else None,
            "value": computed.value,
            "split_gain": computed.split_gain,
            "n_contracts": computed.n_contracts,
            "best_contract_count": computed.best_contract_count,
            "value_by_contract_count": [float(v) for v in computed.value_by_contract_count],
            "single_contract_value": computed.single_contract_value,
            "singleton_value": computed.singleton_value,
            "exhaustive_ran": computed.exhaustive_ran,
            "exhaustive_value": computed.exhaustive_value,
            "interval_value": computed.interval_value,
            "interval_optimal": computed.interval_optimal,
            "searches_agree": computed.searches_agree,
            "effort_assumed_equal": computed.effort_assumed_equal,
            "notes": list(computed.notes),
            "baselines": dict(computed.baselines),
            "says": computed.says(),
        }


__all__ = [
    "DEFAULT_ENUMERATION_LIMIT",
    "N7_BASELINES",
    "Contract",
    "SortingCutoff",
    "SortingReading",
    "contract_value",
    "information_coefficient",
    "noise_to_signal",
    "sorting_cutoff",
]

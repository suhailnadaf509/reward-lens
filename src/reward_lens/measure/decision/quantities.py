"""Six quantities N5 to N8 need, proposed here and **not registered**.

`spec/CATALOGUE.yaml` carries N1 to N4 and stops. The contract layer has five formulas behind it and
no catalogue rows at all, and the quantity rows are the part not to write into the registry
unilaterally, because registering a quantity is a decision about what the library claims to measure.

So the proposals live here as data. Nothing runs at import: `register_proposed()` exists so a test
can show the four instruments are lint-clean apart from the missing registration, `as_yaml_rows()`
emits the exact shape `spec/QUANTITIES.yaml` uses, and `as_catalogue_rows()` emits the four
instrument records in `spec/CATALOGUE.yaml`'s shape, so neither has to be retyped and neither can
drift from the objects the instruments are linted against.

**Five of the six are exactly invariant under `reward.affine`, and that is a result rather than a
convenience.** Rescaling every component's score by `a` sends the noise covariance to `a^2 Sigma` and
the sensitivity matrix to `a M`, and the ratio `sigma^2 / mu'^2` that every one of those five is
built from is blind to it. The exception is the weight vector itself, which is covariant with weight
-1, because a weight is a price per unit of signal and rescaling the signal must rescale the price.
An implementation that reported the weight vector as invariant would be reporting a coordinate
artifact, which is the failure gate 2 exists to prevent, and it is the reason the diagonal formula
must not be applied to raw reward model outputs.
"""

from __future__ import annotations

import math
from typing import Any

from reward_lens.core.quantity import QUANTITIES, Quantity, Unit, register_quantity

# Every token below already appears in `spec/QUANTITIES.yaml`. None is invented.
_WEIGHT = Unit(dimension="1", per="component", scale=None, as_printed="1")
_RATIO = Unit(dimension="ratio", per="component", scale="relative", as_printed="ratio")
_DOSE_PER_COMPONENT = Unit(dimension="dose", per="component", scale=None, as_printed="dose")
_DOSE = Unit(dimension="dose", per=None, scale=None, as_printed="dose")
_COS = Unit(dimension="alignment", per="component", scale=None, as_printed="cos")
_VAR = Unit(dimension="variance", per="component", scale=None, as_printed="var")


OPTIMAL_WEIGHTS = Quantity(
    id="reward.optimal_weights",
    definition=(
        "The vector alpha* maximising the principal's certainty equivalent in the linear-contract "
        "problem of Holmstrom and Milgrom (1991): alpha* = [M C''^-1 M' + r Sigma]^-1 M C''^-1 B', "
        "where B' is the marginal value of effort on each task, C'' is the Hessian of the agent's "
        "effort cost, Sigma is the covariance of the grader noise, r is the agent's absolute risk "
        "aversion and M is the matrix of signal sensitivities to effort. At M = I this reduces to "
        "alpha* = (I + r C'' Sigma)^-1 B', and with both matrices diagonal to "
        "alpha_i = B_i / (1 + r C_ii sigma_i^2). Note the ordering C'' before Sigma: the product "
        "does not commute and transposing it changes the answer. Where one task is unmeasurable "
        "the two-task solution is alpha_1 = [B_1 - B_2 (C_12/C_22)] / "
        "[1 + r sigma_1^2 (C_11 - C_12^2/C_22)] with alpha_2 = 0."
    ),
    unit=_WEIGHT,
    invariance="reward.affine",
    interpretation=(
        "The weight a reward component should carry given how noisy it is, rather than given how "
        "much anyone likes it. The shrinkage factor 1/(1 + r C'' sigma^2) is the same object as the "
        "regressional-Goodhart factor Var(X)/(Var(X) + Var(Z)): contract theory derives it as what "
        "the principal should do and the Goodhart literature derives it as what happens if the "
        "principal does not. The number is a design check with five printed assumptions, not a law: "
        "Prendergast (1999) finds the risk-incentive relationship it rests on weak or wrong-signed "
        "in humans."
    ),
    support=None,
    wedge=True,
)

EQUAL_COMPENSATION_RATIO = Quantity(
    id="reward.equal_compensation_ratio",
    definition=(
        "Per component of a composite reward, the agent's marginal return kappa_i = alpha_i mu'_i "
        "divided by the largest such return across components, where alpha_i is the component's "
        "weight and mu'_i is its dose-response slope, the amount the component's score moves per "
        "unit of policy capacity spent on it. The equal compensation principle (Milgrom and Roberts "
        "1992, not Holmstrom and Milgrom 1991) states that activities the agent treats as "
        "substitutes must offer equal marginal returns, so this ratio is 1 for every component of a "
        "correctly weighted composite."
    ),
    unit=_RATIO,
    invariance="reward.affine",
    interpretation=(
        "The component with the lowest ratio is the one that gets starved: the agent's capacity "
        "goes where its marginal return is highest, and a component offering less return per unit "
        "of capacity receives none of it once effort is fungible. A ratio of 0.3 means that "
        "component pays the policy thirty percent of what the best-paying component pays for the "
        "same effort, and the composite's stated weight on it is not the incentive it delivers."
    ),
    support=(-math.inf, 1.0),
    wedge=True,
)

INFORMATION_COEFFICIENT = Quantity(
    id="reward.information_coefficient",
    definition=(
        "rho_i = n_i / t_i, where n_i = sigma_i^2 / mu'_i^2 is the component's effort-equivalent "
        "noise and t_i is the effort spent on it at the operating point. The numerator converts a "
        "variance in score units into a variance in effort units by dividing by the squared "
        "dose-response slope, which is what makes components with different output scales "
        "comparable at all."
    ),
    unit=_DOSE_PER_COMPONENT,
    invariance="reward.affine",
    interpretation=(
        "How much effort-equivalent noise one unit of effort on this component buys. It is the "
        "quantity the sorting theorem orders components by, and its ordering is what decides which "
        "components belong in the same weighted sum. A common scaling of the effort allocation "
        "cancels from every comparison, so the ordering survives not knowing t."
    ),
    support=(0.0, math.inf),
    wedge=True,
)

SORTING_CUTOFF = Quantity(
    id="reward.sorting_cutoff",
    definition=(
        "The value of the information coefficient rho at which a composite reward's components "
        "split into contracts. Components on opposite sides of the cutoff should not share an "
        "incentive contract; in a reinforcement learning loop that means separate optimisation "
        "stages, separate models or separate gradient paths rather than one weighted sum. Computed "
        "by maximising the total certainty equivalent over partitions, where a contract holding "
        "components with total effort-equivalent noise N has value B^2 / [2 C (1 + r C N)] and the "
        "single shared incentive intensity is what makes one component's noise a tax on every other "
        "component in the same contract. Comparisons across the number of contracts carry the "
        "capacity correction C -> k C, so a given total effort costs the same however many stages "
        "it is spread over and splitting is not free."
    ),
    unit=_DOSE,
    invariance="reward.affine",
    interpretation=(
        "Do not put a noisy judge and a crisp unit test in the same weighted sum. The optimum is "
        "provably asymmetric, so treating components symmetrically is suboptimal whenever their "
        "noise differs at all, and the cutoff says where the asymmetry becomes worth acting on. "
        "For two components, one noiseless and one with x = r C n, splitting beats the single sum "
        "by a factor of (1 + x)^2 / (1 + 2x) at the same total capacity, which is exactly 1 at "
        "x = 0 and above 1 everywhere else."
    ),
    support=(0.0, math.inf),
    wedge=True,
)

COMPONENT_CONGRUITY = Quantity(
    id="reward.component_congruity",
    definition=(
        "cos of the angle between the direction a signal pushes the agent in effort space and the "
        "direction that creates value. For the whole contract it is cos(angle(M' alpha, B')); per "
        "component it is cos(angle(m_i, B')) with m_i the i-th row of M, the gradient of that "
        "component's score in effort space. Distortion is 1 minus congruity."
    ),
    unit=_COS,
    invariance="reward.affine",
    interpretation=(
        "The second of the two numbers every reward component needs. A noisy but congruent "
        "component wants a lower weight, and the shrinkage factor says how much lower. A precise "
        "but distorted component wants a different measure, because no weight fixes an angle. "
        "Congruity is identically the normalised benefit vector when M is diagonal, so a reading "
        "taken on a diagonal M is reporting B' in disguise and says so rather than passing itself "
        "off as a measurement of distortion."
    ),
    support=(-1.0, 1.0),
    wedge=True,
)

COMPONENT_NOISE = Quantity(
    id="reward.component_noise",
    definition=(
        "n_i = sigma_i^2 / mu'_i^2, a component's grader noise variance divided by the square of "
        "its dose-response slope. The variance comes from A2's decomposition, being everything in "
        "the component's score variance that is not the object of measurement; the slope has to be "
        "measured by a weight sweep or stated."
    ),
    unit=_VAR,
    invariance="reward.affine",
    interpretation=(
        "The first of the two numbers every reward component needs, and the one that is comparable "
        "across components where a raw variance is not. Two graders reporting on scales that differ "
        "by two orders of magnitude have variances that differ by four, and dividing by the squared "
        "sensitivity removes exactly that."
    ),
    support=(0.0, math.inf),
    wedge=True,
)

PROPOSED: tuple[Quantity, ...] = (
    OPTIMAL_WEIGHTS,
    EQUAL_COMPENSATION_RATIO,
    INFORMATION_COEFFICIENT,
    SORTING_CUTOFF,
    COMPONENT_CONGRUITY,
    COMPONENT_NOISE,
)

#: Which instrument each proposal belongs to, for the catalogue's `instrument` list.
PROPOSED_INSTRUMENTS: dict[str, tuple[str, ...]] = {
    "reward.optimal_weights": ("N5",),
    "reward.equal_compensation_ratio": ("N6",),
    "reward.information_coefficient": ("N7",),
    "reward.sorting_cutoff": ("N7",),
    "reward.component_congruity": ("N8",),
    "reward.component_noise": ("N8",),
}

#: The minimum access each proposal needs, in the catalogue's `min_access` spelling.
PROPOSED_MIN_ACCESS: dict[str, str] = {
    "reward.optimal_weights": "GRADER:REPLICATE",
    "reward.equal_compensation_ratio": "GRADER:RECORD+POLICY:CONTROL",
    "reward.information_coefficient": "GRADER:REPLICATE+POLICY:CONTROL",
    "reward.sorting_cutoff": "GRADER:REPLICATE+POLICY:CONTROL",
    "reward.component_congruity": "GRADER:REPLICATE+POLICY:CONTROL",
    "reward.component_noise": "GRADER:REPLICATE+POLICY:CONTROL",
}

#: How many rungs each proposal's ladder has as built. One everywhere: the arithmetic is exact and
#: what changes between callers is where the parameters came from, which is provenance rather than a
#: rung. `reward.equal_compensation_ratio` is the exception and has two, because a supplied
#: sensitivity and a measured one are different estimators of the same number.
PROPOSED_RUNGS: dict[str, int] = {
    "reward.optimal_weights": 1,
    "reward.equal_compensation_ratio": 2,
    "reward.information_coefficient": 1,
    "reward.sorting_cutoff": 1,
    "reward.component_congruity": 1,
    "reward.component_noise": 1,
}


def register_proposed() -> list[str]:
    """Register the proposals in this process only. Not called at import, by design.

    A test uses this to show the instruments are otherwise lint-clean, which separates "this
    instrument estimates an unregistered quantity" from "this instrument has four other problems".
    It is not a substitute for the rows landing in `spec/QUANTITIES.yaml`, and nothing in the
    package calls it.
    """
    added = []
    for q in PROPOSED:
        if q.id not in QUANTITIES:
            register_quantity(q)
            added.append(q.id)
    return added


def _folded(key: str, text: str, indent: str = "  ", width: int = 96) -> list[str]:
    """A YAML folded block scalar, which is the only safe way to emit this package's prose.

    Every definition here contains a colon followed by a space, which ends a plain scalar and makes
    the row unparseable, and several contain single and double quotes. A folded block takes the text
    literally and joins its lines with spaces, so nothing has to be escaped and nothing can be
    misread. `spec/QUANTITIES.yaml` already uses `>-` for its longer rows.
    """
    out = [f"{indent}{key}: >-"]
    body, line = indent + "  ", ""
    for word in text.split():
        if line and len(line) + 1 + len(word) > width:
            out.append(body + line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(body + line)
    return out


def _support_yaml(support: tuple[float, float] | None) -> list[str]:
    if support is None:
        return ["  support: null"]
    out = ["  support:"]
    for edge in support:
        out.append(f"  - {'null' if not math.isfinite(edge) else edge}")
    return out


def as_yaml_rows() -> str:
    """The six proposals in `spec/QUANTITIES.yaml`'s own field order, ready to paste.

    Emitted rather than hand-written so the rows cannot drift from the `Quantity` objects the
    instruments are linted against.
    """
    lines: list[str] = []
    for q in PROPOSED:
        lines.append(f"- id: {q.id}")
        lines.append("  unit:")
        lines.append(f"    as_printed: '{q.unit.as_printed}'")
        lines.append(f"    dimension: '{q.unit.dimension}'")
        lines.append(f"    per: {q.unit.per if q.unit.per else 'null'}")
        lines.append(f"    scale: {q.unit.scale if q.unit.scale else 'null'}")
        lines.append(f"  invariance_group: {q.invariance}")
        lines.append(f"  min_access: {PROPOSED_MIN_ACCESS[q.id]}")
        lines.append(f"  rungs: {PROPOSED_RUNGS[q.id]}")
        lines.append("  instrument:")
        for inst in PROPOSED_INSTRUMENTS[q.id]:
            lines.append(f"  - {inst}")
        lines.append(f"  wedge: {'true' if q.wedge else 'false'}")
        lines.extend(_folded("definition", q.definition))
        lines.extend(_folded("interpretation", q.interpretation))
        lines.extend(_support_yaml(q.support))
    return "\n".join(lines)


#: The four instrument records, as data, in `spec/CATALOGUE.yaml`'s field order. Held here as data
#: rather than as prose so `as_catalogue_rows` can emit them and a test can assert that every
#: quantity each record names is one this module actually proposes.
CATALOGUE_RECORDS: tuple[dict[str, Any], ...] = (
    {
        "id": "N5",
        "name": "Optimal component weights",
        "headline": (
            "Nobody sets a reward component's weight from that component's measured noise, and the "
            "formula that does has been available since 1991."
        ),
        "quantities": ["reward.optimal_weights"],
        "says": (
            '"Weight the judge at 0.31, not 0.5. And there is a task you cannot measure, so weight '
            'it at 0.24 instead."'
        ),
        "access_min": "GRADER: REPLICATE, for Sigma. Every other parameter is stated or swept",
        "phases": ["PRE_RUN", "POST_RUN"],
        "envelope_requires": ["STATIONARY_GRADER"],
        "envelope_measured_by": {"STATIONARY_GRADER": "monitor.check_standard_drift"},
        "rung": 1,
        "estimator": (
            "alpha* = [M C''^-1 M' + r Sigma]^-1 M C''^-1 B', with Sigma from A2 and the rest "
            "stated or swept; the diagonal and unmeasurable-task forms are special cases of the "
            "same solve and are checked against it"
        ),
        "bias": (
            "exact given the parameters, and the parameters are where the error lives. The "
            "recommendation is a deterministic function of five inputs, one of which is measured, "
            "so its uncertainty is dominated by the four that are not and the sweep rather than an "
            "interval is what reports it"
        ),
        "invariance_group": "reward.affine",
        "baselines": [
            "equal weights across components, which is what a composite with no analysis uses",
            "weights proportional to B', which is what weighting by importance means and is the "
            "recommendation with the noise term deleted",
        ],
        "kill_condition": (
            "if the recommended weights are within measurement error of the value-proportional "
            "baseline on every real composite tested, the noise term is not doing any work and the "
            "instrument collapses into a restatement of B'."
        ),
        "wedge": True,
    },
    {
        "id": "N6",
        "name": "The equal-compensation table",
        "headline": "Which component of your composite reward is being starved, and by how much.",
        "quantities": ["reward.equal_compensation_ratio"],
        "says": (
            '"format_ok pays the policy 0.18 of what the judge pays for the same capacity. It is '
            'the starved component and its stated weight of 0.2 is not the incentive it delivers."'
        ),
        "access_min": "GRADER: RECORD for the weights; POLICY: CONTROL for a measured sensitivity",
        "phases": ["PRE_RUN", "POST_RUN"],
        "envelope_requires": ["STATIONARY_GRADER"],
        "envelope_measured_by": {"STATIONARY_GRADER": "monitor.check_standard_drift"},
        "rung": 0,
        "estimator": (
            "kappa_i = alpha_i mu'_i from a stated sensitivity at rung 0, from a measured weight "
            "sweep at rung 1; the starved component is argmin kappa"
        ),
        "bias": (
            "exact given mu'. A stated mu' makes the table a conditional statement and the reading "
            "says which of the two it is on every row"
        ),
        "invariance_group": "reward.affine",
        "baselines": [
            "the weights read as commissions, which is what treating a weight as an incentive "
            "assumes and is the table under mu' = 1",
            "equal weights, which is what the table would say if the composite had not been tuned",
        ],
        "kill_condition": (
            "if the commission ratio is within a few percent of 1 for every component of every real "
            "composite tested, composites are already equal-compensating and the table is a "
            "formality."
        ),
        "wedge": True,
    },
    {
        "id": "N7",
        "name": "The sorting cutoff",
        "headline": (
            "Do not put a noisy judge and a crisp unit test in the same weighted sum, with the "
            "threshold at which that stops being a slogan."
        ),
        "quantities": ["reward.information_coefficient", "reward.sorting_cutoff"],
        "says": (
            '"Split at rho = 0.42. The unit tests and the format check belong in one stage and the '
            'judge belongs in another; keeping them together costs 23% of the achievable surplus."'
        ),
        "access_min": "GRADER: REPLICATE for Sigma; POLICY: CONTROL for the sensitivities",
        "phases": ["PRE_RUN", "POST_RUN"],
        "envelope_requires": ["STATIONARY_GRADER"],
        "envelope_measured_by": {"STATIONARY_GRADER": "monitor.check_standard_drift"},
        "rung": 1,
        "estimator": (
            "exhaustive maximisation of the summed contract value B_g^2/[2C(1 + r C N_g)] over "
            "partitions, with the interval partition in rho order computed alongside it and the "
            "two compared, so the sorting theorem's own claim is checked rather than assumed"
        ),
        "bias": (
            "exact given the parameters. The exhaustive search is over all partitions up to a "
            "stated component count and is the interval scan only above it, and the reading says "
            "which of the two produced the answer"
        ),
        "invariance_group": "reward.affine",
        "baselines": [
            "one contract holding every component, which is what every composite reward is",
            "one contract per component, which is the other extreme and is usually worse",
        ],
        "kill_condition": (
            "if the single-contract baseline is optimal on every real composite tested, splitting "
            "never pays and the instrument reports a cutoff nobody should act on."
        ),
        "wedge": True,
    },
    {
        "id": "N8",
        "name": "Noise and angle, per component",
        "headline": (
            "Every reward component needs two numbers rather than one, and no tooling separates "
            "them."
        ),
        "quantities": ["reward.component_noise", "reward.component_congruity"],
        "says": (
            '"The judge is noisy and congruent: lower its weight to 0.31. The length penalty is '
            'precise and distorted at cos = 0.12: no weight fixes it, replace the measure."'
        ),
        "access_min": "GRADER: REPLICATE for Sigma; POLICY: CONTROL for the sensitivity matrix",
        "phases": ["PRE_RUN", "POST_RUN"],
        "envelope_requires": ["STATIONARY_GRADER"],
        "envelope_measured_by": {"STATIONARY_GRADER": "monitor.check_standard_drift"},
        "rung": 1,
        "estimator": (
            "n_i = sigma_i^2/mu'_i^2 from A2 and the sensitivity matrix; congruity is the cosine "
            "between the i-th row of M and B', with the whole-contract cosine between M' alpha and "
            "B' reported beside it"
        ),
        "bias": (
            "exact given the parameters. Congruity is degenerate under a diagonal M, where it "
            "equals the normalised benefit vector, and the reading refuses to call that a "
            "measurement of distortion"
        ),
        "invariance_group": "reward.affine",
        "baselines": [
            "the marginal correlation between each component and the outcome, which is what "
            "practice looks at and which cannot separate a noisy measure from a distorted one",
            "the component's raw variance, which is the noise half with the sensitivity left in "
            "and is therefore not comparable across components",
        ],
        "kill_condition": (
            "if every real composite tested has a diagonal sensitivity matrix, congruity carries no "
            "information beyond B' and only the noise half of the pair ships."
        ),
        "wedge": True,
    },
)


def as_catalogue_rows() -> str:
    """The four instrument records in `spec/CATALOGUE.yaml`'s shape, ready to paste.

    Emitted from `CATALOGUE_RECORDS` so the proposal and the code cannot disagree about which
    quantities an instrument reports.
    """
    lines: list[str] = []
    for rec in CATALOGUE_RECORDS:
        lines.append(f"- id: {rec['id']}")
        lines.append(f"  name: {rec['name']}")
        lines.extend(_folded("headline", str(rec["headline"])))
        lines.append("  quantities:")
        for q in rec["quantities"]:
            lines.append(f"  - {q}")
        lines.extend(_folded("says", str(rec["says"])))
        lines.extend(_folded("access_min", str(rec["access_min"])))
        lines.append("  substrates:")
        for s in ("NEURAL_SCALAR", "NEURAL_GEN", "PROGRAM", "PROCEDURAL", "HUMAN", "COMPOSITE"):
            lines.append(f"  - {s}")
        lines.append("  phases:")
        for p in rec["phases"]:
            lines.append(f"  - {p}")
        lines.append("  envelope_requires:")
        for c in rec["envelope_requires"]:
            lines.append(f"  - {c}")
        lines.append("  envelope_measured_by:")
        for c, q in rec["envelope_measured_by"].items():
            lines.append(f"    {c}: {q}")
        lines.append("  ladder:")
        lines.append(f"  - rung: {rec['rung']}")
        lines.extend(_folded("estimator", str(rec["estimator"]), indent="    "))
        lines.extend(_folded("access", str(rec["access_min"]), indent="    "))
        lines.extend(_folded("bias", str(rec["bias"]), indent="    "))
        lines.extend(
            _folded(
                "cost",
                "free. Arithmetic on parameters the caller already holds",
                indent="    ",
            )
        )
        lines.append(f"  invariance_group: {rec['invariance_group']}")
        lines.append("  baselines:")
        for b in rec["baselines"]:
            lines.append("  - >-")
            lines.extend(_folded("x", str(b), indent="    ")[1:])
        lines.extend(_folded("kill_condition", str(rec["kill_condition"])))
        lines.append(f"  wedge: {'true' if rec['wedge'] else 'false'}")
        lines.append("  status: planned")
    return "\n".join(lines)


__all__ = [
    "CATALOGUE_RECORDS",
    "COMPONENT_CONGRUITY",
    "COMPONENT_NOISE",
    "EQUAL_COMPENSATION_RATIO",
    "INFORMATION_COEFFICIENT",
    "OPTIMAL_WEIGHTS",
    "PROPOSED",
    "PROPOSED_INSTRUMENTS",
    "PROPOSED_MIN_ACCESS",
    "PROPOSED_RUNGS",
    "SORTING_CUTOFF",
    "as_catalogue_rows",
    "as_yaml_rows",
    "register_proposed",
]

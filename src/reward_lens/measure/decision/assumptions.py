"""The five premises the contract layer imports, printed beside every number it produces.

A weight recommendation is not a measurement. It is a measurement pushed through an economic model,
and the model has premises that a reward record can neither confirm nor deny. Holmstrom and Milgrom
(1991) is a theorem about a continuous-time Brownian principal-agent problem with an exponential
utility agent; RLHF is not that, and the distance between them is the whole risk this package
carries.

So the five assumptions are data rather than prose. They travel on every reading, they render in the
same call that renders the recommendation, and a test asserts that no reading in this package can be
produced without them. That is the single property this package is built around: **a weight
recommendation whose model assumptions are invisible is advice wearing a number.**

Each one carries four fields for a reason. ``statement`` is the premise. ``holds_when`` says where it
is actually true, which is usually somewhere other than here. ``breaks`` says what goes wrong and,
where the direction is known, which way the recommended weight is then wrong. ``check`` says what in
this library would test it, or says plainly that nothing can, because "not checkable" is a fact about
the premise and not an omission in the reading.

Two of the five are checkable with instruments that already exist. GAUSSIAN_NOISE is falsified by
N3's tail index on any component whose Hill estimate is above zero. COMMITMENT_ONE_PERIOD is the
envelope condition `STATIONARY_GRADER`, which every instrument here declares and every preflight
consults. The other three are not measurable from a record at any access level, and the package's
response to that is to refuse or to sweep rather than to assume.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# The type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Assumption:
    """One premise of the contract model, with where it holds and what breaks when it does not."""

    key: str
    statement: str
    holds_when: str
    breaks: str
    #: Whether anything in this library can test it on a real subject. Three of five are False, and
    #: that is the honest count rather than a gap to be closed by writing an optimistic checker.
    checkable: bool
    check: str

    def render(self, indent: str = "  ") -> str:
        mark = "checkable" if self.checkable else "not checkable"
        return (
            f"{indent}{self.key}  [{mark}]\n"
            f"{indent}    {self.statement}\n"
            f"{indent}    holds when: {self.holds_when}\n"
            f"{indent}    breaks:     {self.breaks}\n"
            f"{indent}    check:      {self.check}"
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "statement": self.statement,
            "holds_when": self.holds_when,
            "breaks": self.breaks,
            "checkable": self.checkable,
            "check": self.check,
        }


# ---------------------------------------------------------------------------
# The five
# ---------------------------------------------------------------------------

LINEAR_CONTRACT = Assumption(
    key="LINEAR_CONTRACT",
    statement=(
        "The reward is linear in its components, R = sum_i alpha_i x_i + beta, and a linear "
        "contract is the optimum rather than a convenient restriction on the search."
    ),
    holds_when=(
        "Holmstrom and Milgrom (1987) prove linearity optimal in continuous time under Brownian "
        "noise, CARA utility and drift control, where the agent watches the accumulated signal and "
        "adjusts continuously. A composite reward evaluated once at the end of a rollout is not "
        "that setting, and neither is a reward evaluated per step against a fixed rubric."
    ),
    breaks=(
        "Outside the continuous-time setting the optimal contract is generally nonlinear and a "
        "nonlinear one can do strictly better. Every weight reported here is the best linear "
        "answer, which is not the same claim as the best answer."
    ),
    checkable=False,
    check=(
        "not testable from a reward record. It is a restriction on the space of contracts searched, "
        "and the search never leaves that space, so no reading can contradict it."
    ),
)

GAUSSIAN_NOISE = Assumption(
    key="GAUSSIAN_NOISE",
    statement=(
        "Each component's grader noise is normal with covariance Sigma, and the noise does not "
        "depend on how much effort the agent spent."
    ),
    holds_when=(
        "The risk premium term -(r/2) alpha' Sigma alpha is exactly the cumulant generating "
        "function of a Gaussian evaluated at the agent's risk aversion. Under any other noise law "
        "the premium is a functional of the whole distribution rather than of its second moment, "
        "and the covariance stops being a sufficient statistic for the agent's exposure."
    ),
    breaks=(
        "A heavy right tail makes the true risk premium larger than (r/2) alpha' Sigma alpha, so "
        "every weight this layer recommends is too high. The direction of the error is known even "
        "where its size is not."
    ),
    checkable=True,
    check=(
        "testable. Run N3, frontier.tail_index, on each component's scores; a Hill estimate "
        "materially above zero falsifies this for that component and the recommendation for it "
        "should be read as an upper bound."
    ),
)

CARA_AND_COST_CURVATURE = Assumption(
    key="CARA_AND_COST_CURVATURE",
    statement=(
        "The agent has constant absolute risk aversion r, and the cost of effort has a Hessian "
        "C'' that is constant over the range of effort considered."
    ),
    holds_when=(
        "CARA is what makes the agent's problem reduce to a certainty equivalent that is linear in "
        "the mean and linear in the variance, which is what makes the whole problem "
        "finite-dimensional. A constant C'' is the second-order expansion of any smooth convex "
        "cost around the operating point, so it is a local statement and not a global one."
    ),
    breaks=(
        "r and C'' are the two parameters the recommendation is most sensitive to and neither is "
        "identified by anything in a reward record. Prendergast (1999) reports the empirical "
        "risk-incentive relationship in humans as weak or wrong-signed relative to "
        "1/(1 + r C'' sigma^2), so the functional form is not settled outside the model either."
    ),
    checkable=False,
    check=(
        "not measurable from a record at any access level. Either the caller supplies both and the "
        "reading records that they were supplied, or the reading is a function of them over a "
        "stated range. This package refuses rather than defaulting."
    ),
)

COMMITMENT_ONE_PERIOD = Assumption(
    key="COMMITMENT_ONE_PERIOD",
    statement=(
        "The principal commits to the weights for the whole period, there is exactly one period, "
        "and there is no renegotiation and no reputation."
    ),
    holds_when=(
        "The weights and the rubric are fixed across the window the recommendation is about. A "
        "reward schedule that moves mid-run is a different problem: the agent's best response is "
        "to the schedule, not to the weights in force at any instant, and the static optimum is "
        "not the dynamic one."
    ),
    breaks=(
        "A drifting grader makes the first-order condition hold against a moving target, so the "
        "effort allocation the recommendation predicts is never reached and the number describes "
        "a fixed point the run never visits."
    ),
    checkable=True,
    check=(
        "testable, and enforced. Every instrument in this package declares the envelope condition "
        "STATIONARY_GRADER, measured by monitor.check_standard_drift, and refuses outside it."
    ),
)

AGENT_KNOWS_THE_CONTRACT = Assumption(
    key="AGENT_KNOWS_THE_CONTRACT",
    statement=(
        "The agent knows alpha when it chooses effort, so its first-order condition "
        "M' alpha = C'(t) holds exactly at the moment of choosing."
    ),
    holds_when=(
        "The contract is written down and read. A human agent under a commission scheme knows the "
        "commission; that is the setting the theorem was written for."
    ),
    breaks=(
        "A policy does not know the reward weights. It infers them from samples, so the first-order "
        "condition holds in the limit of the learning process rather than during it. Everything "
        "here is a statement about the fixed point, and the path to the fixed point is where "
        "reward hacking is observed, so the layer is silent about exactly the interval people care "
        "most about."
    ),
    checkable=False,
    check=(
        "partly testable and only after the fact. The gap between the realised effort allocation "
        "and the allocation the first-order condition predicts is measurable once a run exists; "
        "before a run exists nothing measures it, and this layer answers before a run exists."
    ),
)

#: The five, in the order they are printed. Order is by how far each one is from the reader's
#: setting: the contract form first, then the noise law, then the agent's preferences, then the
#: timing, then the one about what the agent knows, which is the one an RL reader will recognise
#: as false and the one the source itself calls interesting rather than fatal.
ASSUMPTIONS: tuple[Assumption, ...] = (
    LINEAR_CONTRACT,
    GAUSSIAN_NOISE,
    CARA_AND_COST_CURVATURE,
    COMMITMENT_ONE_PERIOD,
    AGENT_KNOWS_THE_CONTRACT,
)

ASSUMPTION_KEYS: tuple[str, ...] = tuple(a.key for a in ASSUMPTIONS)


def assumption(key: str) -> Assumption:
    """One assumption by key, raising rather than returning None on a typo."""
    for a in ASSUMPTIONS:
        if a.key == key:
            return a
    raise KeyError(
        f"no assumption named {key!r}; this layer has exactly {len(ASSUMPTIONS)}: "
        f"{', '.join(ASSUMPTION_KEYS)}"
    )


def render_assumptions(indent: str = "  ") -> str:
    """All five, rendered. Every reading in this package calls this and none of them may skip it."""
    head = (
        f"{indent}The five assumptions this recommendation imports "
        f"(Holmstrom and Milgrom 1991, linear contracts under CARA and Brownian noise):"
    )
    return "\n".join([head, *(a.render(indent + "  ") for a in ASSUMPTIONS)])


def assumptions_payload() -> list[dict[str, object]]:
    """The five as plain data, for the Evidence payload and for anything that reads a store."""
    return [a.as_dict() for a in ASSUMPTIONS]


__all__ = [
    "AGENT_KNOWS_THE_CONTRACT",
    "ASSUMPTION_KEYS",
    "ASSUMPTIONS",
    "CARA_AND_COST_CURVATURE",
    "COMMITMENT_ONE_PERIOD",
    "GAUSSIAN_NOISE",
    "LINEAR_CONTRACT",
    "Assumption",
    "assumption",
    "assumptions_payload",
    "render_assumptions",
]

"""M7, the uncertainty budget as a first-class reading.

A confidence interval is one number that has already thrown away the thing worth knowing, which is
*which term dominates*. The GUM's alternative is a table: enumerate every contribution, say for each
whether it was evaluated statistically or by judgement, give each a sensitivity coefficient, and
compose in quadrature. The payload is the last line, and the observation that goes with it is
the reason this instrument exists: **the largest term is almost never sampling noise**, and a budget
that cannot say so is not doing its job.

The arithmetic already lives in `core.budget`. What is here is the three things that turn a table
into a reading.

**Composition, in one call.** `compose` takes the pieces the rest of this package produces, the
reading's own sampling standard error, a rung disagreement from M11, a substrate noise floor from
M1, an instrument overhead from M2, a reference material's certificate, and assembles the table. The
seam matters more than the arithmetic: a budget that has to be built by hand is a budget that gets
built from whichever terms the author remembered.

**A lint, as findings rather than as a failure.** A budget whose terms are all Type A has never had
a systematic effect looked at. A budget with one term is an interval with a table drawn round it. A
budget that declares no correlations has assumed independence, which is the commonest way a budget
understates itself. None of those is wrong enough to refuse over and all of them change what the
number means, so they are reported by name on the reading.

**A quantity taken from the subject, and a refusal when there is not one.** The catalogue leaves
M7's quantity list `OPEN` and this is the decision: a combined standard uncertainty is expressed in
the units of the reading it belongs to, every term in the table is a contribution to that reading,
and a budget therefore has no measurand of its own. So `quantity` is set per instance from the
subject, and a budget pointed at nothing refuses rather than emitting a table attributed to no
measurement. The alternative, registering an id like `reading.combined_uncertainty`, would create a
quantity whose unit is the unit of whatever it happens to be pointed at, which is not a quantity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from reward_lens.core.budget import (
    BudgetTerm,
    LimitOfDetection,
    UncertaintyBudget,
)
from reward_lens.core.envelope import EnvelopeSpec
from reward_lens.core.evidence import Uncertainty, register_payload
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.quantity import QUANTITIES, BaselineID
from reward_lens.core.reading import Refusal, RefusalReason, refuse_incomplete
from reward_lens.core.reference import ReferenceMaterial, Transfer
from reward_lens.core.types import (
    Access,
    Capability,
    Component,
    GaugeStatus,
    Phase,
    Substrate,
)
from reward_lens.measure.meta._base import MetaInstrument

#: A budget is arithmetic over numbers other readings already produced.
BUDGET_ACCESS: dict[Component, Access] = {Component.RECORD: Access.RECORD}

#: What a budget is claiming not to be. The first is the interval a paper reports when it has only
#: counted its samples; the second is the same reading with every systematic term dropped, which is
#: what a budget of Type A terms alone amounts to.
BUDGET_BASELINES: tuple[BaselineID, ...] = (
    "baseline.sampling_only",
    "baseline.largest_single_term",
)

BUDGET_ENVELOPE = EnvelopeSpec(
    unconditional=True,
    justification=(
        "quadrature over declared contributions. Every assumption in a budget is carried by the "
        "terms rather than by the composition: a term measured in the wrong regime is wrong where "
        "it was measured, and no regime of this run can make the sum of squares of a stated set of "
        "numbers into a different sum."
    ),
)


# ---------------------------------------------------------------------------
# The lint
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BudgetFinding:
    """One thing about a budget that changes what its number means, with what closes it."""

    field: str
    problem: str
    remedy: str

    def render(self) -> str:
        return f"{self.field}: {self.problem}  ->  {self.remedy}"


def lint_budget(
    budget: UncertaintyBudget, *, sampling_terms: Sequence[str] = ()
) -> list[BudgetFinding]:
    """What is missing from a budget, as findings. Never raises, never refuses.

    ``sampling_terms`` names which terms are sampling noise. It is a parameter rather than a name
    heuristic because guessing from a term's name is exactly the kind of plausible inference this
    library exists not to make, and the author of a budget knows which of their terms came from
    counting samples.
    """
    out: list[BudgetFinding] = []
    terms = budget.terms
    if not terms:
        return [
            BudgetFinding(
                "terms",
                "the budget is empty, so its combined uncertainty is zero by construction",
                "add at least the reading's own sampling standard error. A budget with no terms "
                "composes to zero, which is the one value an uncertainty can never honestly take",
            )
        ]
    if len(terms) == 1:
        out.append(
            BudgetFinding(
                "terms",
                "one term, so the table cannot say which contribution dominates",
                "add the terms the apparatus contributes: the substrate noise floor from M1, the "
                "rung disagreement from M11, the reference certificate, the instrument overhead "
                "from M2. A budget of one term is an interval with a table drawn round it",
            )
        )
    if all(t.kind == "A" for t in terms):
        out.append(
            BudgetFinding(
                "kind",
                "every term is Type A, so nothing in this budget came from judgement, a published "
                "comparison or a stated bound",
                "either the apparatus has no systematic effects, which is worth saying out loud, "
                "or nobody has looked. The engine-to-engine residual and the reference material's "
                "certificate are the two Type B terms almost every reading here has",
            )
        )
    dominant = budget.dominant
    if dominant is not None and dominant.name in set(sampling_terms):
        out.append(
            BudgetFinding(
                "dominant",
                f"the largest term is {dominant.name!r}, which is declared to be sampling noise. "
                f"The largest term is almost never sampling noise",
                "check that a systematic term has not been left out. Sampling noise dominating is "
                "possible and it is the case worth double-checking, because it is also what a "
                "budget looks like when the systematic terms were never measured",
            )
        )
    if len(terms) > 1 and not budget.correlations:
        out.append(
            BudgetFinding(
                "correlations",
                "no correlations are declared, so the composition assumes every pair of terms is "
                "independent",
                "declare the pairs that are not. Two terms measured on the same replicates are "
                "correlated, and silently assuming independence is the commonest way a budget "
                "understates itself",
            )
        )
    if budget.effective_dof() is None:
        missing = sorted(t.name for t in terms if t.dof is None or t.dof <= 0)
        out.append(
            BudgetFinding(
                "dof",
                f"no effective degrees of freedom, because {len(missing)} term(s) carry none "
                f"({', '.join(missing[:3])})",
                "supply `dof` on the Type A terms. Without it the coverage factor is the "
                "large-sample one, which is too small whenever a term came from a handful of "
                "replicates, and substituting infinity would narrow the interval silently",
            )
        )
    return out


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def compose(
    *,
    sampling: float | None = None,
    sampling_name: str = "sampling",
    sampling_dof: float | None = None,
    transfers: Sequence[Transfer] = (),
    limits: LimitOfDetection | None = None,
    overhead: BudgetTerm | None = None,
    reference: ReferenceMaterial | None = None,
    extra: Sequence[BudgetTerm] = (),
    correlations: Mapping[tuple[str, str], float] | None = None,
    coverage_k: float = 2.0,
) -> UncertaintyBudget:
    """Assemble the budget from the pieces the rest of this package produces.

    Every argument is optional and an omitted one contributes nothing, which is honest only because
    `lint_budget` then names what is missing. A name that would collide is suffixed rather than
    dropped: `UncertaintyBudget` refuses duplicate names because two terms under one name compose
    into a number nobody can attribute, and two ladders both contributing a `t21` is a real case.
    """
    terms: list[BudgetTerm] = []
    if sampling is not None:
        terms.append(
            BudgetTerm(
                name=sampling_name,
                value=abs(float(sampling)),
                kind="A",
                dof=sampling_dof,
                note="the reading's own sampling standard error",
            )
        )
    terms.extend(t.as_term() for t in transfers)
    if limits is not None:
        terms.append(limits.as_term())
    if overhead is not None:
        terms.append(overhead)
    if reference is not None:
        terms.extend(reference.as_terms())
    terms.extend(extra)

    seen: dict[str, int] = {}
    unique: list[BudgetTerm] = []
    for term in terms:
        count = seen.get(term.name, 0)
        seen[term.name] = count + 1
        if count:
            term = BudgetTerm(
                name=f"{term.name}#{count + 1}",
                value=term.value,
                kind=term.kind,
                distribution=term.distribution,
                sensitivity=term.sensitivity,
                dof=term.dof,
                note=term.note or f"second contribution named {term.name}",
            )
        unique.append(term)
    return UncertaintyBudget(
        terms=tuple(unique), coverage_k=coverage_k, correlations=dict(correlations or {})
    )


# ---------------------------------------------------------------------------
# The reading
# ---------------------------------------------------------------------------


@register_payload
@dataclass
class BudgetAudit:
    """The composed table, the dominant term named, and what the lint found missing."""

    quantity: str
    combined: float
    expanded: float
    coverage_k: float
    dominant_term: str
    dominant_share: float
    n_terms: int
    type_b_terms: int
    effective_dof: float | None
    shares: Mapping[str, float]
    table: str
    findings: tuple[str, ...]
    value: float | None = None
    baselines: Mapping[str, float] = field(default_factory=dict)

    @property
    def relative(self) -> float | None:
        """The combined uncertainty as a fraction of the reading, where a reading was supplied."""
        if self.value is None or self.value == 0:
            return None
        return self.combined / abs(self.value)

    def says(self) -> str:
        head = (
            f"Combined standard uncertainty {self.combined:.4g} on {self.quantity}, expanded to "
            f"{self.expanded:.4g} at k = {self.coverage_k:g}. The largest of the {self.n_terms} "
            f"terms is {self.dominant_term!r} at {self.dominant_share:.0%} of the variance."
        )
        if self.relative is not None:
            head += f" That is {self.relative:.1%} of the reading itself."
        if self.effective_dof is not None:
            head += f" Effective degrees of freedom {self.effective_dof:.1f}."
        if self.findings:
            head += f" {len(self.findings)} thing(s) the lint says are missing."
        return head

    def render(self) -> str:
        lines = [self.says(), "", self.table]
        if self.findings:
            lines.append("")
            lines.extend(f"    {f}" for f in self.findings)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# The instrument
# ---------------------------------------------------------------------------


class UncertaintyBudgetReading(MetaInstrument):
    """M7. The GUM table as a reading, composed, linted, with the largest term named.

    ``quantity`` is set per instance from the subject the budget belongs to. Constructing one with
    neither a `quantity_id` nor a subject reading leaves it undeclared, which `lint_instrument`
    reports and `compute` refuses on, and both are correct: an unattributed budget is a table of
    numbers in no unit.
    """

    name = "UncertaintyBudget"
    version = "1.0"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    requires = BUDGET_ACCESS
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
    phases = frozenset({Phase.PRE_RUN, Phase.IN_RUN, Phase.POST_RUN, Phase.DEPLOYED})
    envelope = BUDGET_ENVELOPE
    invariance = "units"
    invariance_relation = INVARIANT
    baselines = BUDGET_BASELINES
    rung = 0
    faithful_to = "M7"
    deviations = (
        "the catalogue leaves M7's quantity list OPEN and this instrument takes the quantity of the "
        "reading it decomposes rather than declaring one of its own. A combined standard "
        "uncertainty is in the units of its reading, so a budget has no measurand separate from the "
        "one it is a budget for",
        "the composition drops correlation terms unless the caller declares them, which is the "
        "GUM's law of propagation with an assumption in it. The lint names the assumption on every "
        "reading that makes it rather than leaving it in the arithmetic",
    )

    def __init__(
        self,
        budget: UncertaintyBudget | None = None,
        *,
        quantity_id: str = "",
        subject: Any = None,
        value: float | None = None,
        sampling_terms: Sequence[str] = (),
    ) -> None:
        self.budget = budget
        self.subject = subject
        self.sampling_terms = tuple(sampling_terms)
        resolved = quantity_id or str(getattr(subject, "quantity", "") or "")
        #: Shadows the class attribute on purpose: the quantity is the subject's, per instance.
        self.quantity = resolved
        if value is None and subject is not None:
            candidate = getattr(subject, "value", None)
            value = float(candidate) if isinstance(candidate, (int, float)) else None
        self.value = value

    def compute(self) -> Any:
        if not self.quantity:
            return refuse_incomplete(
                self.name,
                field="a quantity",
                subject="the budget's subject",
                remedy=(
                    "pass `quantity_id=` naming the registered id this budget is the uncertainty "
                    "of, or pass `subject=` the Evidence it belongs to and the quantity is read off "
                    "it. A combined standard uncertainty is in the units of its own reading, so a "
                    "budget with no subject is a table of numbers in no unit."
                ),
            )
        if self.quantity not in QUANTITIES:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.UNIT_MISMATCH,
                detail=(
                    f"the budget is declared for {self.quantity!r}, which spec/QUANTITIES.yaml does "
                    f"not carry, so the units its terms are supposed to share are undefined"
                ),
                remedy=(
                    "name a registered quantity, or register this one. Every term in a budget is a "
                    "contribution in the reading's own units, and an id that resolves to nothing "
                    "cannot say what those are."
                ),
                statistics={"quantity": self.quantity},
            )
        budget = self.budget
        if budget is None or not budget.terms:
            return refuse_incomplete(
                self.name,
                field="any uncertainty term",
                subject=f"the budget for {self.quantity}",
                remedy=(
                    "compose one with `compose(sampling=..., transfers=..., limits=..., "
                    "reference=...)`. An empty budget composes to zero, and zero is the one value "
                    "an uncertainty can never honestly take, so this refuses rather than reporting "
                    "a perfectly certain reading."
                ),
                n_terms=0 if budget is None else len(budget.terms),
            )

        findings = lint_budget(budget, sampling_terms=self.sampling_terms)
        shares = budget.shares()
        dominant = budget.dominant
        assert dominant is not None  # non-empty terms, checked above
        return BudgetAudit(
            quantity=self.quantity,
            combined=budget.combined,
            expanded=budget.expanded,
            # The factor that was applied, not the field that would have been applied before the
            # budget learned to read its own degrees of freedom. `coverage_k` is now the value used
            # only when there are none; with a small `nu_eff` the applied factor is the Student t
            # quantile and can be several times larger. Reporting the field beside a t-expanded
            # number printed "k = 2" next to an interval expanded by 12.7, which was latent only
            # because this reading's own `nu_eff` is 773.
            coverage_k=budget.coverage_factor,
            dominant_term=dominant.name,
            dominant_share=shares.get(dominant.name, float("nan")),
            n_terms=len(budget.terms),
            type_b_terms=sum(1 for t in budget.terms if t.kind == "B"),
            effective_dof=budget.effective_dof(),
            shares=shares,
            table=budget.render(),
            findings=tuple(f.render() for f in findings),
            value=self.value,
            baselines={
                # What the interval would have been with only the terms the caller declared to be
                # sampling noise, which is what a paper reporting a bootstrap interval has.
                "baseline.sampling_only": math.sqrt(
                    sum(t.variance for t in budget.terms if t.name in set(self.sampling_terms))
                ),
                # The largest single contribution, which is what a budget collapses to when the
                # other terms are negligible and is the number to compare the combined one against.
                "baseline.largest_single_term": abs(dominant.contribution),
            },
        )

    def as_uncertainty(self, computed: BudgetAudit) -> Uncertainty:
        """The interval derived from the table, centred on zero, for the reading this belongs to.

        Separate from `uncertainty` on purpose. This instrument's own reading is a budget, and
        wrapping a combined standard uncertainty in a second interval of its own would be a claim
        nothing in the table supports. What the caller wants is this: the interval for the *subject*
        reading, derived from the table so the two cannot disagree.
        """
        assert self.budget is not None
        return Uncertainty.from_budget(self.budget, n=None, method="gum")

    def payload(self, computed: BudgetAudit) -> dict[str, Any]:
        return {
            "quantity": computed.quantity,
            "combined": computed.combined,
            "expanded": computed.expanded,
            "coverage_k": computed.coverage_k,
            "dominant_term": computed.dominant_term,
            "dominant_share": computed.dominant_share,
            "n_terms": computed.n_terms,
            "type_b_terms": computed.type_b_terms,
            "effective_dof": computed.effective_dof,
            "shares": dict(computed.shares),
            "relative": computed.relative,
            "value": computed.value,
            "table": computed.table,
            "findings": list(computed.findings),
            "baselines": dict(computed.baselines),
            "says": computed.says(),
        }


__all__ = [
    "BUDGET_ACCESS",
    "BUDGET_BASELINES",
    "BUDGET_ENVELOPE",
    "BudgetAudit",
    "BudgetFinding",
    "UncertaintyBudgetReading",
    "compose",
    "lint_budget",
]

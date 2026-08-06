"""Reference materials, and the chain that makes a production reading mean something.

Every prior version of this project assumed the answer key was right. It is not, and this module is
the fix.

An analytical laboratory does not calibrate against "a sample someone prepared". It calibrates
against a **certified reference material**, which ships with an assigned value *and an uncertainty
on that value*, decomposed three ways:

    u_CRM² = u_char² + u_bb² + u_stab²
             │         │        └─ stability: does the assigned value drift?
             │         └────────── between-bottle homogeneity: do two preparations agree?
             └──────────────────── characterisation: how well was the value established?

Each has an exact analogue here and none of the three is currently measured anywhere in this
field. `u_char` is how well the planted rule's strength is known, and a LoRA plant at "dose
ρ = 0.75" is a nominal dose rather than a measured one. `u_bb` is whether two plants with different
seeds give the same answer, and the Model Organism Lottery result *is* an uncharacterised
homogeneity term. `u_stab` is whether the plant drifts as the host is finetuned further, which
nobody has measured for any organism.

**The rule that earns this module its place is one line and it is enforced in `compute_trust`
rather than documented:** `u_homogeneity is None` is not a missing field. It renders in every
downstream reading as "reference uncertainty not characterised" and it caps the trust ladder at
`CALIBRATED`. That single rule would have changed how the `CAL-TRANSFER` result reads.

The same treatment applies to labelled corpora, which is where it bites hardest. AISI's
`reward_hacked` column is the best available ground truth for reward hacking and it is a label
produced by a process whose error rate nobody has published. Outcome-propagated step labels are
wrong roughly a third of the time. tau2-bench's zeros are 36% false-success and its ones are 27 to
78% corrupt-success. A corpus is a reference material and it needs a certificate like any other.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from reward_lens.core.budget import BudgetTerm, UncertaintyBudget
from reward_lens.core.types import TrustLevel

#: Matches `reward_lens.core.quantity.ReferenceID`; aliased locally so this module has no import
#: dependency on the catalogue loader.
ReferenceID = str

ReferenceKind = Literal[
    "planted_organism",
    "labelled_corpus",
    "synthetic_ground_truth",
    "held_out_eval",
]


@dataclass(frozen=True)
class MatrixDescription:
    """What system the reference lives in, because a calibration is only valid in its own matrix.

    The metrological name for the library's worst-looking published result is a **matrix effect**:
    the calibration was performed in a clean 0.6B organism and the sample was a real 8B model.
    Recording the matrix is what makes that diagnosable rather than mysterious, and comparing two
    matrices is what `mismatch_with` is for.

    The textbook fix is standard addition: dose the graded structure into the *target* system
    rather than a separate clean one, which removes the mismatch by construction.
    """

    system: str
    scale: str = ""
    note: str = ""

    def mismatch_with(self, other: "MatrixDescription") -> str:
        """A one-line description of how far this matrix is from another, or empty if they match."""
        if self.system == other.system and self.scale == other.scale:
            return ""
        return f"calibrated in {self.render()}, applied to {other.render()}"

    def render(self) -> str:
        return f"{self.system} ({self.scale})" if self.scale else self.system

    def __canonical__(self) -> dict[str, Any]:
        return {"system": self.system, "scale": self.scale, "note": self.note}


@dataclass(frozen=True)
class ReferenceMaterial:
    """An answer key with an uncertainty on the answer.

    Three of the four uncertainty fields may be absent, and absence is a reported state rather than
    a zero. `uncharacterised` names which are missing and `trust_cap` turns that into the ladder
    cap, so an instrument cannot claim more than the reference supports by not looking.
    """

    id: ReferenceID
    kind: ReferenceKind
    assigned_value: Any
    u_characterisation: float
    matrix: MatrixDescription
    #: None means NOT MEASURED, and that is reported rather than treated as zero.
    u_homogeneity: float | None = None
    u_stability: float | None = None
    provenance: Any = None
    note: str = ""
    #: How many degrees of freedom stand behind each term, where the study that produced it is
    #: known. These are not decoration. Without them `UncertaintyBudget.effective_dof()` is None
    #: for every chain built on this reference, the coverage factor falls back to the large-sample
    #: `k = 2`, and the interval is too narrow by exactly the amount the small sample costs: a
    #: factor of 6.35 at the one residual degree of freedom a three-dose calibration line leaves,
    #: and 1.39 at four. A reference material characterised at the floor is the case where this
    #: matters most and the case where it was silently ignored.
    dof_characterisation: float | None = None
    dof_homogeneity: float | None = None
    dof_stability: float | None = None

    def __post_init__(self) -> None:
        if self.u_characterisation < 0:
            raise ValueError("u_characterisation is a standard uncertainty and cannot be negative")
        for name in ("u_homogeneity", "u_stability"):
            v = getattr(self, name)
            if v is not None and v < 0:
                raise ValueError(f"{name} is a standard uncertainty and cannot be negative")
        for name in ("dof_characterisation", "dof_homogeneity", "dof_stability"):
            v = getattr(self, name)
            if v is not None and (not math.isfinite(v) or v <= 0):
                raise ValueError(
                    f"{name} = {v}. Degrees of freedom are a positive count of independent "
                    f"residuals; pass None where the study behind the term is not known rather "
                    f"than zero, so the budget reports no effective dof instead of dividing by it."
                )

    @property
    def uncharacterised(self) -> tuple[str, ...]:
        """Which components of `u_CRM` were never measured."""
        missing = []
        if self.u_homogeneity is None:
            missing.append("u_homogeneity")
        if self.u_stability is None:
            missing.append("u_stability")
        return tuple(missing)

    @property
    def is_certified(self) -> bool:
        """Whether all three components exist. Only a certified reference lifts the trust cap."""
        return not self.uncharacterised

    @property
    def u_crm(self) -> float | None:
        """`sqrt(u_char² + u_bb² + u_stab²)`, or None when a component was never measured.

        None rather than a partial sum, deliberately. Summing the terms that happen to exist and
        presenting the result as `u_CRM` is exactly the understatement this module exists to stop:
        it makes an uncharacterised reference look better than a characterised one with a large
        homogeneity term.

        **The sum is a sum of independent components and the supplier of the three terms has to
        make it one.** ISO Guide 35 splits the certificate this way because `u_char` is the
        uncertainty of *where the assigned value sits*, estimated from the characterisation study,
        and `u_bb` is how far the individual unit in front of you departs from it. Written against
        a calibration line the two are the two halves of one inverse-prediction variance,

            u_CRM² = (s/|b|)² · [ 1/n + (x₀ - x̄)²/S_xx ]   +   (s/|b|)² · 1
                     └─────────── u_char ───────────┘           └─ u_bb ─┘

        so `u_char` is a standard error of the fitted line and `u_bb` is a standard deviation of
        one unit about it. A supplier who passes the raw residual scatter `s/|b|` as `u_char` has
        handed over the `1` twice: the composed `u_CRM` then reads `sqrt(2)·(s/|b|)` where the
        correct value is `sqrt(1 + 1/n)·(s/|b|)`, which is 1.37 times too large at fifteen plants
        and 1.22 times too large at the three-plant floor. This type cannot detect that from the
        numbers it is given, which is why it is written down here.
        """
        if not self.is_certified:
            return None
        return math.sqrt(
            self.u_characterisation**2
            + float(self.u_homogeneity or 0.0) ** 2
            + float(self.u_stability or 0.0) ** 2
        )

    @property
    def u_crm_lower_bound(self) -> float:
        """The best available bound on `u_CRM` when it is not certified: the terms that do exist.

        Explicitly a *lower* bound, and named as one. Useful for saying "at least this bad", never
        for reporting as the uncertainty.
        """
        total = self.u_characterisation**2
        for v in (self.u_homogeneity, self.u_stability):
            if v is not None:
                total += float(v) ** 2
        return math.sqrt(total)

    def trust_cap(self) -> TrustLevel:
        """The highest trust a reading calibrated against this reference may claim.

        An uncertified reference caps at `CALIBRATED`. Note what that means and that it is
        deliberate: a preregistered result calibrated against an uncharacterised organism does not
        reach `REGISTERED`, because freezing a prediction against a ruler of unknown length does
        not make the reading better, it makes the prediction precise about something unmeasured.
        """
        return TrustLevel.ADJUDICATED if self.is_certified else TrustLevel.CALIBRATED

    def status_line(self) -> str:
        """The sentence that appears in every downstream reading."""
        if self.is_certified:
            return f"reference {self.id}: u_CRM = {self.u_crm:.4g} in {self.matrix.render()}"
        missing = " and ".join(self.uncharacterised)
        return (
            f"reference {self.id}: reference uncertainty not characterised ({missing} not "
            f"measured); trust capped at CALIBRATED"
        )

    def as_terms(self) -> tuple[BudgetTerm, ...]:
        """The reference's contributions as budget terms, so they compose with everything else.

        Each term carries its degrees of freedom where the reference knows them, because that is
        what lets the budget compute an effective dof and therefore a coverage factor that is not
        just 2. A reference certified at the three-dose floor contributes a term with one degree of
        freedom, and one degree of freedom is the difference between an interval half-width of
        `2u` and one of `12.7u`.
        """
        terms = [
            BudgetTerm(
                name="u_char",
                value=self.u_characterisation,
                kind="B",
                dof=self.dof_characterisation,
                note=f"characterisation of {self.id}",
            )
        ]
        if self.u_homogeneity is not None:
            terms.append(
                BudgetTerm(
                    name="u_bb",
                    value=self.u_homogeneity,
                    kind="A",
                    dof=self.dof_homogeneity,
                    note="between-seed homogeneity",
                )
            )
        if self.u_stability is not None:
            terms.append(
                BudgetTerm(
                    name="u_stab",
                    value=self.u_stability,
                    kind="A",
                    dof=self.dof_stability,
                    note="stability under further training",
                )
            )
        return tuple(terms)

    def __canonical__(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "u_characterisation": self.u_characterisation,
            "u_homogeneity": self.u_homogeneity,
            "u_stability": self.u_stability,
            "dof_characterisation": self.dof_characterisation,
            "dof_homogeneity": self.dof_homogeneity,
            "dof_stability": self.dof_stability,
            "matrix": self.matrix.__canonical__(),
            "note": self.note,
        }


# ---------------------------------------------------------------------------
# The chain
# ---------------------------------------------------------------------------

#: The three rungs of the calibration chain, cheapest last.
ChainLevel = Literal["primary", "reference_method", "working_method"]


@dataclass(frozen=True)
class Transfer:
    """The disagreement between two rungs of the calibration chain, published as a quantity.

    This is `t₂₁` and `t₃₂` of the chain, and the reason the type exists is that the library's
    worst-looking published result, a maximum absolute AUC difference of 0.419 against a registered
    0.15, is a transfer coefficient. Publishing it as a caveat was the mistake; publishing it as a
    quantity with a method is the service.

    **Correction: 0.419 is not the planted-to-real coefficient it was described as here.**
    Recomputing it from its own parent row (`campaign.organism.scorecard`, reproducing
    0.41898333 to 1e-9 before saying anything) shows both arms are planted: `cpu_auc` is a CPU
    rehearsal and `real_auc` is **the same planted organism** scored by a real reward model, over
    one organism family at three doses. `n_natural_corpora = 0`. So 0.419 is a
    **simulation-to-real-model** transfer, and the planted-to-real coefficient this docstring
    claimed had never been measured at all.

    The measured one, against AISI's `reward_hacked` over 25,664 rollouts, is `t₃₂ = 0.4732
    [0.4543, 0.4933]`. But the number that matters is not that it lands near 0.419: **the
    coefficient depends on the organism design, and the spread is 0.4528.** Under `append`, where
    the hack is added to a working solution, it is 0.47 and almost all of it is `baseline.length`
    inverting. Under `substitute`, where the hack replaces the solution, which is what the policy
    actually did, it is 0.02 and everything transfers. **A transfer coefficient quoted without its
    organism design is not yet a measurement**, and both are published for that reason.

    A transfer also falls out for free anywhere two rungs of an estimator ladder both ran on the
    same data: the difference between the cheap rung and the expensive one is the cheap rung's
    transfer uncertainty, and nobody publishes it.
    """

    from_level: ChainLevel
    to_level: ChainLevel
    value: float
    method: str = ""
    n: int | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    note: str = ""
    evidence: str | None = None

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError(
                "a transfer coefficient is a magnitude of disagreement and cannot be negative; "
                "take the absolute difference."
            )

    @property
    def name(self) -> str:
        """`t32` or `t21`, named by the pair of rungs rather than by the declaration order.

        The chain runs downward, primary to working, and names the transfers `t32` and `t21`:
        the higher rung first, always. Deriving the name from declaration direction instead
        gave the same physical transfer two names, `t32` when written one way and `t23` when written
        the other, and a budget with both spellings would have double-counted it.
        """
        rung = {"primary": 3, "reference_method": 2, "working_method": 1}
        hi, lo = sorted((rung[self.from_level], rung[self.to_level]), reverse=True)
        return f"t{hi}{lo}"

    def as_term(self) -> BudgetTerm:
        return BudgetTerm(
            name=self.name,
            value=self.value,
            kind="A",
            note=self.method or f"{self.from_level} -> {self.to_level}",
            dof=(self.n - 1) if self.n and self.n > 1 else None,
        )

    def render(self) -> str:
        ci = (
            f" [{self.ci_low:.4g}, {self.ci_high:.4g}]"
            if self.ci_low is not None and self.ci_high is not None
            else ""
        )
        return f"{self.name} = {self.value:.4g}{ci}  ({self.from_level} -> {self.to_level})"


@dataclass(frozen=True)
class CalibrationChain:
    """`u_total² = u₁² + t₂₁² + t₃₂² + u_CRM² + u_instrument²`.

    The composition every production reading needs and none currently carries. `u₁` is the working
    method's own uncertainty, the transfers are what each step of the chain costs, `u_CRM` is the
    reference's certificate, and `u_instrument` is the substrate noise floor.

    An uncertified reference does not make this un-composable: the chain still reports, with
    `u_CRM` named as the missing term and the trust capped. What it must not do is quietly drop the
    term and present a smaller total, which is the failure mode.
    """

    reference: ReferenceMaterial
    transfers: tuple[Transfer, ...] = ()
    u_working: float = 0.0
    u_instrument: float = 0.0
    working_matrix: MatrixDescription | None = None
    #: Degrees of freedom behind the two scalar terms, where the caller knows them. `Transfer`
    #: carries its own through `n`, and the reference carries all three of its own. Absent here is
    #: the same statement it is everywhere else: no effective dof for the table, so the coverage
    #: factor stays at the large-sample value and `lint_budget` says which terms are responsible.
    dof_working: float | None = None
    dof_instrument: float | None = None

    @property
    def is_certified(self) -> bool:
        return self.reference.is_certified

    def matrix_mismatch(self) -> str:
        """Whether the reading is being taken in a different matrix from the calibration."""
        if self.working_matrix is None:
            return ""
        return self.reference.matrix.mismatch_with(self.working_matrix)

    def as_budget(
        self, coverage_k: float = 2.0, *, coverage_level: float = 0.95
    ) -> UncertaintyBudget:
        """The chain as a GUM table, so the dominant term is visible rather than buried in a total.

        ``coverage_k`` is the fallback factor for a table with no degrees of freedom anywhere in
        it. When the reference carries them the budget uses `t_p(nu_eff)` instead, which is the
        point of plumbing them through `as_terms`.
        """
        terms: list[BudgetTerm] = []
        if self.u_working:
            terms.append(
                BudgetTerm(
                    name="u_working",
                    value=self.u_working,
                    kind="A",
                    dof=self.dof_working,
                    note="the shipped estimator",
                )
            )
        terms.extend(t.as_term() for t in self.transfers)
        terms.extend(self.reference.as_terms())
        if self.u_instrument:
            terms.append(
                BudgetTerm(
                    name="u_instrument",
                    value=self.u_instrument,
                    kind="B",
                    dof=self.dof_instrument,
                    note="substrate noise floor",
                )
            )
        return UncertaintyBudget(
            terms=tuple(terms), coverage_k=coverage_k, coverage_level=coverage_level
        )

    @property
    def u_total(self) -> float | None:
        """The composed total, or None when the reference is uncertified.

        None rather than a partial total, for the same reason `u_crm` is None: a chain missing a
        term does not have a smaller uncertainty, it has an unknown one. `u_total_lower_bound`
        gives the honest "at least this" number for a reader who needs something to print.
        """
        return self.as_budget().combined if self.is_certified else None

    @property
    def u_total_lower_bound(self) -> float:
        return self.as_budget().combined

    def trust_cap(self) -> TrustLevel:
        return self.reference.trust_cap()

    def render(self) -> str:
        lines = [self.reference.status_line()]
        for t in self.transfers:
            lines.append(f"  {t.render()}")
        mismatch = self.matrix_mismatch()
        if mismatch:
            lines.append(f"  matrix effect: {mismatch}")
        total = self.u_total
        if total is None:
            lines.append(
                f"  u_total: not computable; u_CRM is uncertified. At least "
                f"{self.u_total_lower_bound:.4g} from the terms that were measured."
            )
        else:
            lines.append(f"  u_total = {total:.4g}")
        return "\n".join(lines)


def ladder_disagreement(
    cheap: float,
    expensive: float,
    *,
    from_level: ChainLevel = "working_method",
    to_level: ChainLevel = "reference_method",
    n: int | None = None,
    method: str = "",
) -> Transfer:
    """In one call: two rungs disagreeing on the same data is the cheap rung's transfer term.

    This falls out of the estimator ladder for free and nobody publishes it, so the only work is
    remembering to record it. Making it one call is what makes it get recorded.
    """
    return Transfer(
        from_level=from_level,
        to_level=to_level,
        value=abs(cheap - expensive),
        method=method or "rung disagreement on identical data",
        n=n,
        note=f"cheap rung {cheap:.6g}, expensive rung {expensive:.6g}",
    )


def uncertified_refusal(instrument: str, reference: ReferenceMaterial) -> Any:
    """The refusal for calibrating against a reference that carries no uncertainty of its own."""
    from reward_lens.core.reading import Refusal, RefusalReason

    missing = " and ".join(reference.uncharacterised) or "u_characterisation"
    return Refusal(
        instrument=instrument,
        reason=RefusalReason.REFERENCE_UNCERTIFIED,
        detail=(
            f"reference {reference.id} ({reference.kind}) has no measured {missing}, so it carries "
            f"no uncertainty of its own and cannot certify a reading"
        ),
        remedy=(
            "plant the organism with at least three seeds and report the between-seed spread as "
            "u_homogeneity, or declare the reading exploratory and say the reference is "
            "uncharacterised. You cannot calibrate against an uncalibrated ruler."
        ),
        statistics={
            "u_characterisation": reference.u_characterisation,
            "u_homogeneity": reference.u_homogeneity,
            "u_stability": reference.u_stability,
        },
    )


__all__ = [
    "CalibrationChain",
    "ChainLevel",
    "MatrixDescription",
    "ReferenceID",
    "ReferenceKind",
    "ReferenceMaterial",
    "Transfer",
    "ladder_disagreement",
    "uncertified_refusal",
]

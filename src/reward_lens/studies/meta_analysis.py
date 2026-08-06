"""The aggregation plan: a count over studies is a claim, and it needs freezing too.

Every study in a campaign can be preregistered and the sentence that adds them up can still be
made after the fact. That sentence is where the freedom is: which studies count, what counts as a
confirmation, whether a study that produced three predictions contributes one tally mark or three.
None of that is fixed by freezing the individual predictions, and all of it moves the headline.

This is not a hypothetical failure mode. The flagship "rigour works" paper in psychology, four
labs and sixteen novel findings, was retracted on 2024-09-24 for absent preregistration of the
measures and analyses supporting its titular claim, and for selection of outcome measures with
knowledge of the data. The individual studies were fine. The claim about them was not, and the
paper's own subject was rigour.

This library's evidence base has the same shape. The 2.0 campaign, whose runs and evidence store
live with the campaign rather than in this repository, froze 27 study specs on 2026-07-18 and every
spec hash still verifies. Its published summary reads: "19 of 27 frozen cards adjudicated against
the merged evidence store; 16 of 53 frozen hypotheses confirmed, 21 refuted, 16 inconclusive; 8 kill
criteria fired". Re-derived from the campaign's own evidence store those numbers reproduce exactly.
Every one of the 27 adjudication rows carries ``trust: 2``, which is ``TrustLevel.REGISTERED``.
The sentence that adds them up carries nothing, because there was no object for it to be registered
against. The frozen-spec manifest has eight keys and none of them is an aggregation rule.

**Why this labels rather than raises or refuses.** Plan closure raises, because nothing has run yet
and the useful behaviour is to stop before spending money (see `reward_lens.core.closure`). An
instrument refuses, because a caller who gets a `Refusal` still has a run to look at. Neither fits
here. By the time anyone is counting, the money is spent and the numbers are real, and the reader
of the summary is better served by seeing the count than by not seeing it. What they need is to be
told, in the same line, that the count was assembled after the data arrived. So the count prints,
with `TrustLevel.EXPLORATORY` printed beside it, and the label names what is missing and what
would remove it. Suppressing the number would hide the finding; blocking the document would hide
it from the only person who can act on it.

The trust ladder already had the right word. `TrustLevel.EXPLORATORY` is "anything computed ad hoc"
and `REGISTERED` is "computed under a frozen Study whose predictions predate the run". An aggregate
over registered evidence is not itself registered, and this module is what makes that difference
visible in the one place it matters.

Two further honesty items live here because they are properties of a family rather than of any one
study. Multiplicity: 26 cards each carrying several hypotheses is a large family, and the campaign
corrected within cards only, which it disclosed as inert because no card carried two p-values.
Across cards nothing corrected anything. Null-boundary sensitivity: a verdict produced by comparing
a float to a frozen float can sit close enough to the threshold that a different estimator would
have flipped it, and the reader should be told which ones do rather than left to check 37 rows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable, Sequence

from reward_lens.core.provenance import git_sha
from reward_lens.core.types import TrustLevel, content_hash
from reward_lens.studies.spec import Comparator

# ---------------------------------------------------------------------------
# Why a count is exploratory
# ---------------------------------------------------------------------------


class ExploratoryReason(Enum):
    """The seven ways a campaign-level count fails to be covered by a frozen plan.

    Kept separate rather than collapsed into "not registered" because the remedies differ and the
    remedy is the useful part. "There is no plan" and "there is a plan and it does not include the
    studies you counted" are the same badge and different work.
    """

    #: Nobody froze an aggregation rule. The default state, and the campaign's.
    NO_PLAN = "no_plan"

    #: The plan's content does not hash to the hash it carries, so it was edited after freezing.
    HASH_MISMATCH = "hash_mismatch"

    #: The plan was frozen after the evidence it aggregates. A plan written later is a description.
    PLAN_POSTDATES_EVIDENCE = "plan_postdates_evidence"

    #: The plan states its inclusion rule in prose only, so no program can check what was counted.
    INCLUSION_UNCHECKABLE = "inclusion_uncheckable"

    #: The count ranged over studies the plan does not include, or dropped studies it does.
    OUTSIDE_INCLUSION = "outside_inclusion"

    #: The plan aggregates one unit (studies, say) and the count counts another (hypotheses).
    UNIT_UNDECLARED = "unit_undeclared"

    #: The count uses an outcome label the plan's scoring rule never declared.
    LABEL_UNDECLARED = "label_undeclared"


#: What to do about each reason. Written as instructions to whoever is holding the summary, in the
#: same spirit as `reward_lens.studies.void.DEFAULT_REMEDY`: a badge that says "not registered" and
#: stops has told the reader nothing they can act on.
DEFAULT_REMEDY: dict[ExploratoryReason, str] = {
    ExploratoryReason.NO_PLAN: (
        "Freeze a MetaAnalysisPlan before the next adjudication: `aggregation` for how the count "
        "is computed, `inclusion` for which studies are in it with their ids in `studies`, and "
        "`scoring` for how one outcome becomes one tally mark. The count keeps this label until a "
        "plan that predates the evidence covers it."
    ),
    ExploratoryReason.HASH_MISMATCH: (
        "This plan's fields do not hash to its recorded spec_hash, so it changed after it was "
        "frozen. Restore the frozen content, or freeze the edited plan as a new one and accept "
        "that its frozen_at is now later than the evidence."
    ),
    ExploratoryReason.PLAN_POSTDATES_EVIDENCE: (
        "Freeze the aggregation rule before running the studies it aggregates. A plan written "
        "after the adjudication describes the count rather than committing to it, and there is no "
        "way to tell those apart from the artifact."
    ),
    ExploratoryReason.INCLUSION_UNCHECKABLE: (
        "List the study ids in the plan's `studies` field. The prose inclusion rule is what a "
        "reader needs and it is not something a program can check a count against, so a plan "
        "carrying only prose cannot cover anything."
    ),
    ExploratoryReason.OUTSIDE_INCLUSION: (
        "Count over exactly the studies the plan includes. If the set genuinely changed, say why "
        "in the summary and freeze a new plan; silently dropping a study from a count is the "
        "selection this object exists to make visible."
    ),
    ExploratoryReason.UNIT_UNDECLARED: (
        "Make the count's unit match the plan's. A plan that aggregates studies does not cover a "
        "count of hypotheses, because one study contributing three hypotheses changes the total."
    ),
    ExploratoryReason.LABEL_UNDECLARED: (
        "Add this outcome label to the plan's `labels`, or count under a label the plan declares. "
        "A bucket invented after the data is a scoring choice made with knowledge of the results."
    ),
}


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetaAnalysisPlan:
    """The aggregation rule, frozen before the studies it aggregates run.

    The first five fields are prose, written for a reader. They say what the count is, which
    studies are in it, and how an outcome becomes a tally mark. Prose is what makes the plan
    reviewable and it is not something `cover` can check a count against, so `studies`, `unit` and
    `labels` carry the same three commitments in a form a program can compare. They are not a
    second opinion: the hash covers both, so a plan whose prose and whose machine-readable fields
    disagree is a plan that says two things and a reviewer can see it.

    `frozen_at` is inside the hash. That costs a little (the timestamp has to be fixed before the
    hash can be published) and it buys the property the object exists for: a plan cannot be
    backdated without changing its own identity.
    """

    aggregation: str
    inclusion: str
    scoring: str
    frozen_at: str
    spec_hash: str
    id: str = ""
    studies: tuple[str, ...] = ()
    unit: str = "hypothesis"
    labels: tuple[str, ...] = ()
    alpha: float | None = None
    multiplicity: str = ""
    git_sha: str = ""
    notes: str = ""

    def __canonical__(self) -> dict[str, Any]:
        """Everything the hash covers. `spec_hash` is excluded because it is the hash.

        `git_sha` is excluded too: the same plan frozen from two checkouts of the same content is
        the same plan, and the sha is recorded beside the hash rather than inside it so a dirty
        tree stays visible without changing the plan's identity.
        """
        return {
            "aggregation": self.aggregation,
            "inclusion": self.inclusion,
            "scoring": self.scoring,
            "frozen_at": self.frozen_at,
            "id": self.id,
            "studies": sorted(self.studies),
            "unit": self.unit,
            "labels": sorted(self.labels),
            "alpha": self.alpha,
            "multiplicity": self.multiplicity,
        }

    @property
    def hash_verified(self) -> bool:
        """Whether the plan's fields still hash to the hash it carries.

        False means the content changed after the freeze. That is the edit-after-the-data attack
        this object exists to make visible, and it is why the check recomputes rather than trusting
        the stored value.
        """
        return content_hash(self.__canonical__(), "meta") == self.spec_hash

    @property
    def short_id(self) -> str:
        digest = self.spec_hash.split(":")[-1][:8]
        return f"meta:{self.id}#{digest}" if self.id else f"meta:#{digest}"


def freeze_meta_plan(
    aggregation: str,
    inclusion: str,
    scoring: str,
    *,
    id: str = "",
    studies: Sequence[str] = (),
    unit: str = "hypothesis",
    labels: Sequence[str] = (),
    alpha: float | None = None,
    multiplicity: str = "",
    notes: str = "",
    frozen_at: str | None = None,
    repo_dir: str | None = None,
) -> MetaAnalysisPlan:
    """Freeze an aggregation rule, computing its hash and recording the git sha.

    The two-step shape (build the canonical form, hash it, then build the object carrying the hash)
    is the same one `reward_lens.studies.freeze.freeze` uses on a study spec, for the same reason:
    the hash has to be over the content and the content has to be complete before it is taken.
    """
    ts = frozen_at or datetime.now(timezone.utc).isoformat()
    draft = MetaAnalysisPlan(
        aggregation=aggregation,
        inclusion=inclusion,
        scoring=scoring,
        frozen_at=ts,
        spec_hash="",
        id=id,
        studies=tuple(studies),
        unit=unit,
        labels=tuple(labels),
        alpha=alpha,
        multiplicity=multiplicity,
        notes=notes,
    )
    digest = content_hash(draft.__canonical__(), "meta")
    return MetaAnalysisPlan(
        aggregation=aggregation,
        inclusion=inclusion,
        scoring=scoring,
        frozen_at=ts,
        spec_hash=digest,
        id=id,
        studies=tuple(studies),
        unit=unit,
        labels=tuple(labels),
        alpha=alpha,
        multiplicity=multiplicity,
        git_sha=git_sha(repo_dir),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# What gets counted, and the count itself
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdjudicatedPrediction:
    """One registered prediction, with the verdict and the numbers that produced it.

    This is the row a campaign-level count is a sum over, and it carries the threshold and the
    comparator as well as the outcome so the count and the boundary analysis read the same data.
    `value` is `None` for a prediction whose metric was never computed; those are neither
    confirmations nor refutations and are excluded from every tally here.

    `uncertainty` is the half-width of the recorded interval around `value`, when the evidence
    carries one. It is `None` far more often than it should be, and `boundary_sensitivity` says so
    rather than substituting a scale nobody measured.
    """

    study: str
    owner: str
    kind: str
    metric: str
    comparator: Comparator
    threshold: float
    value: float | None
    outcome: str
    p_value: float | None = None
    uncertainty: float | None = None


@dataclass(frozen=True)
class CountClaim:
    """A campaign-level count, as it will be printed.

    `over` is the set of studies the count actually ranged over, which is the field that makes
    selection checkable. A count that says "16 confirmed" without saying 16 out of what, drawn from
    where, cannot be compared against any inclusion rule.

    `evidence_at` is the latest creation timestamp among the evidence counted. It is what turns
    "the plan is frozen" into "the plan was frozen first", and a claim that leaves it empty gets
    the benefit of the doubt on ordering only because there is nothing to check against.
    """

    label: str
    value: int
    unit: str = "hypothesis"
    over: tuple[str, ...] = ()
    statement: str = ""
    evidence_at: str = ""

    def render_bare(self) -> str:
        return self.statement or f"{self.value} {self.label}"


def tally(
    predictions: Iterable[AdjudicatedPrediction],
    label: str,
    *,
    unit: str = "hypothesis",
    statement: str = "",
    evidence_at: str = "",
) -> CountClaim:
    """Count the predictions carrying one outcome label, recording what was counted over."""
    rows = [p for p in predictions if p.outcome == label]
    studies = sorted({p.study for p in predictions})
    return CountClaim(
        label=label,
        value=len(rows),
        unit=unit,
        over=tuple(studies),
        statement=statement,
        evidence_at=evidence_at,
    )


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Coverage:
    """Whether a frozen plan covers a count, and if not, every reason it does not.

    `reasons` is a tuple rather than a first failure because a plan can be wrong in several ways at
    once and fixing one of them does not help. The campaign's count, checked against a plan that
    does not exist, has exactly one reason; a plan built in a hurry usually has three.
    """

    claim: CountClaim
    plan: MetaAnalysisPlan | None = None
    reasons: tuple[tuple[ExploratoryReason, str], ...] = ()

    @property
    def covered(self) -> bool:
        return not self.reasons

    @property
    def trust(self) -> TrustLevel:
        """The count's own trust level, on the same ladder its evidence sits on.

        A count over registered evidence is registered only if the rule that produced the count was
        itself frozen first. Otherwise it is exactly what `TrustLevel.EXPLORATORY` means: computed
        ad hoc.
        """
        return TrustLevel.REGISTERED if self.covered else TrustLevel.EXPLORATORY

    @property
    def badge(self) -> str:
        """The label that goes next to the number, short enough for a table cell."""
        if not self.covered:
            return "[EXPLORATORY]"
        assert self.plan is not None
        return f"[REGISTERED under {self.plan.short_id}]"

    def render(self) -> str:
        """The count with its badge, and the sentence that says what the badge means."""
        head = f"{self.claim.render_bare()}  {self.badge}"
        if self.covered:
            assert self.plan is not None
            return (
                f"{head}\n"
                f"    Counted under the aggregation rule frozen at {self.plan.frozen_at}, before "
                f"the evidence it counts. {self.plan.aggregation}"
            )
        lines = [
            head,
            "    This count was not preregistered. The individual predictions behind it were "
            "frozen before the run; the rule for counting them was not, so the count is "
            "exploratory even where every number inside it is registered.",
        ]
        for reason, detail in self.reasons:
            lines.append(f"    {reason.name}: {detail}")
            lines.append(f"        do: {DEFAULT_REMEDY[reason]}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render()


def cover(claim: CountClaim, plan: MetaAnalysisPlan | None) -> Coverage:
    """Whether a frozen plan covers this count.

    Five things have to hold and each of them is a way the retraction happened: a plan exists, it
    still hashes to what it was frozen as, it predates the evidence, it enumerates the studies the
    count ranged over, and it declares the unit and the label the count is stated in.

    A plan that covers a different set of studies does not cover this count. That is the clause
    doing the real work: without it, freezing any plan at all would launder any count, and "we
    preregistered an analysis plan" would mean nothing more than that a file exists.
    """
    if plan is None:
        return Coverage(
            claim=claim,
            plan=None,
            reasons=(
                (
                    ExploratoryReason.NO_PLAN,
                    f"no MetaAnalysisPlan covers a count of {claim.value} {claim.label} over "
                    f"{len(claim.over)} studies.",
                ),
            ),
        )

    reasons: list[tuple[ExploratoryReason, str]] = []

    if not plan.hash_verified:
        reasons.append(
            (
                ExploratoryReason.HASH_MISMATCH,
                f"the plan's content hashes to "
                f"{content_hash(plan.__canonical__(), 'meta')} and it carries "
                f"{plan.spec_hash}.",
            )
        )

    if claim.evidence_at and plan.frozen_at and plan.frozen_at > claim.evidence_at:
        reasons.append(
            (
                ExploratoryReason.PLAN_POSTDATES_EVIDENCE,
                f"the plan was frozen at {plan.frozen_at} and the newest evidence it counts was "
                f"created at {claim.evidence_at}.",
            )
        )

    if not plan.studies:
        reasons.append(
            (
                ExploratoryReason.INCLUSION_UNCHECKABLE,
                "the plan enumerates no studies, so there is nothing to check the count's "
                f"{len(claim.over)} against. Its inclusion rule reads: {plan.inclusion!r}",
            )
        )
    else:
        counted, included = set(claim.over), set(plan.studies)
        extra, dropped = sorted(counted - included), sorted(included - counted)
        if extra or dropped:
            bits = []
            if extra:
                bits.append(f"counted but not included: {', '.join(extra)}")
            if dropped:
                bits.append(f"included but not counted: {', '.join(dropped)}")
            reasons.append((ExploratoryReason.OUTSIDE_INCLUSION, "; ".join(bits) + "."))

    if claim.unit != plan.unit:
        reasons.append(
            (
                ExploratoryReason.UNIT_UNDECLARED,
                f"the count is stated per {claim.unit} and the plan aggregates per {plan.unit}.",
            )
        )

    if plan.labels and claim.label not in plan.labels:
        reasons.append(
            (
                ExploratoryReason.LABEL_UNDECLARED,
                f"the count is labelled {claim.label!r} and the plan's scoring rule declares "
                f"{', '.join(sorted(plan.labels))}.",
            )
        )

    return Coverage(claim=claim, plan=plan, reasons=tuple(reasons))


def render_count(claim: CountClaim, plan: MetaAnalysisPlan | None) -> str:
    """One count, rendered with its badge and the sentence explaining it. The public entry point."""
    return cover(claim, plan).render()


def render_counts(claims: Sequence[CountClaim], plan: MetaAnalysisPlan | None) -> str:
    """A block of counts for a summary document.

    The explanation is printed once per uncovered count rather than once per document on purpose.
    A reader who skims to the number they care about should meet the label there, not in a
    methods note further down that they will not reach.
    """
    return "\n\n".join(render_count(c, plan) for c in claims)


# ---------------------------------------------------------------------------
# Multiplicity over the family
# ---------------------------------------------------------------------------


def calibrate_p_to_e(p: float, kappa: float = 0.5) -> float:
    """Turn a p-value into an e-value with the standard calibrator ``f(p) = kappa * p ** (kappa-1)``.

    Every member of this family is an admissible calibrator: if p is a valid p-value then f(p) has
    expectation at most 1 under the null, which is what makes it an e-value. The cost is real and
    it is worth stating plainly, because a reader who sees an e-BH column will assume it is free.
    Calibration throws away power, and `kappa` has to be chosen before seeing p; picking the kappa
    that maximises f for the p you observed is not a calibration, it is a second look at the data.

    At small family sizes the loss dominates. e-BH needs an e-value of at least n/alpha to make a
    single rejection, so a family of 4 at alpha 0.05 needs an e-value of 80, and no admissible
    calibrator maps p = 0.001 above 53.3 (the maximum, at kappa = 0.1448). That is not a defect in
    e-BH; it is the price of validity under arbitrary dependence at n = 4, and the right response
    is to report it rather than to quietly use a procedure whose assumptions do not hold.
    """
    if not 0.0 < kappa < 1.0:
        raise ValueError(f"the calibrator is defined for kappa in (0, 1); got {kappa!r}")
    if p <= 0.0:
        return math.inf
    if p > 1.0:
        raise ValueError(f"a p-value cannot exceed 1; got {p!r}")
    # float() because `**` is typed to admit a complex result; the guards above rule that out.
    return float(kappa * p ** (kappa - 1.0))


def e_bh(e_values: Sequence[float], alpha: float = 0.05) -> tuple[tuple[bool, ...], int]:
    """The e-BH procedure. Returns the rejection mask and the number rejected.

    Sort the e-values downward and reject the largest k, where k is the largest index for which the
    k-th largest e-value is at least ``n / (alpha * k)``. Unlike Benjamini-Hochberg on p-values,
    this controls the false discovery rate under arbitrary dependence between the tests, which is
    the regime a campaign is actually in: cards share models, share prompt banks, and share the
    people who wrote them, and no positive-dependence assumption covers that.

    Non-finite e-values are treated as zero rather than dropped. An undefined test is not evidence
    against its null, and dropping it would shrink n and so loosen the threshold for everything
    else, which is the wrong direction.

    `reward_lens.stats.sequential.ebh` is the same procedure, written for the monitor's ledger of
    alarms and returning an `EBHResult`. This copy exists so that rendering a count does not pull
    `reward_lens.stats` and the second of scipy and sklearn behind it, and the two are held to
    agree by a test. One of them should go once the layering allows it.
    """
    clean = [e if math.isfinite(e) and e > 0 else 0.0 for e in e_values]
    n = len(clean)
    if n == 0:
        return (), 0
    order = sorted(range(n), key=lambda i: clean[i], reverse=True)
    k = 0
    for rank in range(1, n + 1):
        if clean[order[rank - 1]] >= n / (alpha * rank):
            k = rank
    mask = [False] * n
    for i in order[:k]:
        mask[i] = True
    return tuple(mask), k


@dataclass(frozen=True)
class FamilyCorrection:
    """What a family of registered p-values looks like once multiplicity is accounted for.

    Three procedures, because they answer different questions and the difference is the finding.
    Uncorrected is what the campaign reported. Benjamini-Hochberg controls the false discovery rate
    under independence or positive dependence. Benjamini-Yekutieli is the same step-up with the
    harmonic penalty and is valid under arbitrary dependence, and e-BH is valid under arbitrary
    dependence too, by a different route. Reporting all three is the mandatory-comparator
    discipline applied to a procedure rather than to an estimator.
    """

    names: tuple[str, ...] = ()
    p_values: tuple[float, ...] = ()
    alpha: float = 0.05
    kappa: float = 0.5
    uncorrected: tuple[bool, ...] = ()
    bh_q: tuple[float, ...] = ()
    bh_rejected: tuple[bool, ...] = ()
    by_q: tuple[float, ...] = ()
    by_rejected: tuple[bool, ...] = ()
    e_values: tuple[float, ...] = ()
    ebh_rejected: tuple[bool, ...] = ()
    ebh_k: int = 0

    @property
    def n(self) -> int:
        return len(self.p_values)

    @property
    def lost_to_correction(self) -> tuple[str, ...]:
        """The predictions confirmed uncorrected that Benjamini-Hochberg does not reject.

        This is the list a summary has to carry. A confirmation that survives only because nobody
        counted the family is still printed, and it is printed next to the fact that it does not
        survive.
        """
        return tuple(
            name
            for name, raw, bh in zip(self.names, self.uncorrected, self.bh_rejected)
            if raw and not bh
        )

    def render(self) -> str:
        hn = sum(1.0 / i for i in range(1, self.n + 1)) if self.n else 0.0
        lines = [
            f"{self.n} registered p-values in the family, alpha {self.alpha:g}, "
            f"calibrator kappa {self.kappa:g}, harmonic penalty H_{self.n} = {hn:.4f}",
            f"  uncorrected rejections: {sum(self.uncorrected)}",
            f"  Benjamini-Hochberg:     {sum(self.bh_rejected)}   "
            f"(valid under independence or positive dependence)",
            f"  Benjamini-Yekutieli:    {sum(self.by_rejected)}   "
            f"(valid under arbitrary dependence)",
            f"  e-BH:                   {self.ebh_k}   "
            f"(valid under arbitrary dependence; needs e >= {self.n / self.alpha:.1f} "
            f"for one rejection at n = {self.n})",
            "",
            f"  {'prediction':34} {'p':>10} {'q_BH':>10} {'q_BY':>10} {'e':>9}  verdicts",
        ]
        for i, name in enumerate(self.names):
            marks = "".join(
                "*" if m[i] else "."
                for m in (self.uncorrected, self.bh_rejected, self.by_rejected, self.ebh_rejected)
            )
            lines.append(
                f"  {name:34} {self.p_values[i]:10.6f} {self.bh_q[i]:10.5f} "
                f"{self.by_q[i]:10.5f} {self.e_values[i]:9.3f}  {marks}"
            )
        lines.append("  verdict columns, in order: uncorrected, BH, BY, e-BH")
        if self.lost_to_correction:
            lines.append(
                f"  confirmed uncorrected and not rejected by BH: "
                f"{', '.join(self.lost_to_correction)}"
            )
        return "\n".join(lines)


def correct_family(
    entries: Sequence[tuple[str, float]],
    *,
    alpha: float = 0.05,
    kappa: float = 0.5,
) -> FamilyCorrection:
    """Run the three procedures over a family of named p-values.

    Benjamini-Hochberg comes from `reward_lens.stats.multiplicity.bh_fdr` rather than being
    rewritten here. Benjamini-Yekutieli is that procedure's q-values scaled by the harmonic number,
    which is the definition and not a second implementation. e-BH is `e_bh` above.
    """
    # Imported here rather than at module scope. `reward_lens.stats.multiplicity` cannot be
    # reached without executing `reward_lens.stats.__init__`, which costs 1.15 s and pulls in
    # scipy, sklearn, pandas and pyarrow. `reward_lens.studies` imports in 0.26 s today and a
    # module-scope import would quadruple that for every caller, including the ones that only
    # want to render a count.
    from reward_lens.stats.multiplicity import bh_fdr

    names = tuple(name for name, _ in entries)
    p_values = tuple(float(p) for _, p in entries)
    n = len(p_values)
    if n == 0:
        return FamilyCorrection(alpha=alpha, kappa=kappa)

    rejected, q = bh_fdr(list(p_values), alpha)
    bh_q = tuple(float(x) for x in q)
    bh_rejected = tuple(bool(x) for x in rejected)
    hn = sum(1.0 / i for i in range(1, n + 1))
    by_q = tuple(min(x * hn, 1.0) for x in bh_q)
    by_rejected = tuple(x <= alpha for x in by_q)
    e_values = tuple(calibrate_p_to_e(p, kappa) for p in p_values)
    ebh_rejected, ebh_k = e_bh(e_values, alpha)

    return FamilyCorrection(
        names=names,
        p_values=p_values,
        alpha=alpha,
        kappa=kappa,
        uncorrected=tuple(p <= alpha for p in p_values),
        bh_q=bh_q,
        bh_rejected=bh_rejected,
        by_q=by_q,
        by_rejected=by_rejected,
        e_values=e_values,
        ebh_rejected=ebh_rejected,
        ebh_k=ebh_k,
    )


#: Comparators whose predictions are tests against a probability, so a p-value correction applies.
_P_COMPARATORS = ("<", "<=")


def p_value_predictions(
    predictions: Iterable[AdjudicatedPrediction],
) -> tuple[tuple[str, float], ...]:
    """The predictions in a family that are tests against a significance level, named and paired.

    A prediction qualifies when its metric name marks it as a p-value, its comparator is one-sided
    downward, and its threshold sits strictly inside (0, 1). That is a syntactic rule applied after
    the fact, and applying it after the fact is itself the problem this module is about: nothing in
    the campaign's 27 frozen specs marks which predictions carry a test statistic, so the family a
    correction ranges over had to be identified by reading metric names. A `MetaAnalysisPlan` is
    where that identification belongs, which is why `multiplicity` is one of its fields.
    """
    out: list[tuple[str, float]] = []
    for p in predictions:
        name = p.metric.lower()
        looks_like_p = (
            name.endswith("_p")
            or name.endswith("_p_value")
            or name.endswith("_pvalue")
            or "p_value" in name
        )
        if not looks_like_p or p.comparator not in _P_COMPARATORS:
            continue
        if not 0.0 < p.threshold < 1.0 or p.value is None:
            continue
        out.append((f"{p.study}/{p.owner}", float(p.value)))
    return tuple(out)


# ---------------------------------------------------------------------------
# Null-boundary sensitivity
# ---------------------------------------------------------------------------


class Stability(Enum):
    """How much a verdict depends on the exact value of the estimator that produced it."""

    #: The estimate is more than one recorded standard error from the threshold.
    STABLE = "stable"
    #: The estimate is within one recorded standard error of the threshold. A different estimator
    #: of the same quantity could plausibly have produced the other verdict.
    SENSITIVE = "sensitive"
    #: Nothing was recorded to compare the distance against, so stability is not established.
    UNKNOWN = "unknown"
    #: An equality comparator on an indicator. The verdict is a match or it is not, and distance
    #: from the threshold is not a measure of how nearly it went the other way.
    DISCRETE = "discrete"


_DISCRETE_COMPARATORS = ("==", "!=")


@dataclass(frozen=True)
class BoundarySensitivity:
    """One verdict, and how far the estimate would have to move to reverse it.

    `flip_distance` is exact and needs no assumptions: it is the distance from the observed value
    to the frozen threshold, in the metric's own units. `relative_flip` divides that by the larger
    of the threshold and the value, which makes verdicts on different metrics rankable against each
    other and is not a measure of statistical stability. `z` is the one that would be, and it needs
    a recorded uncertainty.
    """

    study: str
    owner: str
    metric: str
    comparator: Comparator
    threshold: float
    value: float
    outcome: str
    flip_distance: float
    relative_flip: float | None
    uncertainty: float | None
    z: float | None
    stability: Stability
    note: str = ""

    def render(self) -> str:
        rel = f"{self.relative_flip:.4f}" if self.relative_flip is not None else "n/a"
        return (
            f"{self.study}/{self.owner}  {self.metric} {self.comparator} {self.threshold:g}  "
            f"observed {self.value:.6g}  {self.outcome.upper()}  "
            f"flip {self.flip_distance:.6g}  relative {rel}  {self.stability.name}"
        )


def boundary_sensitivity(
    predictions: Iterable[AdjudicatedPrediction],
) -> tuple[BoundarySensitivity, ...]:
    """How far each adjudicated verdict sits from the threshold that produced it.

    Returned sorted by relative flip distance, closest first, so the rows a reader should check
    come out at the top. Predictions with no computed value are skipped; they have no verdict to be
    sensitive about.

    Where the evidence records an uncertainty, `stability` is the real answer and `z` is the number
    behind it. Where it does not, `stability` is `UNKNOWN` and the note says what to record. That
    combination is deliberate: substituting an assumed standard error for a missing one would turn
    "nobody measured this" into a stability claim, which is exactly the confident wrong number this
    library exists to refuse.
    """
    rows: list[BoundarySensitivity] = []
    for p in predictions:
        if p.value is None or p.outcome not in ("confirmed", "refuted", "fired", "passed"):
            continue
        lhs = abs(p.value) if p.comparator.startswith("abs") else p.value
        flip = abs(lhs - p.threshold)
        scale = max(abs(p.threshold), abs(lhs))
        relative = flip / scale if scale > 0 else None

        if p.comparator in _DISCRETE_COMPARATORS:
            stability, z, note = (
                Stability.DISCRETE,
                None,
                "an equality check on an indicator: the verdict does not move continuously with "
                "the estimator, so a flip distance of zero here means an exact match rather than a "
                "verdict about to reverse.",
            )
        elif p.uncertainty is not None and p.uncertainty > 0:
            z = flip / p.uncertainty
            stability = Stability.SENSITIVE if z < 1.0 else Stability.STABLE
            note = (
                f"the estimate sits {z:.2f} recorded standard errors from its threshold."
                if z >= 1.0
                else f"the estimate sits {z:.2f} recorded standard errors from its threshold, so "
                f"an estimator of the same quantity could have produced the other verdict."
            )
        else:
            stability, z = Stability.UNKNOWN, None
            note = (
                "no uncertainty was recorded for this metric, so the verdict's stability is not "
                "established. Record ci_low and ci_high on the adjudicating evidence and this "
                "becomes a stability verdict instead of a ranking."
            )

        rows.append(
            BoundarySensitivity(
                study=p.study,
                owner=p.owner,
                metric=p.metric,
                comparator=p.comparator,
                threshold=p.threshold,
                value=p.value,
                outcome=p.outcome,
                flip_distance=flip,
                relative_flip=relative,
                uncertainty=p.uncertainty,
                z=z,
                stability=stability,
                note=note,
            )
        )
    rows.sort(key=lambda r: (r.relative_flip if r.relative_flip is not None else math.inf, r.study))
    return tuple(rows)


def near_threshold(
    rows: Sequence[BoundarySensitivity], *, relative_cut: float = 0.10
) -> tuple[BoundarySensitivity, ...]:
    """The continuously-adjudicated verdicts sitting within `relative_cut` of their threshold.

    `relative_cut` is a reporting cut for ranking rows, not a scientific threshold and not an
    envelope condition. It answers "which rows should a reader look at first" and it does not
    answer "which verdicts are unreliable"; only a recorded uncertainty answers that, and
    `boundary_sensitivity` returns `UNKNOWN` where there is none. Discrete verdicts are excluded
    because a relative distance does not mean the same thing for them.
    """
    return tuple(
        r
        for r in rows
        if r.stability is not Stability.DISCRETE
        and r.relative_flip is not None
        and r.relative_flip <= relative_cut
    )


# ---------------------------------------------------------------------------
# Reading a campaign's own adjudications
# ---------------------------------------------------------------------------


def predictions_from_cards(cards: Iterable[Any]) -> tuple[AdjudicatedPrediction, ...]:
    """Flatten re-adjudicated campaign cards into the family a count is a sum over.

    Takes anything with `card`, `frozen` and `result` attributes, which is what
    `reward_lens.record.convert.readjudicate.readjudicate` returns. Duck-typed rather than
    imported, because that module imports the study runner and importing it back here would close
    a cycle through `reward_lens.studies`.

    Kill criteria come through on the same footing as hypotheses, with `kind` telling them apart.
    A count of fired kills is a campaign-level claim in exactly the same way a count of
    confirmations is, and the campaign published one.
    """
    out: list[AdjudicatedPrediction] = []
    for card in cards:
        spec = card.frozen.spec
        result = card.result
        for h in spec.hypotheses:
            out.append(
                AdjudicatedPrediction(
                    study=card.card,
                    owner=h.id,
                    kind="hypothesis",
                    metric=h.prediction.metric,
                    comparator=h.prediction.comparator,
                    threshold=float(h.prediction.threshold),
                    value=_as_float(result.metrics.get(h.prediction.metric)),
                    outcome=result.outcomes.get(h.id, "void"),
                )
            )
        for k in spec.kill_criteria:
            out.append(
                AdjudicatedPrediction(
                    study=card.card,
                    owner=k.id,
                    kind="kill",
                    metric=k.metric,
                    comparator=k.comparator,
                    threshold=float(k.threshold),
                    value=_as_float(result.metrics.get(k.metric)),
                    outcome=result.kill_outcomes.get(k.id, "void"),
                )
            )
    return tuple(out)


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None if value is None else float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def campaign_counts(
    predictions: Sequence[AdjudicatedPrediction],
    *,
    evidence_at: str = "",
) -> tuple[CountClaim, ...]:
    """The three counts a campaign summary states: confirmations, refutations, fired kills.

    Hypotheses and kill criteria are tallied over their own populations rather than over one pooled
    list, because a kill criterion is not a hypothesis and a count that mixed them would be a
    fourth thing nobody registered.
    """
    hypotheses = [p for p in predictions if p.kind == "hypothesis"]
    kills = [p for p in predictions if p.kind == "kill"]
    confirmed = tally(hypotheses, "confirmed", unit="hypothesis", evidence_at=evidence_at)
    refuted = tally(hypotheses, "refuted", unit="hypothesis", evidence_at=evidence_at)
    fired = tally(kills, "fired", unit="kill criterion", evidence_at=evidence_at)
    return (
        _restate(confirmed, f"{confirmed.value} of {len(hypotheses)} frozen hypotheses confirmed"),
        _restate(refuted, f"{refuted.value} refuted"),
        _restate(fired, f"{fired.value} kill criteria fired"),
    )


def _restate(claim: CountClaim, statement: str) -> CountClaim:
    return CountClaim(
        label=claim.label,
        value=claim.value,
        unit=claim.unit,
        over=claim.over,
        statement=statement,
        evidence_at=claim.evidence_at,
    )


__all__ = [
    "DEFAULT_REMEDY",
    "AdjudicatedPrediction",
    "BoundarySensitivity",
    "CountClaim",
    "Coverage",
    "ExploratoryReason",
    "FamilyCorrection",
    "MetaAnalysisPlan",
    "Stability",
    "boundary_sensitivity",
    "calibrate_p_to_e",
    "campaign_counts",
    "correct_family",
    "cover",
    "e_bh",
    "freeze_meta_plan",
    "near_threshold",
    "p_value_predictions",
    "predictions_from_cards",
    "render_count",
    "render_counts",
    "tally",
]

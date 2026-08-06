"""M3, the dumb-baseline bank: the six numbers every claim in this library ships against.

Use it like this. Build a `DetectionTask` from whatever you have, call `run_bank`, and pass your
own per-item scores to `compare_against_baselines`. Three lines, which is the point: a bank that
takes an afternoon to wire up is a bank that gets skipped on the claim that most needed it.

    task = DetectionTask(labels=y, texts=transcripts, markers=("exit(0)",))
    bank = run_bank(task)
    verdict = compare_against_baselines(my_detector_scores, y, bank)
    print(verdict.render())

The six are string match, length, TF-IDF, n-gram diversity, a scaffolded black-box prompt, and the
gradient-norm peak. They are not a menu. A claim ships against all six, and a baseline that could
not run is recorded as a refusal with a remedy rather than dropped, because a claim that never ran
the black-box comparator and a claim that ran it and won look identical from the outside unless
the refusal is written down.

`lint_claim` is the enforcement. A claim with no baselines fails it, which is the rule "a claim
without a baseline fails lint" made executable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from reward_lens.core.quantity import BaselineID
from reward_lens.stats.baselines.base import (
    BaseBaseline,
    Baseline,
    BaselineBank,
    BaselineComparison,
    BaselineReading,
    BaselineScore,
    DetectionTask,
    Verdict,
    accuracy_at_midpoint,
    auroc,
    compare_against_baselines,
    is_scored,
    oriented_score,
    stratified_folds,
)
from reward_lens.stats.baselines.series import GradNormPeak, gradnorm_peak, smooth
from reward_lens.stats.baselines.text import (
    SCAFFOLD_TEMPLATE,
    SCAFFOLD_VERSION,
    Length,
    NgramDiversity,
    ScaffoldedPrompt,
    StringMatch,
    TfidfLogisticRegression,
    mine_markers,
    render_scaffold,
    scaffold_hash,
)

#: The bank, in the order it is reported in. Keyed by `BaselineID` so an instrument's ``baselines``
#: tuple names entries that exist rather than strings somebody typed.
BASELINES: dict[BaselineID, Baseline] = {
    b.id: b  # type: ignore[misc]
    for b in (
        StringMatch(),
        Length(),
        TfidfLogisticRegression(),
        NgramDiversity(),
        ScaffoldedPrompt(),
        GradNormPeak(),
    )
}

#: The six ids, as a tuple, for an instrument's ``baselines`` declaration.
ALL_SIX: tuple[BaselineID, ...] = tuple(BASELINES)


def run_bank(task: DetectionTask, baselines: Sequence[BaselineID] | None = None) -> BaselineBank:
    """Run every baseline in the bank against one task. Nothing here raises.

    ``baselines`` restricts the run, and restricting it is a decision worth being able to see:
    the returned bank records exactly which ids were asked for, so a claim that ran four of six
    cannot present itself as having run six.
    """
    ids = list(baselines) if baselines is not None else list(BASELINES)
    unknown = [i for i in ids if i not in BASELINES]
    if unknown:
        raise KeyError(
            f"no baseline registered as {unknown}. The bank is {sorted(BASELINES)}; a claim "
            f"naming a baseline that does not exist has named nothing."
        )
    return BaselineBank(task_name=task.name, readings={i: BASELINES[i].run(task) for i in ids})


# ---------------------------------------------------------------------------
# The lint rule: a claim without a baseline is not a claim
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClaimLintFinding:
    """One missing comparison, with what closes it."""

    claim: str
    field: str
    problem: str
    remedy: str

    def render(self) -> str:
        return f"{self.claim}.{self.field}: {self.problem}  ->  {self.remedy}"


def claim_baselines(claim: Any) -> Mapping[BaselineID, float] | None:
    """Find a claim's baseline mapping, wherever the claim keeps it.

    Two shapes are accepted because the library currently has two. `Evidence` carries a
    ``baselines: Mapping[BaselineID, float]`` field directly; what does not yet exist is a path
    that fills it, because `Context.emit` builds Evidence from a value and does not know about
    baselines. So instruments carry them in ``value["baselines"]`` and a populated payload beats an
    empty field until the emit path catches up.

    Returning None means the claim has no place for a baseline mapping at all, which is a
    different finding from having an empty one.
    """
    direct = getattr(claim, "baselines", None)
    if isinstance(direct, Mapping) and direct:
        return direct
    value = getattr(claim, "value", None)
    if isinstance(value, Mapping):
        nested = value.get("baselines")
        if isinstance(nested, Mapping):
            return nested
    if isinstance(claim, Mapping) and isinstance(claim.get("baselines"), Mapping):
        return claim["baselines"]
    return direct if isinstance(direct, Mapping) else None


def lint_claim(
    claim: Any,
    bank: BaselineBank | None = None,
    *,
    require: Sequence[BaselineID] = ALL_SIX,
) -> list[ClaimLintFinding]:
    """A claim with no dumb baseline fails. This is that rule, executable.

    Three rules. A claim carrying no baseline mapping at all fails, because there is nowhere for
    the comparison to have happened. A claim carrying an empty one fails, because an empty mapping
    is the same statement made explicitly. And a claim missing one of the six fails **unless** the
    bank records that baseline refusing, because "the gradient-norm comparison could not run, here
    is why" is a result and silently dropping it is not.

    ``require`` defaults to all six and the default is the rule. It is a parameter because the six
    are transcript-level detection comparators and not every claim is a detection claim: a power
    calculation's dumb comparators are the standard power calculators, and asking it for a string
    match would be asking for a number that does not exist. A claim that overrides this is
    declaring its own comparator set at the call site, in the open. The rule that never bends is
    the first one: the mapping is not empty.

    Returning findings rather than raising follows `lint_instrument`: a gap should be nameable in
    a report, and the test that asserts the list is empty is what closes it.
    """
    name = getattr(claim, "instrument", None) or getattr(claim, "observable", None) or "claim"
    mapping = claim_baselines(claim)
    if mapping is None:
        return [
            ClaimLintFinding(
                str(name),
                "baselines",
                "carries no baselines mapping, so nothing recorded what this was compared against",
                "run `stats.baselines.run_bank` on the same items and attach "
                "`bank.as_mapping()` to the reading",
            )
        ]
    if not mapping:
        return [
            ClaimLintFinding(
                str(name),
                "baselines",
                "carries an empty baselines mapping, and a claim with no dumb baseline is not a "
                "claim",
                "run `stats.baselines.run_bank` on the same items. One published probe reported "
                "AUC 0.998 on a task a zero-parameter string match solves outright",
            )
        ]

    out: list[ClaimLintFinding] = []
    refused = set(bank.refusals()) if bank is not None else set()
    for bid in require:
        if bid in mapping or bid in refused:
            continue
        out.append(
            ClaimLintFinding(
                str(name),
                "baselines",
                f"{bid} is neither scored nor recorded as refused, so it was skipped silently",
                f"run {bid} on the same items, or record the refusal that says what it needed. "
                f"Every claim ships against all six",
            )
        )
    return out


__all__ = [
    "ALL_SIX",
    "BASELINES",
    "SCAFFOLD_TEMPLATE",
    "SCAFFOLD_VERSION",
    "BaseBaseline",
    "Baseline",
    "BaselineBank",
    "BaselineComparison",
    "BaselineReading",
    "BaselineScore",
    "ClaimLintFinding",
    "DetectionTask",
    "GradNormPeak",
    "Length",
    "NgramDiversity",
    "ScaffoldedPrompt",
    "StringMatch",
    "TfidfLogisticRegression",
    "Verdict",
    "accuracy_at_midpoint",
    "auroc",
    "claim_baselines",
    "compare_against_baselines",
    "gradnorm_peak",
    "is_scored",
    "lint_claim",
    "mine_markers",
    "oriented_score",
    "render_scaffold",
    "run_bank",
    "scaffold_hash",
    "smooth",
    "stratified_folds",
]

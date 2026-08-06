"""The thirteen fields the catalogue names for D7, and what each one needs to read.

The card is a composition. Nothing here measures anything: every field names an instrument that
already exists in series A, B or D, says which inputs that instrument needs, and says what a
reader should do when one of them is absent. Adding a measurement to this module would be the
first step towards a second implementation of a quantity that already has one.

**Why the instruments are resolved by name rather than imported.** Five of the thirteen live in
`reward_lens.verifier`, which guards itself with ``require_extra("verifier")``. Importing them at
module scope would make the whole card unimportable on a base install, and the card is the one
artifact that has to render for a reader who has installed nothing. So each spec carries a module
path and a class name, `FieldSpec.resolve` imports on demand, and a missing extra becomes a
refusal naming the install command rather than an ImportError from the middle of a card build.

**On `needs`.** A field's inputs are listed rather than discovered, because three of the composed
instruments (D3, D4, D5) take their subject as a required positional argument and cannot be
constructed at all without it. The card has to know whether it can build an instrument before it
tries, and it has to be able to say which input is missing by name.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Sequence

from reward_lens.core.quantity import QuantityID

if TYPE_CHECKING:  # every one of these is a type, and importing it at runtime is what we avoid
    from reward_lens.measure.composition.hodge import ComparisonFlow
    from reward_lens.measure.composition.revealed import ComparisonSet
    from reward_lens.measure.metrology.flakiness import ReplaySet
    from reward_lens.measure.metrology.gstudy import GroupScores as MetrologyGroups
    from reward_lens.measure.metrology.gstudy import ReplicationDesign
    from reward_lens.record.scores import GroupScores as RecordGroups
    from reward_lens.record.scores import ScoreContext, ScoreTree
    from reward_lens.verifier import (
        ExploitLog,
        MetamorphicRelation,
        Rollout,
        RolloutCorpus,
        RubricInput,
        RubricScorer,
        StrictReference,
        VerifierUnderTest,
    )


# ---------------------------------------------------------------------------
# What the analyst has
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CardInputs:
    """Everything a card can read about one grader. Every field optional, and an absent one is a
    refusal rather than a blank.

    This is deliberately a flat bag rather than a hierarchy. The thirteen instruments were written
    by five packages that never agreed on a subject type, and inventing a unifying one here would
    mean writing an adapter per instrument, which is where a card starts quietly reshaping the
    numbers it is supposed to be reporting. A named slot per input keeps the card a router.

    The two `GroupScores` slots are two different types with one name, and they are not
    interchangeable. ``group_scores`` is `metrology.gstudy.GroupScores`, an array of per-group
    score vectors, and it is what A1 reads. ``record_groups`` is `record.scores.GroupScores`, a
    group's recorded score trees and their contexts, and it is what B4 reads. Collapsing them
    would hand A1 a tree it cannot average and B4 an array with no abstention channel in it.
    """

    # -- the program, and the corpus it graded
    #: The grader's source, for the instruments that read code (D1, D2).
    verifier: "VerifierUnderTest | None" = None
    #: The grader as something callable, for the instruments that query it (D3, D5).
    grader: "Callable[..., float] | VerifierUnderTest | None" = None
    #: The rollouts it graded. The test suite of the mutation literature, under D's substitution.
    corpus: "RolloutCorpus | None" = None
    #: Which metamorphic relations to check. None lets D3 choose its own default set.
    relations: "Sequence[MetamorphicRelation] | None" = None

    # -- the rubric, as a scorer over named numeric inputs
    scorer: "RubricScorer | None" = None
    rubric_inputs: "tuple[RubricInput, ...]" = ()

    # -- the differential oracle D5 measures false positives against
    reference: "StrictReference | None" = None
    fp_seeds: "tuple[Rollout, ...]" = ()

    # -- the logs and the replication designs
    exploit_log: "ExploitLog | None" = None
    replays: "ReplaySet | None" = None
    design: "ReplicationDesign | None" = None
    group_scores: "MetrologyGroups | None" = None
    score_trees: "tuple[ScoreTree | None, ...]" = ()
    score_contexts: "tuple[ScoreContext, ...] | None" = None
    record_groups: "tuple[RecordGroups, ...]" = ()

    # -- the comparison record
    flow: "ComparisonFlow | None" = None
    comparisons: "Sequence[ComparisonSet] | None" = None

    # -- budgets, which belong to the caller and not to the card
    #: D2 stops after this many mutants. None means every mutant the operators generate, which on
    #: a large verifier is the difference between a card and an afternoon.
    mutation_limit: int | None = None
    mutation_rung: int = 1
    mutation_timeout: float | None = 30.0
    #: D4's base sample size. The call count is `N * (2D + 2)`, so this is the field that decides
    #: whether the card costs a thousand grader calls or a hundred thousand.
    sobol_n_base: int = 256
    fuzz_max_examples: int = 200
    #: B1's nulls, and how many draws each gets. The nulls are the expensive half of a curl mass
    #: on a large comparison graph and they are the half that makes the number mean anything, so
    #: they are a knob rather than a default this package chooses.
    curl_nulls: tuple[str, ...] = ("C", "A", "D", "E")
    curl_draws: int = 200
    #: B2's random-tournament baseline draws.
    afriat_baseline_draws: int = 32

    #: A name for the grader, used in the rendered header when nothing else says one.
    grader_name: str = ""

    def has(self, name: str) -> bool:
        """Whether an input is present. An empty sequence is absent, not empty."""
        value = getattr(self, name, None)
        if value is None:
            return False
        if isinstance(value, (tuple, list)) and not value:
            return False
        return True

    def missing(self, names: Sequence[str]) -> tuple[str, ...]:
        return tuple(n for n in names if not self.has(n))


# ---------------------------------------------------------------------------
# A field
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldSpec:
    """One row of the card: which instrument fills it, and what that instrument needs.

    ``quantity`` is written down here as well as declared on the instrument, and the two are
    checked against each other by the package's own test. It is duplicated on purpose: the
    capability report has to be able to say what a field would contain on a machine where the
    instrument's module cannot be imported, and reading it off the class is not available there.
    """

    #: The catalogue's own words for this field, from D7's `says`.
    name: str
    quantity: QuantityID
    module: str
    attr: str
    #: The optional extra this instrument's module needs, or None when the core install has it.
    extra: str | None
    #: `CardInputs` attribute names, all of which must be present before the instrument is built.
    needs: tuple[str, ...]
    #: Builds the instrument from the resolved class and the inputs. Called only when `needs` are
    #: satisfied, so it may index them without guarding.
    build: Callable[[Any, CardInputs], Any]
    #: What to do when an input is absent. An instruction, never a restatement of the failure.
    remedy: str
    #: Extra context for the refusal's detail, naming what the missing input is in the reader's
    #: vocabulary rather than in the constructor's.
    input_names: dict[str, str] = field(default_factory=dict)

    def resolve(self) -> Any:
        """The instrument class. Raises `ExtraRequiredError` when its extra is not installed."""
        return getattr(importlib.import_module(self.module), self.attr)

    def describe_missing(self, inputs: CardInputs) -> str:
        """The absent inputs, phrased to read after "carries no".

        The names are noun phrases without articles and they join with "and no" rather than with a
        comma, because `refuse_incomplete` builds its detail as "X carries no {field}" and a comma
        list there produces "carries no a scorer, the ranges", which is a sentence nobody wrote.
        """
        gaps = inputs.missing(self.needs)
        return " and no ".join(self.input_names.get(g, g) for g in gaps)


# ---------------------------------------------------------------------------
# The thirteen
# ---------------------------------------------------------------------------
#
# The order is the catalogue's, which is not alphabetical and is not arbitrary: it runs from what
# the source says, through what querying it says, to what the record of using it says. A reader
# going down the card is moving away from the program and towards the run.

_VERIFIER = "verifier"


def _coverage(cls: Any, inputs: CardInputs) -> Any:
    return cls(inputs.verifier, inputs.corpus)


def _mutants(cls: Any, inputs: CardInputs) -> Any:
    return cls(
        inputs.verifier,
        inputs.corpus,
        rung=inputs.mutation_rung,
        limit=inputs.mutation_limit,
        timeout=inputs.mutation_timeout,
    )


def _metamorphic(cls: Any, inputs: CardInputs) -> Any:
    return cls(inputs.grader, inputs.corpus, inputs.relations)


def _sensitivity(cls: Any, inputs: CardInputs) -> Any:
    return cls(inputs.scorer, inputs.rubric_inputs, n_base=inputs.sobol_n_base)


def _false_positives(cls: Any, inputs: CardInputs) -> Any:
    return cls(
        inputs.grader,
        inputs.reference,
        inputs.fp_seeds,
        max_examples=inputs.fuzz_max_examples,
    )


def _exploit_families(cls: Any, inputs: CardInputs) -> Any:
    return cls(inputs.exploit_log)


def _flakiness(cls: Any, inputs: CardInputs) -> Any:
    return cls(inputs.replays)


def _variance_components(cls: Any, inputs: CardInputs) -> Any:
    return cls(inputs.design)


def _effective_group_size(cls: Any, inputs: CardInputs) -> Any:
    return cls(inputs.group_scores, inputs.design)


def _census(cls: Any, inputs: CardInputs) -> Any:
    return cls(inputs.score_trees, inputs.score_contexts, groups=inputs.record_groups)


def _curl_mass(cls: Any, inputs: CardInputs) -> Any:
    return cls(inputs.flow, nulls=inputs.curl_nulls, n_draws=inputs.curl_draws)


def _afriat(cls: Any, inputs: CardInputs) -> Any:
    return cls(inputs.comparisons, baseline_draws=inputs.afriat_baseline_draws)


#: The remedy for the two fields B4 fills: instrument the grader so the per-leaf scores are
#: recorded. A score tree is not something a reader can obtain by asking for more access: it exists
#: only if somebody recorded the per-leaf scores while the grader ran.
_TAP_REMEDY = (
    "instrument the grader with `reward_lens.tap` so the per-leaf scores and the abstention "
    "channel are recorded, then pass `score_trees=` or `record_groups=` from the recorded groups. "
    "A composed score written down as one number has no channel to count, so this cannot be "
    "recovered from a record that did not keep one."
)

CARD_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec(
        name="coverage",
        quantity="verifier.decision_coverage",
        module="reward_lens.verifier.coverage",
        attr="DecisionCoverage",
        extra=_VERIFIER,
        needs=("verifier", "corpus"),
        build=_coverage,
        remedy=(
            "supply the grader's source file and the rollouts it graded: "
            "`CardInputs(verifier=VerifierUnderTest(path, entrypoint=...), corpus=ListCorpus.of(...))`. "
            "Coverage is measured by re-running the corpus under a line tracer, so both are needed "
            "and neither can be inferred from the other."
        ),
        input_names={"verifier": "grader source file", "corpus": "corpus of graded rollouts"},
    ),
    FieldSpec(
        name="surviving mutants",
        quantity="verifier.surviving_mutants",
        module="reward_lens.verifier.mutate",
        attr="SurvivingMutants",
        extra=_VERIFIER,
        needs=("verifier", "corpus"),
        build=_mutants,
        remedy=(
            "supply the grader's source and the rollouts it graded. A mutant is killed when "
            "re-grading the corpus with the mutated source changes a score, so a corpus is the "
            "test suite here and there is nothing to kill a mutant with without one."
        ),
        input_names={"verifier": "grader source file", "corpus": "corpus of graded rollouts"},
    ),
    FieldSpec(
        name="metamorphic violations",
        quantity="verifier.metamorphic_violations",
        module="reward_lens.verifier.metamorphic",
        attr="MetamorphicViolations",
        extra=_VERIFIER,
        needs=("grader", "corpus"),
        build=_metamorphic,
        remedy=(
            "supply something callable that scores a rollout, plus the rollouts: "
            "`CardInputs(grader=fn, corpus=ListCorpus.of(...))`. A metamorphic relation is checked "
            "by transforming an input and calling the grader again, so a recorded score is not "
            "enough and the grader has to be reachable."
        ),
        input_names={"grader": "callable grader", "corpus": "corpus of graded rollouts"},
    ),
    FieldSpec(
        name="sensitivity profile",
        quantity="verifier.sobol_ST",
        module="reward_lens.verifier.sensitivity",
        attr="SobolSensitivity",
        extra=_VERIFIER,
        needs=("scorer", "rubric_inputs"),
        build=_sensitivity,
        remedy=(
            "expose the grader as a scorer over named numeric inputs and declare their ranges: "
            "`CardInputs(scorer=fn, rubric_inputs=[RubricInput('helpfulness', 0.0, 1.0), ...])`. "
            "A Sobol decomposition apportions output variance across an input space, so a grader "
            "with no numeric input space has no sensitivity profile rather than a flat one."
        ),
        input_names={
            "scorer": "rubric scorer over named numeric inputs",
            "rubric_inputs": "declared input ranges",
        },
    ),
    FieldSpec(
        name="false-positive catalogue",
        quantity="verifier.false_positive_rate",
        module="reward_lens.verifier.fuzz",
        attr="FalsePositiveFuzzing",
        extra=_VERIFIER,
        needs=("grader", "reference", "fp_seeds"),
        build=_false_positives,
        remedy=(
            "supply a callable grader, a stricter reference that decides the same question, and "
            "seed rollouts to search around: `CardInputs(grader=fn, "
            "reference=StrictReference('exact', decide, basis='...'), fp_seeds=[...])`. A false "
            "positive is a disagreement with a stricter oracle, so without one there is no "
            "quantity here, only a distribution of scores."
        ),
        input_names={
            "grader": "callable grader",
            "reference": "stricter reference oracle",
            "fp_seeds": "seed rollouts to search around",
        },
    ),
    FieldSpec(
        name="silent-zero rate",
        quantity="grader.silent_zero_rate",
        module="reward_lens.measure.composition.abstention",
        attr="SilentZeroRate",
        extra=None,
        needs=("score_trees",),
        build=_census,
        remedy=_TAP_REMEDY,
        input_names={"score_trees": "recorded score trees"},
    ),
    FieldSpec(
        name="flakiness spread",
        quantity="env.flakiness",
        module="reward_lens.measure.metrology.flakiness",
        attr="EnvironmentFlakiness",
        extra=None,
        needs=("replays",),
        build=_flakiness,
        remedy=(
            "replay each task in the harness at least twice and pass the score matrix: "
            "`CardInputs(replays=ReplaySet(scores, task_ids=...))`. The spread is the range across "
            "replays of one task, so a single run per task gives a range of zero for want of a "
            "second observation rather than because the harness is deterministic."
        ),
        input_names={"replays": "replay set carrying at least two replays of each task"},
    ),
    FieldSpec(
        name="exploit-family accounting",
        quantity="verifier.unseen_exploit_mass",
        module="reward_lens.verifier.growth",
        attr="ExploitFamilyCoverage",
        extra=_VERIFIER,
        needs=("exploit_log",),
        build=_exploit_families,
        remedy=(
            "log every exploit you have already found, with its family and the effort it took, and "
            "pass it: `CardInputs(exploit_log=ExploitLog.of([ExploitFind(family=..., "
            "effort=...), ...]))`. The unseen mass is a Good-Turing estimate off the frequency "
            "spectrum of what has been found, so a blacklist with no counts on it cannot produce "
            "one and the fix is where the blacklist is kept."
        ),
        input_names={"exploit_log": "exploit log of finds by family"},
    ),
    FieldSpec(
        name="variance components",
        quantity="grader.variance_components",
        module="reward_lens.measure.metrology.grr",
        attr="VarianceComponents",
        extra=None,
        needs=("design",),
        build=_variance_components,
        remedy=(
            "score each item at least twice under controlled facet variation and pass the crossed "
            "design: `CardInputs(design=ReplicationDesign.from_long(values, objects, raters))`. A "
            "variance decomposition needs replication to separate the grader's disagreement with "
            "itself from the spread across items, and one score per item confounds the two with "
            "nothing able to tell."
        ),
        input_names={"design": "replicated scoring design"},
    ),
    FieldSpec(
        name="effective group size",
        quantity="grader.effective_group_size",
        module="reward_lens.measure.metrology.gstudy",
        attr="EffectiveGroupSize",
        extra=None,
        needs=("group_scores",),
        build=_effective_group_size,
        remedy=(
            "pass the per-group score vectors the run actually produced: "
            "`CardInputs(group_scores=GroupScores.of([...]))`. The effective size is the nominal "
            "group size discounted by how much of the within-group spread is grader noise, so it "
            "is a property of recorded groups and not of the grader alone."
        ),
        input_names={"group_scores": "per-group score vectors"},
    ),
    FieldSpec(
        name="curl mass",
        quantity="grader.curl_mass",
        module="reward_lens.measure.composition.hodge",
        attr="CurlMass",
        extra=None,
        needs=("flow",),
        build=_curl_mass,
        remedy=(
            "build the comparison flow from the grader's observed pairwise verdicts: "
            "`CardInputs(flow=edge_flow(pairs, n_items))`, each pair carrying both win counts. A "
            "curl mass needs a filled triangle to be nonzero at all, so three items compared to "
            "each other is the smallest design that reports anything."
        ),
        input_names={"flow": "pairwise comparison flow"},
    ),
    FieldSpec(
        name="Afriat index",
        quantity="grader.afriat_index",
        module="reward_lens.measure.composition.revealed",
        attr="AfriatIndex",
        extra=None,
        needs=("comparisons",),
        build=_afriat,
        remedy=(
            "pass the comparison sets the grader's verdicts form: "
            "`CardInputs(comparisons=bank_from_scores(...))`. The index is the largest efficiency "
            "at which the recorded choices are still rationalisable by one utility, so it is "
            "computed from choices and there is nothing to rationalise without them."
        ),
        input_names={"comparisons": "recorded comparison sets"},
    ),
    FieldSpec(
        name="abstention channel",
        quantity="grader.abstention_rate",
        module="reward_lens.measure.composition.abstention",
        attr="AbstentionRate",
        extra=None,
        needs=("score_trees",),
        build=_census,
        remedy=_TAP_REMEDY,
        input_names={"score_trees": "recorded score trees"},
    ),
)


#: The catalogue's `says` for D7, split at the commas, in order. The package's own test asserts
#: that `CARD_FIELDS` covers every one of these and adds nothing, so a card that quietly grew a
#: fourteenth field or lost a thirteenth fails rather than shipping.
CATALOGUE_FIELDS: tuple[str, ...] = tuple(spec.name for spec in CARD_FIELDS)


__all__ = ["CARD_FIELDS", "CATALOGUE_FIELDS", "CardInputs", "FieldSpec"]

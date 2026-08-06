"""Acceptance: the sixteen sciences, retyped onto the kernel.

The sciences survive as studies rather than as subsystems, and get retyped rather than rewritten.
The retype is four things, and this file is the clause that closes them:

1. the analysis takes a `record/` object rather than a planted organism,
2. it returns a `Reading`, so a subject that cannot answer produces a refusal with a remedy,
3. every number it reports resolves to a **registered** quantity id,
4. the study spec goes through plan closure, so a prediction nothing can produce is found before
   any work runs rather than after it.

**The population is enumerated from the directory, not from a list here.** A science added later is
caught by `test_every_science_declares_a_retype` instead of being silently exempt, which is the
whole reason it walks the filesystem. That test is expected to fail while the retype is partial, and
its failure message is the work list.

The subject is the recorded GRPO run at `tests/fixtures/grpo_run/short`: a real `GRPOTrainer`, real
weights, real sampling, 96 trajectories over 12 steps. It is deliberately a subject that does *not*
contain most of the phenomena these sciences preregister, because the refusal path is the half of
the retype that a planted organism can never exercise.
"""

from __future__ import annotations

import re

import pytest

from reward_lens.core.closure import ClosureError, GapKind
from reward_lens.core.evidence import Evidence
from reward_lens.core.quantity import QUANTITIES, load_quantities
from reward_lens.core.reading import Refusal
from reward_lens.record.reader import open_run
from reward_lens.record.schema import Run
from reward_lens.studies.plan import check_closure, closure_report
from studies._retype import (
    ScienceRetype,
    frozen_metrics,
    readers,
    retypes,
    science_modules,
    specs,
)

#: What counts as one of the sixteen sciences, by directory name: `sNN_name` plus `atlas_meta`. It
#: is written down here rather than inferred from what happens to be on disk, because `studies/`
#: also holds four compute-gated study packages, and a predicate that merely excludes them today is
#: not the same as one that says what it means.
_SCIENCE_DIR_RE = re.compile(r"^(?:s\d{2}_\w+|atlas_meta)$")

SHORT_RUN = ("tests/fixtures/grpo_run/short", "run:8a8c7e29274db0a681313b48dbd1eb63")
LONG_RUN = ("tests/fixtures/grpo_run/long", "run:f77bf75940ab982bbc35407af99cc094")


@pytest.fixture(scope="module", autouse=True)
def _registry() -> None:
    """The quantity registry loads on demand rather than at import, so load it once."""
    load_quantities()


@pytest.fixture(scope="module")
def record() -> Run:
    return open_run(*SHORT_RUN)


def retyped() -> list[tuple[str, ScienceRetype]]:
    return sorted(retypes().items())


def ids_of(items: list[tuple[str, ScienceRetype]]) -> list[str]:
    return [name for name, _ in items]


RETYPED = retyped()
RETYPED_IDS = ids_of(RETYPED)


# ---------------------------------------------------------------------------
# The population, walked rather than listed
# ---------------------------------------------------------------------------


def test_the_science_directory_is_what_is_walked() -> None:
    """The enumeration reaches every science module, so nothing is exempt by omission."""
    modules = science_modules()
    assert modules, "the walk found no science modules at all"
    for expected in ("studies.s12_hackability.analysis", "studies.s03_thermo.analysis"):
        assert expected in modules, f"{expected} is not in the walk: {modules}"
    # Every directory under `studies/` that ships a spec is in the denominator.
    assert len(specs()) >= 15, f"expected at least fifteen specs, found {sorted(specs())}"


@pytest.mark.xfail(
    strict=False,
    reason=(
        "the retype is partial. This is the work list rather than a bug: the failure message names "
        "the sciences that still ship a frozen spec and no retype."
    ),
)
def test_every_science_declares_a_retype() -> None:
    """The clause that closes the package. Its failure message is the remaining work.

    The denominator is the sixteen sciences by name, and getting there took two corrections.

    It was `science_modules()`, which walks **every** directory under `studies/`. Four
    compute-gated study packages live there (`w6_rate`, `w6_distillation`, `w6_transfer`,
    `w6_monitor`), so the missing list was permanently non-empty and **this test could never have
    flipped to a pass no matter how many sciences landed.** It was found twice independently, from
    opposite ends of the roster, which is what a shared acceptance file is for.

    Both findings proposed `specs()`, on the ground that the compute packages ship no
    `build_spec`. That was true when it was measured and false an hour later: a Phase 6 package's deliverables include a
    frozen study spec, so `w6_distillation` acquired one and walked straight back into the
    denominator. **A predicate that happens to exclude something today is not the same as one that
    says what it means.**

    So the denominator names the thing it is counting. A science is a directory following the
    `sNN_name` convention, plus `atlas_meta`, and that is the only set this test is about. The
    underlying fault is not fixed here: the top-level `studies/` is meant to hold the sixteen
    sciences and four compute-gated packages landed in it as well. Moving them is the right repair
    and is deliberately not being done in the hours before a version tag.
    """
    have = set(retypes())
    want = {name for name in specs() if _SCIENCE_DIR_RE.match(name)}
    assert len(want) == 15, (
        f"expected the fifteen sciences, found {len(want)}: {sorted(want)}. If a science was added "
        f"or renamed, update the convention; if a non-science package landed under studies/ and "
        f"matched, that is the directory collision this docstring describes"
    )
    missing = sorted(want - have)
    assert not missing, (
        f"{len(missing)} of {len(want)} sciences ship a frozen spec and no RETYPE: {missing}. "
        f"Each needs a metric table binding every frozen metric to a registered quantity, and a "
        f"`read(run) -> Reading`."
    )


# ---------------------------------------------------------------------------
# Clause 1 and 2: a record object in, a Reading out
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,retype", RETYPED, ids=RETYPED_IDS)
def test_read_takes_a_record_and_returns_a_reading(
    name: str, retype: ScienceRetype, record: Run
) -> None:
    """`read` accepts a `record.Run` and returns `Evidence | Refusal`, never a dict and never None."""
    reader = readers()[name]
    reading = reader(record)
    assert isinstance(reading, (Evidence, Refusal)), (
        f"{name}.read returned {type(reading).__name__}. A Reading is Evidence or Refusal; a bare "
        f"number or a None is the failure this architecture exists to prevent."
    )


@pytest.mark.parametrize("name,retype", RETYPED, ids=RETYPED_IDS)
def test_a_refusal_carries_a_reason_and_an_actionable_remedy(
    name: str, retype: ScienceRetype, record: Run
) -> None:
    """A refusal that says 'envelope violated' and stops has told the reader nothing.

    The bar asserted here is deliberately low and mechanical (a reason, a non-trivial remedy) because
    the readable version cannot be asserted. The remedy strings are reviewed by eye.
    """
    reading = readers()[name](record)
    if isinstance(reading, Refusal):
        assert reading.reason is not None
        assert len(reading.remedy.split()) >= 6, f"{name}: remedy is too short to act on"
        assert reading.detail.strip(), f"{name}: refusal carries no detail"


@pytest.mark.parametrize("name,retype", RETYPED, ids=RETYPED_IDS)
def test_partial_readings_name_what_they_could_not_answer(
    name: str, retype: ScienceRetype, record: Run
) -> None:
    """Every frozen metric is either reported or refused by name. Silence is not an option.

    A metric that is neither is the failure mode the study runner had to grow `voids` for: under the
    adjudication this project replaced, a kill criterion whose metric was missing looked exactly
    like one that was evaluated and did not fire.
    """
    reading = readers()[name](record)
    if not isinstance(reading, Evidence):
        return
    reported = set(reading.value["metrics"])
    refused = set(reading.value["refusals"])
    unaccounted = sorted(frozen_metrics(retype.spec) - reported - refused)
    assert not unaccounted, (
        f"{name}: {unaccounted} are registered in the frozen spec and this reading neither reports "
        f"nor refuses them."
    )


# ---------------------------------------------------------------------------
# Clause 3: every number resolves to a registered quantity id
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,retype", RETYPED, ids=RETYPED_IDS)
def test_every_bound_metric_names_a_registered_quantity(name: str, retype: ScienceRetype) -> None:
    """The declaration side: no metric binds to an id the registry does not hold."""
    assert retype.unregistered() == (), (
        f"{name} binds metrics to unregistered quantity ids: {retype.unregistered()}. Registering "
        f"an id is the maintainer's decision, so this is a request rather than an edit."
    )


@pytest.mark.parametrize("name,retype", RETYPED, ids=RETYPED_IDS)
def test_every_reported_number_resolves_to_a_registered_quantity(
    name: str, retype: ScienceRetype, record: Run
) -> None:
    """The emitted side, which is the one that matters.

    A field is not wired until every path that emits reaches it, and the cheap check is to assert
    the emitted value rather than the declaration. So this reads the quantity map off the Evidence
    rather than off the metric table, and it caught a real bug: a loop variable named `quantity`
    shadowed the parameter of the same name in
    `ScienceRetype.evidence`, and every reading was stamped with whichever id the last measured
    entry happened to carry.
    """
    reading = readers()[name](record)
    if not isinstance(reading, Evidence):
        return
    resolved = reading.value["quantities"]
    for metric, quantity in resolved.items():
        assert quantity, f"{name}: reported {metric!r} with no quantity id at all"
        assert quantity in QUANTITIES, (
            f"{name}: reported {metric!r} as {quantity!r}, which is not in the registry"
        )
    assert set(resolved) == set(reading.value["metrics"]), (
        f"{name}: the quantity map and the metric map disagree about which numbers exist"
    )
    assert str(reading.quantity) in set(resolved.values()), (
        f"{name}: the Evidence is stamped {reading.quantity!r} and reports none of "
        f"{sorted(set(resolved.values()))}. A reading stamped with a quantity it does not carry is "
        f"a claim about a reading rather than a reading."
    )


# ---------------------------------------------------------------------------
# Clause 4: the spec goes through plan closure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,retype", RETYPED, ids=RETYPED_IDS)
def test_the_study_spec_passes_closure_or_names_its_gap(name: str, retype: ScienceRetype) -> None:
    """A plan with no unregistered metric closes; a plan with one raises, naming it.

    Both halves are required. `check_closure` raising `ClosureError` before any work
    runs, with the prediction, the metric and the gap in the message, is the behaviour the whole
    mechanism exists for, and a science whose metric has no registered quantity id is exactly the
    case that should trigger it. Asserting only the passing half would let a science quietly bind a
    metric to a plausible-looking wrong id in order to go green.
    """
    plan = retype.plan()
    gapped = {m.metric for m in retype.gaps()}
    if not gapped:
        report = check_closure(plan)
        assert report.closed
        assert report.required_arcs, f"{name}: the plan closes and reaches no arcs"
        return

    with pytest.raises(ClosureError) as caught:
        check_closure(plan)
    named = {g.demand.metric for g in caught.value.report.gaps if g.demand is not None}
    assert named == gapped, (
        f"{name}: declared gaps {sorted(gapped)} and closure named {sorted(named)}. The two have to "
        f"agree, or the gap table is documentation rather than a check."
    )
    for gap in caught.value.report.gaps:
        assert gap.kind is GapKind.UNBOUND_METRIC
        assert gap.remedy.strip()


@pytest.mark.parametrize("name,retype", RETYPED, ids=RETYPED_IDS)
def test_every_frozen_metric_is_accounted_for_in_the_plan(name: str, retype: ScienceRetype) -> None:
    """No frozen prediction is left out of the metric table, bound or gapped.

    `ScienceRetype.__post_init__` enforces this at construction; asserting it here means the rule
    survives a refactor of the constructor.
    """
    table = set(retype.by_metric())
    missing = sorted(frozen_metrics(retype.spec) - table)
    assert not missing, f"{name}: frozen spec registers {missing} and the retype binds none of them"


@pytest.mark.parametrize("name,retype", RETYPED, ids=RETYPED_IDS)
def test_the_arc_graph_is_derived_and_acyclic(name: str, retype: ScienceRetype) -> None:
    """Every bound metric has a producing arc, and no arc sits in a dependency loop."""
    report = closure_report(retype.plan())
    assert not report.gaps_of(GapKind.CYCLE), f"{name}: an arc sits in a dependency loop"
    assert not report.gaps_of(GapKind.UNSATISFIED_INPUT), (
        f"{name}: an arc requires an output nothing in the plan produces"
    )
    for arc in retype.arcs():
        assert arc.produces, f"{name}: arc {arc.id!r} produces nothing"


# ---------------------------------------------------------------------------
# The retype does not disturb what already runs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,retype", RETYPED, ids=RETYPED_IDS)
def test_the_original_analysis_survives(name: str, retype: ScienceRetype) -> None:
    """`analyze(StudyRun) -> StudyResult` is untouched, because confirmed results run through it.

    The retype is an addition. A science whose `analyze` stopped resolving would have had its
    confirmed result deleted by a refactor, which is the one outcome this package must not produce.
    """
    import importlib

    module = importlib.import_module(f"studies.{name}.analysis")
    assert callable(getattr(module, "analyze", None)), f"{name}.analyze no longer resolves"
    assert callable(getattr(module, "build_spec", None)), f"{name}.build_spec no longer resolves"
    assert retype.spec.analysis.endswith(".analyze")


def test_s05_reads_a_multi_channel_score_tree() -> None:
    """The compute half of S5, which the GRPO fixture cannot reach because it has one channel.

    Both fixtures score with a single `length_reward` leaf, so `read` correctly refuses them and the
    refusal is all the fixture can exercise. The dark fraction is one minus the R-squared of the
    reward on its named channels, so a two-leaf tree whose total is exactly the sum of its leaves
    has a dark fraction of zero: the channels explain everything because they are everything. That
    is the boundary value worth pinning, because a decomposition that cannot return zero when
    nothing is hidden will not return the truth when something is.
    """
    import numpy as np

    from reward_lens.measure.indices.dark_reward import dark_reward
    from reward_lens.record.scores import GraderCallRef, Leaf, WeightedSum
    from studies.s05_capacity.analysis import _leaves_of

    def call(name: str) -> GraderCallRef:
        return GraderCallRef(grader=name, outcome="returned")

    rng = np.random.default_rng(0)
    rows, totals = [], []
    for _ in range(64):
        a, b = float(rng.normal()), float(rng.normal())
        tree = WeightedSum(
            name="total",
            children=(
                Leaf(name="helpfulness", value=a, grader_call=call("helpfulness")),
                Leaf(name="brevity", value=b, grader_call=call("brevity")),
            ),
        )
        leaves = dict(_leaves_of(tree))
        assert sorted(leaves) == ["brevity", "helpfulness"], "channel extraction lost a leaf"
        rows.append([leaves["helpfulness"], leaves["brevity"]])
        totals.append(a + b)

    dark = float(dark_reward(np.asarray(totals), np.asarray(rows)))
    assert dark == pytest.approx(0.0, abs=1e-9), (
        f"two channels that sum to the total left a dark fraction of {dark}, so the regression is "
        f"not seeing the channels it was handed"
    )

    # And a hidden third channel has to show up as dark, or the quantity measures nothing.
    hidden = [t + float(h) for t, h in zip(totals, rng.normal(size=len(totals)))]
    dark_hidden = float(dark_reward(np.asarray(hidden), np.asarray(rows)))
    assert dark_hidden > 0.2, f"a hidden channel of comparable variance read as {dark_hidden} dark"


def test_readings_are_stable_across_two_real_records() -> None:
    """The same reader over two real runs of different length returns the same shape.

    Twelve steps and two hundred steps, both written by a real `GRPOTrainer`. A reader that happens
    to work on one length and not the other is reading an artifact of the fixture.
    """
    short, long = open_run(*SHORT_RUN), open_run(*LONG_RUN)
    for name, reader in readers().items():
        a, b = reader(short), reader(long)
        assert type(a) is type(b), (
            f"{name}: {type(a).__name__} on 12 steps, {type(b).__name__} on 200"
        )
        if isinstance(a, Evidence) and isinstance(b, Evidence):
            assert set(a.value["metrics"]) == set(b.value["metrics"]), (
                f"{name}: reports different metrics on the two runs"
            )

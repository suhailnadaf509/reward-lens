"""Acceptance: B3's counterfactual composition and B4's silent-zero census.

The clause has two halves and each is asserted on something real rather than on a fixture.

*On a real composite reward, B3 re-evaluates with a node disabled and reports the fraction of
advantages moved by more than one standard deviation and the number whose sign reversed, with no
grader calls.* The real composite is built from the campaign store: eleven of its twelve banks were
scored by more than one reward model, so joining two arms on their shared items gives a two-
component reward whose every leaf is a real reward model's real score on a real response. The
composition is a weighted sum of the two, the counterfactual drops one component, and the question
is the sensitivity analysis nobody publishes: what would this run's advantages have been if we had
not blended that model in. *With no grader calls* is asserted separately against a real verifier
with a counter inside it, because a store has no callable to count and an assertion that cannot
fail is not an assertion.

*B4 reports both the silent-zero rate and the abstention rate on a real grader, with the
advantage-baseline consequence in the interpretation.* Two real graders, and they answer differently
for the same reason. `is_equiv` from `hendrycks/math`, the answer-equivalence checker most open
RLVR maths pipelines still call, has no abstention channel at all: it ends in a bare
`except: return str1 == str2`, so nothing it does can reach a wrapper and the measured rate is 0.0.
The campaign store's fourteen reward models score 201,756 rollouts with not one recorded failure,
and their call records were reconstructed by a converter from a store with no outcome field, so that
0.0 is an assumption made twice over. Both readings say so, in the reading, which is the difference
between an instrument reporting zero and an instrument reporting nothing.

**Two things this file states plainly rather than leaving to be inferred.** The composition in the
B3 clause is assembled here: the leaves are two real reward models' recorded outputs, the shape is a
weighted sum, and no record reachable from this build carries a multi-component composition that
somebody else wrote down. Both real records in reach hold one score per rollout. And the campaign
recorded no estimator at all, so the advantage arm needs one supplied; the refusal that comes back
when it is not supplied is asserted first, because that is the honest default and the assumption is
the departure from it.
"""

from __future__ import annotations

import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from reward_lens.core.quantity import load_quantities
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.types import Access, Component
from reward_lens.measure.base import Context, lint_instrument
from reward_lens.measure.composition.abstention import AbstentionRate, SilentZeroRate
from reward_lens.measure.composition.composition import (
    CompositionTree,
    CounterfactualComposition,
)
from reward_lens.record.schema import EstimatorSpec
from reward_lens.record.scores import (
    GraderCallRef,
    GroupContext,
    GroupScores,
    Leaf,
    Override,
    PredicateRef,
    ScoreContext,
    WeightedSum,
)
from reward_lens.tap import SimpleRun, instrument_grader

load_quantities()

CAMPAIGN_STORE = Path(
    os.environ.get(
        "REWARD_LENS_CAMPAIGN_STORE",
        Path.home() / "final-reward" / "campaign-results" / "runs" / "campaign",
    )
)

needs_campaign = pytest.mark.skipif(
    not (CAMPAIGN_STORE / "evidence.jsonl").exists(),
    reason=(
        f"no campaign evidence store at {CAMPAIGN_STORE}. It is the archive the 2.0 campaign "
        f"produced and it is not in the repository; set REWARD_LENS_CAMPAIGN_STORE to point at it."
    ),
)

#: `is_equiv` from `hendrycks/math`. Fetched rather than vendored, so what is measured is what is
#: actually published today rather than a copy that could have drifted.
REAL_VERIFIER_URL = (
    "https://raw.githubusercontent.com/hendrycks/math/main/modeling/math_equivalence.py"
)

#: Answer pairs in the shapes the MATH benchmark's own normaliser is written for, plus four it is
#: not. The last four are the ones that exercise the bare `except`.
ANSWER_PAIRS = [
    ("\\frac{1}{2}", "\\frac{1}{2}"),
    ("1/2", "\\frac{1}{2}"),
    ("0.5", "\\frac{1}{2}"),
    ("2", "2"),
    ("-3", "-3"),
    ("x=5", "5"),
    ("\\sqrt2", "\\sqrt{2}"),
    ("50\\%", "50"),
    ("\\$5", "5"),
    ("5 \\text{ cm}", "5"),
    ("\\dfrac{3}{4}", "\\frac{3}{4}"),
    ("1 2", "12"),
    ("\\left(3\\right)", "(3)"),
    (".5", "0.5"),
    ("10", "11"),
    ("\\frac{a}{b}", "\\frac{b}{a}"),
    ("", ""),
    ("\\frac{1}{3}", "0.333"),
    ("\\pi", "\\pi"),
    ("3.14", "\\pi"),
    ("[0,1]", "[0, 1]"),
    ("y = 2x", "2x"),
    ("100", "10^2"),
    ("1.0", "1"),
    ("\\frac{", "\\frac{"),
    ("\\text{", "x"),
    ("\\frac{1}{}", "1"),
    ("\\sqrt{", "2"),
]

#: The estimator the counterfactual replays on the campaign join, supplied because the campaign
#: recorded none. Group-centred with no standard-deviation division, which is `verifiers`' own
#: transform and the one this library treats as the reference case. Stated as an assumption in the
#: test that uses it, and the refusal for not supplying one is asserted first.
ASSUMED_GRPO = EstimatorSpec(
    family="grpo",
    group_centred=True,
    std_normalised=False,
    degenerate_policy="skip",
    aggregation="sequence",
    loss_mask_policy="assumed for this counterfactual; the campaign recorded no estimator",
)


# ---------------------------------------------------------------------------
# Subject one: two real reward models on the same real items
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def two_model_blend() -> dict[str, Any]:
    """A two-component reward whose leaves are two real reward models' recorded scores.

    `armorm` and `skywork-v02` both scored the campaign's diagnostic banks, so joining their arms on
    the shared item key gives, for every rollout, two real grader outputs on the same response. That
    is a composite reward with more than one component and no part of it was made up here except the
    weights and the fact of the sum.
    """
    from reward_lens.record.convert import convert_campaign

    arms = {}
    for name in ("armorm", "skywork-v02"):
        run, _ = convert_campaign(CAMPAIGN_STORE, grader=name)
        index: dict[str, Any] = {}
        for step in run.steps:
            for group in step.groups:
                gid = str(group.id)
                index[gid.split("/", 1)[1] if "/" in gid else gid] = group
            if len(index) >= 20_000:
                break
        arms[name] = index

    shared = sorted(set(arms["armorm"]) & set(arms["skywork-v02"]))
    groups: list[GroupScores] = []
    for key in shared:
        a, b = arms["armorm"][key], arms["skywork-v02"][key]
        if len(a.trajectories) != len(b.trajectories) or len(a.trajectories) < 2:
            continue
        trees = []
        ok = True
        for ta, tb in zip(a.trajectories, b.trajectories):
            la, lb = ta.scores, tb.scores
            if not (isinstance(la, Leaf) and isinstance(lb, Leaf)):
                ok = False
                break
            trees.append(
                WeightedSum(
                    name="blend",
                    children=(
                        Leaf("armorm", la.value, la.grader_call, la.abstained),
                        Leaf("skywork_v02", lb.value, lb.grader_call, lb.abstained),
                    ),
                    weights=(1.0, 1.0),
                )
            )
        if not ok:
            continue
        groups.append(
            GroupScores(
                trees=tuple(trees),
                contexts=tuple(
                    ScoreContext(group=GroupContext(k=len(trees), id=key)) for _ in trees
                ),
                estimator=ASSUMED_GRPO,
                id=key,
            )
        )
        if len(groups) >= 4_000:
            break
    if not groups:
        pytest.skip("the two arms share no item with more than one rollout in this store")
    return {"groups": groups, "n_rollouts": sum(len(g.trees) for g in groups)}


@needs_campaign
def test_the_composition_is_real_and_has_more_than_one_component(two_model_blend) -> None:
    """Before the counterfactual means anything, the thing it is run on has to be a composite."""
    groups = two_model_blend["groups"]
    got = CompositionTree.over(groups).compute()
    assert not isinstance(got, Refusal), got
    assert got.n_leaves == 2 * got.n_trees
    assert got.weights_dict_components == 2
    assert got.node_types["Leaf"] == 2 * got.n_trees
    assert got.node_types["WeightedSum"] == got.n_trees
    # A weighted sum of two reward models is the one composition a weights dict *can* express, and
    # the instrument says so rather than manufacturing a wedge that is not there.
    assert got.n_inexpressible == 0
    assert got.is_additive is True
    assert "would have lost nothing" in got.says()
    # Both leaves carry the name of the reward model that produced them.
    names = {n for tree in groups[0].trees for n in ("armorm", "skywork_v02")}
    assert names == {"armorm", "skywork_v02"}


@needs_campaign
def test_the_counterfactual_refuses_the_campaign_s_own_estimator_before_anything_is_assumed(
    two_model_blend,
) -> None:
    """The campaign computed no advantages, so it recorded no estimator, so this refuses.

    Asserted before the arm that supplies one, because the refusal is the default and the
    assumption is the departure. An instrument that quietly picked GRPO here would be inventing the
    transform that decides every number downstream of it.
    """
    from dataclasses import replace

    from reward_lens.record.convert.campaign import NO_ESTIMATOR

    as_recorded = [replace(g, estimator=NO_ESTIMATOR) for g in two_model_blend["groups"][:8]]
    got = CounterfactualComposition(as_recorded, {"skywork_v02"}).compute()
    assert isinstance(got, Refusal)
    assert got.reason is RefusalReason.RECORD_INCOMPLETE
    assert "not group-relative" in got.detail
    assert "family='none'" in got.detail
    assert "Compare the counterfactual on the score scale instead" in got.remedy
    assert "record the per-rollout value-function baseline" in got.remedy


@needs_campaign
def test_b3_reports_both_numbers_on_a_real_two_reward_model_composite(two_model_blend) -> None:
    """The clause, on a real composite reward, at the scale the sentence is written at.

    Every number here is whatever this join does. None of them is a target: what the clause asks is
    that the two are reported, that they are consistent with each other, and that the comparison is
    in advantage space with the score-scale comparison beside it.
    """
    groups = two_model_blend["groups"]
    started = time.perf_counter()
    got = CounterfactualComposition(groups, {"skywork_v02"}).compute()
    elapsed = time.perf_counter() - started
    assert not isinstance(got, Refusal), got

    result = got.result
    assert result.n == two_model_blend["n_rollouts"] >= 1_000
    assert result.n_groups == len(groups)
    assert result.n_trees_with_node == result.n
    assert result.disabled == ("skywork_v02",)

    # The fraction moved by more than one standard deviation, and the count whose sign reversed.
    assert 0.0 < result.fraction_moved <= 1.0
    assert 0 < result.n_sign_reversed <= result.n_comparable
    assert result.n_moved == int(round(result.fraction_moved * result.n_comparable))
    assert result.sd_reference > 0.0
    sentence = result.says()
    assert sentence.startswith("removing skywork_v02 changes")
    assert f"{result.n_sign_reversed} of {result.n} rollouts" in sentence

    # The mandatory baseline: the same comparison before the estimator centred anything.
    assert 0.0 <= got.score_scale_fraction_moved <= 1.0
    assert got.score_sd > 0.0
    assert "before the estimator" in got.interpretation()

    # The declared invariance, measured on this composition rather than asserted about it. A plain
    # weighted sum passes a per-prompt constant through to every rollout, so it cancels.
    assert got.leak is not None
    assert got.leak.cancels is True
    assert got.leak.reach == 1.0
    assert got.leak.resolved is True

    # Free, and it has to be: four thousand groups of real leaves in a second on one core.
    assert elapsed < 30.0


@needs_campaign
def test_b3_touches_no_grader_because_it_declares_no_access_to_one(two_model_blend) -> None:
    """A store has no callable in it, so the structural half of "no grader calls" is the
    declaration: this instrument asks for RECORD and cannot ask for QUERY."""
    inst = CounterfactualComposition(two_model_blend["groups"][:16], {"skywork_v02"})
    assert inst.requires[Component.GRADER] is Access.RECORD
    assert Access.QUERY not in inst.requires[Component.GRADER]
    assert inst.rung == 0
    assert lint_instrument(inst) == []


# ---------------------------------------------------------------------------
# Subject two: a real published verifier, wrapped by the real tap
# ---------------------------------------------------------------------------


class CallCounter:
    """A real verifier with a counter around it, so "no grader calls" is measured."""

    def __init__(self, fn) -> None:
        self.fn = fn
        self.calls = 0

    def __call__(self, *, answer: str, gold: str) -> float:
        self.calls += 1
        return 1.0 if self.fn(answer, gold) else 0.0


@pytest.fixture(scope="module")
def real_verifier():
    try:
        with urllib.request.urlopen(REAL_VERIFIER_URL, timeout=20) as response:
            source = response.read().decode("utf-8")
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        pytest.skip(
            f"no network for the real-verifier run ({type(exc).__name__}). The fixtures in "
            f"tests/test_measure_composition.py prove the code runs; this one proves it measures "
            f"something published, and skipping it leaves that unproven rather than proven."
        )
    namespace: dict[str, Any] = {}
    exec(compile(source, "math_equivalence.py", "exec"), namespace)  # noqa: S102
    assert "except:" in source, "the bare except this instrument is about is no longer there"
    return namespace["is_equiv"]


@pytest.fixture(scope="module")
def verifier_run(real_verifier):
    """Score the real answer pairs through the real tap and keep what it recorded.

    The composition is `is_equiv` for correctness and a length term, in the shape taken
    from Kimi K3: the score is pinned to -1 when the answer runs past a scaled budget. The
    correctness leaf is a real verifier's real output; the length term and the assembly are this
    test's, and that is said here rather than implied.
    """
    grader = CallCounter(real_verifier)
    run = SimpleRun(run_id="w33c")
    tapped = instrument_grader(grader, run=run, name="is_equiv")

    over_budget = PredicateRef(
        name="over_budget", feature="chars", op=">", threshold_feature="budget", scale=1.5
    )
    trees, contexts = [], []
    for answer, gold in ANSWER_PAIRS:
        score = tapped(answer=answer, gold=gold)
        calls = run.ring.drain()
        ref = GraderCallRef.from_call(calls[-1]) if calls else None
        length = min(1.0, len(answer) / 12.0)
        trees.append(
            Override(
                name="length_override",
                condition=over_budget,
                constant=-1.0,
                otherwise=WeightedSum(
                    name="task",
                    children=(Leaf("correct", score, ref), Leaf("brevity", 1.0 - length, ref)),
                    weights=(1.0, 0.25),
                ),
            )
        )
        contexts.append(
            ScoreContext(
                features={"chars": float(len(answer)), "budget": 8.0},
                group=GroupContext(k=len(ANSWER_PAIRS), id="g0"),
            )
        )
    group = GroupScores(
        trees=tuple(trees), contexts=tuple(contexts), estimator=ASSUMED_GRPO, id="g0"
    )
    return {"group": group, "grader": grader, "run": run}


def test_b3_makes_no_grader_calls_on_a_real_wrapped_verifier(verifier_run) -> None:
    """Not "few", not "cheap": none, on the verifier's own counter and on the tap's."""
    group, grader, run = verifier_run["group"], verifier_run["grader"], verifier_run["run"]
    assert grader.calls == len(ANSWER_PAIRS)
    calls_before = grader.calls
    offered_before = run.ring.stats().offered

    got = CounterfactualComposition([group], {"length_override"}).compute()
    assert not isinstance(got, Refusal), got

    assert grader.calls == calls_before
    assert run.ring.stats().offered == offered_before

    # And the reading is a real one: the override binds on the long answers and removing it changes
    # what they scored.
    assert got.result.n == len(ANSWER_PAIRS)
    assert got.result.n_moved >= 0
    assert got.result.scores_before != got.result.scores_after
    assert got.leak is not None and got.leak.cancels is False, (
        "an override pins part of the group, so a constant added to the task reward cannot cancel"
    )
    assert 0.0 < got.leak.reach < 1.0


def test_b3_reports_the_override_as_a_primitive_a_weights_dict_cannot_hold(verifier_run) -> None:
    got = CompositionTree.over([verifier_run["group"]]).compute()
    assert got.inexpressible == ("Override",)
    assert got.is_additive is False
    assert got.weights_dict_components == 2
    assert "gradient dead zone" in got.render()


def test_b4_reports_both_rates_on_a_real_verifier_and_says_what_they_cannot_see(
    verifier_run,
) -> None:
    """The clause's second half, on `is_equiv`.

    The measured rates are both zero and that is the finding, not the absence of one. `is_equiv`
    catches every exception it can raise and returns a string comparison instead, so no failure it
    has ever reaches the wrapper that would count it. The reading leads with that rather than
    printing 0.0% and stopping.
    """
    group = verifier_run["group"]
    silent = SilentZeroRate.over([group])
    abst = AbstentionRate.over([group])

    s = silent.compute()
    a = abst.compute()
    assert not isinstance(s, Refusal) and not isinstance(a, Refusal)

    assert s.n_leaves == 2 * len(ANSWER_PAIRS)
    assert s.n_unattributable == 0
    assert s.n_reconstructed == 0, "these outcomes were observed by the tap, not reconstructed"
    assert s.outcomes == {"returned": s.n_leaves}
    assert s.silent_zero_rate == 0.0
    assert a.abstention_rate == 0.0
    assert s.n_boundary_failures == 0
    assert s.channel_observed is False

    assert "lower bound rather than a measurement" in s.says()
    assert "bare `except: return str1 == str2`" in s.limitation()
    assert "reading the source is what finds the rest" in s.limitation()

    # Both instruments carry the advantage-baseline consequence, and both name the mechanism.
    for reading in (s, a):
        assert "advantage = reward_i - mean(rewards)" in reading.consequence()
        assert "moves the baseline the whole group is measured against" in reading.consequence()

    ev = silent.estimate(Context())
    assert not isinstance(ev, Refusal)
    assert ev.value["value"] == 0.0
    assert ev.value["reports"] == "silent_zero"
    assert "advantage = reward_i - mean(rewards)" in ev.value["interpretation"]
    ev2 = abst.estimate(Context())
    assert ev2.value["reports"] == "abstention"
    assert ev2.value["value"] == 0.0


def test_b4_prices_the_consequence_when_a_wrapped_grader_really_raises(real_verifier) -> None:
    """The same instrument on a record where a real wrapped call really failed.

    `is_equiv` cannot be made to raise: that is its defect and it is the point of the test above.
    So the failing arm wraps it in a caller that rejects an input it will not score, which is what a
    pipeline does, and the exception crosses the real tap. The grader is real, the tap is real, the
    record is real, and the caller is this test's.
    """

    class Rejects(RuntimeError):
        """Raised by name, so nothing has to catch broadly."""

    class Picky:
        def __init__(self, fn) -> None:
            self.fn = fn
            self.calls = 0

        def __call__(self, *, answer: str, gold: str) -> float:
            self.calls += 1
            if not answer.strip():
                raise Rejects("the model returned nothing to grade")
            return 1.0 if self.fn(answer, gold) else 0.0

    grader = Picky(real_verifier)
    run = SimpleRun(run_id="w33c-fail")
    tapped = instrument_grader(grader, run=run, name="is_equiv")

    rows = [("\\frac{1}{2}", "\\frac{1}{2}"), ("2", "2"), ("", "5"), ("0.5", "\\frac{1}{2}")]
    trees, contexts = [], []
    for answer, gold in rows:
        try:
            score = tapped(answer=answer, gold=gold)
            failed = False
        except Rejects:
            # `verifiers`' own line, `rubrics/rubric.py:204-217`: catch it and write a zero.
            score = 0.0
            failed = True
        calls = run.ring.drain()
        ref = GraderCallRef.from_call(calls[-1]) if calls else None
        trees.append(WeightedSum("task", (Leaf("correct", score, ref, failed),), (1.0,)))
        contexts.append(ScoreContext(group=GroupContext(k=len(rows), id="g0")))
    group = GroupScores(
        trees=tuple(trees), contexts=tuple(contexts), estimator=ASSUMED_GRPO, id="g0"
    )

    got = SilentZeroRate.over([group]).compute()
    assert not isinstance(got, Refusal), got
    assert got.n_leaves == 4
    assert got.n_abstained == 1
    assert got.n_silent_zero == 1
    assert got.silent_zero_rate == pytest.approx(0.25)
    assert got.abstention_rate == pytest.approx(0.25)
    assert got.n_boundary_failures == 1, "the exception crossed the real wrapper and was recorded"
    assert got.outcomes == {"returned": 3, "raised": 1}
    assert got.channel_observed is True
    assert got.n_reconstructed == 0

    # The consequence, priced. Three of four rollouts scored 1.0 and the fourth was written down as
    # 0.0, so the framework's group mean is 0.75 against an honest 1.0 and every rollout whose own
    # grader worked had its advantage inflated by 0.25.
    (shift,) = got.shifts
    assert shift.k == 4 and shift.n_scored == 3
    assert shift.mean_as_used == pytest.approx(0.75)
    assert shift.mean_honest == pytest.approx(1.0)
    assert shift.shift == pytest.approx(-0.25)
    assert got.max_abs_shift == pytest.approx(0.25)
    text = got.consequence()
    assert "advantage = reward_i - mean(rewards)" in text
    assert "1 of 1 groups had their mean moved" in text
    assert "0.25" in text
    assert "purely additive" in text
    assert "does not shrink when the group's spread does" in text

    ev = SilentZeroRate.over([group]).estimate(Context())
    assert ev.value["value"] == pytest.approx(0.25)
    assert ev.value["baselines"]["baseline.zeros_counted_as_scores"] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# B4 on the campaign record, at scale
# ---------------------------------------------------------------------------


@needs_campaign
def test_b4_on_the_real_campaign_record_reports_zero_and_says_why_that_is_not_a_measurement(
    two_model_blend,
) -> None:
    """Fourteen real reward models, no recorded failure, and two reasons the zero is soft.

    The campaign scored its banks and stored the scores. It has no field for a call outcome, so the
    converter reconstructs one, and every leaf comes back `returned` because that is what a
    converter can write rather than what anybody observed. The reading counts those separately and
    the limitation names them, which is the difference between reporting a clean run and reporting
    a record that cannot tell you whether the run was clean.
    """
    groups = two_model_blend["groups"]
    got = SilentZeroRate.over(groups).compute()
    assert not isinstance(got, Refusal), got

    assert got.n_leaves == 2 * two_model_blend["n_rollouts"]
    assert got.n_abstained == 0
    assert got.n_silent_zero == 0
    assert got.silent_zero_rate == 0.0
    assert got.abstention_rate == 0.0
    assert got.n_unattributable == 0
    assert got.outcomes == {"returned": got.n_leaves}
    # Every one of them was reconstructed, and that is the whole caveat.
    assert got.n_reconstructed == got.n_leaves
    assert got.channel_observed is False
    assert "lower bound rather than a measurement" in got.says()
    assert "a converter reconstructed it" in got.limitation()
    assert "not what anybody saw" in got.limitation()
    assert set(got.by_grader) == set()

    # And the consequence is still stated, with the honest finding that nothing moved here.
    assert "no group's mean moved" in got.consequence()


@needs_campaign
def test_the_census_refuses_a_record_whose_leaves_carry_no_call_at_all(two_model_blend) -> None:
    """The campaign record does carry calls; a record that does not gets a refusal, not a zero."""
    from dataclasses import replace

    stripped = []
    for g in two_model_blend["groups"][:4]:
        trees = tuple(
            replace(t, children=tuple(replace(c, grader_call=None) for c in t.children))
            for t in g.trees
        )
        stripped.append(replace(g, trees=trees))
    got = SilentZeroRate.over(stripped).compute()
    assert isinstance(got, Refusal)
    assert got.reason is RefusalReason.RECORD_INCOMPLETE
    assert "no outcome is knowable" in got.detail
    assert "would be the reassuring answer and it would be made up" in got.detail
    assert "instrument_grader" in got.remedy
    assert got.instrument == "SilentZeroRate"

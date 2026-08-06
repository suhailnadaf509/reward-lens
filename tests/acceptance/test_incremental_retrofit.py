"""Lint rule four, closed for the battery and the index library (catalogue M9).

An `IncrementalValidity` is mandatory on every white-box reading, and
`measure.base.lint_reading` is the rule. It went live when `policy/` produced the first
white-box reading in the library carrying a measured record, and it was deliberately left ungated in
CI because fifteen shipped instruments declared a white-box capability and emitted nothing. Ungating
made the debt visible instead of making the rule wait on it.

This file is what closes the debt for the paths the retrofit owned, and its shape matters more than
its contents. **The population is enumerated from the registry, not written down here.** A hand-kept
list of white-box instruments is a list that goes stale the first time somebody adds one, and the
failure mode is silent: the new instrument emits no record, no test names it, and the rule that was
supposed to catch it never runs against it. Enumerating `battery.BATTERY` and the index package and
filtering by declared capability means a sixteenth white-box instrument fails this file on the commit
that introduces it.

Every white-box instrument has to do one of two things and both are acceptable outcomes:

  - emit an `IncrementalValidity`, measured against the built black-box bank on its own items; or
  - declare an `incremental_exemption`, a `(reason_id, prose)` pair whose id is one of three
    recognised reasons and whose prose says what the reading is, why that shape has no per-item error
    vector or no competitor, and what would change the answer.

A recorded reason is not a failure and it is not a waiver. What it rules out is the third state, the
one this file exists to make unreachable: an instrument that declares a white-box capability, emits
no record, and says nothing about why.
"""

from __future__ import annotations

import numpy as np
import pytest

from reward_lens.core.budget import IncrementalValidity
from reward_lens.core.types import Capability
from reward_lens.measure.base import WHITE_BOX, Context, declared_capabilities, is_white_box, run
from reward_lens.measure.indices._support import (
    INCREMENTAL_EXEMPTIONS,
    incremental_exemption_findings,
)

torch = pytest.importorskip("torch", reason="the battery requires the white-box extra")

from reward_lens.measure import battery as battery_pkg  # noqa: E402
from reward_lens.measure import indices as index_pkg  # noqa: E402

#: The eighteen index classes, in the order `measure/indices/__init__.py` documents them. Kept in
#: step with `tests/acceptance/test_w1_5_retrofit.py`, which is where the population is established.
INDEX_NAMES = (
    "KUI",
    "Distortion",
    "CoverageDisparity",
    "TeacherCompatibility",
    "TailIndex",
    "VerificationScore",
    "StyleShare",
    "ReceiptReliance",
    "Skepticism",
    "Coherence",
    "DarkReward",
    "InterpCoverage",
    "Chi",
    "VCE",
    "Legibility",
    "EvalAwareness",
    "RobustnessSNR",
    "Contested",
)

#: Held by other packages and retrofitted there. Named rather than filtered
#: silently, so the exclusion is a statement somebody has to delete rather than a gap.
NOT_IN_THIS_PACKAGE = frozenset({"Chi", "FeatureRewardAlignment"})


def all_instruments() -> list:
    return [cls() for cls in battery_pkg.BATTERY] + [
        getattr(index_pkg, name)() for name in INDEX_NAMES
    ]


def white_box_instruments() -> list:
    """Every shipped instrument the rule applies to, from the registry and by declaration."""
    return [i for i in all_instruments() if is_white_box(i)]


def owned_white_box() -> list:
    return [i for i in white_box_instruments() if i.name not in NOT_IN_THIS_PACKAGE]


# ---------------------------------------------------------------------------
# The population, measured rather than asserted from a note
# ---------------------------------------------------------------------------


def test_the_white_box_population_is_fifteen_and_the_mask_is_the_shipped_one() -> None:
    """Fifteen shipped instruments declare a white-box capability under `measure.base.WHITE_BOX`.

    The count is here because it was the retrofit's first open question and it had two answers.
    `WHITE_BOX` is `ACTIVATIONS | GRADIENTS | HVP`, which reaches fifteen. Widening it to include
    `LINEAR_READOUT` reaches seventeen, adding `FeatureRewardAlignment` and
    `MultiObjectiveGeometry`, which declare a readout and no activations. Whether reading a head's
    weight vector counts as opening the network is a decision for the library and not for this
    file; the test pins what is true of the shipped mask so that widening it is a visible change
    with a failing test attached rather than a silent one.
    """
    assert WHITE_BOX == (Capability.ACTIVATIONS | Capability.GRADIENTS | Capability.HVP)
    assert len(white_box_instruments()) == 15

    wider = Capability.ACTIVATIONS | Capability.GRADIENTS | Capability.LINEAR_READOUT
    widened = {i.name for i in all_instruments() if declared_capabilities(i) & wider}
    assert widened - {i.name for i in white_box_instruments()} == {
        "FeatureRewardAlignment",
        "MultiObjectiveGeometry",
    }


# ---------------------------------------------------------------------------
# The rule, on every white-box instrument this package owns
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("inst", owned_white_box(), ids=lambda i: i.name)
def test_every_white_box_instrument_pays_or_says_why_not(inst) -> None:
    """The clause: a record on the reading, or a reason in the declaration. Never neither.

    This is the declaration half and it is checked on every white-box instrument in the package.
    The emission half needs a subject and is checked separately, on the one instrument that has one.
    An instrument that emits a record needs no exemption; an instrument that cannot emit one needs a
    reason whose id is recognised and whose prose is an argument rather than a phrase.
    """
    exemption = getattr(inst, "incremental_exemption", None)
    emits = getattr(inst, "emits_incremental", False) or inst.name in _EMITTERS
    assert exemption is not None or emits, (
        f"{inst.name} declares {declared_capabilities(inst)!r}, which makes its readings white-box "
        f"under lint rule four, and it neither emits an IncrementalValidity nor declares an "
        f"incremental_exemption saying why it cannot. Measure one against stats.baselines, or "
        f"declare (reason_id, prose) with reason_id in {sorted(INCREMENTAL_EXEMPTIONS)}."
    )
    if exemption is not None:
        assert incremental_exemption_findings(inst) == []


#: The instruments that pay rather than say. One, and the file is written so that adding a second is
#: a one-line change here plus a measurement below.
_EMITTERS = frozenset({"EvalAwareness"})


def test_the_exemptions_use_every_recognised_reason_and_invent_none() -> None:
    """The three reasons are all in use, and no instrument has coined a fourth.

    Both halves matter. An unused reason is a category nobody needed, which is worth knowing before
    it is defended; an invented one is a lint rule routed around by a string, which is the failure
    a checkable id exists to prevent.
    """
    used = {
        inst.incremental_exemption[0]
        for inst in owned_white_box()
        if getattr(inst, "incremental_exemption", None) is not None
    }
    assert used == INCREMENTAL_EXEMPTIONS, (
        f"reasons in use: {sorted(used)}; recognised: {sorted(INCREMENTAL_EXEMPTIONS)}"
    )


def test_a_fabricated_reason_is_rejected_and_a_short_one_is_too() -> None:
    """The checker is checked, because an exemption nobody validates is a comment.

    Two ways to route around lint rule four with a string, and both are refused: a reason id the
    integrator never adjudicated, and a recognised id with nothing behind it.
    """

    class Invented:
        name = "Invented"
        incremental_exemption = ("BECAUSE_IT_IS_HARD", "x" * 400)

    class Terse:
        name = "Terse"
        incremental_exemption = ("NO_PER_ITEM_VERDICT", "not applicable")

    assert any("not one of" in f for f in incremental_exemption_findings(Invented()))
    assert any("characters of reason" in f for f in incremental_exemption_findings(Terse()))


# ---------------------------------------------------------------------------
# The emission half: the record on the reading, not the one on the class
# ---------------------------------------------------------------------------


def test_eval_awareness_emits_a_measured_record_through_the_emit_path() -> None:
    """The one instrument in this package with a subject emits a record, asserted where it lands.

    **Asserted on `Evidence.incremental` rather than on anything the instrument declares**, which is
    the whole lesson of the two defects before it. Three of `make_evidence`'s optional fields have each
    been found declared, plumbed, tested at one call site and dead at another, and `incremental` was
    the third: the rule made it mandatory, `make_evidence` had accepted one since it was written,
    and `Context.emit` did not forward it, so the mandatory field was unreachable rather than merely
    unset. A field is not wired until every path that emits reaches it, and the cheap check is to
    assert the emitted value.

    The subject is the built diagnostic set on the tiny classifier, and the numbers this produces are
    reported rather than pinned: the point of the assertion is that four measured numbers arrive on
    the reading and that `lint_reading` is satisfied by them, not that they take particular values on
    a randomly initialised grader.
    """
    from reward_lens.data.builtin.diagnostic_v3 import load_diagnostic_v3
    from reward_lens.measure.base import lint_reading
    from reward_lens.measure.indices import EvalAwareness
    from reward_lens.signals import from_tiny

    pairs = [p for view in load_diagnostic_v3().values() for p in view.items]
    graded = {"math_correctness", "code_correctness", "correctness", "factuality"}
    items, labels = [], []
    for pair in pairs:
        for side in (pair.chosen, pair.rejected):
            items.append((pair.prompt_text, side.text))
            labels.append(1 if pair.axis in graded else 0)
    y = np.asarray(labels)

    inst = EvalAwareness(is_benchmark=y, seed=0)
    reading = run(inst, Context(signal=from_tiny(seed=0), view=items, readout="reward"))

    record = reading.incremental
    assert isinstance(record, IncrementalValidity)
    assert record.baseline_id.startswith("baseline.")
    for value in (
        record.own_score,
        record.baseline_score,
        record.error_correlation,
        record.ensemble_score,
    ):
        assert np.isfinite(value)
    assert -1.0 <= record.error_correlation <= 1.0

    # The rule the whole file is about, run against the reading it applies to.
    assert lint_reading(reading, inst) == []


def test_the_record_names_a_baseline_the_bank_actually_ran(record_case=None) -> None:
    """The baseline named on the record is one the bank scored, not a string chosen for the field.

    A record whose `baseline_id` does not correspond to a baseline that ran is the same failure as a
    claim with no baseline, wearing a name. The notes carry every baseline's AUROC and every
    refusal's reason, so the correspondence is checkable from the payload alone.
    """
    from reward_lens.data.builtin.diagnostic_v3 import load_diagnostic_v3
    from reward_lens.measure.indices import EvalAwareness
    from reward_lens.signals import from_tiny

    pairs = [p for view in load_diagnostic_v3().values() for p in view.items]
    graded = {"math_correctness", "code_correctness", "correctness", "factuality"}
    items, labels = [], []
    for pair in pairs:
        for side in (pair.chosen, pair.rejected):
            items.append((pair.prompt_text, side.text))
            labels.append(1 if pair.axis in graded else 0)

    inst = EvalAwareness(is_benchmark=np.asarray(labels), seed=0)
    reading = run(inst, Context(signal=from_tiny(seed=0), view=items, readout="reward"))

    notes = reading.value["incremental_notes"]
    assert reading.incremental.baseline_id in notes["baseline_auroc"]
    # Two of the six cannot run on a task with no logged series and no judge, and the reading says
    # which and why rather than dropping them.
    assert set(notes["baseline_refusals"]) == {
        "baseline.gradnorm_peak",
        "baseline.scaffolded_prompt",
    }
    assert len(notes["baseline_auroc"]) == 4


def test_a_baseline_at_ceiling_is_named_at_the_reading() -> None:
    """The ceiling trap, closed where it can be seen.

    The library's first incremental record was measured against a baseline whose accuracy was 1.0,
    because the reference run's grader is a length function and the length baseline therefore solves
    its task outright. Every number in that record was right and it established that the four-number
    shape works, not that opening the network bought anything. So the helper says so at the reading
    when the best baseline is at ceiling, and it says so again when neither method clears chance,
    which is the other way four correct numbers can describe the fixture instead of the instrument.
    """
    from reward_lens.measure.indices._support import measure_incremental_validity

    # A task the length baseline solves outright: the positives are long and the negatives short.
    texts = tuple(("word " * 40).strip() if i % 2 else "no" for i in range(40))
    labels = np.asarray([1 if i % 2 else 0 for i in range(40)])
    # A white-box score that is exactly right, so the ensemble cannot improve on either.
    record, notes = measure_incremental_validity(
        "probe.perfect", labels.astype(float), labels, texts=texts, n_resamples=200
    )
    assert record is not None
    assert record.baseline_score == pytest.approx(1.0)
    assert "ceiling" in notes, "a baseline at ceiling has to be named at the reading"
    assert "not about what opening the network bought" in notes["ceiling"]

    assert "floor" not in notes  # a method at 1.0 is not at the floor

    # And the floor, asserted as the rule rather than as a hand-tuned instance. Chance is the
    # majority-class rate: on a task that is 75% negative, a method at 0.62 is worse than answering
    # with the majority and calling that "above chance" is how a floor gets reported as a finding.
    # Ten seeds of pure noise, and the note has to fire exactly when the condition holds.
    rng = np.random.default_rng(0)
    fired = 0
    for seed in range(10):
        y = np.asarray([i % 2 for i in range(40)])
        record, floor_notes = measure_incremental_validity(
            "probe.noise",
            rng.normal(size=40),
            y,
            # Real words, not single characters: sklearn's default token pattern needs two word
            # characters, and a corpus of one-character tokens makes `baseline.tfidf` raise
            # `ValueError: empty vocabulary` out of `run`, which the bank's own contract says no
            # baseline does. That is a defect in `stats/baselines/text.py` and it is reported
            # rather than worked around here; this fixture just avoids provoking it.
            texts=tuple(" ".join(["word"] * int(n)) for n in rng.integers(1, 30, size=40)),
            n_resamples=200,
            seed=seed,
        )
        if record is None:
            continue
        chance = floor_notes["chance"]
        assert chance == pytest.approx(0.5)
        at_floor = max(record.own_score, record.baseline_score) <= chance + 1e-12
        assert ("floor" in floor_notes) is at_floor
        if at_floor:
            fired += 1
            assert "two noise processes" in floor_notes["floor"]
    assert fired, "ten draws of pure noise and the floor note never fired; the rule is unreachable"

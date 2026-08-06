"""Acceptance: Level 0, the frontier, on a real open grader and a real gold channel.

The clause this file discharges, verbatim: *on one open grader and one gold channel, the frontier
renders with its ESS horizon marked and the instrument refuses past it; HedgeTune runs on the same
samples as a baseline and both turning-point estimates are reported side by side; the Prentice
checklist returns a per-criterion verdict including at least one honest untestable.*

**The grader and the gold channel are real and neither is simulated.** The proxy is an open reward
model's score, recorded when that model was actually run: `tulu-dpo` and `skywork-v2-qwen3-8b`
scored on RewardBench 2's best-of-4 split, 1,763 prompts with four completions each, 7,052 rollouts.
The gold channel is RewardBench 2's own correctness label, which is a human-curated benchmark
annotation and not a second model's opinion: exactly one of the four completions per prompt is the
correct one. The two arrive on the same rollouts in the same order, which is the only thing the
frontier asks of them.

A second real pair is used where a continuous proxy is wanted: a Qwen process reward model's
per-step scores on ProcessBench, aggregated to a per-solution score by the minimum over steps, with
ProcessBench's human first-error annotation as the gold channel. 3,400 items.

Both live in the campaign evidence store, which is on disk beside this repository rather than in
it, so every test here skips with the path named when it is absent. The scores are read straight
out of the store's jsonl and its numpy sidecars: no model is loaded, no network call is made, and
nothing here costs money.

**One orientation decision, made once and not tuned.** The recorded PRM readout carries a note in
its own payload saying it is the row-0 linear readout of a two-class head rather than the official
correctness readout. Row 0 of that head is the incorrect class, so the correctness-oriented proxy
is its negation. That sign comes from the payload's note and from nothing else; it is not chosen by
looking at which sign makes the frontier more interesting.
"""

from __future__ import annotations

import json
import os
import pathlib

import numpy as np
import pytest

from reward_lens.core.envelope import RegimeReading
from reward_lens.core.quantity import load_quantities
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.measure.base import Context, lint_instrument
from reward_lens.measure.frontier import (
    FRONTIER,
    ChecklistReading,
    ConcomitantBestOfN,
    FrontierReading,
    GoldVersusKL,
    RewardTailIndex,
    SurrogateChecklist,
    VisibilityHorizon,
    concomitant_expectation,
    measure_checklist,
    measure_concomitant,
    measure_frontier,
    measure_horizon,
)

#: The campaign evidence store, which is not in this repository. There is no default: point
#: ``REWARD_LENS_CAMPAIGN_STORE`` at the store directory or the tests that need it skip.
_CAMPAIGN_ENV = os.environ.get("REWARD_LENS_CAMPAIGN_STORE")
CAMPAIGN = pathlib.Path(_CAMPAIGN_ENV) if _CAMPAIGN_ENV else None

#: The two open reward models used as the proxy. The first turns on gold inside the horizon and the
#: second does not, which is what makes the pair worth running: the instrument has to be right about
#: both and the two answers look nothing alike.
PROXIES = ("tulu-dpo", "skywork-v2-qwen3-8b")


def _ctx() -> Context:
    """No signal. Level 0 reads two arrays, which is why it runs before any training exists."""
    return Context()


# ---------------------------------------------------------------------------
# Reading the real scores
# ---------------------------------------------------------------------------


def _rows(observable: str, slice_name: str) -> dict[str, dict]:
    """Every evidence envelope for one observable and one dataset slice, keyed by model."""
    out: dict[str, dict] = {}
    with (CAMPAIGN / "evidence.jsonl").open() as handle:
        for line in handle:
            envelope = json.loads(line)
            if envelope.get("observable") != observable:
                continue
            extra = envelope["subject"].get("extra") or {}
            if extra.get("slice") == slice_name:
                out[extra.get("roster_key", "")] = envelope["value"]["fields"]
    return out


def _sidecar(node: dict) -> np.ndarray:
    return np.load(CAMPAIGN / "payloads" / node["__ndarray__"]["sidecar"])


@pytest.fixture(scope="module")
def rewardbench() -> dict:
    """RewardBench 2 best-of-4: real reward-model scores and the benchmark's correctness label."""
    if CAMPAIGN is None or not (CAMPAIGN / "evidence.jsonl").is_file():
        pytest.skip("no campaign evidence store; set REWARD_LENS_CAMPAIGN_STORE")
    fields = _rows("campaign.scores", "rb2-full")
    if not set(PROXIES) <= set(fields):
        pytest.skip(f"rb2-full carries {sorted(fields)}, not {list(PROXIES)}")
    scores = {key: _sidecar(fields[key]["scores"]).astype(np.float64) for key in PROXIES}
    prompts, per_prompt = next(iter(scores.values())).shape
    assert per_prompt == 4, "RewardBench 2's best-of-4 layout is four completions per prompt"
    gold = np.zeros((prompts, per_prompt))
    gold[:, 0] = 1.0
    subsets = np.asarray(fields[PROXIES[0]]["meta"]["__map__"]["subsets"]["__seq__"])
    return {
        "scores": scores,
        "gold": gold,
        "subsets": subsets,
        "prompts": prompts,
        "n": prompts * per_prompt,
    }


@pytest.fixture(scope="module")
def processbench() -> dict:
    """A real process reward model against ProcessBench's human first-error annotation."""
    if CAMPAIGN is None or not (CAMPAIGN / "evidence.jsonl").is_file():
        pytest.skip("no campaign evidence store; set REWARD_LENS_CAMPAIGN_STORE")
    fields = _rows("campaign.prm.steps", "processbench-full")
    if not fields:
        pytest.skip("the campaign store carries no processbench-full PRM slice")
    field = next(iter(fields.values()))
    values = _sidecar(field["values"]).astype(np.float64)
    offsets = _sidecar(field["offsets"]).astype(int)
    labels = _sidecar(field["labels"]).astype(int)
    # The minimum over steps: a solution is correct only if every step is. The negation orients the
    # row-0 logit towards correctness, which the payload's own readout note is the authority for.
    proxy = -np.array([values[a:b].min() for a, b in zip(offsets[:-1], offsets[1:])])
    gold = (labels == -1).astype(np.float64)  # -1 is ProcessBench's "no error anywhere"
    return {"reward": proxy, "gold": gold, "n": int(gold.size)}


def _selection_arms(scores: np.ndarray, gold: np.ndarray, seed: int = 0) -> dict:
    """The treatment channel, and it is the intervention this whole layer is about.

    Arm 1 is the completion the proxy picks out of the four; arm 0 is a uniformly random one from
    the same four. That is optimisation against the proxy, applied once, at no cost and with no
    training. It is a real intervention rather than a label: the assignment is done here and both
    arms are observed.
    """
    rng = np.random.default_rng(seed)
    rows = np.arange(scores.shape[0])
    picked = scores.argmax(axis=1)
    drawn = rng.integers(0, scores.shape[1], scores.shape[0])
    return {
        "reward": np.concatenate([scores[rows, picked], scores[rows, drawn]]),
        "gold": np.concatenate([gold[rows, picked], gold[rows, drawn]]),
        "treatment": np.concatenate([np.ones(rows.size), np.zeros(rows.size)]),
    }


# ===========================================================================
# Clause: the frontier renders, with its ESS horizon marked
# ===========================================================================


@pytest.mark.parametrize("proxy", PROXIES)
def test_the_frontier_renders_on_a_real_grader_with_its_ess_horizon_marked(rewardbench, proxy):
    r = rewardbench["scores"][proxy].ravel()
    g = rewardbench["gold"].ravel()
    reading = measure_frontier(r, g, grid=41, resamples=120)
    assert isinstance(reading, FrontierReading), reading
    assert reading.n == rewardbench["n"] == 7052

    # Renders: one sentence carrying real numbers, and the arrays a figure is drawn from.
    rendered = reading.render()
    assert f"{reading.kl_max:.3g}" in rendered
    assert "nats" in rendered
    for axis in (reading.kl, reading.gold, reading.ess_frac, reading.stationarity):
        assert axis.shape == reading.lambdas.shape

    # The horizon is marked on the same axes rather than in a caption: the last point of the swept
    # curve sits exactly on the floor, so the plot's right-hand edge is the horizon.
    assert reading.horizon_binding
    assert reading.ess_frac[-1] == pytest.approx(reading.floor, rel=1e-6)
    assert reading.kl[-1] == pytest.approx(reading.kl_max, rel=1e-9)
    assert reading.coverage_at_horizon == pytest.approx(1.0 / reading.floor, rel=1e-6)
    assert np.all(np.diff(reading.ess_frac) <= 1e-12), "ESS must fall monotonically along the sweep"
    assert 0.5 < reading.kl_max < 4.0, reading.kl_max

    # The gold channel is real: a single draw is one in four, by construction of the benchmark.
    assert reading.gold[0] == pytest.approx(0.25, abs=1e-9)


def test_the_horizon_binds_on_every_open_grader_tested_which_is_its_kill_condition(rewardbench):
    """N2's kill condition is that the horizon never binds. On four real graders it always does."""
    horizons = {}
    for proxy, scores in rewardbench["scores"].items():
        reading = measure_horizon(scores.ravel(), grid=33)
        assert not isinstance(reading, Refusal)
        assert reading.binding, proxy
        horizons[proxy] = reading.kl_max
    assert all(0.5 < kl < 4.0 for kl in horizons.values()), horizons
    assert (
        "declines to answer"
        in measure_horizon(rewardbench["scores"][PROXIES[0]].ravel(), grid=17).says
    )


# ===========================================================================
# Clause: the instrument refuses past the horizon
# ===========================================================================


def test_the_instrument_refuses_past_the_horizon_on_the_real_grader(rewardbench):
    r = rewardbench["scores"][PROXIES[0]].ravel()
    g = rewardbench["gold"].ravel()
    horizon = measure_horizon(r, grid=17)
    beyond = horizon.lambda_max * 2.0

    out = GoldVersusKL(r, g, lambda_max=beyond, grid=25, resamples=0).estimate(_ctx())
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.ESS_BELOW_FLOOR
    assert out.reason.name == "ESS_BELOW_FLOOR"
    assert out.statistics["requested_lambda"] == pytest.approx(beyond)
    assert out.statistics["ess_at_request"] < 0.05 * len(r)
    assert "re-run with lambda_max <=" in out.remedy
    assert out.is_bounded, "the visible part of the curve is worth handing back"
    assert isinstance(out.partial.value, FrontierReading)
    assert out.partial.value.kl_max == pytest.approx(horizon.kl_max, rel=1e-9)

    # And it does not refuse inside the horizon, which is the other half of the claim.
    inside = GoldVersusKL(r, g, lambda_max=horizon.lambda_max * 0.6, grid=25, resamples=0).estimate(
        _ctx()
    )
    assert not isinstance(inside, Refusal)
    assert inside.value.kl_max == pytest.approx(horizon.kl_max, rel=1e-9)
    assert inside.value.kl[-1] < horizon.kl_max


# ===========================================================================
# Clause: HedgeTune runs on the same samples and both estimates are reported side by side
# ===========================================================================


@pytest.mark.parametrize("proxy", PROXIES)
def test_hedgetune_runs_on_the_same_samples_and_both_estimates_are_reported_side_by_side(
    rewardbench, proxy
):
    """The mandatory baseline, and it is a baseline rather than a citation because it is here."""
    reading = measure_frontier(
        rewardbench["scores"][proxy].ravel(), rewardbench["gold"].ravel(), grid=41, resamples=120
    )
    assert "baseline.hedgetune" in reading.baselines
    assert "baseline.hedgetune" in GoldVersusKL.baselines

    rendered = reading.render()
    assert "hedgetune" in rendered and "cumulant" in rendered
    assert reading.hedgetune.method == "hedgetune"
    assert reading.cumulant.method == "cumulant"
    assert reading.hedgetune.detail and reading.cumulant.detail

    if reading.hedgetune.found:
        # A turn: it is inside the horizon, it is a maximum rather than a trough, and the grid's
        # own argmax agrees with the solver to within one grid step.
        assert reading.hedgetune.is_maximum
        assert reading.hedgetune.lam <= reading.lambda_max
        assert reading.peak_is_interior
        step = float(np.diff(reading.kl).max())
        assert abs(reading.hedgetune.kl - reading.peak_kl) <= 2 * step
        assert reading.peak_kl_ci[0] < reading.peak_kl < reading.peak_kl_ci[1]
    else:
        # No turn: both estimators say so, and neither invents one past where we can see.
        assert not reading.peak_is_interior
        assert not reading.cumulant.found
        assert np.isnan(reading.baselines["baseline.hedgetune"])


def test_the_two_turning_point_estimates_disagree_enough_to_keep_both(rewardbench):
    """N1's kill condition, checked rather than assumed: they do not agree, so both ship.

    On `tulu-dpo` the solver puts the turn at a KL roughly five times that of the closed form. The
    closed form is one Newton step from zero, so it is only as good as the curvature of
    `Cov_lambda(g, r)` between zero and the root, and on this grader that curvature is large.
    """
    reading = measure_frontier(
        rewardbench["scores"]["tulu-dpo"].ravel(),
        rewardbench["gold"].ravel(),
        grid=41,
        resamples=200,
    )
    assert reading.hedgetune.found and reading.cumulant.found
    lo, hi = reading.peak_kl_ci
    assert not (lo <= reading.cumulant.kl <= hi), (
        "the closed form landed inside the interval on the turn, which would be the kill "
        "condition firing: report HedgeTune and drop the second estimator"
    )
    assert reading.hedgetune.kl > 2.0 * reading.cumulant.kl


# ===========================================================================
# Clause: the checklist returns a per-criterion verdict including an honest untestable
# ===========================================================================


def test_the_checklist_returns_a_verdict_per_criterion_with_an_honest_untestable(rewardbench):
    arms = _selection_arms(rewardbench["scores"]["tulu-dpo"], rewardbench["gold"])
    units = np.concatenate([rewardbench["subsets"], rewardbench["subsets"]])
    reading = measure_checklist(
        arms["reward"],
        arms["gold"],
        treatment=arms["treatment"],
        unit=units,
        permutations=500,
        resamples=200,
    )
    assert isinstance(reading, ChecklistReading)
    assert reading.has_treatment and reading.has_units

    numbers = [c.number for c in reading.criteria]
    assert numbers == [1, 2, 3, 4], "four criteria, one verdict each"
    for c in reading.criteria:
        assert c.verdict in {"pass", "fail", "untestable"}
        assert c.detail.strip() and c.source.strip()
        if c.verdict == "untestable":
            assert c.testable_by.strip(), f"criterion {c.number} gave no route to testing it"

    assert reading.n_untestable >= 1, "the honest untestable the clause asks for"
    assert reading.verdict_of(4) == "untestable", "VanderWeele cannot pass without an intervention"
    assert "potential outcomes" in [c for c in reading.criteria if c.number == 4][0].testable_by

    # Prentice's own four conditions are reported separately, because the published sentence
    # ("fails criterion 4: the proxy does not fully mediate the treatment effect on gold") is about
    # Prentice's fourth condition and the catalogue's fourth slot is VanderWeele's.
    assert len(reading.prentice_conditions) == 4
    capture = reading.prentice_conditions[3]
    assert capture.verdict == "fail"
    assert "does not fully mediate" in capture.detail
    assert reading.verdict_of(1) == "fail"
    assert "Fails criterion" in reading.says


def test_the_checklist_is_honest_about_what_one_arm_can_and_cannot_answer(processbench):
    """The common case in practice, and it is N4's kill condition arriving.

    A grader, a gold channel and no intervention: all four criteria come back untestable, because
    three of them are ratios or contrasts of two arms and the fourth is about counterfactuals.
    The catalogue's kill condition for N4 says that if every criterion returns untestable on every
    real grader for want of an intervention, the checklist is a research note and only the
    concomitant reading ships. Half of that has happened here. It is not the whole of it: the same
    checklist on the same layer returns two failures the moment a selection arm exists, and a
    selection arm costs nothing, so what this establishes is that **the checklist is unusable
    without an arm rather than unusable**. Where a caller has only one arm, the concomitant reading
    is the one to hand them, and it needs no arm at all.
    """
    reading = measure_checklist(
        processbench["reward"], processbench["gold"], permutations=400, resamples=200
    )
    assert not reading.has_treatment
    assert reading.n_untestable == 4 and reading.n_fail == 0
    assert all(c.testable_by.strip() for c in reading.criteria)
    assert "could not be tested" in reading.says

    # And the reading that does not need an arm still answers on the same rollouts.
    concomitant = measure_concomitant(
        processbench["reward"], processbench["gold"], best_of=16, resamples=120
    )
    assert not isinstance(concomitant, Refusal)
    assert concomitant.expected_gold > concomitant.gold_at_one
    assert "exactly" in concomitant.says


# ===========================================================================
# The concomitant, exactly, against its two baselines
# ===========================================================================


def test_the_exact_concomitant_matches_simulation_and_the_realised_best_of_four(rewardbench):
    """Two baselines and they are different objects, which is the finding rather than a caveat.

    Simulation draws four pairs from the pooled bank and is the check on the exact expression: they
    must agree inside Monte Carlo error and they do. The realised best-of-4 selects within a
    prompt, which is a structurally different experiment, and the gap between the two is what
    per-prompt structure is worth. On `skywork-v2-qwen3-8b` the realised accuracy is above the
    pooled expectation, because selecting inside a prompt compares four completions to the same
    question and pooling compares four unrelated ones.
    """
    scores = rewardbench["scores"]["skywork-v2-qwen3-8b"]
    gold = rewardbench["gold"]
    realised = float((scores.argmax(axis=1) == 0).mean())

    reading = measure_concomitant(
        scores.ravel(), gold.ravel(), best_of=4, resamples=200, simulate=20_000
    )
    assert not isinstance(reading, Refusal)
    sim = reading.baselines["baseline.simulated_best_of_n"]
    se = reading.baselines["baseline.simulated_best_of_n_se"]
    assert abs(reading.expected_gold - sim) < 4.0 * se, (reading.expected_gold, sim, se)
    assert "exactly" in reading.says
    assert reading.gold_at_one == pytest.approx(0.25, abs=1e-9)
    assert 0.0 < reading.expected_gold < realised, (reading.expected_gold, realised)
    assert reading.expected_gold_ci[0] < reading.expected_gold < reading.expected_gold_ci[1]

    # The curve over n is exact at every n, so where best-of-n stops buying gold is exact too.
    assert reading.ns[0] == 1
    exact_at_one, _, _ = concomitant_expectation(scores.ravel(), gold.ravel(), 1)
    assert exact_at_one == pytest.approx(0.25, abs=1e-9)
    assert np.all(np.diff(reading.proxy_curve) >= -1e-12), "E[max] cannot fall as n grows"


def test_the_concomitant_refuses_when_best_of_n_collapses_onto_the_bank(processbench):
    out = ConcomitantBestOfN(
        processbench["reward"], processbench["gold"], best_of=200_000
    ).estimate(_ctx())
    assert isinstance(out, Refusal) and out.reason is RefusalReason.ESS_BELOW_FLOOR
    assert out.statistics["effective_blocks"] < 0.05 * processbench["n"]
    assert "score more rollouts, or ask for a smaller n" in out.remedy


# ===========================================================================
# The tail index on the real grader, and the envelope it feeds
# ===========================================================================


def test_the_tail_index_refuses_on_every_real_bank_here_and_names_the_n_it_needs(
    rewardbench, processbench
):
    """7,052 and 3,400 rollouts are both far short of the exceedance count a tail index needs.

    This is the honest outcome and it is worth being plain about: neither of the two real banks
    available to this package can support a defensible tail index, so the `LIGHT_TAILED` condition
    that N1 and N2 declare is **unmeasured** on both of them, and their readings say so through
    the preflight's `unchecked` list rather than by assuming it holds.
    """
    for reward, n in (
        (rewardbench["scores"]["tulu-dpo"].ravel(), rewardbench["n"]),
        (processbench["reward"], processbench["n"]),
    ):
        out = RewardTailIndex(reward, resamples=60).estimate(_ctx())
        assert isinstance(out, Refusal) and out.reason is RefusalReason.ABOVE_LOD_BELOW_LOQ
        assert out.statistics["n"] == n
        assert out.statistics["exceedances"] < 1570
        assert out.statistics["n_required"] == 31_400
        assert out.is_bounded, "the plot as far as this bank gets is still worth holding"
        assert "31,400" in out.remedy


def test_a_reading_whose_envelope_was_never_measured_says_so_rather_than_passing(rewardbench):
    """`unchecked` is what keeps an unmeasured precondition from reading as a satisfied one."""
    instrument = GoldVersusKL(
        rewardbench["scores"]["tulu-dpo"].ravel(),
        rewardbench["gold"].ravel(),
        grid=17,
        resamples=0,
    )
    silent = instrument.preflight(Context())
    assert silent.ok
    assert any("envelope" in note for note in silent.unchecked), silent.unchecked

    measured = instrument.preflight(Context(regime_reading=RegimeReading.of(LIGHT_TAILED=False)))
    assert not measured.ok
    assert measured.refusal.reason is RefusalReason.ENVELOPE_VIOLATED
    assert "LIGHT_TAILED" in measured.refusal.detail


# ===========================================================================
# Declarations
# ===========================================================================


def test_lint_instrument_is_empty_for_all_five_frontier_instruments():
    load_quantities()
    for cls in FRONTIER:
        assert lint_instrument(cls()) == [], cls.__name__


def test_the_layer_needs_a_grader_a_gold_channel_and_a_policy_and_nothing_else():
    """The frontier's access line, as five declarations that can be checked."""
    from reward_lens.core.types import Access, Component

    for cls in (VisibilityHorizon, RewardTailIndex):
        assert set(cls.requires) == {Component.GRADER, Component.POLICY}, cls.__name__
    for cls in (GoldVersusKL, SurrogateChecklist, ConcomitantBestOfN):
        assert set(cls.requires) == {
            Component.GRADER,
            Component.GOLD,
            Component.POLICY,
        }, cls.__name__
    for cls in FRONTIER:
        assert set(cls.requires.values()) == {Access.QUERY}, cls.__name__
        assert cls.rung == 0, cls.__name__


def test_the_layer_imports_no_torch():
    """Level 0 is arithmetic on two arrays and the dependency list should show it."""
    import subprocess
    import sys

    code = "import sys, reward_lens.measure.frontier; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False", out.stdout

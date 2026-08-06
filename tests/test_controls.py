"""Unit tests for the four control instruments (M3, M4, M5, M10 in `reward_lens.measure.controls`).

The tests that matter here are the ones that check a gate refuses rather than warns. A placebo arm
that is missing, a null whose control was never run, a control run at a different n, a control that
missed its own planted effect: each of those has to come back as a `Refusal` carrying a remedy, and
each is checked separately, because they fail for different reasons and need different
instructions.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.measure.base import Context, lint_instrument
from reward_lens.measure.controls import (
    CONTROLS,
    PLACEBO_BANK,
    ControlDesign,
    DumbBaselineBank,
    InterventionArm,
    MatchedControl,
    MatchedPositiveControl,
    NullClaim,
    PowerAndMDE,
    SemanticPlacebo,
    compare_to_placebo,
    contrast,
    default_contrast,
    gate_null,
    guard_null,
    random_gaussian_direction,
    register_proposed,
    resolve_row,
    semantic_placebo,
)
from reward_lens.stats.baselines import DetectionTask, lint_claim
from reward_lens.stats.power import PairedBinaryDesign, resolution_ratio


def _ctx() -> Context:
    """A context with no signal at all, which is the honest shape for these four.

    None of them touches a network: they read injected data, which is why they need no signal. The
    kernel now makes `Context.signal` optional, so the tests say so rather than manufacturing a
    stub reward signal to satisfy a field nobody reads.
    """
    return Context()


def _encoder(dim: int = 64):
    """A deterministic word embedder: each word gets a fixed pseudo-random vector, phrases average.

    Enough geometry to test norm matching and cosines without a model. Two phrase sets that share
    no words come out close to orthogonal, which is what a coherent contrast looks like.
    """

    def vector(word: str) -> np.ndarray:
        digest = hashlib.sha256(word.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big")
        return np.random.default_rng(seed).standard_normal(dim)

    def encode(phrases):
        return np.stack([np.mean([vector(w) for w in p.lower().split()], axis=0) for p in phrases])

    return encode


# ---------------------------------------------------------------------------
# M4: the placebo bank
# ---------------------------------------------------------------------------


def test_the_bank_holds_coherent_contrasts_each_with_a_stated_rationale():
    assert len(PLACEBO_BANK) >= 6
    for c in PLACEBO_BANK:
        assert len(c.positive) >= 4 and len(c.negative) >= 4
        assert len(c.rationale.split()) >= 6
        assert c.id == c.id.lower()


def test_the_two_published_contrasts_are_in_the_bank_and_say_where_they_came_from():
    for cid in ("vampires_vs_werewolves", "top_shelf_vs_bottom_shelf"):
        c = contrast(cid)
        assert "published" in c.source


def test_a_contrast_with_no_rationale_cannot_be_constructed():
    from reward_lens.measure.controls.placebo import PlaceboContrast

    with pytest.raises(ValueError, match="rationale"):
        PlaceboContrast(id="x", positive=("a",), negative=("b",), rationale="  ")


def test_selecting_a_contrast_is_the_default_path():
    """No contrast argument, and a direction comes back. That inversion is the whole design."""
    direction = semantic_placebo(_encoder())
    assert direction.contrast in {c.id for c in PLACEBO_BANK}
    assert direction.vector.shape == (64,)
    assert direction.norm > 0


def test_the_default_contrast_is_deterministic_and_respects_exclusions():
    assert default_contrast(seed=3).id == default_contrast(seed=3).id
    picked = default_contrast(seed=0).id
    assert default_contrast(seed=0, exclude=(picked,)).id != picked


def test_the_placebo_is_norm_matched_to_the_claimed_direction():
    claimed = np.random.default_rng(0).standard_normal(64) * 7.5
    direction = semantic_placebo(_encoder(), match_to=claimed)
    assert direction.is_norm_matched
    assert direction.norm == pytest.approx(float(np.linalg.norm(claimed)))
    assert direction.cosine_to_target is not None
    assert abs(direction.cosine_to_target) < 0.6  # coherent, and not aligned with the claim


def test_the_random_gaussian_baseline_is_also_norm_matched_so_the_two_are_comparable():
    claimed = np.random.default_rng(1).standard_normal(64) * 3.0
    v = random_gaussian_direction(64, match_to=claimed, seed=0)
    assert float(np.linalg.norm(v)) == pytest.approx(float(np.linalg.norm(claimed)))


def test_a_claim_with_no_placebo_arm_is_refused():
    out = compare_to_placebo(InterventionArm("hacking direction", effect=-0.42), None)
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.NO_MATCHED_CONTROL
    assert "semantic_placebo" in out.remedy
    assert "vampires" in out.remedy


def test_a_placebo_that_does_the_same_job_reports_the_claim_as_non_specific():
    rng = np.random.default_rng(2)
    claimed = InterventionArm("real", effect=-0.40, per_item=rng.normal(-0.40, 0.10, 200))
    placebo = InterventionArm("vampires", effect=-0.39, per_item=rng.normal(-0.39, 0.10, 200))
    out = compare_to_placebo(claimed, placebo, seed=0)
    assert out.verdict == "non_specific"
    assert "NOT SPECIFIC" in out.render()
    assert out.ratio == pytest.approx(0.975, abs=0.01)


def test_a_placebo_that_does_nothing_lets_the_claim_be_specific():
    rng = np.random.default_rng(3)
    claimed = InterventionArm("real", effect=-0.40, per_item=rng.normal(-0.40, 0.10, 200))
    placebo = InterventionArm("vampires", effect=-0.02, per_item=rng.normal(-0.02, 0.10, 200))
    out = compare_to_placebo(claimed, placebo, seed=0)
    assert out.verdict == "specific"


def test_two_bare_scalars_are_unresolved_rather_than_specific():
    out = compare_to_placebo(
        InterventionArm("real", effect=-0.40), InterventionArm("vampires", effect=-0.02)
    )
    assert out.verdict == "unresolved"
    assert "UNRESOLVED" in out.render()


def test_a_placebo_that_was_not_norm_matched_is_flagged_in_the_render():
    rng = np.random.default_rng(4)
    direction = semantic_placebo(_encoder())  # no match_to
    out = compare_to_placebo(
        InterventionArm("real", effect=-0.4, per_item=rng.normal(-0.4, 0.1, 100)),
        InterventionArm("placebo", effect=-0.05, per_item=rng.normal(-0.05, 0.1, 100)),
        direction=direction,
    )
    assert "NOT norm-matched" in out.render()


# ---------------------------------------------------------------------------
# M5: the matched positive control
# ---------------------------------------------------------------------------


def _design(n: int = 40, **kw) -> ControlDesign:
    return ControlDesign(n=n, alpha=0.05, tails=2, statistic="paired-permutation", **kw)


def _null(mde: float | None = 0.18) -> NullClaim:
    return NullClaim(
        instrument="Susceptibility", effect=0.01, p_value=0.62, design=_design(), mde=mde
    )


def test_a_result_that_is_not_a_null_passes_the_gate_untouched():
    hit = NullClaim(instrument="X", effect=0.4, p_value=0.001, design=_design())
    verdict = gate_null(hit, None)
    assert verdict.ok and "not a null" in verdict.note


def test_a_null_with_no_matched_control_is_refused():
    verdict = gate_null(_null(), None)
    assert not verdict.ok
    assert verdict.refusal.reason is RefusalReason.NO_MATCHED_CONTROL
    assert "matched positive control at the same n" in verdict.refusal.remedy
    assert "underpowered" in verdict.refusal.remedy


def test_the_refusal_carries_the_minimum_detectable_effect_as_an_honest_bound():
    assert "minimum detectable effect is 0.18" in gate_null(_null(), None).refusal.detail
    assert "minimum detectable effect" not in gate_null(_null(mde=None), None).refusal.detail


def test_a_control_at_a_different_n_is_not_a_matched_control():
    control = MatchedControl(
        id="planted", design=_design(n=400), planted_effect=0.3, observed_effect=0.31, p_value=1e-6
    )
    verdict = gate_null(_null(), control)
    assert not verdict.ok
    assert "n 40.0 against 400.0" in verdict.refusal.detail


def test_a_control_at_the_same_n_but_a_different_statistic_is_not_matched():
    control = MatchedControl(
        id="planted",
        design=ControlDesign(n=40, alpha=0.05, tails=2, statistic="t-test"),
        planted_effect=0.3,
        observed_effect=0.31,
        p_value=1e-6,
    )
    assert "statistic" in gate_null(_null(), control).refusal.detail


def test_a_control_that_missed_its_own_planted_effect_refuses_hardest():
    control = MatchedControl(
        id="planted", design=_design(), planted_effect=0.3, observed_effect=0.05, p_value=0.44
    )
    verdict = gate_null(_null(), control)
    assert not verdict.ok
    assert "missed it" in verdict.refusal.detail
    assert "raise n" in verdict.refusal.remedy


def test_a_matched_control_that_worked_lets_the_null_stand():
    control = MatchedControl(
        id="planted", design=_design(), planted_effect=0.3, observed_effect=0.29, p_value=1e-4
    )
    verdict = gate_null(_null(), control)
    assert verdict.ok
    assert "informative" in verdict.render()


def test_effective_n_rather_than_row_count_decides_whether_two_designs_match():
    """Fifty fresh stimuli and fifty mutations of five seeds have the same n and are not matched."""
    claim = NullClaim(
        instrument="X",
        effect=0.01,
        p_value=0.6,
        design=ControlDesign.from_lineage([f"s{i // 10}" for i in range(50)]),
    )
    control = MatchedControl(
        id="planted",
        design=ControlDesign.from_lineage([f"s{i}" for i in range(50)]),
        planted_effect=0.3,
        observed_effect=0.3,
        p_value=1e-6,
    )
    assert "effective n" in gate_null(claim, control).refusal.detail


# ---------------------------------------------------------------------------
# The wrapper that makes the gate real
# ---------------------------------------------------------------------------


class _NullReporter:
    """An instrument that reports a null and does not think about controls."""

    name = "NullReporter"
    version = "1.0"

    def preflight(self, ctx):  # pragma: no cover - trivial
        from reward_lens.measure.base import PreflightResult

        return PreflightResult(instrument=self.name, ok=True)

    def estimate(self, ctx):
        return ctx.emit({"effect": 0.01, "p_value": 0.62, "n": 40, "alpha": 0.05})


def test_wrapping_an_instrument_makes_its_null_refuse_without_a_control():
    wrapped = guard_null(_NullReporter())
    out = wrapped.estimate(_ctx())
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.NO_MATCHED_CONTROL


def test_the_same_wrapped_instrument_passes_when_a_matched_control_is_in_the_context():
    ctx = _ctx()
    ctx.stats["matched_control"] = MatchedControl(
        id="planted",
        design=ControlDesign(n=40, alpha=0.05),
        planted_effect=0.3,
        observed_effect=0.3,
        p_value=1e-6,
    )
    out = guard_null(_NullReporter(), design=ControlDesign(n=40, alpha=0.05)).estimate(ctx)
    assert not isinstance(out, Refusal)


def test_a_wrapped_instrument_still_forwards_its_declarations():
    inner = MatchedPositiveControl()
    wrapped = guard_null(inner)
    assert wrapped.name == inner.name
    assert wrapped.quantity == inner.quantity
    assert wrapped.baselines == inner.baselines


# ---------------------------------------------------------------------------
# M10 as an instrument
# ---------------------------------------------------------------------------


def test_the_power_instrument_emits_every_number_with_the_calculators_as_baselines():
    design = PairedBinaryDesign(n=500, accuracy_a=0.82, accuracy_b=0.85, rho=0.5)
    reading = PowerAndMDE(design, replicates=2000).estimate(_ctx())
    payload = reading.value
    assert payload["validated_against"] == "simulation"
    assert payload["q"] < 1.0 and payload["resolved"] is False
    assert len(payload["baselines"]) == 5


def test_the_power_instrument_refuses_with_no_design_rather_than_assuming_one():
    out = PowerAndMDE().estimate(_ctx())
    assert isinstance(out, Refusal)
    assert "PairedBinaryDesign" in out.remedy


def test_an_unresolved_row_refuses_rather_than_returning_a_ranking():
    out = resolve_row("Leaderboard", resolution_ratio(500, 1274), observed_gap=0.03, mde=0.06)
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.BELOW_LOD
    assert "not resolved" in out.detail
    assert out.statistics["q"] < 1.0


def test_a_resolved_row_returns_the_resolution_rather_than_a_refusal():
    out = resolve_row("Leaderboard", resolution_ratio(2000, 1274))
    assert not isinstance(out, Refusal)
    assert out.resolved


# ---------------------------------------------------------------------------
# M3 as an instrument
# ---------------------------------------------------------------------------


def _detection_task(n: int = 120, seed: int = 0) -> DetectionTask:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.5).astype(int)
    words = ("alpha", "beta", "gamma", "delta")
    texts = tuple(
        " ".join(rng.choice(words, size=25)) + (" exit(0)" if label else "") for label in y
    )
    return DetectionTask(
        labels=y, texts=texts, series=rng.normal(0, 1, n) + 0.8 * y, markers=("exit(0)",)
    )


def test_the_bank_instrument_emits_a_baselines_mapping_that_passes_the_claim_lint():
    task = _detection_task()
    rng = np.random.default_rng(5)
    own = 2.0 * task.labels + rng.normal(0, 1, task.n)
    reading = DumbBaselineBank(task, own).estimate(_ctx())
    assert reading.value["best_baseline"] == "baseline.string_match"
    assert reading.value["best_score"] == pytest.approx(1.0)
    assert lint_claim(reading, require=("baseline.string_match", "baseline.tfidf")) == []


def test_the_bank_instrument_refuses_with_no_task():
    out = DumbBaselineBank().estimate(_ctx())
    assert isinstance(out, Refusal)
    assert "DetectionTask" in out.remedy


def test_measure_raises_when_estimate_would_have_refused():
    """`measure` is the Observable half and returns Evidence by contract; `estimate` refuses."""
    with pytest.raises(ValueError, match="declines to produce"):
        DumbBaselineBank().measure(_ctx())


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------


def test_all_four_instruments_lint_clean_once_the_proposed_quantities_are_registered():
    from reward_lens.core.quantity import load_quantities

    load_quantities()
    register_proposed()
    for cls in CONTROLS:
        assert lint_instrument(cls()) == [], cls.__name__


def test_every_control_declares_a_baseline_of_its_own():
    for cls in CONTROLS:
        assert cls.baselines, f"{cls.__name__} declares no baseline"


def test_preflight_names_the_checks_it_could_not_run_rather_than_passing_them():
    result = SemanticPlacebo().preflight(_ctx())
    assert result.ok
    assert "access" in result.unchecked
    assert any("envelope" in u for u in result.unchecked)

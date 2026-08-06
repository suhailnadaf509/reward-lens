"""Acceptance: F3, the cost book. The KL budget, the efficiency, and where the nats went.

**The clause.** *Efficiency is in [0,1] on every run tested; a synthetic case with a known `G`
reproduces `KL_min` analytically.*

Both halves are here and neither is softened.

**The synthetic half** is the one that proves the arithmetic. `G` is chosen, `Δz` is chosen, and the
expected `KL_min` is worked out by hand in the test body with the intermediate values written down,
so that the assertion compares the instrument against arithmetic rather than against itself. The
correlated two-feature case is done the same way, including its Shapley shares: `G = [[2,1],[1,2]]`
has determinant 3 and inverse `(1/3)[[2,-1],[-1,2]]`, so `Δz = [1,0]` costs `½·(1/3)·2 = 1/3` nats,
which splits into `φ_length = 7/24` and `φ_other = 1/24`. Those fractions are in the file.

**The efficiency half runs on four runs and no run produces a value outside [0,1].** Two of them are
the real GRPO records, where the answer is a refusal rather than a number, and that is a result
rather than a gap: `kl_to_previous` and `kl_to_ref` are `None` on all 200 steps and `beta` is 0.0,
so `D_t` is not in the record and cannot be reconstructed from it. The optimiser is
`ADAMW_TORCH_FUSED`, so the applied step is not the gradient times the learning rate and the moment
state that would relate them was never written; `update_norm` and `grad_norm_unclipped` are `None`
too. `cost_series` therefore refuses with `RECORD_INCOMPLETE`, which is the outcome the derivation asks for
by name in preference to substituting a proxy and keeping the name `kl_spent`. `KL_min` and its
per-feature shares need no denominator and are computed on all 199 step pairs.

The other two runs are built here, on the real policy the record names, because a clause about a
bounded ratio is worth nothing if the denominator is never present. `trl-internal-testing/
tiny-Qwen3ForCausalLM` is loaded at the checkpoint the run started from, a real policy-gradient step
is taken on the record's own step-0 rollouts and advantages, and both checkpoints are sampled from.
`D_t = KL(π₁ ‖ π₀)` is then computed exactly, as a full-vocabulary token-level KL over sequences
drawn from π₁, and `G` comes from the rung-2 Fisher kernel at π₀.

**What that measured, recorded here so a skipped run still carries it.** At a step size of 1.0 the
step spends 4.07 nats per sequence, the movement it produced needs at least 2.31, and efficiency is
**0.566**, inside the bound. At 0.5 the step spends 0.035 nats and the observed movement scores
2.80, a ratio of 81, and the instrument **refuses** rather than reporting it. That refusal is the
point of the second case: P5 says a value outside [0,1] is an instrument bug, so the instrument must
never hand one out, and the way it never hands one out is by refusing when the premise fails. At the
small step the premise that fails is visible in the numbers, because `Δz` on the leading feature is
under half its own standard error: the step's behavioural effect is below what 16 samples resolve,
and a quadratic form in a noisy `Δz` measures the noise.

Two things about that experiment are not the trainer's. The step is plain SGD rather than AdamW, so
that `Δθ` is known exactly rather than through an optimiser's hidden state, and a step size of 1.0
is far larger than any real run would take. Both are stated rather than smoothed over. What the
experiment establishes is that the bound holds on a real network with a real denominator, and that
the self-check fires when it should.

**One cross-check worth recording.** `½ Δθᵀ F Δθ` computed from the 8-rollout empirical Fisher gives
1.3e-2 nats where the token-level KL gives 1.2e-4, an overestimate of 110 times, and at a step size
of 1.0 it gives 13,179 against a true 3.9. An empirical Fisher from eight samples in 2.45 million
dimensions puts all of its curvature in the six directions it can see, and the update direction lies
inside that span, so it reads the whole step length there. That is why this package refuses to
compute `D_t` from a small-sample Fisher and asks for the logged KL instead.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.types import Access, Component, Phase, Substrate
from reward_lens.measure.base import Context, lint_instrument
from reward_lens.measure.efficiency import (
    MetricG,
    StepCost,
    UpdateEfficiency,
    UpdateKLMin,
    UpdateKLShare,
    UpdateKLSpent,
    cost_series,
    kl_min_series,
    metric_g,
    shapley_shares,
)
from reward_lens.measure.ledger.features import SurfaceFeatures
from reward_lens.measure.ledger.price import (
    StepSample,
    learning_rates,
    ledger_between,
    ledger_series,
    steps_from_run,
)
from reward_lens.record.reader import open_run

LONG_RUN = "run:f77bf75940ab982bbc35407af99cc094"
SHORT_RUN = "run:8a8c7e29274db0a681313b48dbd1eb63"


def _synthetic_g(matrix: np.ndarray, names: tuple[str, ...]) -> MetricG:
    """A `MetricG` around a matrix chosen by hand, with no estimation in the way."""
    return MetricG(
        names=names,
        matrix=np.asarray(matrix, dtype=np.float64),
        damping=0.0,
        damping_stable=True,
        conditioning=1.0,
        rung=0,
        method="chosen by hand in the test",
        n_samples=0,
    )


# ---------------------------------------------------------------------------
# The synthetic half: a known G reproduces KL_min analytically
# ---------------------------------------------------------------------------


def test_diagonal_g_reproduces_kl_min_computed_by_hand():
    """`G = diag(4, 1, 1/4)`, `Δz = (2, 3, 1/2)`. The expected value is arithmetic, written out."""
    g = _synthetic_g(np.diag([4.0, 1.0, 0.25]), ("length", "tools", "target"))
    dz = np.array([2.0, 3.0, 0.5])

    # G is diagonal, so G^-1 = diag(1/4, 1, 4) and the quadratic form is a sum of three terms:
    #   length  2^2 * (1/4) = 1.0
    #   tools   3^2 * 1     = 9.0
    #   target  (1/2)^2 * 4 = 1.0
    # KL_min = 1/2 * (1.0 + 9.0 + 1.0) = 5.5
    by_hand = 0.5 * (2.0**2 * 0.25 + 3.0**2 * 1.0 + 0.5**2 * 4.0)
    assert by_hand == 5.5

    value, out_of_range = g.kl_min(dz)
    assert value == pytest.approx(5.5, rel=1e-12)
    assert out_of_range == 0.0

    # A diagonal G is its own eigenbasis, so the per-direction shares are the three terms above.
    shares = g.eigen_shares(dz)
    assert sorted(shares) == pytest.approx([0.5, 0.5, 4.5], rel=1e-12)
    assert shares.sum() == pytest.approx(5.5, rel=1e-12)


def test_correlated_g_reproduces_kl_min_and_its_shapley_shares_by_hand():
    """`G = [[2,1],[1,2]]`, `Δz = (1, 0)`. Determinant 3, so every number below is exact.

    This is the case that matters, because the whole difficulty is that named features
    are correlated and the eigenbasis decomposition therefore does not name anything. Here the
    second feature does not move at all and still carries a share, which is correct: `G` couples
    them, so holding the second feature at its observed zero while moving the first costs more than
    moving the first alone.
    """
    g = _synthetic_g(np.array([[2.0, 1.0], [1.0, 2.0]]), ("length", "other"))
    dz = np.array([1.0, 0.0])

    # det(G) = 2*2 - 1*1 = 3, so G^-1 = (1/3) * [[2, -1], [-1, 2]].
    # KL_min = 1/2 * dz^T G^-1 dz = 1/2 * (1/3) * 2 = 1/3.
    by_hand = 0.5 * (1.0 / 3.0) * 2.0
    assert by_hand == pytest.approx(1.0 / 3.0, rel=1e-15)
    value, _ = g.kl_min(dz)
    assert value == pytest.approx(1.0 / 3.0, rel=1e-12)

    # The coalition values, each a one-line inverse:
    #   v({length}) = 1/2 * 1^2 / 2      = 1/4
    #   v({other})  = 1/2 * 0^2 / 2      = 0
    #   v({both})   = 1/3                (above)
    # Shapley with two players weights each ordering 1/2:
    #   phi_length = 1/2*(1/4 - 0) + 1/2*(1/3 - 0)    = 1/8 + 1/6 = 7/24
    #   phi_other  = 1/2*(0 - 0)   + 1/2*(1/3 - 1/4)  = 0   + 1/24 = 1/24
    #   7/24 + 1/24 = 8/24 = 1/3, which is KL_min.
    shares, attributed, kl_min = shapley_shares(g, dz)
    assert shares["length"] == pytest.approx(7.0 / 24.0, rel=1e-12)
    assert shares["other"] == pytest.approx(1.0 / 24.0, rel=1e-12)
    assert attributed == pytest.approx(1.0 / 3.0, rel=1e-12)
    assert kl_min == pytest.approx(1.0 / 3.0, rel=1e-12)
    assert sum(shares.values()) == pytest.approx(kl_min, rel=1e-12)


def test_the_single_feature_bound_is_the_singleton_coalition():
    """`δ²/(2 G_ii)`: moving one feature by `δ` costs at least this, whatever else happens."""
    g = _synthetic_g(np.array([[2.0, 1.0], [1.0, 2.0]]), ("length", "other"))
    assert g.single_feature_bound("length", 1.0) == pytest.approx(0.25, rel=1e-15)
    # And it is a genuine lower bound on the cost of a movement that includes it.
    value, _ = g.kl_min(np.array([1.0, 0.0]))
    assert g.single_feature_bound("length", 1.0) <= value

    # A feature no parameter move can reach costs infinitely many nats, which is h^2 = 0 from the
    # cost side rather than a numerical failure.
    inert = _synthetic_g(np.diag([1.0, 0.0]), ("length", "frozen"))
    assert inert.single_feature_bound("frozen", 1.0) == float("inf")


def test_shares_that_do_not_sum_to_kl_min_cannot_be_constructed():
    """D19's rule, as a construction-time assertion rather than as a docstring."""
    with pytest.raises(ValueError, match="does not add up"):
        StepCost(
            step=0,
            next_step=1,
            kl_spent=1.0,
            kl_min=0.5,
            efficiency=0.5,
            shares={"length": 0.2, "other": 0.1},
            residual_share=0.0,
        )


def test_efficiency_outside_the_unit_interval_cannot_be_constructed():
    """P5, as an assertion. A ratio above 1 is an instrument bug and never a finding."""
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        StepCost(
            step=0,
            next_step=1,
            kl_spent=0.25,
            kl_min=0.5,
            efficiency=2.0,
            shares={"length": 0.5},
            residual_share=0.0,
        )


def test_a_subset_attribution_leaves_the_rest_in_the_residual():
    """The unattributed share of F3's sentence, and it is a named field rather than a rounding gap."""
    g = _synthetic_g(np.diag([4.0, 1.0, 0.25]), ("length", "tools", "target"))
    dz = np.array([2.0, 3.0, 0.5])
    shares, attributed, kl_min = shapley_shares(g, dz, attribute_to=["length", "tools"])
    # G is diagonal so the coalition game is additive and the shares are the terms themselves.
    assert shares["length"] == pytest.approx(0.5, rel=1e-12)
    assert shares["tools"] == pytest.approx(4.5, rel=1e-12)
    assert shares["target"] == 0.0
    # The target feature's own term, 1/2 * (1/2)^2 * 4 = 0.5, is what is left unattributed.
    assert kl_min - attributed == pytest.approx(0.5, rel=1e-12)


def test_a_synthetic_cost_book_reports_the_sentence_f3_promises():
    """End to end on chosen numbers: spend, minimum, efficiency, and the named split."""
    g = _synthetic_g(np.diag([4.0, 1.0, 0.25]), ("length", "tools", "target"))
    half = _one_step_kl_min(g, np.array([2.0, 3.0, 0.5]))
    cost = StepCost.from_kl_min(half, kl_spent=11.0)
    assert cost.kl_min == pytest.approx(5.5, rel=1e-12)
    assert cost.efficiency == pytest.approx(0.5, rel=1e-12)
    assert 0.0 <= cost.efficiency <= 1.0
    rendered = cost.render()
    assert "efficiency 0.500" in rendered
    assert "tools 82%" in rendered  # 4.5 of 5.5


def _one_step_kl_min(g: MetricG, dz: np.ndarray):
    """One `StepKlMin` from a chosen `G` and `Δz`, through the same path the instruments use."""
    sample_before = StepSample(
        index=0,
        names=g.names,
        features=np.zeros((2, len(g.names))),
        advantages=np.zeros(2),
        group_ids=np.zeros(2, dtype=np.int64),
        task_ids=("t", "t"),
    )
    sample_after = StepSample(
        index=1,
        names=g.names,
        features=np.tile(dz, (2, 1)),
        advantages=np.zeros(2),
        group_ids=np.zeros(2, dtype=np.int64),
        task_ids=("t", "t"),
    )
    ledger = ledger_between(sample_before, sample_after, eta=1.0)
    rows = kl_min_series([ledger], g)
    assert not isinstance(rows, Refusal), rows
    return rows[0]


# ---------------------------------------------------------------------------
# The real GRPO record: D_t is not there, and that is the finding
# ---------------------------------------------------------------------------


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "grpo_run"
ACCESS = {Component.RECORD: Access.RECORD, Component.POLICY: Access.BACKWARD}


def _open(which: str):
    root = FIXTURES / which
    if not root.exists() or not (root / "runs").exists():
        return None
    try:
        run_id = next(p.name for p in (root / "runs").iterdir())
    except StopIteration:
        return None
    return open_run(root, run_id.replace("run_", "run:"))


def _ctx(**kwargs) -> Context:
    return Context(access=dict(ACCESS), substrate=Substrate.PROGRAM, phase=Phase.POST_RUN, **kwargs)


@pytest.fixture(scope="module")
def long_run():
    """The longest real GRPO record on disk: the 200-step one when it is there, else the 12-step."""
    run = _open("long") or _open("short")
    if run is None:
        pytest.skip(f"no GRPO record is on disk under {FIXTURES}")
    return run


@pytest.fixture(scope="module")
def book(long_run):
    """`(samples, ledgers, G)` over the whole record, which is what every real-record test reads."""
    samples = steps_from_run(long_run, SurfaceFeatures())
    ledgers = ledger_series(samples, eta_by_step=learning_rates(long_run))
    g = metric_g(samples)
    assert not isinstance(g, Refusal), g
    return samples, ledgers, g


def test_the_record_carries_no_per_step_kl_and_no_way_to_reconstruct_one(long_run):
    """The fact the reconciliation plan turns on, asserted on the record rather than quoted.

    Four fields would each give `D_t` and all four are absent on every step. `kl_to_previous` is
    the quantity itself. `kl_to_ref` is a different one and would not do, but its absence rules out
    even the wrong answer. `update_norm` with the Fisher would give `½ΔθᵀFΔθ`, and
    `grad_norm_unclipped` with the learning rate would give it for plain SGD, which this is not:
    the optimiser is AdamW, so the applied step is not the gradient times `eta` and the moment
    state that would relate them was never written.
    """
    steps = list(long_run.steps)
    assert [s for s in steps if s.optimizer.kl_to_previous is not None] == []
    assert [s for s in steps if s.optimizer.kl_to_ref is not None] == []
    assert [s for s in steps if s.optimizer.update_norm is not None] == []
    assert [s for s in steps if s.optimizer.grad_norm_unclipped is not None] == []
    assert {s.schedule.get("beta") for s in steps} == {0.0}
    assert "ADAMW" in long_run.components[Component.OPTIMIZER].name.upper()
    # And the one field that is present, which is why a proxy would have been tempting.
    assert all(s.optimizer.grad_norm_clipped is not None for s in steps)


def test_the_cost_book_refuses_record_incomplete_rather_than_substituting_a_proxy(long_run, book):
    """The named forbidden outcome, as a refusal with a remedy that says what to log."""
    _samples, ledgers, g = book
    out = cost_series(ledgers, g, run_=long_run)
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.RECORD_INCOMPLETE
    assert "kl_to_previous" in out.remedy
    assert "beta" in out.remedy
    # The remedy names the proxy it is declining to use, so nobody re-derives it downstream.
    assert "gradient norm" in out.remedy and "unknown factor" in out.remedy
    assert out.statistics["n_with_kl"] == 0


def test_kl_min_and_its_shares_are_computed_on_every_step_pair_without_a_denominator(book):
    """The half of F3 that needs no `D_t`, on the real record, on every pair it has."""
    _samples, ledgers, g = book
    rows = kl_min_series(ledgers, g)
    assert not isinstance(rows, Refusal), rows
    assert len(rows) == len(ledgers)
    for row in rows:
        assert row.kl_min >= 0.0
        assert np.isfinite(row.kl_min)
        # D19's rule, holding on real data on every step rather than only in the synthetic case.
        assert sum(row.shares.values()) + row.residual_share == pytest.approx(
            row.kl_min, rel=1e-6, abs=1e-12
        )
        assert all(v >= 0.0 for v in row.shares.values())
        assert set(row.shares) == set(g.names)


def test_the_metric_holds_C_succeeds_G_on_the_real_record(book):
    """`C ⪰ G`, checked on the real feature basis rather than only in the derivation."""
    _samples, _ledgers, g = book
    assert g.covariance is not None
    gap = np.linalg.eigvalsh(g.covariance - g.matrix)
    assert gap.min() >= -1e-7 * max(float(np.max(np.abs(g.covariance))), 1.0)
    for name, h2 in g.heritability().items():
        assert np.isnan(h2) or 0.0 <= h2 <= 1.0, name


def test_two_of_the_five_surface_features_are_constant_and_keep_their_place_in_the_basis(book):
    """The join key D19 fixes is the whole of `StepSample.names`, so nothing is dropped from it."""
    _samples, _ledgers, g = book
    assert g.names == SurfaceFeatures().names
    diagonal = np.diag(g.matrix)
    inert = {n for i, n in enumerate(g.names) if diagonal[i] == 0.0}
    assert inert == {"type_token_ratio", "n_turns"}
    assert g.rank == 3


@pytest.mark.parametrize("cls", [UpdateKLSpent, UpdateKLMin, UpdateEfficiency, UpdateKLShare])
def test_every_instrument_passes_lint_and_declares_the_access_its_metric_needed(
    cls, long_run, book
):
    _samples, ledgers, g = book
    instrument = cls(ledgers, g, run_=long_run)
    assert lint_instrument(instrument) == []
    # A rung-0 `G` needs no checkpoint, and the declaration says so rather than over-claiming.
    assert instrument.requires == {Component.RECORD: Access.RECORD}
    assert instrument.quantity in {
        "update.kl_spent",
        "update.kl_min",
        "update.efficiency",
        "update.kl_share",
    }


def test_a_rung_two_metric_makes_the_instrument_declare_policy_backward(book):
    """The access an estimate needed is a property of the estimate, not of the instrument class."""
    _samples, ledgers, g = book
    lifted = MetricG(
        names=g.names,
        matrix=g.matrix,
        damping=1e-3,
        damping_stable=True,
        conditioning=g.conditioning,
        rung=2,
        method="fisher_kernel",
        n_samples=g.n_samples,
    )
    instrument = UpdateKLMin(ledgers, lifted)
    assert instrument.requires == {
        Component.RECORD: Access.RECORD,
        Component.POLICY: Access.BACKWARD,
    }


def test_the_emitted_reading_carries_the_quantity_it_declares(book):
    """E35/E44/E51: assert the emitted quantity, not the declared one."""
    _samples, ledgers, g = book
    reading = UpdateKLMin(ledgers, g).estimate(_ctx())
    assert not isinstance(reading, Refusal), reading
    assert reading.quantity == "update.kl_min"
    assert reading.value["n_pairs"] == len(ledgers)
    assert reading.baselines is not None

    shares = UpdateKLShare(ledgers, g).estimate(_ctx())
    assert not isinstance(shares, Refusal), shares
    assert shares.quantity == "update.kl_share"
    assert np.asarray(shares.value["shares"]).shape == (len(ledgers), len(g.names))


def test_efficiency_on_the_real_record_is_a_refusal_and_never_a_number_outside_the_bound(
    book, long_run
):
    """The clause's first half on the two records that exist: no run produces a value outside [0,1]."""
    _samples, ledgers, g = book
    reading = UpdateEfficiency(ledgers, g, run_=long_run).estimate(_ctx())
    assert isinstance(reading, Refusal)
    assert reading.reason is RefusalReason.RECORD_INCOMPLETE


def test_a_ledger_and_a_metric_in_different_bases_refuse_rather_than_pairing_up(book):
    """The join key, enforced. A quadratic form across two bases is a confident wrong number."""
    _samples, ledgers, g = book
    wrong = _synthetic_g(np.eye(2), ("length", "other"))
    out = kl_min_series(ledgers, wrong)
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.UNIT_MISMATCH


def test_a_supplied_denominator_makes_the_book_readable_on_the_real_record(book):
    """The remedy the refusal names, exercised: pass `kl_spent` and the same record produces a book.

    The denominators here are not measurements of this run and the test does not treat them as one.
    They are set to ten times each step's own `KL_min`, which is the smallest thing that makes the
    path testable, and what is asserted is the plumbing and the bound rather than a number about
    the run.
    """
    _samples, ledgers, g = book
    rows = kl_min_series(ledgers, g)
    assert not isinstance(rows, Refusal)
    supplied = {row.step: max(row.kl_min * 10.0, 1e-9) for row in rows}
    out = cost_series(ledgers, g, kl_spent=supplied)
    assert not isinstance(out, Refusal), out
    assert len(out) == len(ledgers)
    for cost in out:
        assert 0.0 <= cost.efficiency <= 1.0
        assert cost.efficiency == pytest.approx(0.1, rel=1e-9)


def test_a_denominator_below_kl_min_fires_the_kill_condition_instead_of_reporting(book):
    """F3's kill condition. `KL_min` above `D_t` means the book does not balance, and it says so."""
    _samples, ledgers, g = book
    rows = kl_min_series(ledgers, g)
    assert not isinstance(rows, Refusal)
    supplied = {row.step: max(row.kl_min * 0.5, 1e-12) for row in rows}
    out = cost_series(ledgers, g, kl_spent=supplied)
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.ENVELOPE_VIOLATED
    assert out.statistics["worst_ratio"] == pytest.approx(2.0, rel=1e-6)
    assert "per sequence" in out.remedy


# ---------------------------------------------------------------------------
# Two real checkpoints of the real policy, where a denominator actually exists
# ---------------------------------------------------------------------------

#: The step size that makes the behavioural signal beat the sampling noise on this policy. It is
#: far larger than any real run would take and that is stated rather than smoothed over: a 0.6M
#: model against a length grader at `lr` 1e-06 moves its features by less than one standard error
#: per step, and a quadratic form in a `Δz` that small measures the batch size.
BIG_STEP = 1.0
#: A step small enough that `Δz` is inside its own noise, which is where the self-check must fire.
SMALL_STEP = 0.5
#: Completions sampled per prompt at each checkpoint. Sixteen keeps the pair of experiments near a
#: minute on CPU; the measured efficiency at 32 is 0.576 against 0.566 here, so the conclusion does
#: not turn on it.
N_SAMPLES = 16


def _two_checkpoint_experiment(eta: float):
    """One real policy-gradient step on the real policy, and the cost book across it.

    Returns `(kl_spent, ledger, G)`. The step is plain SGD rather than the record's AdamW, so that
    `Δθ` is exactly `-eta * grad` and nothing depends on an optimiser's hidden state.
    `D_t = KL(π₁ ‖ π₀)` is the full-vocabulary token-level KL summed over completion positions and
    averaged over sequences drawn from **π₁**, which is the direction the definition takes. Taking
    it over the record's own stale completions instead reads 40% low here, because the step moves
    probability onto tokens those completions do not contain.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from reward_lens.measure.efficiency.scores import sequence_scores
    from reward_lens.measure.ledger.features import surface_features

    name = "trl-internal-testing/tiny-Qwen3ForCausalLM"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
    model.eval()
    featuriser = SurfaceFeatures()

    run = _open("long") or _open("short")
    step = list(run.steps.slice(0, 1))[0]
    pairs, advantages, groups, tasks = [], [], [], []
    for ordinal, group in enumerate(step.groups):
        for trajectory in group.trajectories:
            pairs.append(
                (
                    "\n".join(t.text for t in trajectory.turns if t.role == "user" and t.text),
                    "\n".join(t.text for t in trajectory.turns if t.role == "assistant" and t.text),
                )
            )
            advantages.append(
                float("nan") if trajectory.advantage is None else float(trajectory.advantage)
            )
            groups.append(ordinal)
            tasks.append(str(group.task_ref))
    prompts = sorted({p for p, _ in pairs})
    parameters = [p for p in model.parameters() if p.requires_grad]

    def sequence_logprob(prompt: str, completion: str):
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
        ids = torch.tensor([prompt_ids + completion_ids])
        logits = model(input_ids=ids).logits.float()
        log_probs = torch.log_softmax(logits[0, len(prompt_ids) - 1 : -1], dim=-1)
        return log_probs.gather(1, torch.tensor(completion_ids)[:, None]).sum()

    @torch.no_grad()
    def draw(seed: int) -> dict[str, list[str]]:
        torch.manual_seed(seed)
        out: dict[str, list[str]] = {}
        for prompt in prompts:
            ids = torch.tensor(
                [tokenizer(prompt, add_special_tokens=False)["input_ids"]] * N_SAMPLES
            )
            generated = model.generate(
                input_ids=ids,
                do_sample=True,
                max_new_tokens=12,
                attention_mask=torch.ones_like(ids),
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            out[prompt] = [
                tokenizer.decode(g[ids.shape[1] :], skip_special_tokens=True) for g in generated
            ]
        return out

    @torch.no_grad()
    def log_prob_rows(sequences):
        rows = []
        for prompt, completion in sequences:
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
            ids = torch.tensor([prompt_ids + completion_ids])
            logits = model(input_ids=ids).logits.float()
            rows.append(torch.log_softmax(logits[0, len(prompt_ids) - 1 : -1], dim=-1))
        return rows

    def as_sample(index: int, drawn: dict[str, list[str]]) -> StepSample:
        rows, labels, refs = [], [], []
        for ordinal, prompt in enumerate(prompts):
            for text in drawn[prompt]:
                values = surface_features(text, 2)
                if values is None:
                    continue
                rows.append([values[n] for n in featuriser.names])
                labels.append(ordinal)
                refs.append(prompt)
        return StepSample(
            index=index,
            names=featuriser.names,
            features=np.asarray(rows, dtype=np.float64),
            advantages=np.zeros(len(rows)),
            group_ids=np.asarray(labels, dtype=np.int64),
            task_ids=tuple(refs),
        )

    # `G` at the checkpoint the step starts from, from the record's own eight rollouts.
    features = np.vstack(
        [np.asarray([surface_features(c, 2)[n] for n in featuriser.names]) for _, c in pairs]
    )
    base = StepSample(
        index=0,
        names=featuriser.names,
        features=features,
        advantages=np.asarray(advantages),
        group_ids=np.asarray(groups, dtype=np.int64),
        task_ids=tuple(tasks),
    )
    g = metric_g([base], method="fisher_kernel", scores=sequence_scores(model, tokenizer, pairs))
    assert not isinstance(g, Refusal), g

    before = as_sample(0, draw(1234))
    initial = {k: v.clone() for k, v in model.state_dict().items()}

    model.zero_grad(set_to_none=True)
    loss = sum(
        (-a) * sequence_logprob(p, c) for (p, c), a in zip(pairs, advantages) if np.isfinite(a)
    )
    loss.backward()
    with torch.no_grad():
        for parameter in parameters:
            parameter -= eta * parameter.grad
    model.zero_grad(set_to_none=True)

    drawn_after = draw(4321)
    after = as_sample(1, drawn_after)
    on_policy = [(p, c) for p in prompts for c in drawn_after[p] if c.strip()]
    at_one = log_prob_rows(on_policy)
    model.load_state_dict(initial)
    at_zero = log_prob_rows(on_policy)
    kl_spent = float(np.mean([float((a.exp() * (a - b)).sum()) for a, b in zip(at_one, at_zero)]))
    return kl_spent, ledger_between(before, after, eta=eta), g


@pytest.mark.whitebox
def test_efficiency_is_inside_the_bound_on_a_real_policy_with_a_real_denominator():
    """The clause's first half, on a run where `D_t` exists because this test made it exist.

    Measured: the step spends 4.07 nats per sequence, the movement it produced needs at least 2.31,
    and efficiency is 0.566. The assertion is the bound, and the recorded numbers are there so that
    a change in either direction is visible rather than absorbed.
    """
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    kl_spent, ledger, g = _two_checkpoint_experiment(BIG_STEP)

    book = cost_series([ledger], g, kl_spent={ledger.step: kl_spent})
    assert not isinstance(book, Refusal), book.render()
    (cost,) = book

    assert 0.0 <= cost.efficiency <= 1.0
    assert cost.kl_min <= cost.kl_spent
    assert cost.kl_spent == pytest.approx(4.07, rel=0.35)
    assert cost.efficiency == pytest.approx(0.57, abs=0.25)
    # The shares still add up on a real network, which is the property D19 asserts at construction.
    assert sum(cost.shares.values()) + cost.residual_share == pytest.approx(cost.kl_min, rel=1e-6)
    assert isinstance(cost, StepCost)


@pytest.mark.whitebox
def test_a_step_below_the_noise_refuses_rather_than_reporting_an_efficiency_above_one():
    """P5 as a self-check: the instrument never hands out a value outside [0,1], it refuses.

    At this step size the policy's real behavioural effect is smaller than what sixteen samples
    resolve, so `Δz` is mostly noise and the quadratic form in it comes out above `D_t`. Measured:
    `D_t` 0.035 nats against a `KL_min` of 2.80, a ratio of 81. The instrument refuses, and F3's
    kill condition is the reason it gives.
    """
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    kl_spent, ledger, g = _two_checkpoint_experiment(SMALL_STEP)

    out = cost_series([ledger], g, kl_spent={ledger.step: kl_spent})
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.ENVELOPE_VIOLATED
    assert out.statistics["worst_ratio"] > 1.0
    assert out.statistics["worst_kl_spent"] == pytest.approx(kl_spent, rel=1e-9)
    # And nothing constructed a StepCost on the way, so no value outside the bound ever existed.
    assert "premises failed" in out.remedy

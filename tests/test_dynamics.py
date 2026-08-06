"""Acceptance tests for ``reward_lens.dynamics`` (M9).

Four properties carry the milestone and each is proven here on the CPU-provable synthetic vehicle
(`synthetic_planted_sequence`), a handful of tiny reward models that differ only in a planted, growing
reward loading:

  1. The fingerprint chain verifies an honest sequence and rejects a tampered checkpoint, both a
     shallow edit of a recorded fingerprint (caught by the hash chain) and a deep weight swap under an
     unchanged record (caught by recomputing the fingerprint).
  2. The sweep is resumable: a second run over the same (observable, view, chain) skips every computed
     step and appends nothing new to the store.
  3. The bias-entry curve is monotone rising on the planted sequence, because the reward's loading onto
     the planted feature grows by construction and the estimator must track it faithfully.
  4. The stabilization detector reports a finite step, the point at which the canonicalized reward
     direction stops rotating even as its raw magnitude keeps growing (rotation vs rescaling).

The collapse autopsy and the E04 rho trajectory are proven on synthetic trajectories, and the
GPU-scale and optional-package entry points are proven to refuse rather than fabricate.
"""

from __future__ import annotations

import numpy as np
import pytest

# The synthetic checkpoint sequence is a real (tiny) transformer per checkpoint, so torch is required;
# the pure chain/autopsy/rho/gating assertions below could run without it, but the headline vehicle
# needs it, and torch is present in this environment.
torch = pytest.importorskip(
    "torch",
    reason="the dynamics synthetic checkpoint vehicle builds tiny transformers and needs torch.",
)

from reward_lens.core.errors import ProvenanceError  # noqa: E402
from reward_lens.core.store import EvidenceStore  # noqa: E402
from reward_lens.core.types import GaugeStatus  # noqa: E402
from reward_lens.dynamics import (  # noqa: E402
    LayerwiseProjection,
    Probe,
    bias_entry_curve,
    devinterp,  # noqa: E402
    faithfulness_rho_trajectory,
    second_epoch_collapse_autopsy,
    stabilization_report,
    sweep_over_checkpoints,
    synthetic_planted_sequence,
)
from reward_lens.dynamics import checkpoints as ck  # noqa: E402


@pytest.fixture(scope="module")
def synthetic():
    """Build the planted-feature checkpoint sequence once for the module."""
    return synthetic_planted_sequence(n_checkpoints=8, seed=0)


# ---------------------------------------------------------------------------
# 1. The fingerprint chain verifies and rejects a tampered checkpoint
# ---------------------------------------------------------------------------


def test_chain_verifies_honest_sequence(synthetic):
    """An honest sequence passes the shallow chain check and the deep fingerprint check."""
    seq = synthetic.sequence
    assert seq.verify_chain().ok
    assert seq.verify_fingerprints().ok
    # verify() returns the passing result and does not raise on an honest chain.
    assert seq.verify(deep=True).ok


def test_chain_rejects_shallow_tamper(synthetic):
    """Editing a recorded fingerprint breaks the hash chain at that step."""
    seq = synthetic.sequence
    tampered = seq.tampered(3, model_fp="mfp:deadbeefdeadbeefdeadbeefdeadbeef")

    result = tampered.verify_chain()
    assert not result.ok
    assert result.first_bad_step == seq[3].step
    with pytest.raises(ProvenanceError):
        tampered.verify()


def test_chain_rejects_deep_weight_swap(synthetic):
    """Swapping weights under an unchanged record passes the chain but fails the fingerprint."""
    seq = synthetic.sequence
    # A loader from an unrelated sequence returns a model whose fingerprint differs from the record.
    other = synthetic_planted_sequence(n_checkpoints=1, seed=99)
    swapped = seq.tampered(2, loader=other.sequence[0].loader)

    # The recorded fingerprint was not touched, so the hash chain still verifies ...
    assert swapped.verify_chain().ok
    # ... but recomputing the fingerprint from the loaded weights catches the swap.
    fp_result = swapped.verify_fingerprints()
    assert not fp_result.ok
    assert fp_result.first_bad_step == seq[2].step
    with pytest.raises(ProvenanceError):
        swapped.verify(deep=True)


def test_reseal_restores_a_valid_chain(synthetic):
    """An honest re-release reseals the chain around a change, and it verifies again."""
    seq = synthetic.sequence
    other = synthetic_planted_sequence(n_checkpoints=1, seed=123)
    resealed = seq.tampered(
        1,
        model_fp=other.sequence[0].model_fp,
        loader=other.sequence[0].loader,
        reseal=True,
    )
    assert resealed.verify_chain().ok


# ---------------------------------------------------------------------------
# 2. The sweep is resumable
# ---------------------------------------------------------------------------


def test_sweep_is_resumable(synthetic, tmp_path):
    """A second sweep over the same chain skips every step and appends nothing new."""
    store = EvidenceStore(tmp_path / "store")
    observable = LayerwiseProjection()

    first = sweep_over_checkpoints(synthetic.sequence, observable, view=synthetic.view, store=store)
    assert first.n_computed == len(synthetic.sequence)
    assert first.n_cached == 0
    size_after_first = len(store)
    lines_after_first = sum(1 for _ in open(store.jsonl, encoding="utf-8"))
    assert size_after_first == len(synthetic.sequence)

    second = sweep_over_checkpoints(
        synthetic.sequence, observable, view=synthetic.view, store=store
    )
    # Nothing recomputed, everything served from cache.
    assert second.n_computed == 0
    assert second.n_cached == len(synthetic.sequence)
    assert all(point.from_cache for point in second.points)
    # The store did not grow: no new Evidence was appended.
    assert len(store) == size_after_first
    assert sum(1 for _ in open(store.jsonl, encoding="utf-8")) == lines_after_first
    # The resumed run returns the identical Evidence (same content ids), step by step.
    assert [e.id for e in first.evidence] == [e.id for e in second.evidence]
    assert first.steps == synthetic.sequence.steps


def test_sweep_verifies_chain_before_running(synthetic, tmp_path):
    """A sweep refuses to run over a chain that does not verify."""
    store = EvidenceStore(tmp_path / "store")
    tampered = synthetic.sequence.tampered(2, model_fp="mfp:00000000000000000000000000000000")
    with pytest.raises(ProvenanceError):
        sweep_over_checkpoints(tampered, LayerwiseProjection(), view=synthetic.view, store=store)


def test_builtin_observable_emits_gated_evidence(synthetic, tmp_path):
    """The built-in crystallization observable yields well-formed, invariant Evidence."""
    store = EvidenceStore(tmp_path / "store")
    traj = sweep_over_checkpoints(
        synthetic.sequence, LayerwiseProjection(), view=synthetic.view, store=store
    )
    ev = traj.evidence[0]
    assert ev.observable == "dynamics.wr_projection"
    assert ev.gauge == GaugeStatus.INVARIANT
    payload = ev.value
    assert set(payload) >= {"profile", "layers", "crystal_layer", "crystal_frac", "n_layers"}
    assert len(payload["profile"]) == payload["n_layers"]
    assert 0.0 <= payload["crystal_frac"] <= 1.0


# ---------------------------------------------------------------------------
# 3. The bias-entry curve is monotone rising on the planted sequence
# ---------------------------------------------------------------------------


def test_bias_entry_curve_is_monotone_rising(synthetic, tmp_path):
    """The planted feature's effect size rises monotonically across training (M9)."""
    store = EvidenceStore(tmp_path / "store")
    curves = bias_entry_curve(synthetic.sequence, [synthetic.probe], synthetic.view, store=store)
    d = curves.effect_size[synthetic.probe.name]

    assert len(d) == len(synthetic.sequence)
    assert all(np.isfinite(x) for x in d)
    # Non-decreasing across the whole run (a tiny tolerance for floating-point noise).
    assert all(d[i + 1] >= d[i] - 1e-6 for i in range(len(d) - 1)), d
    # The bias genuinely enters: it is near-absent early and large late.
    assert d[-1] > d[0] + 1.0
    # It crosses the entry threshold at a locatable step.
    assert curves.entry_step[synthetic.probe.name] is not None
    # The correlation-scale effect size rises too.
    r = curves.effect_r[synthetic.probe.name]
    assert r[-1] > r[0]


def test_bias_entry_curve_with_labels(synthetic, tmp_path):
    """A probe defined by a binary label grouping produces the same rising curve."""
    store = EvidenceStore(tmp_path / "store")
    labels = (synthetic.feature >= float(np.median(synthetic.feature))).astype(int)
    probe = Probe(name="planted-label", labels=labels)
    curves = bias_entry_curve(synthetic.sequence, [probe], synthetic.view, store=store)
    d = curves.effect_size["planted-label"]
    assert d[-1] > d[0] + 1.0
    assert all(d[i + 1] >= d[i] - 1e-6 for i in range(len(d) - 1))


# ---------------------------------------------------------------------------
# 4. The stabilization detector reports a finite stabilization step
# ---------------------------------------------------------------------------


def test_stabilization_detector_reports_finite_step(synthetic, tmp_path):
    """The reward direction stops rotating at a finite step while its magnitude keeps growing."""
    store = EvidenceStore(tmp_path / "store")
    report = stabilization_report(synthetic.sequence, synthetic.view, store=store)

    step = report.stabilization_step
    assert step is not None
    assert isinstance(step, int)
    # It is an interior step: the direction rotates early and settles before the end.
    assert synthetic.sequence.steps[0] < step <= synthetic.sequence.steps[-1]
    # By construction the raw magnitude keeps changing after the direction settles: this is rescaling,
    # not still-forming rotation, which is the distinction the report exists to draw.
    assert report.rescaling_continues
    # The final consecutive canonical cosines are essentially one (no residual rotation).
    assert report.canonical_cos[-1] > 1.0 - report.eps
    # The raw norms are not constant (the reward vector is still being rescaled).
    assert max(report.raw_norm) - min(report.raw_norm) > 1e-3


def test_stabilization_step_near_schedule_expectation(synthetic, tmp_path):
    """The detected stabilization step matches the planted schedule's rotation knee."""
    store = EvidenceStore(tmp_path / "store")
    report = stabilization_report(synthetic.sequence, synthetic.view, store=store)
    # The detector and the schedule's own direction-rotation expectation agree within a step or two;
    # both read the same logistic knee, one through the canonical frame and one through raw directions.
    assert abs(report.stabilization_step - synthetic.expected_stabilization_step) <= 2


# ---------------------------------------------------------------------------
# The collapse autopsy skeleton and the E04 rho trajectory (synthetic trajectories)
# ---------------------------------------------------------------------------


def test_collapse_autopsy_reports_growth_and_alignment():
    """The autopsy names growing components and w_r's drift toward a memorization direction."""
    component_magnitudes = {
        0: {"mlp_L0": 1.0, "attn_L1": 1.0},
        1: {"mlp_L0": 1.0, "attn_L1": 1.0},
        2: {"mlp_L0": 3.0, "attn_L1": 1.02},
    }
    w_r_by_step = {
        0: np.array([1.0, 0.0, 0.0]),
        1: np.array([0.9, 0.4, 0.0]),
        2: np.array([0.3, 0.95, 0.0]),
    }
    memorization = {"memo": np.array([0.0, 1.0, 0.0])}

    autopsy = second_epoch_collapse_autopsy(
        component_magnitudes, w_r_by_step, memorization, epoch_boundary_step=1
    )
    growing = dict(autopsy.growing_components)
    assert "mlp_L0" in growing and growing["mlp_L0"] >= 1.2
    assert "attn_L1" not in growing
    assert autopsy.alignment_increased["memo"] is True
    # The held-out restoration term is never fabricated; it needs the GPU-scale evaluation.
    assert autopsy.held_out_restored is None


def test_faithfulness_rho_trajectory_is_developmental():
    """The E04 attribution-vs-patching rho can be tracked per checkpoint and flagged developmental."""
    rng = np.random.default_rng(0)
    n_components = 128  # enough components that the w=0 checkpoint's rho is genuinely near zero
    pairs = []
    n_steps = 5
    for step in range(n_steps):
        attribution = rng.standard_normal(n_components)
        # Early (w=0): patching is independent of attribution, so rho starts near zero. Late (w=1):
        # patching is exactly the negated attribution, so rho approaches -1. The anti-correlation emerges
        # from ~0 and deepens across training: the planted developmental trend the curve must recover.
        w = step / (n_steps - 1)
        patching = -w * attribution + (1.0 - w) * rng.standard_normal(n_components)
        pairs.append((step, attribution, patching))

    traj = faithfulness_rho_trajectory(pairs, n_resamples=400, seed=0)
    assert traj.steps == list(range(n_steps))
    assert len(traj.rho) == n_steps
    # The recovered curve genuinely starts near zero and deepens to a strong anti-correlation.
    assert abs(traj.rho[0]) < 0.2  # near zero at the first checkpoint
    assert traj.rho[-1] < 0  # anti-correlated at the final checkpoint (the v1 finding)
    assert traj.rho[-1] < traj.rho[0]  # the anti-correlation deepened across training
    assert traj.developmental

    # Not a tautology of "ends anti-correlated": a reward that is already anti-correlated at every
    # checkpoint and never deepens is a constant, not a developmental trend, and must not be flagged.
    flat_pairs = []
    for step in range(n_steps):
        attribution = rng.standard_normal(n_components)
        patching = -0.7 * attribution + 0.7 * rng.standard_normal(n_components)
        flat_pairs.append((step, attribution, patching))
    flat = faithfulness_rho_trajectory(flat_pairs, n_resamples=400, seed=0)
    assert flat.rho[-1] < 0  # anti-correlated throughout ...
    assert not flat.developmental  # ... but flat, so not developmental


# ---------------------------------------------------------------------------
# The GPU-scale and optional-package entry points refuse rather than fabricate
# ---------------------------------------------------------------------------


def test_train_rm_pythia_is_gpu_gated():
    """The RM-Pythia training run refuses on non-flagship hardware and never fabricates."""
    with pytest.raises(RuntimeError, match="GPU-gated"):
        ck.train_rm_pythia()


def test_from_hf_revisions_is_download_gated():
    """Building a chain from HF revisions refuses to fan out downloads unless allowed."""
    with pytest.raises(NotImplementedError, match="download"):
        ck.from_hf_revisions("org/rm-pythia", [(0, "step-0"), (1, "step-100")])


def test_devinterp_bridge_is_a_marked_stub():
    """The optional LLC bridge reports unavailable and refuses cleanly when the package is absent."""
    # The external package is not a dependency and is not installed here.
    assert devinterp.is_available() is False
    with pytest.raises(ImportError, match="devinterp"):
        # A one-checkpoint synthetic sequence is enough; the bridge refuses before doing any work.
        seq = synthetic_planted_sequence(n_checkpoints=1, seed=7).sequence
        devinterp.estimate_llc(seq)

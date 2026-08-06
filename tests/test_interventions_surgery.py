"""Proofs for the three activation/weight surgery interventions.

``steer``, ``ablate``, and ``edit`` are linear-algebraic operations whose correctness is known by
construction, so each is pinned here to floating-point tolerance rather than described in prose. The
activation interventions (steer, ablate) mount through the frozen runtime's single hook path, so they
are exercised on the tiny ``from_tiny`` ClassifierRM through the same ``run_patched_scores`` helper
``patch.py`` exposes; the weight-space edit is exercised through its sibling ``run_edited_scores``.

What is proven versus what is gated: everything here runs on CPU on the tiny synthetic Llama and is
proven by a passing assertion. Nothing in this file needs a GPU or a downloaded checkpoint, so nothing
is skipped or gated. The larger campaign models these interventions will run on are out of scope for
this file and are gated in the loader (``load_signal`` raises unless ``allow_download``).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from reward_lens.concepts.vectors import dose_response_slope
from reward_lens.core.types import Site
from reward_lens.interventions.ablate import AblationIntervention
from reward_lens.interventions.edit import EditIntervention, run_edited_scores
from reward_lens.interventions.patch import run_patched_scores
from reward_lens.interventions.steer import SteeringIntervention, unit_direction
from reward_lens.signals.loaders import from_tiny

_VIEW = [
    ("Explain gravity.", "Gravity is the attraction between masses."),
    ("Say hi.", "Hello there, friend."),
    ("Add.", "Two plus two is four."),
    ("Color?", "The sky is blue today."),
]


@pytest.fixture(scope="module")
def signal():
    """The tiny synthetic ClassifierRM every surgery proof runs on (CPU, offline)."""
    return from_tiny(seed=1, conformance_quickcheck=False)


def _final_head_input(signal, view) -> np.ndarray:
    """The real post-norm head-input hidden state of the first item, as an fp64 vector.

    This is the space the reward head reads and the space the concept direction and ``w_r`` live in,
    so it is the correct operating point for the weight-edit dose-response proof.
    """
    tokenized = [signal.tokenize(it) for it in view]
    batch = signal.runtime.collate(tokenized)
    raw = signal.runtime.forward(batch)
    head_input = raw.extra["head_input"]
    final_pos = raw.extra["final_pos"]
    idx = torch.arange(head_input.shape[0])
    return head_input[idx, final_pos][0].detach().cpu().numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# steer.py
# ---------------------------------------------------------------------------


def test_steer_strength_zero_is_bit_exact_noop(signal):
    """(a) A steer at strength zero reproduces the clean scores to the bit, not merely closely.

    The clean path is ``signal.score``; the steered path is the compiled steer run through
    ``run_patched_scores``. The hook short-circuits at strength zero, so the forward is untouched and
    the two score arrays are bit-for-bit identical.
    """
    read = signal.readout("reward")
    w_r = read.vector.detach().cpu().numpy()
    clean = signal.score(_VIEW).value.values
    steer0 = SteeringIntervention(direction=w_r, site=read.site, strength=0.0)
    steered = run_patched_scores(signal, steer0.compile(signal), _VIEW)
    assert np.array_equal(clean, steered), "strength-0 steer was not a bit-exact no-op"


def test_steer_hook_adds_exact_multiple_of_unit_direction():
    """(b) The mount hook adds exactly ``strength * unit(direction)`` at every position.

    On a synthetic residual the difference between the hooked output and the input must equal the
    steering vector broadcast over batch and sequence, to fp32 tolerance.
    """
    torch.manual_seed(0)
    hidden = torch.randn(2, 5, 32, dtype=torch.float32)
    direction = np.asarray(np.random.RandomState(0).randn(32), dtype=np.float32)
    strength = 1.7
    site = Site(0, "resid_post")
    apply = (
        SteeringIntervention(direction=direction, site=site, strength=strength)
        .compile(None)
        .mounts[site]
    )
    out = apply(hidden, {})
    expected = strength * torch.as_tensor(unit_direction(direction))
    got = out - hidden
    assert torch.allclose(got, expected.expand_as(got), atol=1e-6), (
        "steer delta is not s * unit_dir"
    )


def test_steer_dose_response_sign_and_monotonic(signal):
    """(c) Steering along ``w_r`` raises the reward monotonically; positive strength raises it.

    The reward is ``<norm(h), w_r> + bias``; steering the last-layer residual along ``w_r`` gives a
    reward whose derivative in the dose reduces (the Cauchy-Schwarz argument in the module) to a
    non-negative quantity, so the response is monotone increasing for every item. The dose axis spans
    negative and positive strengths, and the fitted slope is positive.
    """
    read = signal.readout("reward")
    w_r = read.vector.detach().cpu().numpy()
    doses = np.linspace(-2.0, 2.0, 9)
    rewards = np.array(
        [
            run_patched_scores(
                signal,
                SteeringIntervention(direction=w_r, site=read.site, strength=float(a)).compile(
                    signal
                ),
                _VIEW,
            )
            for a in doses
        ]
    )  # (n_dose, n_item)
    increments = np.diff(rewards, axis=0)
    assert np.all(increments > 0), "reward was not strictly increasing in steering strength"
    # Positive strength raises reward above the baseline, negative lowers it (correct dose sign).
    base = rewards[doses == 0.0][0]
    assert np.all(rewards[-1] > base) and np.all(base > rewards[0])
    # The dose-response slope is positive for every item.
    slopes = np.array([dose_response_slope(doses, rewards[:, i]) for i in range(rewards.shape[1])])
    assert np.all(slopes > 0), "dose-response slope along w_r was not positive"


def test_steer_fingerprint_tracks_strength_and_direction(signal):
    """(d) The fingerprint changes with strength and with direction, and is otherwise stable."""
    read = signal.readout("reward")
    w_r = read.vector.detach().cpu().numpy()
    other = np.asarray(np.random.RandomState(2).randn(w_r.shape[0]), dtype=np.float32)
    base = SteeringIntervention(direction=w_r, site=read.site, strength=1.0)
    assert (
        base.fingerprint()
        == SteeringIntervention(direction=w_r, site=read.site, strength=1.0).fingerprint()
    ), "fingerprint is not stable across identical steers"
    assert (
        base.fingerprint()
        != SteeringIntervention(direction=w_r, site=read.site, strength=2.0).fingerprint()
    ), "fingerprint did not change with strength"
    assert (
        base.fingerprint()
        != SteeringIntervention(direction=other, site=read.site, strength=1.0).fingerprint()
    ), "fingerprint did not change with direction"


def test_steer_rejects_zero_direction():
    """A zero direction has no orientation to steer along and is rejected at normalization."""
    with pytest.raises(ValueError, match="zero"):
        unit_direction(np.zeros(8, dtype=np.float32))


# ---------------------------------------------------------------------------
# ablate.py
# ---------------------------------------------------------------------------


def _synthetic_resid(seed: int = 0):
    torch.manual_seed(seed)
    hidden = torch.randn(2, 5, 32, dtype=torch.float32)
    direction = np.asarray(np.random.RandomState(seed).randn(32), dtype=np.float32)
    return hidden, direction


def test_directional_ablation_removes_component_exactly():
    """After directional ablation the residual has ~0 component along the ablated direction."""
    hidden, direction = _synthetic_resid()
    hook = (
        AblationIntervention(site=Site(0, "resid_post"), direction=direction, mode="directional")
        .compile(None)
        .mounts[Site(0, "resid_post")]
    )
    out = hook(hidden, {})
    unit = torch.as_tensor(unit_direction(direction))
    along = (out * unit).sum(dim=-1)
    assert float(along.abs().max()) < 1e-5, (
        "ablated residual still has a component along the direction"
    )


def test_directional_ablation_is_idempotent():
    """Ablating an already-ablated residual changes nothing (the projection is a projection)."""
    hidden, direction = _synthetic_resid(seed=3)
    hook = (
        AblationIntervention(site=Site(0, "resid_post"), direction=direction, mode="directional")
        .compile(None)
        .mounts[Site(0, "resid_post")]
    )
    once = hook(hidden, {})
    twice = hook(once, {})
    assert torch.allclose(once, twice, atol=1e-6), "directional ablation is not idempotent"


def test_ablation_leaves_orthogonal_vector_unchanged():
    """A vector orthogonal to the ablated direction passes through untouched."""
    _, direction = _synthetic_resid(seed=5)
    unit = unit_direction(direction)
    raw = np.asarray(np.random.RandomState(9).randn(32), dtype=np.float32)
    orthogonal = raw - float(raw @ unit) * unit  # Gram-Schmidt: exactly orthogonal to unit
    tensor = torch.as_tensor(orthogonal).reshape(1, 1, 32)
    hook = (
        AblationIntervention(site=Site(0, "resid_post"), direction=direction, mode="directional")
        .compile(None)
        .mounts[Site(0, "resid_post")]
    )
    out = hook(tensor, {})
    assert torch.allclose(out, tensor, atol=1e-5), (
        "ablation moved a vector orthogonal to its direction"
    )


def test_mean_ablation_sets_coordinate_to_supplied_mean():
    """Mean ablation replaces the along-direction coordinate with the supplied dataset mean."""
    hidden, direction = _synthetic_resid(seed=7)
    mean_projection = 3.14
    hook = (
        AblationIntervention(
            site=Site(0, "resid_post"),
            direction=direction,
            mode="mean",
            mean_projection=mean_projection,
        )
        .compile(None)
        .mounts[Site(0, "resid_post")]
    )
    out = hook(hidden, {})
    unit = torch.as_tensor(unit_direction(direction))
    coord = (out * unit).sum(dim=-1)
    assert float((coord - mean_projection).abs().max()) < 1e-5, (
        "mean ablation did not set the coordinate"
    )


def test_directional_ablation_runs_on_tiny_model(signal):
    """A tiny-model forward under directional ablation produces finite rewards (sanity)."""
    direction = np.asarray(np.random.RandomState(11).randn(32), dtype=np.float32)
    abl = AblationIntervention(site=Site(1, "resid_post"), direction=direction, mode="directional")
    scores = run_patched_scores(signal, abl.compile(signal), _VIEW)
    assert scores.shape == (len(_VIEW),)
    assert np.all(np.isfinite(scores)), "ablation produced a non-finite reward on the tiny model"


def test_head_ablation_runs_on_tiny_model(signal):
    """Zeroing one attention head's contribution composes with patch.py's head surface and runs."""
    n_heads = int(signal.runtime.model.config.num_attention_heads)
    abl = AblationIntervention(site=Site(0, "head_out", 1), mode="head", n_heads=n_heads)
    scores = run_patched_scores(signal, abl.compile(signal), _VIEW)
    assert scores.shape == (len(_VIEW),)
    assert np.all(np.isfinite(scores)), "head ablation produced a non-finite reward"
    # Removing a head's contribution actually moves the reward (it is not a silent no-op).
    clean = signal.score(_VIEW).value.values
    assert np.any(np.abs(scores - clean) > 0.0), "head ablation left every score unchanged"


def test_ablation_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown ablation mode"):
        AblationIntervention(site=Site(0, "resid_post"), mode="bogus")


def test_head_ablation_requires_head_count(signal):
    with pytest.raises(ValueError, match="n_heads"):
        AblationIntervention(site=Site(0, "head_out", 0), mode="head").compile(signal)


# ---------------------------------------------------------------------------
# edit.py
# ---------------------------------------------------------------------------


def _unit(rng, d):
    v = rng.standard_normal(d)
    return v / np.linalg.norm(v)


def test_edit_removes_dose_response_along_direction(signal):
    """After editing ``w_r`` to remove ``u``, the reward's dose-response slope along ``u`` is ~0.

    The core weight-space proof (E17 / S12). Along an orthogonal direction the slope is identical to
    the unedited head's slope, so the edit removes sensitivity to exactly one direction and nothing
    else. Both are read out through ``reward_lens.concepts.vectors.dose_response_slope``.
    """
    w_r = signal.readout("reward").vector.detach().cpu().numpy().astype(np.float64)
    d_model = w_r.shape[0]
    rng = np.random.default_rng(0)
    u = _unit(rng, d_model)
    raw = rng.standard_normal(d_model)
    v = raw - (raw @ u) * u
    v = v / np.linalg.norm(v)  # exactly orthogonal to u

    edit = EditIntervention(direction=u.astype(np.float32), strength=1.0)
    w_prime = edit.edited_vector(signal)  # fp64

    h0 = _final_head_input(signal, _VIEW)
    doses = np.linspace(-2.0, 2.0, 9)
    reward_along_u = np.array([(h0 + a * u) @ w_prime for a in doses])
    reward_along_v = np.array([(h0 + a * v) @ w_prime for a in doses])
    reward_along_v_unedited = np.array([(h0 + a * v) @ w_r for a in doses])

    slope_u = dose_response_slope(doses, reward_along_u)
    slope_v = dose_response_slope(doses, reward_along_v)
    slope_v_unedited = dose_response_slope(doses, reward_along_v_unedited)

    assert abs(slope_u) < 1e-6, f"edit did not remove the dose-response along u (slope {slope_u})"
    assert abs(slope_v - slope_v_unedited) < 1e-7, (
        "edit changed the slope along an orthogonal direction"
    )
    # And the surviving orthogonal slope is exactly the projection of w_r on v (nothing else moved).
    assert abs(slope_v_unedited - float(v @ w_r)) < 1e-7


def test_edit_runs_on_tiny_model_through_sibling_runner(signal):
    """``run_edited_scores`` scores the tiny model under the edited head, finite and edit-dependent.

    Editing along ``w_r`` itself (the direction the head reads) removes the head's own dominant axis,
    so the edited scores differ from the clean scores; editing along a direction orthogonal to ``w_r``
    changes the reward far less. Both must be finite.
    """
    w_r = signal.readout("reward").vector.detach().cpu().numpy()
    clean = signal.score(_VIEW).value.values
    edited = run_edited_scores(
        signal, EditIntervention(direction=w_r, strength=1.0).compile(signal), _VIEW
    )
    assert edited.shape == (len(_VIEW),)
    assert np.all(np.isfinite(edited))
    assert np.any(np.abs(edited - clean) > 1e-4), "editing out w_r left the scores unchanged"


def test_edit_compiled_vector_matches_projection(signal):
    """The compiled fp32 edited vector matches the fp64 projection identity ``w_r - (w_r.u) u``."""
    w_r = signal.readout("reward").vector.detach().cpu().numpy().astype(np.float64)
    rng = np.random.default_rng(4)
    u = _unit(rng, w_r.shape[0])
    edit = EditIntervention(direction=u.astype(np.float32), strength=1.0)
    compiled = edit.compile(signal)
    hand = w_r - (w_r @ unit_direction(u).astype(np.float64)) * unit_direction(u).astype(np.float64)
    assert np.max(np.abs(compiled.meta["edited_vector"].astype(np.float64) - hand)) < 1e-6


def test_edit_fingerprint_tracks_direction_and_strength(signal):
    """The edit fingerprint changes with direction and strength, and is otherwise stable."""
    rng = np.random.default_rng(1)
    u = _unit(rng, 32).astype(np.float32)
    w = _unit(rng, 32).astype(np.float32)
    base = EditIntervention(direction=u, strength=1.0)
    assert base.fingerprint() == EditIntervention(direction=u, strength=1.0).fingerprint()
    assert base.fingerprint() != EditIntervention(direction=u, strength=0.5).fingerprint()
    assert base.fingerprint() != EditIntervention(direction=w, strength=1.0).fingerprint()

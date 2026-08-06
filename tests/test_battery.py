"""Property tests for the measurement battery.

These are not parity tests (that is ``test_e_parity.py``); they are the properties every ported
Observable must have: it runs on a tiny ``from_tiny`` ClassifierRM, it returns gated Evidence at the
right trust and gauge, and its own science holds in the small. Direct linear attribution reconstructs
the reward, patching a component moves the score, the multi-objective geometry reads every objective
rather than a row mean, and the honest E07 refuses the degenerate cascade the v1 detector produced.

The E07 negative fixture is the load-bearing honesty check. The v1 misalignment-cascade detector shipped
two preference pairs per dimension, and a correlation of two points is forced to plus or minus one by
the arithmetic (documented in ``docs/content/caveats.md``); that pinned-to-one number is the "cascade"
artifact. The v3 matched-prompt block supplies enough matched stimuli that the correlation is free to
take any value, so it is no longer a degenerate one. The true 8B noise-floor result needs the real
reward model and is proven on the data design in ``tests/test_data_e07.py``, which this references.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch

from reward_lens.core.types import GaugeStatus, TrustLevel
from reward_lens.data.builtin.diagnostic_v3 import load_diagnostic_v3, matched_prompt_views
from reward_lens.data.schema import DataView
from reward_lens.measure import base as mb
from reward_lens.measure.battery import (
    BiasBattery,
    CircuitJaccard,
    ConceptDoseResponse,
    ConflictMatrix,
    DirectLinearAttribution,
    FeatureRewardAlignment,
    LensCrystallization,
    MultiObjectiveGeometry,
    PatchGrid,
    PathEffect,
    PromptSNR,
)
from reward_lens.signals import from_tiny

# Import the data-design noise-floor test so the honest E07 result it proves is referenced here.
from tests.test_data_e07 import (
    test_cross_dimension_cascade_is_at_noise_floor as _data_e07_noise_floor,
)


@pytest.fixture(scope="module")
def signal():
    # No global torch.set_grad_enabled(False): that is process-wide and would break later grad/hvp
    # tests. Scoring and capture disable grad internally, so the observables need no global toggle.
    return from_tiny(seed=0)


@pytest.fixture(scope="module")
def signal_b():
    return from_tiny(seed=1)


class _RandomDictionary:
    """A random decoder, supplied through ``ctx.regime['sae']``.

    This is what ``FeatureRewardAlignment`` used to build for itself when no dictionary was
    supplied, and the numbers it produces are identical in kind: alignments of random directions
    with the reward, which the observable reports with ``trained_sae=False``. It moved into the test
    when ``reward_lens.sae`` moved behind the ``[dict]`` extra, because an instrument that
    conjures a sparse dictionary out of a module it happens to be able to import is claiming a
    dependency it never declared. The documented contract is one method, so this is the whole of it.
    """

    def __init__(self, d_model: int, n_features: int, seed: int = 0):
        generator = torch.Generator().manual_seed(seed)
        self.W_dec = torch.randn(n_features, d_model, generator=generator)
        self.W_dec /= self.W_dec.norm(dim=1, keepdim=True)

    def feature_reward_alignments(self, reward_direction: torch.Tensor) -> torch.Tensor:
        return self.W_dec @ reward_direction.to(self.W_dec.dtype)


@pytest.fixture(scope="module")
def multi_label_signal():
    from transformers import LlamaConfig, LlamaForSequenceClassification

    from reward_lens.signals import wrap_hf_model
    from reward_lens.signals.loaders import _build_tokenizer

    tok = _build_tokenizer("gpt2")
    torch.manual_seed(2)
    cfg = LlamaConfig(
        vocab_size=getattr(tok, "vocab_size", 1000),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=256,
        pad_token_id=getattr(tok, "pad_token_id", 0) or 0,
        num_labels=19,
        attn_implementation="eager",
    )
    model = LlamaForSequenceClassification(cfg).eval()
    return wrap_hf_model(
        model,
        tok,
        device="cpu",
        architecture="LlamaForSequenceClassification",
        conformance_quickcheck=False,
    )


@pytest.fixture(scope="module")
def one_axis_view():
    return DataView(list(load_diagnostic_v3()["helpfulness"].items)[:5])


@pytest.fixture(scope="module")
def multi_axis_view():
    views = load_diagnostic_v3()
    return DataView(list(views["helpfulness"].items)[:4] + list(views["verbosity"].items)[:4])


# ---------------------------------------------------------------------------
# Every Observable returns gated Evidence at the right trust and gauge
# ---------------------------------------------------------------------------


def test_every_observable_returns_gated_evidence(
    signal, signal_b, multi_label_signal, one_axis_view, multi_axis_view
):
    """Each Observable runs on the tiny model and returns EXPLORATORY Evidence at its declared gauge.

    Trust is EXPLORATORY because no scorecard is registered (gate 1), and the gauge is exactly what the
    Observable declares. This is the executable form of "no ungated number leaves an Observable".
    """
    cases = [
        (LensCrystallization(), mb.Context(signal=signal, view=one_axis_view)),
        (DirectLinearAttribution(), mb.Context(signal=signal, view=one_axis_view)),
        (PatchGrid(), mb.Context(signal=signal, view=one_axis_view)),
        (PatchGrid("head"), mb.Context(signal=signal, view=one_axis_view)),
        (PathEffect(), mb.Context(signal=signal, view=one_axis_view)),
        (ConceptDoseResponse(), mb.Context(signal=signal, view=one_axis_view)),
        (BiasBattery(), mb.Context(signal=signal, view=multi_axis_view)),
        (PromptSNR(), mb.Context(signal=signal, view=multi_axis_view)),
        (ConflictMatrix(), mb.Context(signal=signal, view=multi_axis_view)),
        (
            CircuitJaccard(),
            mb.Context(signal=signal, view=one_axis_view, others=(signal_b,), is_comparison=True),
        ),
        (
            FeatureRewardAlignment(),
            mb.Context(
                signal=signal,
                view=one_axis_view,
                regime={
                    "sae": _RandomDictionary(int(signal.meta.d_model), 64),
                    "sae_trained": False,
                },
            ),
        ),
        (MultiObjectiveGeometry(), mb.Context(signal=multi_label_signal, view=one_axis_view)),
    ]
    for obs, ctx in cases:
        ev = mb.run(obs, ctx)
        assert ev.trust is TrustLevel.EXPLORATORY, (
            f"{obs.name} should be EXPLORATORY, got {ev.trust}"
        )
        assert ev.gauge is obs.gauge_status, (
            f"{obs.name} gauge {ev.gauge} != declared {obs.gauge_status}"
        )
        assert obs.faithful_to is not None, f"{obs.name} must declare faithful_to"
        assert isinstance(ev.value, dict) and ev.value, f"{obs.name} produced no payload"


def test_gauges_are_declared_honestly():
    """The declared gauges match the science: depth fractions and effect sizes are invariant, raw
    cosines and SAE alignments are raw-coordinate."""
    assert LensCrystallization.gauge_status is GaugeStatus.INVARIANT
    assert DirectLinearAttribution.gauge_status is GaugeStatus.INVARIANT
    assert BiasBattery.gauge_status is GaugeStatus.INVARIANT
    assert MultiObjectiveGeometry.gauge_status is GaugeStatus.RAW_ONLY
    assert ConflictMatrix.gauge_status is GaugeStatus.RAW_ONLY
    assert FeatureRewardAlignment.gauge_status is GaugeStatus.RAW_ONLY


# ---------------------------------------------------------------------------
# DLA recovers the reward; patching moves the score
# ---------------------------------------------------------------------------


def test_dla_reconstructs_the_reward_and_names_a_dominant_component(signal):
    """DLA's summed contributions track the reward margin and it names a real dominant component.

    The residual decomposition reconstructs the reward up to the final-norm gap, so the summed
    contributions correlate strongly with the actual score margin across pairs, and the dominant
    component per pair is a real component name. This is the "DLA recovers the final-layer dominant
    contribution" property in the small.
    """
    pairs = list(load_diagnostic_v3()["helpfulness"].items)[:8]
    view = DataView(pairs)
    ev = mb.run(DirectLinearAttribution(), mb.Context(signal=signal, view=view))
    dla_sum = np.array(ev.value["reward_differential"])
    margin = (
        signal.score([(p.prompt_text, p.chosen.text) for p in pairs]).value.values
        - signal.score([(p.prompt_text, p.rejected.text) for p in pairs]).value.values
    )
    corr = float(np.corrcoef(dla_sum, margin)[0, 1])
    assert corr > 0.8, f"DLA reconstruction correlates only {corr} with the reward margin"
    names = set(ev.value["component_names"])
    assert all(dom in names for dom in ev.value["dominant_component"])


def test_patching_a_component_changes_the_score(signal, one_axis_view):
    """A component patch moves the reward: at least one component has a nonzero patch effect."""
    ev = mb.run(PatchGrid(), mb.Context(signal=signal, view=one_axis_view))
    assert ev.value["max_abs_effect"] > 0.0, "no component patch changed the reward"
    # And directly: patching one component gives a different score than the clean forward.
    from reward_lens.core.types import Site
    from reward_lens.interventions.patch import ComponentPatch, run_patched_scores
    from reward_lens.measure.battery._common import capture_sites

    pair = one_axis_view.items[0]
    chosen = [(pair.prompt_text, pair.chosen.text)]
    rejected = [(pair.prompt_text, pair.rejected.text)]
    site = Site(0, "attn_out")
    source = capture_sites(signal, rejected, (site,), full_sequence=True)[site]
    clean = signal.score(chosen).value.values[0]
    patched = run_patched_scores(
        signal, ComponentPatch(site=site, source=source).compile(signal), chosen
    )[0]
    assert abs(float(patched) - float(clean)) > 0.0, "patching attn_L0 left the score unchanged"


def test_head_patching_ranks_a_top_head(signal, one_axis_view):
    """Head-granularity patching runs and identifies a top head (the E15 mechanism, tiny scale)."""
    ev = mb.run(PatchGrid("head"), mb.Context(signal=signal, view=one_axis_view))
    assert ev.value["top_component"] is not None
    assert ev.value["top_component"].startswith("head_L")
    assert ev.value["max_abs_effect"] >= 0.0


# ---------------------------------------------------------------------------
# Multi-objective geometry reads every objective, not the row mean
# ---------------------------------------------------------------------------


def test_multiobjective_reads_all_objectives_not_row_mean(multi_label_signal, one_axis_view):
    """MultiObjectiveGeometry returns the full objective cosine matrix, never a single row-mean scalar."""
    ev = mb.run(MultiObjectiveGeometry(), mb.Context(signal=multi_label_signal, view=one_axis_view))
    n = ev.value["n_objectives"]
    assert n == 19, f"expected 19 objective readouts, got {n}"
    cos = np.array(ev.value["cosine_matrix"])
    assert cos.shape == (19, 19)
    assert np.allclose(np.diag(cos), 1.0, atol=1e-5)  # every objective is unit-aligned with itself
    assert np.allclose(cos, cos.T, atol=1e-6)  # symmetric


def test_multiobjective_refuses_a_scalar_head(signal, one_axis_view):
    """A scalar reward model has no objective geometry; the Observable refuses rather than row-mean it."""
    from reward_lens.core.errors import CapabilityError

    with pytest.raises(CapabilityError):
        mb.run(MultiObjectiveGeometry(), mb.Context(signal=signal, view=one_axis_view))


# ---------------------------------------------------------------------------
# Negative fixture: honest E07 vs the degenerate cascade
# ---------------------------------------------------------------------------


def _cross_dimension_correlations(signal, n_per_dim: int) -> list[float]:
    """Cross-dimension correlations of per-pair reward deltas over the matched-prompt block.

    Each dimension's delta vector is ``score(chosen) - score(rejected)`` over the first ``n_per_dim``
    matched prompts; the correlations are the pairwise Pearson correlations across dimensions. With
    ``n_per_dim == 2`` this is the two-point correlation the v1 cascade detector computed.
    """
    views = matched_prompt_views()
    dims = sorted(views)
    deltas = {}
    for dim in dims:
        items = views[dim].items[:n_per_dim]
        chosen = [(p.prompt_text, p.chosen.text) for p in items]
        rejected = [(p.prompt_text, p.rejected.text) for p in items]
        deltas[dim] = (
            signal.score(chosen).value.values - signal.score(rejected).value.values
        ).astype(np.float64)
    corrs = []
    for a, b in itertools.combinations(dims, 2):
        if np.std(deltas[a]) < 1e-12 or np.std(deltas[b]) < 1e-12:
            continue
        corrs.append(float(np.corrcoef(deltas[a], deltas[b])[0, 1]))
    return corrs


def test_e07_honest_matched_block_is_not_the_degenerate_cascade(signal):
    """The honest E07 correlation is free to be non-degenerate; the v1 two-point cascade is pinned to 1.

    This is the documented direction. v1's misalignment-cascade detector used two pairs per dimension,
    and a two-point correlation is forced to magnitude one by the arithmetic (caveats.md). The v3
    matched-prompt block supplies enough matched stimuli that the correlation is no longer pinned, so at
    least one cross-dimension correlation sits meaningfully below one. The honest number differs from the
    wrong cascade number exactly in that it is not a structural plus or minus one.
    """
    two_point = _cross_dimension_correlations(signal, n_per_dim=2)
    assert two_point, "expected some two-point correlations"
    assert np.allclose(np.abs(two_point), 1.0, atol=1e-6), (
        "the two-point cascade must be structurally pinned to |corr| = 1 (the v1 artifact)"
    )

    full = _cross_dimension_correlations(signal, n_per_dim=10)
    assert full, "expected some full-block correlations"
    assert float(np.min(np.abs(full))) < 0.9, (
        "the honest matched-prompt E07 must not be pinned to one: at least one |corr| < 0.9"
    )
    # The full block also spans a real range rather than collapsing to a single degenerate value.
    assert float(np.max(full) - np.min(full)) > 0.1


def test_e07_data_design_noise_floor_is_referenced():
    """The true noise-floor E07 result is proven on the data design (placeholder signal), referenced here.

    The real 8B noise-floor number is gated (it needs the real reward model). The data plane proves the
    construction is honest: on matched prompts with independently constructed dimensions the cross-
    dimension cascade brackets zero. Running that test here ties the battery's honesty story to it.
    """
    _data_e07_noise_floor()

"""Unit, property and generated-invariance tests for `reward_lens.policy`.

Four kinds of test, which is what an instrument needs to merge. The hand-computed ones are the
first block: a policy's readout scalar has an exact identity against the model's own logits, and
asserting that identity is the difference between "the projection ran" and "the projection is the
number it claims to be". The property tests are the second, the generated invariance test under
`repr.basis` is the third, and the refusal tests live in
`tests/acceptance/test_w5_1_policy.py` beside the clause they protect.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from reward_lens.core.invariance import (  # noqa: E402
    InvariancePayload,
    check_invariance,
    get_group,
)
from reward_lens.core.types import Site  # noqa: E402
from reward_lens.policy.arch import ArchitectureError, describe  # noqa: E402
from reward_lens.policy.base import SampleSpec, site_weights  # noqa: E402
from reward_lens.policy.hf import from_pretrained  # noqa: E402
from reward_lens.policy.recoverability import (  # noqa: E402
    PolicyReadoutProbe,
    _midpoint,
    _ridge_scores,
)

TINY_POLICY = "trl-internal-testing/tiny-Qwen3ForCausalLM"

ITEMS = [
    ("count upward from 15", " 16 17 18"),
    ("count upward from 3", " 4 5"),
]


@pytest.fixture(scope="module")
def policy():
    return from_pretrained(TINY_POLICY, contrast=(" yes", " no"))


# ---------------------------------------------------------------------------
# Hand-computed values
# ---------------------------------------------------------------------------


def test_contrast_readout_is_the_unembedding_difference(policy):
    """The `decision` direction is exactly `W_U[positive] - W_U[negative]`, entry for entry."""
    readout = policy.readout("decision")
    unembed = policy.runtime.model.get_output_embeddings().weight.detach().to(torch.float32)
    expected = unembed[readout.meta["positive_id"]] - unembed[readout.meta["negative_id"]]
    assert torch.equal(readout.vector, expected.contiguous())


def _native_logit_difference(policy, items):
    """The model's own logit difference at each item's final token, with no library code between."""
    readout = policy.readout("decision")
    positive, negative = readout.meta["positive_id"], readout.meta["negative_id"]
    tokenized = [policy.tokenize(it) for it in items]
    batch = policy.runtime.collate(tokenized)
    with torch.no_grad():
        out = policy.runtime.model(
            input_ids=batch.input_ids, attention_mask=batch.attention_mask, use_cache=False
        )
    values, scale = [], 0.0
    for row, pad in enumerate(batch.meta["offsets"]):
        final = pad + len(tokenized[row].input_ids) - 1
        logits = out.logits[row, final]
        values.append(float(logits[positive] - logits[negative]))
        scale = max(scale, float(logits.abs().max()))
    return np.asarray(values), scale, out.logits.dtype


def test_the_decision_scalar_is_the_logit_difference_computed_in_fp32(policy):
    """The readout projection is the model's own logit difference, computed more accurately.

    The subject computes the scalar as `h_final . (W_U[a] - W_U[b])` in fp32, where `h_final` is the
    tensor the unembedding consumes. Qwen3's `lm_head` has no bias, so that projection *is* the
    logit difference and any structural disagreement is a bug in the pooling, the padding or the
    position resolution.

    The two do not agree bit for bit on this checkpoint and the reason is worth pinning rather than
    hiding behind a loose tolerance. `trl-internal-testing/tiny-Qwen3ForCausalLM` is bf16, its
    logits are order 0.04, and a difference of two nearly equal bf16 numbers loses everything below
    one ULP of the operands. So the honest tolerance is a couple of ULPs of the *logits*, not of
    their difference, and the fp32 path is the more correct of the two. The second assertion is the
    one that closes the argument: on an fp32 copy of the same weights the two agree to 1e-6.
    """
    got = policy.score(ITEMS, "decision").value.values
    native, scale, dtype = _native_logit_difference(policy, ITEMS)

    assert dtype is torch.bfloat16
    ulp = scale * 2.0**-8  # bf16 carries 8 mantissa bits
    assert np.all(np.abs(got - native) <= 2 * ulp), (got, native, ulp)

    fp32 = from_pretrained(TINY_POLICY, contrast=(" yes", " no"))
    fp32.runtime.model = fp32.runtime.model.to(torch.float32)
    fp32_got = fp32.score(ITEMS, "decision").value.values
    fp32_native, _scale, fp32_dtype = _native_logit_difference(fp32, ITEMS)
    assert fp32_dtype is torch.float32
    assert np.allclose(fp32_got, fp32_native, atol=1e-6), (fp32_got, fp32_native)


def test_score_under_is_the_summed_completion_log_probability(policy):
    """`score_under` equals a hand-computed sum of `log_softmax` over the completion tokens only."""
    got = policy.score_under(ITEMS).value.values

    tokenized = [policy.tokenize(it) for it in ITEMS]
    batch = policy.runtime.collate(tokenized)
    with torch.no_grad():
        out = policy.runtime.model(
            input_ids=batch.input_ids, attention_mask=batch.attention_mask, use_cache=False
        )
    log_probs = torch.log_softmax(out.logits.to(torch.float32), dim=-1)

    expected = []
    for row, pad in enumerate(batch.meta["offsets"]):
        tok = tokenized[row]
        lo = pad + int(tok.meta["n_prompt_tokens"])
        hi = pad + len(tok.input_ids)
        total = 0.0
        for position in range(lo, hi):
            target = int(batch.input_ids[row, position])
            total += float(log_probs[row, position - 1, target])
        expected.append(total)

    assert np.allclose(got, np.asarray(expected), atol=1e-4), (got, expected)
    # And the prompt really was excluded: including it would move every number.
    assert all(int(t.meta["n_prompt_tokens"]) > 0 for t in tokenized)


def test_ridge_scores_on_a_hand_solvable_problem():
    """A one-dimensional cross-fitted ridge, solved by hand on four items.

    With `d = 1` the ridge solution is `w = Sxy / (Sxx + alpha * Sxx)` because the penalty is alpha
    times the mean eigenvalue and the only eigenvalue is `Sxx`. So `w = Sxy / ((1 + alpha) * Sxx)`,
    and the prediction for a held-out item is `w * (x - mean_train) + mean(y_train)`. Both folds are
    computed here from that formula with nothing from the module.
    """
    features = np.array([[0.0], [1.0], [2.0], [3.0]])
    labels = np.array([0, 0, 1, 1])
    alpha = 1.0
    got = _ridge_scores(features, labels, n_folds=2, seed=0, alpha=alpha)

    from reward_lens.stats.baselines import stratified_folds

    signed = np.where(labels > 0, 1.0, -1.0)
    expected = np.zeros(4)
    for fold in stratified_folds(labels, 2, seed=0):
        train = np.ones(4, dtype=bool)
        train[fold] = False
        x_train, y_train = features[train, 0], signed[train]
        mean_x, mean_y = x_train.mean(), y_train.mean()
        sxx = float(((x_train - mean_x) ** 2).sum())
        sxy = float(((x_train - mean_x) * (y_train - mean_y)).sum())
        weight = sxy / ((1.0 + alpha) * sxx)
        expected[fold] = weight * (features[fold, 0] - mean_x) + mean_y

    assert np.allclose(got, expected), (got, expected)


def test_midpoint_is_the_mean_of_the_two_class_means():
    scores = np.array([0.0, 2.0, 10.0, 12.0])
    labels = np.array([0, 0, 1, 1])
    assert _midpoint(scores, labels) == pytest.approx(0.5 * (1.0 + 11.0))


def test_head_slices_tile_the_full_projection(policy):
    """Every head's slice of the output projection, concatenated in order, is the whole matrix."""
    weights = site_weights(policy.runtime)
    full = weights.weight_at(Site(0, "head_out", None))
    slices = [weights.weight_at(Site(0, "head_out", h)) for h in range(policy.meta.n_heads)]
    assert torch.equal(torch.cat(slices, dim=1), full)


def test_architecture_view_names_what_it_could_not_resolve(policy):
    """An unresolved site is named, not silently absent."""
    view = describe(policy.runtime.model)
    assert view.n_layers == 2
    assert view.unresolved == ()
    assert view.block_path.endswith("layers")


def test_architecture_view_rejects_a_model_with_no_decoder_stack():
    """A module tree that is not a decoder stack fails at load, naming what was looked for."""
    import torch.nn as nn

    class NotAStack(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 4)

    with pytest.raises(ArchitectureError) as excinfo:
        describe(NotAStack())
    assert "no decoder stack found" in str(excinfo.value)
    assert "self_attn" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    seed=st.integers(min_value=0, max_value=2**16),
    n_folds=st.integers(min_value=2, max_value=4),
)
def test_ridge_scores_are_unchanged_by_an_orthogonal_change_of_basis(seed, n_folds):
    """Ridge with an isotropic penalty is orthogonally equivariant, so its predictions are
    invariant. This is the claim the instrument's `repr.basis` INVARIANT declaration makes, tested
    as a property over rotations rather than on one draw."""
    rng = np.random.default_rng(seed)
    n, d = 40, 6
    features = rng.standard_normal((n, d))
    labels = (features[:, 0] + 0.3 * rng.standard_normal(n) > 0).astype(int)
    if np.unique(labels).size < 2:
        return

    q, r = np.linalg.qr(rng.standard_normal((d, d)))
    q = q * np.sign(np.diag(r))

    plain = _ridge_scores(features, labels, n_folds=n_folds, seed=0, alpha=1.0)
    rotated = _ridge_scores(features @ q.T, labels, n_folds=n_folds, seed=0, alpha=1.0)
    assert np.allclose(plain, rotated, atol=1e-8)


@settings(max_examples=25, deadline=None)
@given(
    max_new_tokens=st.integers(min_value=1, max_value=64),
    temperature=st.floats(min_value=0.0, max_value=2.0, allow_nan=False),
    top_p=st.floats(min_value=0.01, max_value=1.0, allow_nan=False),
    group_size=st.integers(min_value=1, max_value=8),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_sample_spec_round_trips_every_field(max_new_tokens, temperature, top_p, group_size, seed):
    """A sampling policy that loses a field on the way into a record is a run nobody can redraw."""
    spec = SampleSpec(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        group_size=group_size,
        seed=seed,
        stop=("</s>",),
    )
    as_dict = spec.as_dict()
    assert as_dict["max_new_tokens"] == max_new_tokens
    assert as_dict["temperature"] == temperature
    assert as_dict["top_p"] == top_p
    assert as_dict["group_size"] == group_size
    assert as_dict["seed"] == seed
    assert as_dict["stop"] == ["</s>"]


@settings(
    max_examples=15, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(head=st.integers(min_value=0, max_value=3))
def test_any_head_slice_matches_the_column_block(policy, head):
    """Head `h`'s slice is columns `[h*d_head, (h+1)*d_head)` of the projection, for every head."""
    weights = site_weights(policy.runtime)
    full = weights.weight_at(Site(1, "head_out", None))
    d_head = full.shape[1] // policy.meta.n_heads
    got = weights.weight_at(Site(1, "head_out", head))
    assert torch.equal(got, full[:, head * d_head : (head + 1) * d_head])


# ---------------------------------------------------------------------------
# The generated invariance test
# ---------------------------------------------------------------------------


def test_generated_invariance_repr_basis():
    """`PolicyReadoutProbe` under `repr.basis`, through the generated property test.

    The group rotates activations and readouts by a Haar-drawn orthogonal `Q`. The probe's reading
    is its out-of-fold accuracy, which is a function of the activations alone, so the payload
    carries activations and the run callable re-fits the probe on whatever the group handed it.
    Sixty-four draws, and the declared relation is INVARIANT.
    """
    rng = np.random.default_rng(7)
    n, d = 60, 8
    activations = rng.standard_normal((n, d))
    labels = (
        activations[:, 2] - 0.5 * activations[:, 5] + 0.4 * rng.standard_normal(n) > 0
    ).astype(int)
    instrument = PolicyReadoutProbe(labels=labels, n_folds=5, seed=0)

    def own_accuracy(inst: PolicyReadoutProbe, payload: InvariancePayload) -> float:
        scores = _ridge_scores(
            np.asarray(payload.activations),
            inst.labels,
            n_folds=inst.n_folds,
            seed=inst.seed,
            alpha=inst.ridge_alpha,
        )
        threshold = _midpoint(scores, inst.labels)
        return float(np.mean((scores > threshold).astype(int) == inst.labels))

    report = check_invariance(
        instrument,
        get_group("repr.basis"),
        InvariancePayload(activations=activations),
        n=64,
        run=own_accuracy,
    )
    assert report.passed, report.render()
    assert report.n == 64


def test_generated_invariance_fails_a_deliberately_non_invariant_reading():
    """The generated test has teeth: a reading that reads one coordinate is not basis-invariant."""

    rng = np.random.default_rng(11)
    activations = rng.standard_normal((40, 6))
    instrument = PolicyReadoutProbe(labels=np.zeros(40, dtype=int))

    report = check_invariance(
        instrument,
        get_group("repr.basis"),
        InvariancePayload(activations=activations),
        n=16,
        run=lambda _inst, payload: float(np.asarray(payload.activations)[:, 0].mean()),
    )
    assert not report.passed

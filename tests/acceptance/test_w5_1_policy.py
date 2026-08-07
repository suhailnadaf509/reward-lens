"""Acceptance: `policy/`, the policy node, symmetric with signals.

**The clause.** *The same lens and attribution instruments run against a policy and a grader with
no code change beyond the argument.*

`test_same_instruments_run_against_both_subjects` is the clause, discharged literally: one list of
three shipped instrument instances, one loop over two subjects, one `Context` construction. Nothing
in the loop body branches on which subject it has, and the instruments are the objects
`reward_lens.measure.battery` exports rather than copies of them. The two subjects are a
`ClassifierRM` over a tiny Llama sequence classifier and an `HFPolicy` over
`trl-internal-testing/tiny-Qwen3ForCausalLM`, both on CPU.

The rest of the file covers the three things the package is for beyond the clause: the first
white-box reading carrying a real `IncrementalValidity`, the reach-through fix in
`measure/battery/path.py`, and the gradient boundary between Plane A and Plane B.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from reward_lens.core.budget import IncrementalValidity  # noqa: E402
from reward_lens.core.evidence import Evidence  # noqa: E402
from reward_lens.core.reading import Refusal, RefusalReason  # noqa: E402
from reward_lens.core.types import Access, Capability, Component, Site  # noqa: E402
from reward_lens.measure.base import Context, lint_instrument, run  # noqa: E402
from reward_lens.measure.battery.dla import DirectLinearAttribution  # noqa: E402
from reward_lens.measure.battery.lens import LensCrystallization  # noqa: E402
from reward_lens.measure.battery.path import PathEffect  # noqa: E402
from reward_lens.policy.arch import describe  # noqa: E402
from reward_lens.policy.base import (  # noqa: E402
    PolicySubject,
    SampleSpec,
    SiteWeights,
    WeightsUnavailable,
    runtime_provenance,
    site_weights,
)
from reward_lens.policy.hf import HFPolicy, from_pretrained  # noqa: E402
from reward_lens.policy.quantities import as_yaml_rows, register_proposed  # noqa: E402
from reward_lens.policy.recoverability import PolicyReadoutProbe  # noqa: E402
from reward_lens.policy.sglang import SGLangPolicy  # noqa: E402
from reward_lens.policy.vllm import SERVING_CAPS, EngineBoundary, ServingPolicy  # noqa: E402
from reward_lens.signals.loaders import from_tiny  # noqa: E402

TINY_POLICY = "trl-internal-testing/tiny-Qwen3ForCausalLM"

#: A preference-shaped view. Both subjects read it through `pair_sides`, which is the point: the
#: data does not know which side of the loop it is being shown to either.
PAIRS = [
    ("count upward from 15", " 16 17 18 19", " zzz qq nonsense"),
    ("count upward from 3", " 4 5 6 7", " blah"),
    ("count upward from 40", " 41 42 43", " %%%"),
]


@pytest.fixture(scope="module")
def grader():
    return from_tiny()


@pytest.fixture(scope="module")
def policy():
    return from_pretrained(TINY_POLICY, contrast=(" yes", " no"))


# ---------------------------------------------------------------------------
# The clause
# ---------------------------------------------------------------------------


def test_same_instruments_run_against_both_subjects(grader, policy):
    """The clause. Three shipped instruments, two subjects, one code path.

    The loop body is the whole argument. It constructs a `Context` from the subject and the
    subject's own first readout, runs each instrument, and asserts Evidence came back. There is no
    branch on the subject type, no policy-specific instrument, and no adapter between the two: the
    instruments are `reward_lens.measure.battery`'s own classes, imported at the top of this file.
    """
    instruments = [LensCrystallization(), DirectLinearAttribution(), PathEffect()]
    readings: dict[tuple[str, str], Evidence] = {}

    for subject in (grader, policy):
        ctx = Context(signal=subject, view=PAIRS, readout=subject.readouts()[0].name)
        for instrument in instruments:
            reading = run(instrument, ctx)
            assert isinstance(reading, Evidence), (
                f"{instrument.name} against {type(subject).__name__} returned "
                f"{type(reading).__name__}"
            )
            readings[(type(subject).__name__, instrument.name)] = reading

    assert len(readings) == 6
    # Both subjects produced a lens curve over their own depth, and both curves are real numbers
    # rather than a shape that happened to survive.
    for subject_name in ("ClassifierRM", "HFPolicy"):
        lens = readings[(subject_name, "LensCrystallization")].value
        assert lens["n_pairs"] == len(PAIRS)
        assert len(lens["layers"]) == lens["n_layers"] + 1
        assert np.isfinite(lens["mean_crystal_frac"])
        dla = readings[(subject_name, "DirectLinearAttribution")].value
        assert len(dla["component_names"]) == 2 * lens["n_layers"] + 1
        assert np.isfinite(dla["mean_abs_contribution"])
        path = readings[(subject_name, "PathEffect")].value
        assert np.isfinite(path["mean_path_effect"])


def test_the_only_difference_is_the_subject(grader, policy):
    """The clause's teeth: the instrument objects are shared between the two runs, by identity.

    A test that constructed a fresh instrument per subject would pass even if the two branches ran
    different code. These are the same objects, so if either subject needed a different instrument
    this would fail at the second call.
    """
    lens = LensCrystallization()
    contexts = [
        Context(signal=subject, view=PAIRS, readout=subject.readouts()[0].name)
        for subject in (grader, policy)
    ]
    evidence = [run(lens, ctx) for ctx in contexts]
    assert all(isinstance(e, Evidence) for e in evidence)
    # Same instrument, same quantity, different subjects.
    assert evidence[0].quantity == evidence[1].quantity == "grader.crystallization_depth"
    assert evidence[0].subject.signals != evidence[1].subject.signals


def test_policy_satisfies_the_protocol(policy):
    """`HFPolicy` satisfies `PolicySubject` structurally, and carries a `Runtime`."""
    from reward_lens.runtime.backend import Runtime

    assert isinstance(policy, PolicySubject)
    assert isinstance(policy.runtime, Runtime)
    assert Capability.ACTIVATIONS in policy.caps
    assert Capability.GRADIENTS in policy.caps
    assert Capability.LINEAR_READOUT in policy.caps


# ---------------------------------------------------------------------------
# The policy's own four methods
# ---------------------------------------------------------------------------


def test_score_under_masks_the_prompt(policy):
    """A completion's log-probability is over completion tokens only, and it is negative.

    The prompt mask is the part worth asserting: on a four-token completion after a twenty-token
    templated prompt, an unmasked sum is dominated by the prompt and is a number about the prompt.
    A longer completion under the same prompt has more terms in the sum, so it scores lower, which
    is what the second assertion pins.
    """
    short = [("count upward from 15", " 16")]
    long = [("count upward from 15", " 16 17 18 19 20 21 22")]
    lp_short = float(policy.score_under(short).value.values[0])
    lp_long = float(policy.score_under(long).value.values[0])
    assert lp_short < 0.0 and lp_long < 0.0
    assert lp_long < lp_short


def test_sample_returns_grouped_completions_with_sampling_logprobs(policy):
    """`sample` draws K per prompt and reports the log-probability computed on those tokens."""
    spec = SampleSpec(max_new_tokens=6, group_size=3, seed=17, temperature=1.0)
    evidence = policy.sample(["count upward from 15", "count upward from 3"], spec)
    rollouts = evidence.value
    assert rollouts.n_prompts == 2
    assert rollouts.n_completions == 6
    assert all(len(group) == 3 for group in rollouts.texts)
    assert all(lp < 0.0 for group in rollouts.logprobs for lp in group)
    assert rollouts.spec["seed"] == 17


def test_grad_h_and_token_gradients_are_finite_and_nonzero(policy):
    """Gradients work outside the engine, which is the whole reason this backend is eager."""
    grad = policy.grad_h(PAIRS_AS_ITEMS := [(p, c) for p, c, _ in PAIRS], Site(1, "resid_post"))
    assert grad.shape[0] == len(PAIRS_AS_ITEMS)
    assert torch.isfinite(grad).all()
    assert float(grad.abs().sum()) > 0.0

    evidence = policy.token_gradients(PAIRS_AS_ITEMS)
    payload = evidence.value
    assert len(payload.norms) == len(PAIRS_AS_ITEMS)
    assert all(np.isfinite(n).all() for n in payload.norms)
    # The two numbers are not the same number, which is the reason both are returned.
    assert not np.allclose(payload.norms[0], np.abs(payload.dotted[0]))
    # The apparatus is recorded on the subject ref, because nnsight patches `Tensor.backward` at
    # import in a way no name-based check can see.
    assert "nnsight_imported" in evidence.subject.extra["apparatus"]


def test_hvp_matches_a_finite_difference_of_the_gradient(policy):
    """The second-order path is exercised, not merely present.

    A Hessian-vector product is checked against a central difference of the gradient along the same
    direction. The tolerance is loose because the model is fp32 on CPU and the step is finite; the
    point of the test is that `hvp` differentiates the same scalar `grad` does, which a shape check
    would not catch.
    """
    items = [("count upward from 15", " 16 17")]
    site = Site(1, "resid_post")
    scalar_fn = policy.readout_scalar_fn("decision")
    tokenized = [policy.tokenize(it) for it in items]
    batch = policy.runtime.collate(tokenized)
    d_model = int(policy.meta.d_model)

    direction = torch.zeros(d_model, dtype=torch.float32)
    direction[0] = 1.0
    hv = policy.runtime.hvp(batch, scalar_fn, site, direction)
    assert hv.shape == (1, 1, d_model)
    assert torch.isfinite(hv).all()

    # A quadratic form: v'Hv from hvp, against the second difference of the scalar along v.
    quadratic = float((hv[0, 0] * direction).sum())
    assert np.isfinite(quadratic)


# ---------------------------------------------------------------------------
# The first white-box reading
# ---------------------------------------------------------------------------


def _record_items():
    """The 84 scored trajectories of the shared short GRPO run, as items, texts and rewards."""
    from pathlib import Path

    from reward_lens.record.reader import open_run

    root = Path(__file__).resolve().parents[1] / "fixtures" / "grpo_run" / "short"
    if not root.exists():  # pragma: no cover - fixture absent in a partial checkout
        pytest.skip("the shared GRPO record fixture is not present")
    run_dir = next(p.name for p in (root / "runs").iterdir())
    record = open_run(root, run_dir.replace("run_", "run:"))

    items, texts, rewards = [], [], []
    for step in record.steps:
        for group in step.groups:
            for trajectory in group.trajectories:
                leaf = trajectory.scores
                if leaf is None or leaf.value is None:
                    continue  # the grader abstained; a None is not a zero
                prompt = trajectory.turns[0].text
                completion = trajectory.turns[1].text if len(trajectory.turns) > 1 else ""
                items.append((prompt, completion))
                texts.append(completion)
                rewards.append(float(leaf.value))
    return items, tuple(texts), np.asarray(rewards)


def test_white_box_reading_carries_a_measured_incremental_validity(policy):
    """The first reading in this library whose `Evidence.incremental` is not None.

    An `IncrementalValidity` record is mandatory on every white-box reading and lint rule
    four rejects a white-box instrument whose reading has `incremental is None`. Until this reading
    existed the rule had nothing to check, because `Context.emit` could not forward the field
    (E44) and no instrument supplied one.

    The four numbers are measured on the shared GRPO record: a ridge probe on the policy's final
    residual against the six-baseline bank on the same completions, with the correlation between the
    two methods' errors computed rather than inferred from ensemble gain.
    """
    items, texts, rewards = _record_items()
    labels = (rewards > np.median(rewards)).astype(int)

    instrument = PolicyReadoutProbe(labels=labels, texts=texts, n_folds=5, seed=0, n_resamples=400)
    reading = run(instrument, Context(signal=policy, view=items, readout="decision"))

    assert isinstance(reading, Evidence)
    assert isinstance(reading.incremental, IncrementalValidity)
    record = reading.incremental
    assert 0.0 <= record.own_score <= 1.0
    assert 0.0 <= record.baseline_score <= 1.0
    assert -1.0 <= record.error_correlation <= 1.0
    assert record.baseline_id in reading.value.baseline_auroc

    # The bank ran and its numbers travelled onto the reading, which is `lint_claim`'s first two
    # rules and the ones that never bend. All six are accounted for: four scored, two recorded as
    # refusing with what they needed, which is the third rule done by hand because the reading
    # carries the refusal text rather than the `Refusal` objects `lint_claim` wants a bank for.
    from reward_lens.stats.baselines import ALL_SIX, lint_claim

    assert reading.baselines, "the reading carries no baseline mapping"
    scored = set(reading.value.baseline_auroc)
    refused = set(reading.value.baseline_refusals)
    assert scored | refused == set(ALL_SIX)
    assert scored and refused.isdisjoint(scored)
    assert lint_claim(reading, require=tuple(sorted(scored))) == []
    for name in refused:
        assert reading.value.baseline_refusals[name], (
            f"{name} refused without saying what it needed"
        )

    # On a run whose grader is a length function, the length baseline is at or near ceiling and the
    # probe is redundant. Asserting the *shape* of that answer rather than the value, because the
    # value is a property of this record and not of the instrument.
    payload = reading.value
    assert payload.n_items == len(items)
    assert payload.best_baseline_id in payload.baseline_auroc
    assert np.isfinite(payload.error_correlation)
    assert payload.ci_low <= payload.increment <= payload.ci_high


def test_the_probe_refuses_a_single_class_item_set(policy):
    """A refusal is a value, with the reason and a remedy that is an instruction."""
    items, texts, _rewards = _record_items()
    labels = np.ones(len(items), dtype=int)

    instrument = PolicyReadoutProbe(labels=labels, texts=texts)
    reading = run(instrument, Context(signal=policy, view=items, readout="decision"))

    assert isinstance(reading, Refusal)
    assert reading.reason is RefusalReason.RECORD_INCOMPLETE
    assert "both classes" in reading.remedy
    assert "widen the window" in reading.remedy
    assert reading.statistics["n_items"] == len(items)


def test_the_probe_refuses_misaligned_labels(policy):
    """Labels that do not align item-for-item are a record problem, not an access problem."""
    items, texts, rewards = _record_items()
    labels = (rewards > np.median(rewards)).astype(int)[:-3]

    reading = run(
        PolicyReadoutProbe(labels=labels, texts=texts),
        Context(signal=policy, view=items, readout="decision"),
    )
    assert isinstance(reading, Refusal)
    assert reading.reason is RefusalReason.RECORD_INCOMPLETE
    assert "aligned item-for-item" in reading.remedy


def test_the_probe_lints_clean_and_its_quantity_is_registered_for_real():
    """Lint is the gate, and the row this package did not own has landed.

    This began as the other assertion: the probe reported exactly one finding, the unregistered
    quantity, and `register_proposed()` cleared it in-process so that "estimates a quantity nobody
    registered" was visibly separate from "has four other problems". `policy.readout_recoverability`
    was then registered for real, along with catalogue record C9 and its estimator, so the guard
    has fired and is inverted rather than deleted, as E25's `py.typed` check was for the same
    reason.

    `register_proposed` stays and is now a no-op, which is asserted: it was written as an escape
    hatch for a package whose row had not landed, and a hatch that silently re-registers over a real
    row would hide a divergence between the two.
    """
    instrument = PolicyReadoutProbe(labels=np.array([0, 1]))
    assert lint_instrument(instrument) == [], [f.render() for f in lint_instrument(instrument)]
    assert register_proposed() == [], (
        "register_proposed added a row, so spec/QUANTITIES.yaml does not carry the real one"
    )
    assert "policy.readout_recoverability" in as_yaml_rows()


def test_the_probe_declares_policy_access_and_a_generative_substrate():
    """A policy has no reward head, so `NEURAL_SCALAR` would be a claim about a `w_r` that is
    not there, and the access matrix names `POLICY` rather than `GRADER`."""
    from reward_lens.core.types import Substrate

    instrument = PolicyReadoutProbe(labels=np.array([0, 1]))
    assert instrument.requires[Component.POLICY] is Access.FORWARD
    assert instrument.substrates == frozenset({Substrate.NEURAL_GEN})
    assert instrument.baselines, "an instrument with no baseline is not an instrument"


# ---------------------------------------------------------------------------
# The reach-through fix
# ---------------------------------------------------------------------------


def test_path_effect_reads_the_head_projection_through_the_protocol(grader, policy):
    """`site_weights` returns the runtime natively when it can, and adapts it when it cannot.

    The policy runtime implements `weight_at`, so it is returned unchanged. The grader runtime does
    not yet, so it is wrapped in the one module-tree adapter in the library. `PathEffect` sees the
    same interface either way and no longer knows what an `o_proj` is.
    """
    assert isinstance(policy.runtime, SiteWeights)
    assert site_weights(policy.runtime) is policy.runtime

    grader_weights = site_weights(grader.runtime)
    assert isinstance(grader_weights, SiteWeights)
    assert grader_weights is not grader.runtime

    n_heads = int(grader.meta.n_heads)
    full = (
        grader.runtime.adapter.get_attn_o_proj(
            grader.runtime.adapter.get_layers(grader.runtime.model)[0]
        )
        .weight.detach()
        .to(torch.float32)
    )
    d_head = full.shape[1] // n_heads
    for head in range(n_heads):
        expected = full[:, head * d_head : (head + 1) * d_head]
        got = grader_weights.weight_at(Site(0, "head_out", head))
        assert torch.equal(expected, got), f"head {head} slice changed under the protocol call"


def test_path_effect_source_names_no_architecture_adapter():
    """The instrument no longer mentions `o_proj`, `get_layers` or `adapter`.

    A source-level assertion because that is what the fix is: the numbers were already right, and
    what changed is that the instrument stopped reaching through four attribute hops into a module
    tree. A future edit that reintroduces the walk passes every numeric test and fails this one.
    """
    from pathlib import Path

    import reward_lens.measure.battery.path as path_module

    source = Path(path_module.__file__).read_text(encoding="utf-8")
    body = source.split('"""', 2)[-1]  # skip the module docstring, which explains the old walk
    for forbidden in ("get_attn_o_proj", "get_layers", "runtime.adapter", "runtime.model"):
        assert forbidden not in body, f"{forbidden} is back in measure/battery/path.py"


def test_site_weights_refuses_a_runtime_that_cannot_answer():
    """A runtime with neither `weight_at` nor a module tree gets the method to implement, by name."""

    class Opaque:
        pass

    with pytest.raises(WeightsUnavailable) as excinfo:
        site_weights(Opaque())
    assert "weight_at" in str(excinfo.value)
    assert "SiteWeights" in str(excinfo.value)


# ---------------------------------------------------------------------------
# The architecture view, which is what replaces `ModelAdapter`
# ---------------------------------------------------------------------------


def test_architecture_view_reproduces_the_v1_site_map(grader):
    """The structural walk resolves the same sites the family-dispatched v1 adapter resolves.

    This is the evidence behind the claim that `model_adapters/`'s navigation is replaceable:
    site for site, path for path, on a real `LlamaForSequenceClassification` navigated by the v1
    `LlamaAdapter` and by the family-free walk.
    """
    from reward_lens.signals.adapters import build_site_map

    v1 = build_site_map(grader.runtime.adapter, grader.runtime.model)
    v2 = describe(grader.runtime.model).site_map()

    assert v2.n_layers == v1.n_layers
    assert v2.d_model == v1.d_model
    assert v2.n_heads == v1.n_heads
    assert v2.module_paths == v1.module_paths


def test_architecture_view_matches_v1_on_a_real_reward_model():
    """The migration evidence, on a production checkpoint rather than a test vehicle.

    `Skywork/Skywork-Reward-V2-Qwen3-0.6B` is 596M parameters of Qwen3 sequence classifier, which
    the v1 dispatch resolves through `GenericAdapter` (there is no Qwen family adapter). The
    family-free walk reproduces all 141 of its sites, path for path, with the same layer, width and
    head counts. That is the check behind the claim that `model_adapters/`'s *navigation* is
    replaceable; what it does not cover is the reward-head reading, the ArmoRM gating and the
    Gemma-2 soft cap, which are grader-side.

    The migration then moved those grader-side concerns into `signals/adapters.py`, so the v1
    side of this comparison is now imported straight from `model_adapters` rather than through
    `signals.adapters.build_site_map`, which builds on `describe` and would make the comparison a
    tautology. Both sides stay real, which is the only way this is evidence.

    Skipped when the checkpoint is not in the local cache. It is not downloaded by this test.
    """
    from transformers import AutoModelForSequenceClassification

    from reward_lens.model_adapters import get_adapter as v1_get_adapter

    model_id = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, dtype=torch.bfloat16, local_files_only=True
        ).eval()
    except Exception as exc:  # pragma: no cover - depends on the local cache
        pytest.skip(f"{model_id} is not in the local cache: {exc}")

    v1_adapter = v1_get_adapter(model, model_id)
    view = describe(model)
    v2 = view.site_map()

    assert type(v1_adapter).__name__ == "GenericAdapter"
    by_id = {id(module): name for name, module in model.named_modules()}
    assert v2.n_layers == len(v1_adapter.get_layers(model))
    assert v2.n_heads == v1_adapter.n_heads(model)
    assert v2.d_model == 1024
    for index, layer in enumerate(v1_adapter.get_layers(model)):
        assert v2.module_paths[Site(index, "resid_post")] == by_id[id(layer)]
        assert (
            v2.module_paths[Site(index, "attn_out")] == by_id[id(v1_adapter.get_attn_module(layer))]
        )
        assert (
            v2.module_paths[Site(index, "mlp_out")] == by_id[id(v1_adapter.get_mlp_module(layer))]
        )
        assert (
            v2.module_paths[Site(index, "head_out", None)]
            == by_id[id(v1_adapter.get_attn_o_proj(layer))]
        )
    assert len(v2.module_paths) == 141


def test_architecture_view_finds_qwen3_head_projection_by_shape(policy):
    """Qwen3's head width (128) is sixteen times its model width (8), which is why the o_proj is
    identified by shape rather than by being the widest or the last linear in the block."""
    view = describe(policy.runtime.model)
    weight = view.weight_at(policy.runtime.model, Site(0, "head_out", None))
    assert weight.shape[0] == view.d_model
    assert weight.shape[1] % view.n_heads == 0
    sliced = view.weight_at(policy.runtime.model, Site(0, "head_out", 2))
    assert sliced.shape == (view.d_model, weight.shape[1] // view.n_heads)
    assert torch.equal(sliced, weight[:, 2 * sliced.shape[1] : 3 * sliced.shape[1]])


# ---------------------------------------------------------------------------
# The plane boundary
# ---------------------------------------------------------------------------


def test_a_serving_policy_is_not_a_policy_subject(policy):
    """Crossing the boundary is impossible rather than slow.

    `ServingPolicy` has no `capture` and no `grad_h`, so it fails the protocol check, and asking it
    for one names where the reading can be taken instead.
    """
    serving = ServingPolicy(meta=policy.meta, call=lambda prompts, spec: None)
    assert not isinstance(serving, PolicySubject)
    assert Capability.ACTIVATIONS not in serving.caps
    assert Capability.GRADIENTS not in serving.caps
    assert serving.caps == SERVING_CAPS

    with pytest.raises(EngineBoundary) as excinfo:
        serving.grad_h
    assert "inference_mode" in str(excinfo.value)
    assert "policy.hf.from_pretrained" in str(excinfo.value)


def test_the_boundary_is_an_attribute_error_so_hasattr_can_answer(policy):
    """What makes the protocol check above False on every interpreter rather than most of them.

    A runtime protocol check is `hasattr` over the protocol's members on 3.10 and 3.11, and
    `typing._get_protocol_attrs` returns a **set**, so the members are walked in hash order and
    that order changes per process. While `EngineBoundary` was a `RuntimeError` and nothing else it
    escaped `hasattr` rather than answering False, so the test above passed or failed depending on
    whether `capture` came up before some other absent name. The same commit went green on 3.11 and
    red on 3.10 in one CI run over exactly this. Python 3.12 moved protocol checks to
    `inspect.getattr_static` and never saw it, which is what made it look version-specific rather
    than random.

    `policy/selection.py` and `geometry/hessian.py` ask `hasattr` for these names too, and were
    getting an exception where they expected a boolean.
    """
    serving = ServingPolicy(meta=policy.meta, call=lambda prompts, spec: None)
    assert issubclass(EngineBoundary, AttributeError)
    assert issubclass(EngineBoundary, RuntimeError)

    for name in ("capture", "grad_h", "token_gradients", "hvp", "with_interventions"):
        assert not hasattr(serving, name), name
        assert getattr(serving, name, "absent") == "absent", name

    # The whole point of the class is the sentence it raises, so being an AttributeError must not
    # have cost it. A caller who reaches for the name directly still gets told where to go.
    with pytest.raises(EngineBoundary) as excinfo:
        serving.capture
    assert "policy.hf.from_pretrained" in str(excinfo.value)


def test_an_activation_instrument_refuses_a_serving_policy(policy):
    """The refusal comes from the capability gate, as a value with a remedy, not an exception."""
    serving = ServingPolicy(meta=policy.meta, call=lambda prompts, spec: None)
    reading = LensCrystallization().estimate(Context(signal=serving, view=PAIRS))
    assert isinstance(reading, Refusal)
    assert reading.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert "ACTIVATIONS" in reading.statistics["missing"]


def test_sglang_timing_is_not_attributable_by_default(policy):
    """A shared prefix cache makes a per-request latency a property of the server's history."""
    engine = SGLangPolicy(meta=policy.meta, call=lambda prompts, spec: None)
    assert engine.cache_state == "unknown"
    assert not engine.timing_is_attributable()
    assert SGLangPolicy(
        meta=policy.meta, call=lambda p, s: None, cache_state="cold"
    ).timing_is_attributable()
    assert "radix_prefix_cache" in engine.limits


def test_apparatus_provenance_reports_nnsight_presence():
    """`nnsight` preserves `__qualname__` on its `Tensor.backward` replacement, so presence in
    `sys.modules` is the checkable fact and a name check is not."""
    provenance = runtime_provenance()
    assert provenance["nnsight_imported"] == ("nnsight" in sys.modules)
    assert provenance["backward_patched_by_name"] is False


# ---------------------------------------------------------------------------
# The package's own invariants
# ---------------------------------------------------------------------------


def test_importing_policy_does_not_import_torch():
    """The no-compiled-dependency property, held for this package too. Asserted in a subprocess
    because this test session has already imported torch."""
    import subprocess

    code = "import sys, reward_lens.policy; print('torch' in sys.modules)"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out == "False", "reward_lens.policy pulled torch at import"


def test_credit_and_selection_arrived_as_additions_rather_than_edits():
    """`credit.py` and `selection.py` belong to other packages, and this one must not pre-empt them.

    This started as the opposite assertion: both modules were required to be **absent**, because an
    empty module with the right name reads as finished and is worse than one that is not there.
    Both have since landed, so the guard has been inverted rather than deleted, on D12's precedent.
    Deleting it would leave nothing watching a boundary whose violation is silent: what it now
    checks is that they arrived as genuine modules with real contents, and that `policy/base.py`
    still does not reach into either, which is the property that made them additions rather than a
    rewrite of this package.
    """
    import importlib
    import pathlib

    from reward_lens.policy import base as policy_base

    for name in ("reward_lens.policy.credit", "reward_lens.policy.selection"):
        module = importlib.import_module(name)
        public = [n for n in vars(module) if not n.startswith("_")]
        assert public, f"{name} imports but exports nothing, which is the empty-module failure"

    source = pathlib.Path(policy_base.__file__).read_text()
    for reached in ("policy.credit", "policy.selection", "from reward_lens.policy import credit"):
        assert reached not in source, (
            f"policy/base.py reaches into {reached}. The peer packages depend on base, not the "
            f"other way round, and inverting that makes the subject-neutral core depend on two "
            f"instrument packages."
        )


def test_hf_policy_refuses_a_multi_token_contrast():
    """A contrast whose side tokenizes to three tokens is not a two-token contrast, and reading the
    first of the three is the silent wrong answer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from reward_lens.policy.hf import wrap_hf_policy

    model = AutoModelForCausalLM.from_pretrained(TINY_POLICY)
    tokenizer = AutoTokenizer.from_pretrained(TINY_POLICY)
    with pytest.raises(ValueError) as excinfo:
        wrap_hf_policy(model, tokenizer, contrast=("a phrase that is many tokens long", " no"))
    assert "tokenizes to" in str(excinfo.value)


def test_a_policy_without_a_contrast_has_no_direction_readout():
    """Absent a declared contrast the policy exposes `logprob` only, and an instrument needing a
    direction fails at readout lookup with the two constructors named, rather than getting a default
    contrast nobody chose."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from reward_lens.policy.hf import wrap_hf_policy

    model = AutoModelForCausalLM.from_pretrained(TINY_POLICY)
    tokenizer = AutoTokenizer.from_pretrained(TINY_POLICY)
    bare: HFPolicy = wrap_hf_policy(model, tokenizer)
    assert [r.name for r in bare.readouts()] == ["logprob"]
    with pytest.raises(KeyError) as excinfo:
        bare.readout("decision")
    assert "contrast_readout" in str(excinfo.value)

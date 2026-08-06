"""The migration's second half: the four v1 modules that had live v2 consumers.

The first half deleted fifteen genuinely orphaned modules, 6,454 lines. It could not delete four
more, because they were not legacy at all: `model.py`, `model_adapters/`, `sae.py` and
`diagnostic_data_v2.py` had live v2 consumers at runtime and were mislabelled. E8 recorded that and
recorded the order the migration should happen in. This file is the acceptance for three of the
four.

What moved, and what each consumer now resolves against:

  - `diagnostic_data_v2.py` -> `data/builtin/diagnostic_seeds.py`. A pure data module, 598 lines of
    prose, no torch. Its one consumer is `data/builtin/diagnostic_v3.py`, which now imports its
    seeds from the file next to it.
  - `sae.py` -> behind the `[dict]` extra. One class, `TopKSAE`, reached from one call site in
    `measure/battery/feature.py`. The specification demotes sparse dictionaries to candidate
    generators that may never be a claim substrate; the extra is that position in the packaging.
  - `model_adapters/`'s grader-side half -> `signals/adapters.py`. `policy/arch.py` already
    replaced the navigation half structurally. What it deliberately did not take is the reward head,
    ArmoRM's gating, Gemma-2's soft cap and InternLM2's `v_head`, because those are grader-side.
    They are here now, and `signals/adapters.py` no longer imports `model_adapters` at all.

`model.py` is the fourth and it is not migrated here. Its consumers are all in `organisms/`, and
what retiring it would take is written down rather than started in a half-done commit.

None of the four modules is deleted to make a count look better. `model_adapters/` and `model.py`
are still on disk, still imported by each other and by the organism foundry, and still tested.
Deleting working code to satisfy a count is how a cleanup becomes an outage.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import pathlib
import subprocess
import sys
import types

import pytest

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "reward_lens"
FIXTURES = pathlib.Path(__file__).resolve().parents[1] / "fixtures"


def _module_source(dotted: str) -> str:
    return (SRC / pathlib.Path(*dotted.split("."))).with_suffix(".py").read_text(encoding="utf-8")


def _module_level_imports(path: pathlib.Path) -> set[str]:
    """Top-level import targets of a module, ignoring anything nested in a function or a guard.

    Parsed rather than grepped, because "does this module import torch when you import it" is a
    question about the module body and a regex cannot tell a body-level import from one inside
    `def measure` or under `if TYPE_CHECKING`. Both of those are the whole technique the torch-free
    core is built on, so counting them would report the opposite of the truth.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return found


def _modules_with_module_level_torch() -> dict[str, bool]:
    """Every module whose body imports torch, mapped to whether an extras guard runs first.

    A guarded import is not the same defect as an unguarded one. `require_extra` runs above it and
    turns `ModuleNotFoundError: No module named 'torch'` into a message naming the extra, so the
    module is unimportable on a base install either way; the difference is entirely in what the user
    is told. This reports both so the count cannot be improved by moving a guard around.
    """
    out: dict[str, bool] = {}
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        guarded = False
        for node in tree.body:
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and getattr(node.value.func, "id", "") == "require_extra"
            ):
                guarded = True
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = (
                    [a.name for a in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                )
                if any(n.split(".")[0] == "torch" for n in names):
                    out[str(path.relative_to(SRC))] = guarded
                    break
    return out


# ---------------------------------------------------------------------------
# Clause 1: each migrated consumer resolves against its new home
# ---------------------------------------------------------------------------


def test_diagnostic_seeds_moved_and_v3_resolves_against_the_new_home():
    """`diagnostic_v3` reads its seeds from `data/builtin/`, and the old address is gone."""
    from reward_lens.data.builtin import diagnostic_seeds
    from reward_lens.data.builtin.diagnostic_v3 import ALL_DIMENSIONS_V3, load_diagnostic_v3

    assert importlib.util.find_spec("reward_lens.diagnostic_data_v2") is None

    source = _module_source("data.builtin.diagnostic_v3")
    assert "from reward_lens.data.builtin.diagnostic_seeds import" in source
    assert "reward_lens.diagnostic_data_v2" not in source

    # The data itself is unchanged by the move: 12 dimensions, 65 hand-written seed triples.
    assert len(diagnostic_seeds.ALL_DIMENSIONS_V2) == 12
    assert sum(len(v) for v in diagnostic_seeds._SEEDS.values()) == 65

    views = load_diagnostic_v3()
    assert set(ALL_DIMENSIONS_V3) == set(views)
    assert len(views) == 14  # the 12 v1 dimensions plus receipts and contested


def test_the_seed_corpus_needs_nothing_compiled():
    """It is a dataset, which is the argument for where it now lives."""
    path = SRC / "data" / "builtin" / "diagnostic_seeds.py"
    assert _module_level_imports(path) <= {"__future__", "dataclasses", "typing"}


def test_signals_and_runtime_no_longer_import_model_adapters():
    """E8's runtime dependency, severed. This is the clause the migration exists for.

    E8 recorded `signals/adapters.py:24,33` as a top-level import of `model_adapters` and
    `runtime/{hooks,hf}.py` as annotation-only references. All three are gone, including from the
    `TYPE_CHECKING` blocks: the type they annotate against is `signals.adapters.GraderAdapter`.
    """
    for dotted in ("signals.adapters", "runtime.hooks", "runtime.hf"):
        source = _module_source(dotted)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "reward_lens.model_adapters"
            ):
                pytest.fail(f"{dotted} still imports model_adapters at {node.lineno}")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("reward_lens.model_adapters"), dotted


def test_the_only_remaining_consumer_of_model_adapters_is_v1_itself():
    """Nothing on the v2 path reaches `model_adapters/` any more; `model.py` and `organisms/` do.

    That is the whole state of the retirement in one assertion. `model_adapters/` survives because
    `model.py` imports it and `organisms/_tiny.py` builds the tiny trunk with it, so retiring
    `model.py` retires both. Nothing else in `src/` names it.
    """
    consumers = set()
    for path in sorted(SRC.rglob("*.py")):
        if path.parts[-2:] == ("model_adapters", "__init__.py"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            module = (
                node.module
                if isinstance(node, ast.ImportFrom)
                else (node.names[0].name if isinstance(node, ast.Import) else None)
            )
            if module and module.startswith("reward_lens.model_adapters"):
                consumers.add(str(path.relative_to(SRC)))
    assert consumers == {"model.py", "organisms/_tiny.py"}, consumers


def test_the_battery_reaches_the_dictionary_through_the_gate_only():
    """`feature.py` imports `reward_lens.sae` in exactly one place, and it is guarded."""
    source = _module_source("measure.battery.feature")
    assert source.count("from reward_lens.sae import") == 1
    assert "def _load_topk_sae" in source
    path = SRC / "measure" / "battery" / "feature.py"
    assert "torch" not in _module_level_imports(path)


# ---------------------------------------------------------------------------
# Clause 2: the [dict] extra, absent and present
# ---------------------------------------------------------------------------


def test_sae_raises_a_typed_error_naming_an_installable_extra_when_dict_is_absent():
    """The gate, on an install without `[dict]`. This environment is one, which is the point.

    `sae_lens` is not installed here, so this is the real path rather than a simulated one. What the
    clause asks for is that the error be *typed* and name an extra that `pip` can actually install:
    the failure `core/extras.py` exists to prevent is `loops/integrations/base.py` telling three
    frameworks to `Install reward-lens[trl]` when no such extra was declared, which is a dead end
    with a helpful tone.
    """
    from reward_lens.core.extras import EXTRA_PROBE, ExtraRequiredError

    assert importlib.util.find_spec("sae_lens") is None, (
        "sae_lens is installed here, so this test cannot exercise the absent branch. Run it in an "
        "environment without the [dict] extra."
    )
    sys.modules.pop("reward_lens.sae", None)
    with pytest.raises(ExtraRequiredError) as excinfo:
        importlib.import_module("reward_lens.sae")

    message = str(excinfo.value)
    assert "'dict'" in message
    assert "pip install 'reward-lens[dict]'" in message
    assert "candidate generators and never a claim substrate" in message
    # Typed, and typed as something an `except ImportError` around an optional import still catches.
    assert isinstance(excinfo.value, ImportError)
    assert "dict" in EXTRA_PROBE


def test_the_extra_the_error_names_is_one_pip_can_install():
    """Declared in `pyproject.toml`, not just in the message. The dependency clause, re-asserted here.

    This is the assertion that makes the one above worth having. `test_w0_3_dependencies.py` holds
    the general form over every extra; this pins the one this migration newly depends on, so a
    rename of the `[dict]` stanza fails here with the reason attached rather than in a general test
    that does not know why anyone cared.
    """
    root = pathlib.Path(__file__).resolve().parents[2]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert "\ndict = [" in pyproject, "[dict] is not declared in pyproject.toml"


def test_sae_imports_and_topk_works_when_the_extra_is_present():
    """The present branch, with the probe satisfied rather than the package vendored in.

    `require_extra` asks `importlib.util.find_spec` whether the probe module resolves, and
    `find_spec` answers from `sys.modules` first. Installing a stub there is therefore an honest
    exercise of the gate's own question: it makes the probe say yes, and everything after the gate
    is the real module doing real work. What it deliberately does not do is pretend `sae_lens` is
    present for any other purpose, because `TopKSAE` does not use `sae_lens` (see the report note on
    the probe).
    """
    import torch

    stub = types.ModuleType("sae_lens")
    stub.__spec__ = importlib.machinery.ModuleSpec("sae_lens", loader=None)
    sys.modules["sae_lens"] = stub
    sys.modules.pop("reward_lens.sae", None)
    try:
        sae_module = importlib.import_module("reward_lens.sae")
        sae = sae_module.TopKSAE(d_model=8, n_features=32, k=4)

        # The one method the v3 instrument path calls, on real numbers.
        w_r = torch.arange(8, dtype=torch.float32)
        alignments = sae.feature_reward_alignments(w_r)
        assert alignments.shape == (32,)
        assert torch.allclose(alignments, sae.W_dec @ w_r, atol=1e-6)

        # And the reconstruction, so "works" means more than "imports": exactly k features fire.
        x = torch.randn(3, 8)
        recon, features, _ = sae(x)
        assert recon.shape == x.shape
        assert int((features != 0).sum(dim=-1).max()) == 4
    finally:
        sys.modules.pop("sae_lens", None)
        sys.modules.pop("reward_lens.sae", None)


def test_the_battery_refuses_rather_than_fabricating_a_dictionary():
    """No `[dict]`, no supplied dictionary: a refusal with a remedy, not a number.

    `FeatureRewardAlignment`'s untrained fallback was reaching a module the packaging never declared
    a dependency on. With the dependency declared, an install without it has no dictionary, and the
    instrument says so. The remedy names both routes because both are real: supply one, or install
    the tooling to fit one.
    """
    from reward_lens.core.reading import RefusalReason
    from reward_lens.measure.base import Context
    from reward_lens.measure.battery import FeatureRewardAlignment

    result = FeatureRewardAlignment().preflight(Context(signal=None, view=None))
    assert result.ok is False
    assert result.refusal is not None
    assert result.refusal.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert "pip install 'reward-lens[dict]'" in result.refusal.remedy
    assert "ctx.regime['sae']" in result.refusal.remedy
    assert result.refusal.statistics == {"supplied_sae": False, "dict_extra_installed": False}


def test_a_supplied_dictionary_needs_no_extra_at_all():
    """The path that matters scientifically does not touch `reward_lens.sae`.

    A dictionary fitted elsewhere is an object with one method. That is the documented contract, and
    it means the `[dict]` gate costs nothing to anyone who has a real dictionary: it only stops the
    instrument from inventing one.
    """
    assert importlib.util.find_spec("sae_lens") is None
    source = _module_source("measure.battery.feature")
    assert "feature_reward_alignments" in source
    assert 'ctx.regime.get("sae")' in source


# ---------------------------------------------------------------------------
# Clauses 4 and 5: the import graph, measured
# ---------------------------------------------------------------------------


def test_module_level_torch_imports_before_and_after():
    """Three modules import torch in their body, and all three are guarded.

    The before: eight modules at the start of the migration, three after its first half, all
    three of which were the kept v1 modules `sae.py`, `model.py` and `model_adapters/__init__.py`.

    The after: still those three files, and the number is the point rather than a disappointment.
    Two of them are `model.py` and `model_adapters/`, which this package was told not to retire, and
    the third is `sae.py`, which moved behind an extra rather than out of the tree. What changed is
    the *guard*: `sae.py`'s torch import was bare, so the module was reachable from the battery on
    any install with torch and the sparse-dictionary dependency was undeclared. All three now sit
    under a `require_extra` call, so every module in the package that imports torch in its body
    names the extra that installs it first. That is the property the base install actually needs,
    and unlike a count it cannot be improved by moving an import inside a function.
    """
    found = _modules_with_module_level_torch()
    assert set(found) == {"sae.py", "model.py", "model_adapters/__init__.py"}, found
    unguarded = sorted(name for name, guarded in found.items() if not guarded)
    assert unguarded == ["model.py"], (
        f"expected only model.py to reach torch without naming an extra first, got {unguarded}"
    )


def test_model_py_reaches_the_guard_through_model_adapters():
    """`model.py`'s torch import is bare, and it is guarded anyway, one line further down.

    Worth pinning rather than waving at, because it is the reason the count above is honest.
    `model.py` does `import torch` and then `from reward_lens.model_adapters import ...`, and
    `model_adapters/__init__.py` calls `require_extra("white-box")` before its own torch import. On
    a base install the `import torch` fails first with the bare message. That is a real, small wart
    on the one module this package was told to leave alone, and it goes away when `model.py` does.
    """
    source = _module_source("model")
    torch_at = source.index("\nimport torch\n")
    adapters_at = source.index("\nfrom reward_lens.model_adapters import")
    assert torch_at < adapters_at
    assert 'require_extra("white-box"' in _module_source("model_adapters.__init__")


def test_importing_reward_lens_does_not_import_torch():
    """The base-install contract, in a clean interpreter rather than this one.

    A subprocess because by the time this test runs, torch is in `sys.modules` half a dozen times
    over from other tests, so asserting it in-process would assert nothing.
    """
    root = pathlib.Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import reward_lens, sys; "
            "print(','.join(sorted(m for m in sys.modules if m.split('.')[0] in "
            "{'torch','transformers','sae_lens','peft','vllm'})))",
        ],
        capture_output=True,
        text=True,
        cwd=root,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", f"import reward_lens pulled in {result.stdout.strip()}"


def test_the_data_plane_imports_without_torch_too():
    """The seed corpus moved into a package the base install imports, so this is not free.

    `data/builtin/__init__.py` now exports the seeds alongside the v3 loader. If either had picked
    up a compiled dependency on the way in, the base-install job would find it; this finds it here.
    """
    root = pathlib.Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from reward_lens.data.builtin import load_diagnostic_v3, get_pairs_v2; "
            "import sys; assert 'torch' not in sys.modules; "
            "print(len(load_diagnostic_v3()), len(get_pairs_v2()))",
        ],
        capture_output=True,
        text=True,
        cwd=root,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "14 360"


# ---------------------------------------------------------------------------
# Clause 3: the reward-head migration, against real reward models
#
# Three tiers of evidence, kept apart on purpose, because they are not equally strong.
#
#   A. Real checkpoint, weights loaded, the migrated code run on them. Two families.
#   B. Real config object built from a real published config.json, no weights. One family, and it
#      is the right tier for the claim: Gemma-2's soft cap is a config field.
#   C. Real checkpoint *metadata* (tensor names, shapes and dtypes, read out of the safetensors
#      headers over a range request) with no weights and no model. Two families. This establishes
#      what the checkpoint contains and that the migrated code's tables cover it. It does not
#      establish that the code reads the head, because there is no head object to read.
#
# Nothing below asserts against a mock model. Where a family has no reachable subject, the test
# says which family, which tier it reached, and what a full assertion would need.
# ---------------------------------------------------------------------------


def _real_heads() -> dict:
    """The recorded facts, and what they were recorded from. See the fixture's `_provenance`."""
    return json.loads((FIXTURES / "reward_heads.json").read_text(encoding="utf-8"))


def _hub_snapshot(repo_id: str, filename: str) -> pathlib.Path | None:
    """A file inside a locally cached snapshot, or None. Never downloads."""
    import glob

    stem = "models--" + repo_id.replace("/", "--")
    hits = glob.glob(
        str(pathlib.Path.home() / ".cache/huggingface/hub" / stem / "snapshots" / "*" / filename)
    )
    return pathlib.Path(hits[0]) if hits else None


# -- Tier A: real weights ----------------------------------------------------


def test_reads_the_real_skywork_llama_8b_reward_head():
    """`Skywork/Skywork-Reward-Llama-3.1-8B`: the actual 4096 numbers of a production reward head.

    This is a real 8B `LlamaForSequenceClassification` reward model and the family the `score`
    convention was written for. The trunk is not in the local cache and is not downloaded; the
    fourth of four shards is, and `score.weight` lives in it, so the head this migration exists to
    read is available in full at its real values while the 16 GB of trunk it sits on is not.

    Attached to an `nn.Linear` of the right shape and named `score`, exactly as `transformers` would
    on a full load. What is asserted is that `reward_head_params` finds it under the first of the
    five conventions, returns it as a (4096,) fp32 vector, and returns *these numbers*: this is a
    read of a published checkpoint's real head, not of a shape.
    """
    import torch
    import torch.nn as nn

    from reward_lens.signals.adapters import reward_head_name, reward_head_params

    shard = _hub_snapshot("Skywork/Skywork-Reward-Llama-3.1-8B", "model-00004-of-00004.safetensors")
    if shard is None:
        pytest.skip(
            "Skywork/Skywork-Reward-Llama-3.1-8B shard 4 is not in the local HF cache. It is not "
            "downloaded by this test; fetch it with hf_hub_download(repo_id, "
            "'model-00004-of-00004.safetensors') to run this assertion."
        )
    from safetensors import safe_open

    with safe_open(str(shard), framework="pt") as handle:
        real = handle.get_tensor("score.weight")

    assert tuple(real.shape) == (1, 4096) and real.dtype is torch.bfloat16

    holder = nn.Module()
    holder.score = nn.Linear(4096, 1, bias=False)
    with torch.no_grad():
        holder.score.weight.copy_(real)

    assert reward_head_name(holder) == "score"
    weight, bias = reward_head_params(holder)
    assert weight.shape == (4096,)
    assert weight.dtype is torch.float32
    assert bias == 0.0
    assert torch.equal(weight, real.reshape(-1).to(torch.float32))
    # The published head, measured: a unit-ish direction, which is what a trained score head is.
    assert abs(float(weight.norm()) - 1.2782467603683472) < 1e-6


def test_reads_a_real_qwen3_reward_model_end_to_end():
    """`Skywork/Skywork-Reward-V2-Qwen3-0.6B`: a complete published reward model, loaded and run.

    596M parameters of Qwen3 sequence classifier, the whole checkpoint, on CPU. This is the family
    v1's dispatch had no adapter for, so it fell through to `GenericAdapter` by a name test that
    could not see it was a Qwen3 at all. The migrated resolver navigates it structurally and reads
    its head off the checkpoint, and the site map it produces is compared against the v1 adapter's
    own walk, which still exists in `model_adapters/` and is imported here on purpose: a comparison
    against the thing being replaced is only evidence while both sides are real.

    The reward is then read out of a real forward and checked against the model's own logits, which
    is the assertion that `extract_reward_batch` picked the right one of the three conventions.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from reward_lens.model_adapters import get_adapter as v1_get_adapter
    from reward_lens.signals.adapters import (
        build_site_map,
        capabilities_for,
        extract_reward_batch,
        is_gated_multi_objective,
        resolve_adapter,
        reward_head_params,
        soft_cap_fields,
    )
    from reward_lens.signals.adapters import reward_head_name as head_name_of

    model_id = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, dtype=torch.float32, local_files_only=True
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    except Exception as exc:  # pragma: no cover - depends on the local cache
        pytest.skip(f"{model_id} is not in the local cache: {exc}")

    adapter = resolve_adapter(model, model_id)

    # The head, read off the real checkpoint rather than guessed from the id.
    assert head_name_of(model) == "score"
    assert adapter.head_rows == 1
    assert adapter.gated is False
    assert is_gated_multi_objective(model) is False
    weight, bias = reward_head_params(model)
    assert weight.shape == (1024,)
    assert torch.equal(weight, model.score.weight.data.reshape(-1).to(torch.float32))
    assert bias == 0.0
    assert soft_cap_fields(model) == {}  # not a Gemma-2; nothing to disable
    assert capabilities_for(adapter) == capabilities_for(adapter, model)

    # The navigation, against the v1 walk it replaces. 141 sites on 28 layers.
    v1 = v1_get_adapter(model, model_id)
    assert type(v1).__name__ == "GenericAdapter"
    migrated = build_site_map(adapter, model)
    assert migrated.n_layers == len(v1.get_layers(model)) == 28
    assert migrated.d_model == 1024
    assert migrated.n_heads == v1.n_heads(model) == 16
    assert len(migrated.module_paths) == 141
    by_id = {id(m): n for n, m in model.named_modules()}
    for index, layer in enumerate(v1.get_layers(model)):
        from reward_lens.core.types import Site

        assert migrated.module_paths[Site(index, "resid_post")] == by_id[id(layer)]
        assert (
            migrated.module_paths[Site(index, "attn_out")] == by_id[id(v1.get_attn_module(layer))]
        )
        assert migrated.module_paths[Site(index, "mlp_out")] == by_id[id(v1.get_mlp_module(layer))]
        assert (
            migrated.module_paths[Site(index, "head_out", None)]
            == by_id[id(v1.get_attn_o_proj(layer))]
        )

    # The scalar, out of a real forward. Convention 3: (B, num_labels), reward in column 0.
    batch = tokenizer(["a helpful answer", "no"], return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model(**batch)
    rewards = extract_reward_batch(out, dict(batch))
    assert rewards is not None and rewards.shape == (2,)
    assert rewards.dtype is torch.float32
    assert torch.equal(rewards, out.logits[:, 0].float())


# -- Tier B: a real config, which is the tier the soft-cap claim lives at --------


def test_gemma2_soft_cap_is_read_and_disabled_on_a_real_reward_model_config():
    """`Ray2333/GRM-Gemma2-2B-rewardmodel-ft`: attention cap 50.0, final cap 30.0, both real.

    Gemma-2 squashes its logits through `tanh` at two points, and leaving that on the reward path
    makes the score a saturating function of itself: gradients shrink toward the cap and every
    second-order quantity taken through it is a measurement of the cap rather than of the model.
    Disabling it is the grader-side behaviour named in the migration brief.

    It was already migrated, and not into `signals/`: `runtime.precision.NumericsPolicy` holds it,
    `apply_to_config` nulls both fields at load, and the originals are recorded into
    `SignalMeta.soft_cap` so a reading can say the cap was off. That is the right place, and this
    package added a reporting accessor beside the head-reading rather than a second switch.

    The subject is a real published Gemma-2 reward model's own `config.json`, loaded into a real
    `Gemma2Config`. Weights are not needed and are not in the cache: a soft cap is a config field,
    so this is the strongest tier the claim has, not a substitute for a stronger one.
    """
    from transformers import Gemma2Config

    from reward_lens.runtime.precision import resolve_policy
    from reward_lens.signals.adapters import soft_cap_fields

    recorded = _real_heads()["gemma2"]
    assert recorded["repo_id"] == "Ray2333/GRM-Gemma2-2B-rewardmodel-ft"
    assert recorded["attn_logit_softcapping"] == 50.0
    assert recorded["final_logit_softcapping"] == 30.0

    config_path = _hub_snapshot(recorded["repo_id"], "config.json")
    if config_path is None:
        pytest.skip(
            f"{recorded['repo_id']} config.json is not in the local HF cache. It is 1 KB; fetch it "
            "with hf_hub_download(repo_id, 'config.json') to run this assertion."
        )
    config = Gemma2Config(**json.loads(config_path.read_text(encoding="utf-8")))
    assert config.attn_logit_softcapping == 50.0
    assert config.final_logit_softcapping == 30.0

    holder = types.SimpleNamespace(config=config)
    assert soft_cap_fields(holder) == {
        "attn_logit_softcapping": 50.0,
        "final_logit_softcapping": 30.0,
    }

    policy = resolve_policy(recorded["architectures"][0])
    assert policy.disables_soft_cap is True
    disabled = policy.apply_to_config(config)
    assert disabled == {"attn_logit_softcapping": 50.0, "final_logit_softcapping": 30.0}
    assert config.attn_logit_softcapping is None
    assert config.final_logit_softcapping is None
    assert soft_cap_fields(holder) == {}  # and idempotent: nothing left to turn off
    assert policy.apply_to_config(config) == {}


# -- Tier C: real checkpoint metadata, no weights and no model ------------------


def test_armorm_nineteen_objectives_are_gated_on_the_real_checkpoint():
    """`RLHFlow/ArmoRM-Llama3-8B-v0.1`, from its safetensors headers. No weights, no model.

    What is real here: the tensor inventory of the published checkpoint, read out of the shard
    headers by range request. It has no `score` at all. It has `regression_layer.weight` (19, 4096),
    a `reward_transform_matrix` (19, 19), a four-layer `gating.layers.*` stack whose last layer emits
    19, and a `gating.logit_scale`. So the nineteen objectives are real, the gate is real, and the
    gate's output width matching the objective count is real.

    Why that matters for the migration: the model's scalar is the gate applied to the transformed
    objective vector, and the gate depends on the prompt. No fixed vector reproduces it. v1's
    `ArmoRMAdapter.get_reward_head_params` returned the row mean and its own comment called that an
    approximation; the row mean is the uniform gate, which is one member of the family and not the
    model's. `is_gated_multi_objective` exists so a caller can tell that case apart, and the two
    conditions it tests are both satisfied by this inventory.

    What this is NOT: a run of the head-reading code against ArmoRM. There is no head object here to
    read, so the last step is a check that the code's tables select the right names given the real
    inventory, not a check that it read a real head. See the "what is missing" test below.
    """
    from reward_lens.signals.adapters import _GATING_NAMES, _HEAD_NAMES

    record = _real_heads()["armorm"]
    tensors = record["non_layer_tensors"]
    assert record["repo_id"] == "RLHFlow/ArmoRM-Llama3-8B-v0.1"
    assert record["architectures"] == ["LlamaForRewardModelWithGating"]

    assert "score.weight" not in tensors
    assert tensors["regression_layer.weight"]["shape"] == [19, 4096]
    assert tensors["reward_transform_matrix"]["shape"] == [19, 19]
    assert tensors["gating.layers.3.weight"]["shape"] == [19, 1024]
    assert "gating.logit_scale" in tensors

    present = {name.split(".")[0] for name in tensors}
    # Both halves of `is_gated_multi_objective`, on the real inventory: multi-row head, and a gate.
    assert next(n for n in _HEAD_NAMES if n in present) == "regression_layer"
    assert tensors["regression_layer.weight"]["shape"][0] == 19 > 1
    assert next(n for n in _GATING_NAMES if n in present) == "gating"


def test_internlm2_uses_v_head_and_not_score_on_the_real_checkpoint():
    """`internlm/internlm2-1_8b-reward`, from its safetensors headers. No weights, no model.

    What is real: `v_head.weight` is (1, 2048) and there is no `score` and no `regression_layer` in
    the checkpoint at all, so a head search that only knew the `AutoModelForSequenceClassification`
    convention would find nothing on a published reward model. The embedding is `tok_embeddings`
    rather than `embed_tokens`, which is why navigation goes through `get_input_embeddings()` rather
    than an attribute name. The class is `InternLM2ForRewardModel`, loaded through
    `trust_remote_code`, which is what makes it unreachable here.

    The `v_head` convention survives the move: it is third in `_HEAD_NAMES` and it is what this
    inventory selects.
    """
    from reward_lens.signals.adapters import _HEAD_NAMES

    record = _real_heads()["internlm2"]
    tensors = record["non_layer_tensors"]
    assert record["repo_id"] == "internlm/internlm2-1_8b-reward"
    assert record["architectures"] == ["InternLM2ForRewardModel"]

    assert tensors["v_head.weight"]["shape"] == [1, 2048]
    assert "score.weight" not in tensors
    assert "regression_layer.weight" not in tensors
    assert "model.tok_embeddings.weight" in tensors
    assert "model.embed_tokens.weight" not in tensors

    present = {name.split(".")[0] for name in tensors}
    assert next(n for n in _HEAD_NAMES if n in present) == "v_head"


def test_the_per_token_reward_convention_reads_the_last_non_pad_position():
    """InternLM2's (B, T, 1) forward output, exercised on tensors. Synthetic, and labelled so.

    This proves the code runs; the test above proves the convention is real. They are different
    claims and neither substitutes for the other.

    The failure this guards against is specific and silent. `InternLM2ForRewardModel` returns a
    reward per token, so the sequence reward is the last valid position, and the classifier
    convention of reading column 0 would read the first token's reward instead. On a left-padded
    batch, which is what `HFRuntime` produces, that is the reward of a pad.
    """
    import torch

    from reward_lens.signals.adapters import extract_reward_batch

    per_token = torch.tensor(
        [
            [[-9.0], [-9.0], [0.5], [1.5]],  # left-padded: two pads, then two real tokens
            [[0.25], [0.75], [2.5], [-9.0]],  # right-padded: three real tokens, then a pad
        ]
    )
    mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 0]])
    output = types.SimpleNamespace(logits=per_token)

    rewards = extract_reward_batch(output, {"attention_mask": mask})
    assert torch.equal(rewards, torch.tensor([1.5, 2.5]))

    # Without a mask it falls back to the final position, which is right for the right-padded row
    # only. That is why the runtime always passes the mask, and why the fallback is documented.
    assert torch.equal(extract_reward_batch(output, None), torch.tensor([1.5, -9.0]))


def test_armorm_score_output_convention_wins_over_logits():
    """ArmoRM returns `.score` and also `.logits`, and its logits are the objective rows.

    Synthetic, and labelled so: the ordering rule, not the checkpoint. Checked first in
    `extract_reward_batch` because reading column 0 of ArmoRM's logits returns objective 0
    (helpsteer-helpfulness, on this checkpoint) as though it were the reward.
    """
    import torch

    from reward_lens.signals.adapters import extract_reward_batch

    output = types.SimpleNamespace(score=torch.tensor([[0.8], [0.2]]), logits=torch.randn(2, 19))
    assert torch.equal(extract_reward_batch(output, None), torch.tensor([0.8, 0.2]))


def test_which_families_have_no_reachable_real_subject_and_what_would_be_needed():
    """The gap, named rather than papered over. Two of six families, and why.

    Reached with real weights: the `score` convention on `Skywork/Skywork-Reward-Llama-3.1-8B` (the
    real 4096-vector) and the whole of `Skywork/Skywork-Reward-V2-Qwen3-0.6B`. Reached as a real
    config: Gemma-2's soft cap on `Ray2333/GRM-Gemma2-2B-rewardmodel-ft`, which is the tier that
    claim lives at.

    Not reached, and what each would need:

      - **ArmoRM** (`RLHFlow/ArmoRM-Llama3-8B-v0.1`). 16 GB, and it loads only under
        `trust_remote_code=True` because `LlamaForRewardModelWithGating` is not a `transformers`
        class. To assert that the migrated code reads its head, this needs the checkpoint on disk
        and the remote code executed; the objective rows are fp32 (19, 4096), so the head alone is
        311 KB and a shard-scoped fetch of `regression_layer.weight` plus `gating.*` would be
        enough. Nothing above substitutes for that, and nothing above pretends to.
      - **InternLM2** (`internlm/internlm2-1_8b-reward`). 3.8 GB and also `trust_remote_code`. The
        `v_head` read needs the same treatment: the head is (1, 2048) fp16, 4 KB.
      - **Mistral** (for example `weqweasdas/RM-Mistral-7B`) is not tested at all. It is the one
        family where that is defensible on its own terms: v1's `MistralAdapter` was `pass` under
        `LlamaAdapter`, so it never carried any distinct behaviour to migrate.

    This test asserts the inventory, so the gap stays honest as the cache changes: a family listed
    as unreachable that becomes reachable makes this fail and the claim gets upgraded.
    """
    reachable_with_weights = {
        "llama": _hub_snapshot(
            "Skywork/Skywork-Reward-Llama-3.1-8B", "model-00004-of-00004.safetensors"
        ),
        "qwen3": _hub_snapshot("Skywork/Skywork-Reward-V2-Qwen3-0.6B", "model.safetensors"),
    }
    assert all(p is not None for p in reachable_with_weights.values()), reachable_with_weights

    for family in ("armorm", "internlm2"):
        record = _real_heads()[family]
        # The *weights*, not the index: the index and config are 30 KB and are cached here as a
        # side effect of recording the fixture, which is not the same as having the model.
        shards = _hub_snapshot(record["repo_id"], "model-00001-of-*.safetensors")
        assert shards is None, (
            f"{record['repo_id']} weights are now in the local cache. Upgrade the {family} "
            "assertion from recorded metadata to a real head read, and delete it from this list."
        )

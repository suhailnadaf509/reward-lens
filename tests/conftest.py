"""Pytest configuration for reward-lens tests.

Three pieces of global hygiene. Matplotlib runs headless when it is installed at all. The evidence
store and activation cache are redirected to a per-session temporary directory before any
reward_lens module reads the setting, so tests never write to the developer's real
``~/.reward_lens`` and never see each other's state through the default store. And the modules
that need an optional extra are dropped from collection when that extra is absent, so the same
suite runs against a base install and reports what it actually covered there.

That last one is what makes "the torch-free test subset passes" a thing you can run rather than a
thing you assert. Marking sixty files by hand would drift; asking whether the extra's probe module
is importable does not.
"""

import importlib.util
import os
import sys
import tempfile

import pytest

# matplotlib moved to the [viz] extra, so the base install does not have it. Nothing in the
# torch-free half plots, and the modules that do are dropped from collection below.
if importlib.util.find_spec("matplotlib") is not None:
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend for tests

#: Extra name -> the module importable if and only if that extra is installed, and the subsystems
#: `require_extra` guards behind it. A test module that names any of those subsystems needs the
#: extra, directly or through a science that imports one of them, and is dropped from collection
#: when the probe says the extra is absent.
#:
#: The probes repeat `reward_lens.core.extras.EXTRA_PROBE` instead of importing it, because
#: `reward_lens.core.__init__` pulls in the config layer, and collection has to be decided before
#: this file is allowed to touch a setting: the store redirection below has not run yet.
#:
#: The token lists are static on purpose: importing every test module to find out would be precise
#: and would also run their import-time side effects twice. The real check that they are complete
#: is the base-install job in CI, which fails loudly if a module slips through. That is how the
#: `verifier` group and the three white-box entries marked below were found, all of them modules
#: that arrived with 3.0 and had never been collected against a base install.
_GUARDED_SUBSYSTEMS: dict[str, tuple[str, tuple[str, ...]]] = {
    "white-box": (
        "torch",
        (
            "import torch",
            "reward_lens.attribution",
            "reward_lens.concepts",
            "reward_lens.dynamics",
            "reward_lens.geometry",
            "reward_lens.interventions",
            "reward_lens.measure.battery",
            # `indices/__init__` imports `coherence`, which imports `participation_ratio` from
            # `reward_lens.geometry`. The function is pure numpy and the guard is on the geometry
            # package rather than the function, so the whole index catalogue sits behind the extra
            # over one import. See the note in `measure/indices/__init__.py`.
            "reward_lens.measure.indices",
            # ...and `frontier.covector` imports `reward_lens.policy` and `runtime.backend`. Its
            # siblings do not, and `frontier/__init__` does not import it, so naming the module
            # rather than the package keeps the rest of the frontier tests running here.
            "reward_lens.measure.frontier.covector",
            "reward_lens.model_adapters",
            "reward_lens.model",
            "reward_lens.organisms",
            # `policy/__init__` imports `arch`, which imports `runtime.backend`.
            "reward_lens.policy",
            "reward_lens.runtime",
            "reward_lens.sae",
            "reward_lens.signals",
            "studies.",  # every science analysis reaches a guarded subsystem
        ),
    ),
    # Nothing compiled is missing here: the verifier series is pure Python, and `coverage` and the
    # rest are simply their own extra that a base install does not carry. `verifier/__init__`
    # guards the whole package, so the one token covers every module that reaches it.
    "verifier": ("coverage", ("reward_lens.verifier",)),
}


def _extra_is_absent(probe: str) -> bool:
    """Is an extra's probe module missing?

    `find_spec` runs the finders and a finder can raise, which is the hazard `require_extra`
    guards against too: a half-removed distribution, or a meta_path hook standing in for an
    absent dependency. A probe that cannot answer is not a probe that says yes.
    """
    try:
        return importlib.util.find_spec(probe) is None
    except Exception:
        return True


_NEEDS_A_MISSING_EXTRA = tuple(
    token
    for _probe, _tokens in _GUARDED_SUBSYSTEMS.values()
    if _extra_is_absent(_probe)
    for token in _tokens
)

collect_ignore: list[str] = []
if _NEEDS_A_MISSING_EXTRA:
    import pathlib

    _here = pathlib.Path(__file__).parent
    for _p in sorted(_here.rglob("test_*.py")):
        _src = _p.read_text(encoding="utf-8")
        if any(tok in _src for tok in _NEEDS_A_MISSING_EXTRA):
            collect_ignore.append(str(_p.relative_to(_here)))

# Redirect the store/cache root before reward_lens.core.config resolves it. Set at import time so
# the very first `get_settings()` in any test already points at the throwaway home.
_TEST_HOME = tempfile.mkdtemp(prefix="reward_lens_test_home_")
os.environ.setdefault("REWARD_LENS_HOME", _TEST_HOME)


@pytest.fixture(autouse=True)
def _restore_torch_grad_state():
    """Restore global autograd state after every test.

    Some tests deliberately disable gradients for scoring or E-parity work with the global
    ``torch.set_grad_enabled(False)`` rather than a scoped ``with torch.no_grad()``. Under a full
    pytest run that leaks into later tests, and the HVP, Hessian, and organism-training tests then
    fail with "does not require grad". This fixture returns autograd to its default enabled state
    after each test, so test ordering cannot make a grad-requiring test fail. It touches torch only
    if a test already imported it, so pure-numpy tests stay torch-free.
    """
    yield
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.set_grad_enabled(True)

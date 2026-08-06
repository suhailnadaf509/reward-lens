"""Pytest configuration for reward-lens tests.

Three pieces of global hygiene. Matplotlib runs headless when it is installed at all. The evidence
store and activation cache are redirected to a per-session temporary directory before any
reward_lens module reads the setting, so tests never write to the developer's real
``~/.reward_lens`` and never see each other's state through the default store. And the modules
that need the ``[white-box]`` extra are dropped from collection when it is absent, so the same
suite runs against a base install and reports what it actually covered there.

That last one is what makes "the torch-free test subset passes" a thing you can run rather than a
thing you assert. Marking sixty files by hand would drift; asking whether torch is importable does
not.
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

_HAS_TORCH = importlib.util.find_spec("torch") is not None

#: The subsystems guarded by `require_extra("white-box", ...)`, plus the two v1 modules the
#: guarded subsystems still lean on. A test module that names any of these needs torch, directly
#: or through a science that imports one of them, and is dropped from collection on a base
#: install. The list is static on purpose: importing every test module to find out would be
#: precise and would also run their import-time side effects twice. The real check that it is
#: complete is the base-install job in CI, which fails loudly if a module slips through.
_WHITE_BOX_TOKENS = (
    "import torch",
    "reward_lens.attribution",
    "reward_lens.concepts",
    "reward_lens.dynamics",
    "reward_lens.geometry",
    "reward_lens.interventions",
    "reward_lens.measure.battery",
    "reward_lens.model_adapters",
    "reward_lens.model",
    "reward_lens.organisms",
    "reward_lens.runtime",
    "reward_lens.sae",
    "reward_lens.signals",
    "studies.",  # every science analysis reaches a guarded subsystem
)

collect_ignore: list[str] = []
if not _HAS_TORCH:
    import pathlib

    _here = pathlib.Path(__file__).parent
    for _p in sorted(_here.rglob("test_*.py")):
        _src = _p.read_text(encoding="utf-8")
        if any(tok in _src for tok in _WHITE_BOX_TOKENS):
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

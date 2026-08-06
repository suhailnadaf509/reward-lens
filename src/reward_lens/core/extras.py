"""Optional dependency groups, and a guard that names one that actually exists.

The core installs with no compiled dependency, so a subsystem that needs torch has to say so at
the boundary rather than dying inside itself with ``ModuleNotFoundError: No module named 'torch'``.
That error is technically accurate and practically useless: it does not say which extra installs
torch, and it appears identically whether the user is missing an extra or has a broken
environment.

The failure this replaces was worse than an unhelpful message. ``loops/integrations/base.py``
raised ``Install reward-lens[trl]`` for three frameworks, and none of ``trl``, ``verl`` or
``openrlhf`` was declared in ``pyproject.toml``, so the instruction the error gave could not be
followed. An error naming an extra that does not exist is a dead end with a helpful tone.

So the mapping below is the single source of truth for what an extra is called, and
``tests/acceptance/test_w0_3_dependencies.py`` asserts it matches ``pyproject.toml`` exactly. A
typo in an extra name is then a failing test rather than a user stuck in a loop.
"""

from __future__ import annotations

import importlib.util

from reward_lens.core.errors import RewardLensError

#: Extra name -> an importable module that is present if and only if the extra is installed.
#: The probe is one module rather than the whole group because the group is only ever needed as a
#: whole: an environment with torch but not transformers is broken in a way this cannot diagnose
#: and should not try to.
EXTRA_PROBE: dict[str, str] = {
    "white-box": "torch",
    "organisms": "peft",
    "sampling": "vllm",
    "record": "pyarrow",
    "dict": "sae_lens",
    "verifier": "coverage",
    "fuzz": "atheris",
    "trl": "trl",
    "verl": "",  # no runtime dependency; the adapter reads a record
    "viz": "matplotlib",
    "dev": "pytest",
}

#: What each extra is for, in the words a user needs to decide whether they want it.
EXTRA_PURPOSE: dict[str, str] = {
    "white-box": "reading a model's activations and gradients",
    "organisms": "planting and training model organisms",
    "sampling": "generating rollouts through vLLM",
    "record": "Parquet scalar tables and safetensors tensor shards, rather than the JSON Lines and .npy the record writes by default",
    "dict": "sparse dictionary methods, which are candidate generators and never a claim substrate",
    "verifier": "the verifier series: coverage, mutation, metamorphic relations, sensitivity",
    "fuzz": "coverage-guided fuzzing for D5 rung 2, which needs a clang toolchain",
    "trl": "attaching the tap to a live TRL training run",
    "verl": "reading a veRL record",
    "viz": "rendering figures",
    "dev": "running the test suite and the linters",
}


class ExtraRequiredError(RewardLensError, ImportError):
    """A subsystem needs an optional dependency group that is not installed.

    Subclasses ``ImportError`` as well as ``RewardLensError`` so that ``except ImportError``
    around an optional import keeps working, which is the shape most callers already have.
    """


def require_extra(extra: str, *, subsystem: str) -> None:
    """Raise `ExtraRequiredError` if ``extra`` is not installed, naming what to install.

    Called at the top of a subsystem's ``__init__``, so importing any module beneath it fails at
    the boundary with an actionable message instead of somewhere in the middle with a bare
    ``ModuleNotFoundError``.

    ``extra`` must be a key of `EXTRA_PROBE`; an unknown one is a `KeyError` at import time rather
    than a message pointing at an extra nobody can install.
    """
    probe = EXTRA_PROBE[extra]
    if not probe:
        return
    try:
        found = importlib.util.find_spec(probe) is not None
    except Exception:
        # find_spec runs the finders, and a finder can raise: a half-removed distribution, a
        # broken namespace package, or a meta_path hook standing in for an absent dependency.
        # A probe that cannot answer is not a probe that says yes.
        found = False
    if found:
        return
    raise ExtraRequiredError(
        f"{subsystem} needs the optional {extra!r} extra, which is not installed. "
        f"Install it with:  pip install 'reward-lens[{extra}]'  "
        f"({EXTRA_PURPOSE[extra]}). "
        f"The core install is deliberately free of compiled dependencies, so most of the "
        f"instrument catalogue, including the whole grader card, runs without this."
    )


__all__ = ["EXTRA_PROBE", "EXTRA_PURPOSE", "ExtraRequiredError", "require_extra"]

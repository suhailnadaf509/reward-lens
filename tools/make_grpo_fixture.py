"""Regenerate the real GRPO record fixtures under ``tests/fixtures/grpo_run/``.

The in-run instruments need a record produced by a real trainer rather than a hand-built one, and
regenerating it per test costs seven minutes of CPU and a TRL install. So it is written to disk
once and read as a record. This script is what wrote it, kept so the fixture is reproducible
rather than a binary nobody can account for.

The run is deliberately the same one the TRL tap acceptance test builds, through the same
helpers: a real ``GRPOTrainer``, a real model with real weights that really change, real
sampling, real advantages, on CPU. ``trl-internal-testing/tiny-Qwen3ForCausalLM`` is 0.6M
parameters over four layers with a real Qwen3 architecture, and it is TRL's own test model, so
the framework is exercised against it upstream too.

What these runs are, and are not. They are a real optimisation trace of a tiny model against a
length grader. They are **not** a reward-hacking transition, and no instrument should report a
lead time or a transition width off them as though they contained one.

Needs the ``[trl]`` extra:

    python tools/make_grpo_fixture.py short     # 12 steps, about 13 seconds
    python tools/make_grpo_fixture.py long      # 200 steps, about 7 minutes

Both are idempotent: an existing fixture is left alone rather than rewritten, because the run id
is content-derived and rewriting one would orphan every reference to it.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "tests" / "fixtures" / "grpo_run"
SCRATCH = REPO / ".fixture-scratch"

#: (name, steps). Twelve is enough for anything that reads a handful of steps and is fast to
#: iterate against; two hundred is what a per-step regression or a changepoint fit needs.
SIZES = {"short": 12, "long": 200}


def _trl_tap_helpers():
    """Import the TRL tap acceptance test as a module, for its trainer helpers.

    Importing a test module is unusual and it is the right call here: the alternative is a second
    copy of the trainer construction, which would drift from the one the acceptance test asserts
    against, and then the fixture would stop being the run that test measured.
    """
    path = REPO / "tests" / "acceptance" / "test_w4_1_trl_tap.py"
    spec = importlib.util.spec_from_file_location("trl_tap_helpers", path)
    if spec is None or spec.loader is None:  # pragma: no cover - a broken checkout
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["trl_tap_helpers"] = module
    spec.loader.exec_module(module)
    return module


def generate(name: str) -> None:
    steps = SIZES[name]
    root = OUT / name
    if root.exists() and any(root.iterdir()):
        print(f"{name}: already present at {root}, leaving it alone")
        return

    helpers = _trl_tap_helpers()
    from reward_lens.record.writer import RecordWriter
    from reward_lens.tap.adapters.trl import TRLTap

    started = time.time()
    tap = TRLTap(run_id=f"grpo-{name}", budget=helpers.GENEROUS, emit_extra=True)
    trainer = helpers.build_trainer(
        SCRATCH / name, tap.wrap(helpers.length_reward), steps=steps, log_completions=True
    )
    tap.attach(trainer)
    trainer.train()
    run = tap.finish()
    report = RecordWriter(root).write(run)

    print(
        f"{name}: {steps} steps in {time.time() - started:.1f}s -> {report.steps} steps, "
        f"{report.trajectories} trajectories, {report.turns} turns"
    )
    print(f"    run id: {run.id}")
    print(f"    at:     {root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("which", choices=(*SIZES, "all"), nargs="?", default="all")
    args = parser.parse_args()
    for name in SIZES if args.which == "all" else (args.which,):
        generate(name)


if __name__ == "__main__":
    main()

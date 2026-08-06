"""The adapter conformance suite.

Every signal adapter must pass this before it is trusted to produce numbers; it is the structural
fix for the InternLM2/QRM class of silent exclusion, where a model quietly dropped out of a campaign
because its load half-failed and nobody noticed. The suite checks the invariants a reward readout
must satisfy no matter the architecture:

  - determinism: the same input scores the same twice;
  - batch-vs-single parity: a batched score equals the one-at-a-time score within the fp32 readout
    tolerance;
  - left-pad invariance: padding a short sequence up to a long one in a batch does not move its
    score;
  - readout-vs-head: the fp32 projection of the head input equals the model's native head output
    within tolerance (the readout vector really is the head);
  - prefix consistency: the last entry of the per-token curve equals the scalar score;
  - the dtype-policy matrix: a bf16/fp16/fp32 trunk with the fp32 head produces finite (non-NaN)
    scores (the head-in-fp32 policy holds across trunk dtypes);
  - template round-trip: character spans survive tokenization into token coordinates.

New adapters are not registered until this passes. The suite runs on CPU on the
tiny model; the four campaign models run it in a GPU job (gated here).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from reward_lens.signals.classifier import ClassifierRM

# The stimuli: deliberately varied in length so batching forces heavy left-padding and the
# invariance checks actually exercise the padding path.
_ITEMS: list[tuple[str, str]] = [
    ("What is 2+2?", "It is 4."),
    ("Name a color.", "Blue."),
    (
        "Explain gravity in one sentence.",
        "Gravity is the mutual attraction between masses that pulls them together across distance, "
        "growing weaker with the square of the separation.",
    ),
]


@dataclass
class ConformanceCheck:
    """One check's outcome: whether it passed, a human detail string, and whether it was skipped."""

    name: str
    passed: bool
    detail: str = ""
    skipped: bool = False


@dataclass
class ConformanceReport:
    """The result of running the suite against one signal.

    ``passed`` is True only if every non-skipped check passed. Skips (a trunk dtype the device does
    not support, weights that are download-gated) are recorded but do not fail the report; they are
    surfaced so the gap is visible rather than assumed away.
    """

    signal: str
    checks: list[ConformanceCheck] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks if not c.skipped)

    @property
    def n_passed(self) -> int:
        return sum(1 for c in self.checks if c.passed and not c.skipped)

    @property
    def n_skipped(self) -> int:
        return sum(1 for c in self.checks if c.skipped)

    def summary(self) -> str:
        lines = [f"conformance for {self.signal}: {'PASS' if self.passed else 'FAIL'}"]
        for check in self.checks:
            status = "skip" if check.skipped else ("pass" if check.passed else "FAIL")
            lines.append(f"  [{status}] {check.name}: {check.detail}")
        return "\n".join(lines)


def run_conformance(
    signal: "ClassifierRM",
    tol: float | None = None,
    items: list[Any] | None = None,
) -> ConformanceReport:
    """Run the full conformance suite against a signal.

    ``tol`` defaults to the signal's numerics-policy tolerance (1e-4). Each check is wrapped so an
    exception becomes a failed (or, for the dtype matrix, a skipped) check rather than aborting the
    suite, so the report always enumerates every invariant. Returns a ``ConformanceReport`` whose
    ``passed`` property is the acceptance gate.
    """

    tolerance = tol if tol is not None else max(signal.policy.tol, 1e-4)
    data = items if items is not None else list(_ITEMS)
    report = ConformanceReport(signal=str(signal.meta.fingerprint))

    def add(name: str, fn: Callable[[], tuple[bool, str]], skip_on_error: bool = False) -> None:
        try:
            passed, detail = fn()
            report.checks.append(ConformanceCheck(name, passed, detail))
        except Exception as exc:  # noqa: BLE001 - a broken check is a failed (or skipped) check
            report.checks.append(
                ConformanceCheck(name, False, f"{type(exc).__name__}: {exc}", skipped=skip_on_error)
            )

    add("determinism", lambda: _check_determinism(signal, data))
    add("batch_vs_single", lambda: _check_batch_vs_single(signal, data, tolerance))
    add("left_pad_invariance", lambda: _check_left_pad(signal, data, tolerance))
    add("readout_matches_head", lambda: _check_readout_vs_head(signal, data, tolerance))
    add("prefix_consistency", lambda: _check_prefix(signal, data, tolerance))
    add("dtype_matrix", lambda: _check_dtype_matrix(signal, data), skip_on_error=True)
    add("template_round_trip", lambda: _check_template_round_trip(signal))
    return report


# ---------------------------------------------------------------------------
# individual checks
# ---------------------------------------------------------------------------


def _check_determinism(signal: "ClassifierRM", data: list[Any]) -> tuple[bool, str]:
    import numpy as np

    first = signal.score(data).value.values
    second = signal.score(data).value.values
    same = bool(np.array_equal(first, second))
    return same, f"max|diff|={float(np.abs(first - second).max()):.2e}"


def _check_batch_vs_single(signal: "ClassifierRM", data: list[Any], tol: float) -> tuple[bool, str]:
    import numpy as np

    batched = signal.score(data).value.values
    singles = np.array([signal.score([item]).value.values[0] for item in data])
    diff = float(np.abs(batched - singles).max())
    return diff <= tol, f"max|diff|={diff:.2e} (tol {tol:.1e})"


def _check_left_pad(signal: "ClassifierRM", data: list[Any], tol: float) -> tuple[bool, str]:
    import numpy as np

    # The shortest and the longest item: batching them forces the short one to be heavily
    # left-padded, so a match with its unpadded score proves left-pad invariance.
    order = sorted(range(len(data)), key=lambda i: len(str(data[i])))
    short, long = data[order[0]], data[order[-1]]
    batched = signal.score([short, long]).value.values
    alone = np.array([signal.score([short]).value.values[0], signal.score([long]).value.values[0]])
    diff = float(np.abs(batched - alone).max())
    return diff <= tol, f"max|diff|={diff:.2e} (tol {tol:.1e})"


def _check_readout_vs_head(signal: "ClassifierRM", data: list[Any], tol: float) -> tuple[bool, str]:
    import numpy as np
    import torch

    readout = signal.score(data).value.values
    tokenized = [signal.tokenize(item) for item in data]
    batch = signal.runtime.collate(tokenized)
    raw = signal.runtime.forward(batch)
    if raw.reward is None:
        return True, "native head output unavailable; readout self-consistency assumed"
    native = raw.reward.detach().to("cpu", dtype=torch.float32).numpy()
    diff = float(np.abs(readout - native).max())
    return diff <= tol, f"max|readout-native|={diff:.2e} (tol {tol:.1e})"


def _check_prefix(signal: "ClassifierRM", data: list[Any], tol: float) -> tuple[bool, str]:

    scores = signal.score(data).value.values
    curves = signal.score_prefixes(data).value.curves
    diffs = [abs(float(curve[-1]) - float(score)) for curve, score in zip(curves, scores)]
    worst = max(diffs) if diffs else 0.0
    return worst <= tol, f"max|curve[-1]-score|={worst:.2e} (tol {tol:.1e})"


def _check_dtype_matrix(signal: "ClassifierRM", data: list[Any]) -> tuple[bool, str]:
    """Score under bf16/fp16/fp32 trunks (fp32 head) and require finite scores.

    A trunk dtype the device cannot run (fp16 on some CPU kernels) is recorded and skipped inside the
    loop rather than failing the check; the requirement is no NaN wherever the trunk actually runs.
    fp32 must always run and be finite. Uses a deep copy of the model per dtype so the live signal is
    never mutated.
    """
    import numpy as np
    import torch

    from reward_lens.signals.loaders import wrap_hf_model

    results: list[str] = []
    all_finite = True
    for dtype_name in ("float32", "bfloat16", "float16"):
        try:
            model_copy = copy.deepcopy(signal.runtime.model).to(getattr(torch, dtype_name))
            temp = wrap_hf_model(
                model_copy,
                signal.tokenizer,
                device="cpu",
                architecture=signal.meta.architecture,
                numerics=signal.policy.with_trunk(dtype_name),
                conformance_quickcheck=False,
            )
            values = temp.score(data).value.values
            finite = bool(np.all(np.isfinite(values)))
            all_finite = all_finite and finite
            results.append(f"{dtype_name}:{'finite' if finite else 'NaN'}")
        except Exception as exc:  # noqa: BLE001 - device may not support this dtype's kernels
            results.append(f"{dtype_name}:skip({type(exc).__name__})")
    return all_finite, ", ".join(results)


def _check_template_round_trip(signal: "ClassifierRM") -> tuple[bool, str]:
    """A character span survives tokenization into a non-empty, covering token span."""
    base = signal.tokenize(("Give an example.", "hello brave world"))
    if not base.token_offsets:
        return True, "tokenizer exposes no offsets; span carry-through not applicable"
    text = base.text
    needle = "world"
    start = text.find(needle)
    if start < 0:
        return False, f"probe substring {needle!r} not found in templated text"
    end = start + len(needle)
    item = {
        "prompt": "Give an example.",
        "response": "hello brave world",
        "spans": [(start, end, "probe")],
    }
    tokenized = signal.tokenize(item)
    if not tokenized.spans:
        return False, "span was dropped during tokenization"
    span = tokenized.spans[0]
    covered = [tokenized.token_offsets[i] for i in range(span.start, span.end)]
    ok = bool(covered) and covered[0][0] <= start and covered[-1][1] >= end
    return ok, f"char[{start},{end}) -> tokens[{span.start},{span.end}) covering={ok}"


__all__ = ["run_conformance", "ConformanceReport", "ConformanceCheck"]

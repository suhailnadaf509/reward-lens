"""Per-adapter conformance for the v3 signal adapters.

``signals.conformance.run_conformance`` is written against ``ClassifierRM``: its readout-vs-head
check reads the native scalar reward, and its dtype matrix rebuilds the signal through ``wrap_hf_model``,
both of which assume a sequence classifier with a ``score`` head. The seven adapters here include
generative judges (an ``lm_head``, not a ``score`` head), paired-model implicit rewards, and composites,
so this module provides a conformance runner that holds each adapter to the invariants that *apply* to it:

  - determinism, batch-vs-single parity, and left-pad invariance (every scoring adapter);
  - readout-vector extraction matches a direct head/logit computation (every adapter with a linear or
    logit_diff readout: the fp32 projection must equal running the actual head module on the pooled
    hidden state, which is how "the readout vector really is the head" is made executable for the judge
    as well as the classifiers);
  - prefix consistency where ``score_prefixes`` is exposed (``score_prefixes(...)[-1] == score(...)``);
  - template round-trip: a typed span survives tokenization into covering token coordinates.

A composite (ensemble, dense) or a log-ratio signal (implicit) has no single head direction, so the
readout-vs-head check is declared not-applicable (skipped, recorded) rather than forced. New adapters are
not registered until the applicable checks pass.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from reward_lens.signals.conformance import ConformanceCheck, ConformanceReport


def run_adapter_conformance(
    signal: Any,
    *,
    items: Sequence[Any],
    readout: str | None = None,
    tol: float | None = None,
    check_head: bool = True,
    check_prefix: bool = True,
    probe: tuple[Any, str, str] | None = None,
    probe_inject: bool = False,
) -> ConformanceReport:
    """Run the applicable conformance checks against one adapter.

    ``items`` is the stimulus set (varied in length so batching forces left-padding). ``readout`` names
    the readout to score under (the adapter's default when None). ``check_head`` runs the
    readout-vs-head check for adapters with a linear/logit_diff readout; pass False for composites.
    ``probe`` is ``(item, needle, kind)`` for the template round-trip: the ``item`` must tokenize with a
    span of ``kind`` covering ``needle`` in the rendered text. ``probe_inject`` handles adapters that
    carry explicit item spans rather than auto-typing them (a rubric grader): the check renders the
    item, locates ``needle``, injects an explicit char span over the rendered text, and re-tokenizes.
    Each check is wrapped so a raised exception becomes a recorded failure rather than aborting the
    suite.
    """
    policy = getattr(signal, "policy", None)
    tolerance = tol if tol is not None else max(getattr(policy, "tol", 1e-4), 1e-4)
    data = list(items)
    report = ConformanceReport(signal=str(signal.meta.fingerprint))

    def add(name: str, fn: Callable[[], tuple[bool, str]], skip_on_error: bool = False) -> None:
        try:
            passed, detail = fn()
            report.checks.append(ConformanceCheck(name, passed, detail))
        except Exception as exc:  # noqa: BLE001 - a broken check is a failed (or skipped) check
            report.checks.append(
                ConformanceCheck(name, False, f"{type(exc).__name__}: {exc}", skipped=skip_on_error)
            )

    add("determinism", lambda: _determinism(signal, data, readout))
    add("batch_vs_single", lambda: _batch_vs_single(signal, data, readout, tolerance))
    add("left_pad_invariance", lambda: _left_pad(signal, data, readout, tolerance))
    if check_head:
        add("readout_matches_head", lambda: _readout_vs_head(signal, data, readout, tolerance))
    if check_prefix and hasattr(signal, "score_prefixes"):
        add("prefix_consistency", lambda: _prefix(signal, data, readout, tolerance))
    if probe is not None:
        add(
            "template_round_trip", lambda: _template_round_trip(signal, *probe, inject=probe_inject)
        )
    return report


# ---------------------------------------------------------------------------
# individual checks
# ---------------------------------------------------------------------------


def _score_vals(signal: Any, items: Sequence[Any], readout: str | None) -> np.ndarray:
    return np.asarray(signal.score(items, readout).value.values, dtype=np.float64)


def _determinism(signal: Any, data: Sequence[Any], readout: str | None) -> tuple[bool, str]:
    first = _score_vals(signal, data, readout)
    second = _score_vals(signal, data, readout)
    same = bool(np.array_equal(first, second))
    return same, f"max|diff|={float(np.abs(first - second).max()):.2e}"


def _batch_vs_single(
    signal: Any, data: Sequence[Any], readout: str | None, tol: float
) -> tuple[bool, str]:
    batched = _score_vals(signal, data, readout)
    singles = np.array([_score_vals(signal, [item], readout)[0] for item in data])
    diff = float(np.abs(batched - singles).max())
    return diff <= tol, f"max|diff|={diff:.2e} (tol {tol:.1e})"


def _left_pad(
    signal: Any, data: Sequence[Any], readout: str | None, tol: float
) -> tuple[bool, str]:
    order = sorted(range(len(data)), key=lambda i: len(str(data[i])))
    short, long = data[order[0]], data[order[-1]]
    batched = _score_vals(signal, [short, long], readout)
    alone = np.array(
        [_score_vals(signal, [short], readout)[0], _score_vals(signal, [long], readout)[0]]
    )
    diff = float(np.abs(batched - alone).max())
    return diff <= tol, f"max|diff|={diff:.2e} (tol {tol:.1e})"


def _resolve_readout(signal: Any, name: str | None) -> Any:
    """Resolve a readout by name across adapters that expose ``readout()`` or only ``readouts()``."""
    if name is None:
        name = signal.readouts()[0].name
    getter = getattr(signal, "readout", None)
    if callable(getter):
        try:
            return getter(name)
        except KeyError:
            pass
    for read in signal.readouts():
        if read.name == name:
            return read
    raise KeyError(f"no readout {name!r} on {type(signal).__name__}")


def _readout_vs_head(
    signal: Any, data: Sequence[Any], readout: str | None, tol: float
) -> tuple[bool, str]:
    """The fp32 readout must equal running the actual head module on the pooled hidden state.

    Pools the head input at the final token, runs the head (the ``score`` head for a classifier-style
    adapter, the ``lm_head`` for a judge), and extracts the readout's target (a logit difference for a
    logit_diff readout, a head row for a criterion, the weighted row-sum for a rubric aggregate, row 0
    for a single reward). Equality within ``tol`` proves the readout vector is the head.
    """
    import torch

    read = _resolve_readout(signal, readout)
    if read.kind not in ("linear", "logit_diff") or read.vector is None:
        return True, f"readout kind {read.kind!r} has no head direction; check not applicable"
    runtime = signal.runtime
    head = getattr(runtime, "head_module", None)
    if head is None:
        return True, "runtime exposes no head module; readout self-consistency assumed"

    scores = _score_vals(signal, data, readout)
    tokenized = [signal.tokenize(it) for it in data]
    pooled = runtime.final_head_inputs(tokenized)
    out = head(pooled.to(next(head.parameters()).dtype))
    out = out.detach().to(torch.float32)
    if out.ndim == 1:
        out = out.unsqueeze(-1)
    native = _extract_native(out, read).cpu().numpy()
    diff = float(np.abs(scores - native).max())
    return diff <= tol, f"max|readout-head|={diff:.2e} (tol {tol:.1e})"


def _extract_native(out: Any, read: Any) -> Any:
    """Pull the readout's target out of the head output ``out`` (shape ``(n, out_dim)``)."""
    meta = read.meta
    if "a_id" in meta and "b_id" in meta:  # logit_diff verdict
        return out[:, int(meta["a_id"])] - out[:, int(meta["b_id"])]
    if "weights" in meta:  # rubric weighted aggregate
        import torch

        w = torch.tensor(meta["weights"], dtype=out.dtype, device=out.device)
        return (out[:, : w.shape[0]] * w).sum(dim=-1) + float(meta.get("bias", 0.0))
    if "row" in meta:  # a specific head row (criterion / quantile)
        return out[:, int(meta["row"])]
    return out[:, 0]


def _prefix(signal: Any, data: Sequence[Any], readout: str | None, tol: float) -> tuple[bool, str]:
    scores = _score_vals(signal, data, readout)
    curves = signal.score_prefixes(data, readout).value.curves
    diffs = [abs(float(curve[-1]) - float(score)) for curve, score in zip(curves, scores)]
    worst = max(diffs) if diffs else 0.0
    return worst <= tol, f"max|curve[-1]-score|={worst:.2e} (tol {tol:.1e})"


def _template_round_trip(
    signal: Any, item: Any, needle: str, kind: str, inject: bool = False
) -> tuple[bool, str]:
    """A span of ``kind`` survives tokenization into token coordinates covering ``needle``.

    With ``inject`` the item's explicit spans are populated by rendering the item, locating ``needle``
    in the rendered text, and adding a char span there; this exercises the carry-through of an
    explicit item span for adapters that do not auto-type one (a rubric grader).
    """
    if inject and isinstance(item, dict):
        rendered = signal.tokenize(item).text
        pos = rendered.find(needle)
        if pos >= 0:
            item = {**item, "spans": [(pos, pos + len(needle), kind)]}
    tokenized = signal.tokenize(item)
    if not tokenized.token_offsets:
        return True, "tokenizer exposes no offsets; span carry-through not applicable"
    text = tokenized.text
    start = text.find(needle)
    if start < 0:
        return False, f"probe substring {needle!r} not found in rendered text"
    end = start + len(needle)
    matches = [s for s in tokenized.spans if s.kind == kind]
    if not matches:
        return False, f"no span of kind {kind!r} survived tokenization"
    for span in matches:
        covered = [tokenized.token_offsets[i] for i in range(span.start, span.end)]
        if covered and covered[0][0] <= start and covered[-1][1] >= end:
            return True, f"char[{start},{end}) covered by {kind!r} tokens[{span.start},{span.end})"
    return False, f"a {kind!r} span exists but none covers char[{start},{end})"


__all__ = ["run_adapter_conformance"]

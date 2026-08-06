"""Controlled stimulus builders: error planting with generator edit scripts.

This is the one home for controlled stimulus construction. Each builder returns a
lineage-complete item together with the exact character-level edit script it applied, so token
alignment (`data.align`) is exact by construction rather than re-inferred. That coupling is the
whole point: an error-localization or receipt-reliance measurement is only as trustworthy as the
alignment beneath it, and an alignment built from the generator's own edits cannot drift.

What is built fully here is the mechanical family: planting an error at a known step of a math or
code solution by swapping a number or negating an operator, the three receipt-falsification arms on
a trajectory, and the deterministic paraphrase battery. What is deliberately stubbed is the
family that needs an oracle to preserve a confound (style-matched or confidence-matched rewrites,
oracle-authored quadruples, judge tournaments): those need oracle provenance stamping, and their
stubs raise `NotImplementedError` naming exactly what they need. A stub here never masquerades as a
finished builder.
"""

from __future__ import annotations

import re
from typing import Any

from reward_lens.core import Span
from reward_lens.core.errors import DataError
from reward_lens.data.align import CharEdit, align, apply_edits
from reward_lens.data.lineage import make_lineage
from reward_lens.data.schema import (
    Pair,
    Trajectory,
    TrajStep,
    trajectory_content,
)
from reward_lens.data.spans import (
    DEFAULT_TOKENIZER,
    ERROR,
    Tokenizer,
    make_span,
)

# ---------------------------------------------------------------------------
# Step splitting
# ---------------------------------------------------------------------------

_INLINE_STEP_DELIMS = re.compile(r"→|->|;")


def _segment_inline(text: str, base: int) -> list[tuple[str, int, int]]:
    """Split a single line into steps on arrow/semicolon delimiters, preserving char offsets.

    Returns ``(stripped_text, abs_start, abs_end)`` for each non-empty segment, where the offsets are
    into the original solution (``base`` is the line's offset). Math solutions are commonly written
    as ``a → b → c`` on one line, so this is what makes "step k" addressable there.
    """
    out: list[tuple[str, int, int]] = []
    last = 0
    bounds: list[tuple[int, int]] = []
    for m in _INLINE_STEP_DELIMS.finditer(text):
        bounds.append((last, m.start()))
        last = m.end()
    bounds.append((last, len(text)))
    for lo, hi in bounds:
        seg = text[lo:hi]
        lead = len(seg) - len(seg.lstrip())
        stripped = seg.strip()
        if stripped:
            start = base + lo + lead
            out.append((stripped, start, start + len(stripped)))
    return out


def _split_steps(solution: str) -> list[tuple[str, int, int]]:
    """Split a solution into addressable steps with character offsets.

    Newlines delimit steps first (the code case); a single-line solution is then split on arrow and
    semicolon delimiters (the inline-math case). Returns ``(stripped_text, abs_start, abs_end)`` per
    step. Offsets are into ``solution`` so an edit can be placed exactly.
    """
    lines_steps: list[tuple[str, int, int]] = []
    off = 0
    for line in solution.split("\n"):
        stripped = line.strip()
        if stripped:
            lead = len(line) - len(line.lstrip())
            start = off + lead
            lines_steps.append((stripped, start, start + len(stripped)))
        off += len(line) + 1  # +1 for the newline that split consumed
    if len(lines_steps) > 1:
        return lines_steps
    if len(lines_steps) == 1:
        text, start, _ = lines_steps[0]
        inline = _segment_inline(text, start)
        if len(inline) > 1:
            return inline
        return lines_steps
    return []


# ---------------------------------------------------------------------------
# Mechanical corruptions
# ---------------------------------------------------------------------------

_DIGIT_RUN = re.compile(r"\d+")
_OPERATOR_FLIP = {"+": "-", "-": "+", "*": "/", "/": "*", "=": "≠"}
_OPERATOR_ORDER = ("+", "-", "*", "/", "=")


def _swap_number_edit(solution: str, step_start: int, step_end: int) -> CharEdit:
    """Plant a wrong number: change the last digit of the last number in the step.

    Changing a single digit (by +5 mod 10, which differs from the original for every digit) keeps the
    length and the token count, so it is a clean length-preserving and token-count-preserving control.
    """
    step_text = solution[step_start:step_end]
    runs = list(_DIGIT_RUN.finditer(step_text))
    if not runs:
        raise DataError(
            f"corrupt_step(mode='swap_number'): step [{step_start}:{step_end}] contains no digits "
            "to swap; pick a numeric step or a different mode"
        )
    run = runs[-1]
    last_digit_pos = step_start + run.end() - 1
    old = solution[last_digit_pos]
    new = str((int(old) + 5) % 10)
    return CharEdit(start=last_digit_pos, end=last_digit_pos + 1, replacement=new)


def _negate_edit(solution: str, step_start: int, step_end: int) -> CharEdit:
    """Negate a step: flip its first arithmetic operator (``+`` <-> ``-``, ``=`` -> not-equal).

    Length-preserving and token-count-preserving (one single-character operator token for another).
    """
    step_text = solution[step_start:step_end]
    for op in _OPERATOR_ORDER:
        pos = step_text.find(op)
        if pos != -1:
            abs_pos = step_start + pos
            return CharEdit(start=abs_pos, end=abs_pos + 1, replacement=_OPERATOR_FLIP[op])
    raise DataError(
        f"corrupt_step(mode='negate'): step [{step_start}:{step_end}] contains no operator "
        f"({', '.join(_OPERATOR_ORDER)}) to negate; pick a different step or mode"
    )


_MECHANICAL_MODES = {
    "swap_number": _swap_number_edit,
    "negate": _negate_edit,
}


def corrupt_step(
    solution: str,
    k: int,
    mode: str = "swap_number",
    *,
    tokenizer: Tokenizer | None = None,
) -> tuple[str, list[CharEdit], Span]:
    """Plant a mechanical error at step ``k`` of a math or code solution.

    Returns ``(corrupted_text, edit_script, error_span)`` where ``error_span`` is a token span (kind
    ``error``) over the *corrupted* text marking the planted mistake, and ``edit_script`` is the exact
    character-level edits (over the *clean* text) that produced it. Feed that script to
    `data.align.align` to get an exact clean/corrupted alignment, which is what makes error
    localization exact.

    ``k`` is a zero-based step index; it raises `DataError` if out of range. Fully implemented modes:

    - ``"swap_number"`` changes the last digit of the last number in the step (length- and
      token-count-preserving).
    - ``"negate"`` flips the step's first arithmetic operator.

    TODO: length/style/confidence-*matched* control variants that keep everything but
    correctness fixed generally need an oracle rewrite (they cannot be produced by a purely
    mechanical edit) and are not built here; they need oracle provenance stamping.
    Requesting an unimplemented mode raises with that note rather than silently degrading.
    """
    steps = _split_steps(solution)
    if not steps:
        raise DataError("corrupt_step: solution has no addressable steps")
    if k < 0 or k >= len(steps):
        raise DataError(
            f"corrupt_step: step index {k} out of range [0, {len(steps)}) for this solution"
        )
    _, step_start, step_end = steps[k]
    edit_fn = _MECHANICAL_MODES.get(mode)
    if edit_fn is None:
        raise DataError(
            f"corrupt_step: mode {mode!r} is not a mechanical mode. "
            f"Built modes: {sorted(_MECHANICAL_MODES)}. Style/confidence-matched and other "
            "oracle-assisted control variants are not implemented."
        )
    edit = edit_fn(solution, step_start, step_end)
    corrupted = apply_edits(solution, [edit])
    tok = tokenizer or DEFAULT_TOKENIZER
    corrupted_tok = tok.tokenize(corrupted)
    c_start = edit.start
    c_end = edit.start + len(edit.replacement)
    error_span = make_span(corrupted_tok, c_start, c_end, ERROR, mode=mode, step_index=k)
    return corrupted, [edit], error_span


# ---------------------------------------------------------------------------
# Receipt / narrative falsification arms
# ---------------------------------------------------------------------------

_TRUTH_FLIPS = {
    "pass": "fail",
    "fail": "pass",
    "true": "false",
    "false": "true",
    "yes": "no",
    "no": "yes",
    "success": "failure",
    "failure": "success",
    "correct": "incorrect",
    "valid": "invalid",
}


def _falsify_text(text: str) -> str:
    """Produce a truth-inverted version of a receipt or narrative fragment.

    Deterministic and mechanical: swap the last digit of the last number if there is one; else flip a
    truth-bearing keyword (pass/fail, true/false, yes/no); else prefix ``not `` so the claim is
    negated. This is enough to make a receipt stop supporting (or a narrative stop matching) the
    evidence, which is all the receipt sciences need to plant the manipulation.
    """
    runs = list(_DIGIT_RUN.finditer(text))
    if runs:
        run = runs[-1]
        pos = run.end() - 1
        new_digit = str((int(text[pos]) + 5) % 10)
        return text[:pos] + new_digit + text[pos + 1 :]
    for m in re.finditer(r"[A-Za-z]+", text):
        word = m.group(0)
        flip = _TRUTH_FLIPS.get(word.lower())
        if flip is not None:
            repl = flip.capitalize() if word[0].isupper() else flip
            return text[: m.start()] + repl + text[m.end() :]
    return "not " + text


def _span_char_range(step: TrajStep, span: Span, tokenizer: Tokenizer) -> tuple[int, int]:
    """The character range in a step's text that a token span covers."""
    tok = tokenizer.tokenize(step.text)
    if span.end <= span.start or span.end > len(tok.tokens):
        raise DataError("receipt_edits: span does not resolve within the step text")
    return tok.tokens[span.start].start, tok.tokens[span.end - 1].end


def _edit_step(
    step: TrajStep, char_start: int, char_end: int, replacement: str, tokenizer: Tokenizer
) -> tuple[TrajStep, list[CharEdit]]:
    """Apply a single character edit to a step's text and re-map its spans through the alignment.

    Returns the rewritten step and the edit script. Spans that survive are carried through exactly
    with a `SpanMap` built from that very script; a span whose content was deleted maps to None and is
    dropped, which is the correct behaviour for the delete-receipt arm.
    """
    edit = CharEdit(start=char_start, end=char_end, replacement=replacement)
    new_text = apply_edits(step.text, [edit])
    spanmap = align(step.text, new_text, edit_script=[edit], tokenizer=tokenizer)
    new_receipts = tuple(s for s in (spanmap.map_span(sp) for sp in step.receipts) if s is not None)
    new_narrative = tuple(
        s for s in (spanmap.map_span(sp) for sp in step.narrative) if s is not None
    )
    new_step = TrajStep(
        action=step.action,
        tool_call=step.tool_call,
        text=new_text,
        receipts=new_receipts,
        narrative=new_narrative,
        meta=dict(step.meta),
    )
    return new_step, [edit]


def _find_step_with(steps: tuple[TrajStep, ...], attr: str) -> int:
    for idx, step in enumerate(steps):
        if getattr(step, attr):
            return idx
    return -1


def receipt_edits(
    traj: Trajectory, *, tokenizer: Tokenizer | None = None
) -> dict[str, dict[str, Any]]:
    """Build the three receipt-manipulation arms of a trajectory (the N1 stimulus).

    Returns a dict keyed by arm name, each value ``{"trajectory": Trajectory, "edit_script":
    [CharEdit], "step_index": int}``:

    - ``"falsify_receipt"`` inverts the evidence in the first step that has a receipt span, so the
      receipt no longer supports the narrative.
    - ``"falsify_narrative"`` inverts the model's account in the first step that has a narrative span,
      so the narrative no longer matches the receipt.
    - ``"delete_receipt"`` removes the receipt span entirely, so there is no evidence at all.

    Each arm carries an updated lineage (the seed id is preserved; the op records the arm) and an
    exact edit script. The receipt reliance score reads the reward difference across these arms and
    attributes it to the receipt span via span patching; that attribution is only sound because the
    alignment here is exact.
    """
    tok = tokenizer or DEFAULT_TOKENIZER
    arms: dict[str, dict[str, Any]] = {}

    def build_arm(name: str, step_idx: int, new_step: TrajStep, script: list[CharEdit]) -> None:
        new_steps = tuple(new_step if i == step_idx else s for i, s in enumerate(traj.steps))
        op = f"receipt_edit:{name}"
        lineage = make_lineage(
            traj.lineage.seed_id,
            traj.lineage.builder_id,
            (*traj.lineage.ops, op),
            trajectory_content(traj.prompt_text, new_steps, traj.outcome),
        )
        new_traj = Trajectory(
            steps=new_steps,
            outcome=dict(traj.outcome),
            lineage=lineage,
            prompt=traj.prompt,
            meta=dict(traj.meta),
        )
        arms[name] = {"trajectory": new_traj, "edit_script": script, "step_index": step_idx}

    receipt_idx = _find_step_with(traj.steps, "receipts")
    if receipt_idx >= 0:
        step = traj.steps[receipt_idx]
        span = step.receipts[0]
        cs, ce = _span_char_range(step, span, tok)
        falsified = _falsify_text(step.text[cs:ce])
        new_step, script = _edit_step(step, cs, ce, falsified, tok)
        build_arm("falsify_receipt", receipt_idx, new_step, script)

        # Delete arm: also strip the space before the receipt where present, to avoid a dangling gap.
        del_start = cs - 1 if cs > 0 and step.text[cs - 1] == " " else cs
        new_step_del, script_del = _edit_step(step, del_start, ce, "", tok)
        build_arm("delete_receipt", receipt_idx, new_step_del, script_del)

    narrative_idx = _find_step_with(traj.steps, "narrative")
    if narrative_idx >= 0:
        step = traj.steps[narrative_idx]
        span = step.narrative[0]
        cs, ce = _span_char_range(step, span, tok)
        falsified = _falsify_text(step.text[cs:ce])
        new_step, script = _edit_step(step, cs, ce, falsified, tok)
        build_arm("falsify_narrative", narrative_idx, new_step, script)

    if not arms:
        raise DataError(
            "receipt_edits: trajectory has no step with a receipt or narrative span to edit; "
            "type the spans (data.spans) before requesting receipt arms"
        )
    return arms


# ---------------------------------------------------------------------------
# Paraphrase battery
# ---------------------------------------------------------------------------

# Extends v1's E16 prompt-robustness templates. Each is a (prefix, suffix) pair; the identity is
# first so element 0 is always the untouched prompt. Combining prefixes and suffixes gives a large
# deterministic space of surface variants that hold the request fixed, which is exactly the
# robustness-SNR stimulus the E16 study consumes.
_PARAPHRASE_PREFIXES = (
    "",
    "Could you tell me: ",
    "Question: ",
    "Quick question, ",
    "I'm curious, ",
    "Please help with this: ",
)
_PARAPHRASE_SUFFIXES = (
    "",
    " Thanks!",
    " Please answer briefly.",
    " I'd appreciate your input.",
    " Can you explain?",
    " (asking for a project)",
)


def paraphrase_battery(prompt: str, k: int) -> list[str]:
    """Return up to ``k`` deterministic surface paraphrases of ``prompt``.

    Element 0 is always the untouched prompt. The rest are distinct prefix/suffix rewrites drawn from
    the E16 template space in a fixed order, so the battery is reproducible across runs. If ``k``
    exceeds the number of distinct variants the templates can produce, the full distinct set is
    returned (fewer than ``k``) rather than fabricating near-duplicates; the caller sees the honest
    count. Style- or register-shifting paraphrases that need an oracle are out of scope here.
    """
    if k <= 0:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for suffix in _PARAPHRASE_SUFFIXES:
        for prefix in _PARAPHRASE_PREFIXES:
            variant = f"{prefix}{prompt}{suffix}"
            if variant not in seen:
                seen.add(variant)
                out.append(variant)
            if len(out) >= k:
                return out
    return out


# ---------------------------------------------------------------------------
# Oracle-assisted builders (stubbed with explicit provenance requirements)
# ---------------------------------------------------------------------------


def style_controls(pair: Pair) -> Any:
    """STUB: verbosity/format/confidence-matched rewrites of a pair.

    Matching everything but the labelled axis (so a verbosity-controlled correctness pair reads the
    same length and register on both sides) requires an oracle rewrite that a mechanical edit cannot
    produce, with oracle provenance stamping. It is stubbed rather than approximated so no study
    mistakes an uncontrolled pair for a controlled one.
    """
    raise NotImplementedError(
        "style_controls needs an oracle rewrite to hold style/verbosity/confidence fixed while "
        "varying only the labelled axis, with oracle provenance. "
        "Use the mechanical corrupt_step / receipt_edits builders instead."
    )


def quadruples(topic_bank: Any) -> Any:
    """STUB: L2 agree/disagree x correct/incorrect controlled quadruples.

    The 2x2 sycophancy design needs oracle-authored responses that vary agreement and correctness
    independently while holding topic and register fixed. Stubbed until the oracle layer exists.
    """
    raise NotImplementedError(
        "quadruples needs oracle-authored cells (agree/disagree x correct/incorrect) with matched "
        "register and oracle provenance."
    )


def tournament_from_judges(prompts: Any, pool: Any, k: int) -> Any:
    """STUB: k-wise judge tournaments with position-debiasing rotation.

    Building a tournament from a judge pool requires calling the judges (oracles) with position
    rotation and stamping each edge with its judge id and prompt hash. Stubbed until the oracle
    layer exists; the `Tournament` schema and `EdgeObs.judge_id` are already in place for it.
    """
    raise NotImplementedError(
        "tournament_from_judges needs a judge oracle pool with position-debiasing rotation and "
        "per-edge provenance stamping. The Tournament/EdgeObs schema is "
        "ready to receive its output."
    )


__all__ = [
    "corrupt_step",
    "receipt_edits",
    "paraphrase_battery",
    "style_controls",
    "quadruples",
    "tournament_from_judges",
]

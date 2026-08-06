"""SpanMap fuzz and correctness tests (the mandatory alignment property test).

Misalignment is the quiet killer of every pairwise causal method, so the alignment carries a
property test rather than a couple of examples. The fuzz generates random token-level edits to a
random token sequence, renders the edited text, and feeds the generator's own character-level edit
script to `align`. Because the edits are known, we have a ground-truth correspondence to check
against: every surviving token must align one-to-one to its image, every equal-aligned pair must have
identical surface, and `map_span` must carry a replaced region onto exactly its replacement.

The second half checks `map_span` on the real mechanical corruptions from `data.corruptions`, which is
where an alignment error would actually bite a verification measurement.
"""

from __future__ import annotations

import random

from reward_lens.core import Span
from reward_lens.data import CharEdit, align, apply_edits, corrupt_step
from reward_lens.data.spans import DEFAULT_TOKENIZER

_VOCAB = [
    "the",
    "quick",
    "brown",
    "fox",
    "jumps",
    "over",
    "lazy",
    "dog",
    "alpha",
    "beta",
    "gamma",
    "delta",
    "one",
    "two",
    "three",
    "42",
    "pi",
    "sum",
    "foo",
    "bar",
    "baz",
    "node",
    "edge",
    "graph",
    "red",
    "blue",
    "green",
    "step",
    "value",
    "token",
]


def _gen_edits(n: int, rng: random.Random) -> list[tuple[int, int, list[str]]]:
    """Random non-overlapping token-level edits over interior indices, separated by gaps.

    Index 0 and n-1 are never edited, so a surviving token always brackets every edit and the edited
    text never collapses to empty; that keeps single-space boundaries clean so no token merges and the
    ground truth stays exact.
    """
    edits: list[tuple[int, int, list[str]]] = []
    i = 1
    while i < n - 1:
        if rng.random() < 0.4:
            te = min(n - 1, i + rng.randint(1, 2))
            kind = rng.choice(["del", "rep", "exp"])
            if kind == "del":
                repl: list[str] = []
            elif kind == "rep":
                repl = [rng.choice(_VOCAB) for _ in range(te - i)]
            else:
                repl = [rng.choice(_VOCAB) for _ in range(te - i + rng.randint(1, 2))]
            edits.append((i, te, repl))
            i = te + 1  # leave a gap so edits never abut
        else:
            i += 1
    return edits


def _splice(a_words: list[str], edits: list[tuple[int, int, list[str]]]):
    """Apply token-level edits, returning ``(b_words, survivor_map)``.

    ``survivor_map`` maps each surviving ``a`` token index to its index in ``b``. This is the
    independent ground-truth alignment the test checks the SpanMap against.
    """
    by_start = {ts: (ts, te, r) for (ts, te, r) in edits}
    b_words: list[str] = []
    survivor_map: dict[int, int] = {}
    i = 0
    n = len(a_words)
    while i < n:
        if i in by_start:
            _, te, repl = by_start[i]
            b_words.extend(repl)
            i = te
        else:
            survivor_map[i] = len(b_words)
            b_words.append(a_words[i])
            i += 1
    return b_words, survivor_map


def _char_script(a_text: str, edits: list[tuple[int, int, list[str]]]) -> list[CharEdit]:
    """The character-level edit script a generator would emit for these token-level edits."""
    toks = DEFAULT_TOKENIZER.tokenize(a_text).tokens
    starts = [t.start for t in toks]
    ends = [t.end for t in toks]
    script: list[CharEdit] = []
    for ts, te, repl in edits:
        if repl:
            script.append(CharEdit(starts[ts], ends[te - 1], " ".join(repl)))
        else:
            # Deletion: also remove the single preceding space (ts >= 1 is guaranteed) so the edited
            # text is single-space clean and apply_edits reproduces it exactly.
            script.append(CharEdit(ends[ts - 1], ends[te - 1], ""))
    return script


def test_spanmap_fuzz_exact_from_edit_script() -> None:
    rng = random.Random(20260709)
    cases = 0
    for _ in range(400):
        n = rng.randint(4, 14)
        a_words = [rng.choice(_VOCAB) for _ in range(n)]
        a_text = " ".join(a_words)
        edits = _gen_edits(n, rng)
        if not edits:
            continue
        b_words, survivor_map = _splice(a_words, edits)
        if not b_words:
            continue
        b_text = " ".join(b_words)
        script = _char_script(a_text, edits)

        # The generator's script must reproduce b exactly (the contract align relies on).
        assert apply_edits(a_text, script) == b_text

        sm = align(a_text, b_text, edit_script=script)
        assert sm.exact is True

        # Every surviving token aligns one-to-one to its known image, with identical surface.
        aligned = set(sm.aligned_pairs)
        for a_idx, b_idx in survivor_map.items():
            assert (a_idx, b_idx) in aligned, (a_text, b_text, a_idx, b_idx)
            assert sm.a.tokens[a_idx].norm == sm.b.tokens[b_idx].norm

        # The alignment is order-preserving on both sides.
        ai = [p[0] for p in sm.aligned_pairs]
        bi = [p[1] for p in sm.aligned_pairs]
        assert ai == sorted(ai)
        assert bi == sorted(bi)

        # map_span carries each replaced (non-empty) region onto exactly its replacement.
        for ts, te, repl in edits:
            if repl:
                mapped = sm.map_span(Span(ts, te, "error"))
                assert mapped is not None
                assert sm.b.text_for_span(mapped) == " ".join(repl)
            else:
                # A fully deleted region has no image.
                assert sm.map_span(Span(ts, te, "error")) is None
        cases += 1
    assert cases > 50  # the fuzz actually exercised many non-trivial cases


def test_spanmap_map_span_on_corrupt_step_number_swap() -> None:
    """map_span carries the clean digit span onto the planted wrong digit, exactly."""
    solution = "3x + 7 = 22 → 3x = 15 → x = 5"
    corrupted, script, error_span = corrupt_step(solution, 2, "swap_number")

    sm = align(solution, corrupted, edit_script=script)
    assert sm.exact is True

    # The clean token containing the edited character.
    clean_tok = sm.a
    edited_char = script[0].start
    tok_idx = clean_tok.char_to_token_map()[edited_char]
    assert tok_idx >= 0
    clean_span = Span(tok_idx, tok_idx + 1, "error")

    mapped = sm.map_span(clean_span)
    assert mapped is not None
    # The mapped span is the corruption's own error span, covering the wrong value.
    assert (mapped.start, mapped.end) == (error_span.start, error_span.end)
    assert (
        sm.b.text_for_span(mapped)
        == corrupted[script[0].start : script[0].start + len(script[0].replacement)]
    )


def test_spanmap_map_span_on_receipt_falsification() -> None:
    """map_span carries a receipt span from the clean step onto its falsified image."""
    from reward_lens.data import Trajectory, TrajStep, make_lineage, receipt_edits
    from reward_lens.data.spans import RECEIPT, make_span

    text = "the report says PASS for all cases"
    tok = DEFAULT_TOKENIZER.tokenize(text)
    r0 = text.index("PASS")
    receipt = make_span(tok, r0, r0 + len("PASS"), RECEIPT)
    step = TrajStep(action="run", text=text, receipts=(receipt,))
    traj = Trajectory(
        steps=(step,),
        outcome={"ok": True},
        lineage=make_lineage("t:0", "test", (), ["t"]),
    )

    arm = receipt_edits(traj)["falsify_receipt"]
    sm = align(text, arm["trajectory"].steps[0].text, edit_script=arm["edit_script"])
    mapped = sm.map_span(receipt)
    assert mapped is not None
    # The receipt content changed from PASS to a falsified token, mapped exactly onto it.
    assert sm.b.text_for_span(mapped) != "PASS"
    assert sm.b.text_for_span(mapped).lower() in {"fail", "false", "no"}


def test_align_inferred_path_surfaces_match() -> None:
    """The difflib-inferred path (no edit script) still produces surface-consistent equal pairs."""
    a = "the quick brown fox jumps over the lazy dog"
    b = "the quick red fox leaps over the lazy dog"
    sm = align(a, b)
    assert sm.exact is False
    for ai, bi in sm.aligned_pairs:
        assert sm.a.tokens[ai].norm == sm.b.tokens[bi].norm
    # "quick" survives and aligns; its map is a single matched token.
    q_idx = sm.a.normalized().index("quick")
    mapped = sm.map_span(Span(q_idx, q_idx + 1))
    assert mapped is not None
    assert sm.b.text_for_span(mapped) == "quick"

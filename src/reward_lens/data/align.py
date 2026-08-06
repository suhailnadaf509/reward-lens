"""Token alignment across paired or edited sequences: SpanMap.

Span-level patching, differential attribution, receipt-reliance, and error localization all rest on
knowing, exactly, which token of the rejected response corresponds to which token of the chosen one,
or which token of the corrupted solution corresponds to which token of the clean one. Get that wrong
and every pairwise causal method quietly measures the wrong thing. Misalignment is the quiet
killer of every pairwise causal method, so this module is small, exact where it can be, and
fuzz-tested.

There are two paths to an alignment:

- The **inferred** path uses ``difflib.SequenceMatcher`` on the normalized token strings. This is
  the honest best effort for two responses that were written independently (a real chosen/rejected
  pair from a dataset), where no generator recorded what changed.
- The **exact** path consumes a generator's character-level edit script. Every corruption builder in
  `data.corruptions` emits the exact edits it made, and when that script is available the alignment
  is computed from it directly rather than re-inferred, which makes it exact by construction. That
  is the guarantee: exactness wherever the corruption generator recorded its edit.

Both paths produce the same object, a `SpanMap`, over a uniform list of token-level opcodes, so
`map_span` and `aligned_pairs` behave identically regardless of how the alignment was obtained.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Sequence

from reward_lens.core import Span
from reward_lens.data.spans import DEFAULT_TOKENIZER, TokenizedInput, Tokenizer

# A token-level opcode: (tag, i1, i2, j1, j2) with tag in {equal, replace, delete, insert},
# a-indices [i1, i2) mapping to b-indices [j1, j2). This is difflib's opcode shape, reused for the
# edit-script path so both alignments share all downstream logic.
Opcode = tuple[str, int, int, int, int]


# ---------------------------------------------------------------------------
# Character-level edit scripts (the exactness contract with the generators)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CharEdit:
    """One character-level edit: replace ``text[start:end]`` with ``replacement``.

    A pure insertion has ``start == end``; a pure deletion has ``replacement == ""``. Edits in a
    script are over the *original* (``a``) text's coordinates and must be non-overlapping. A
    generator emits these as it builds a corrupted item so the alignment consuming them is exact.
    """

    start: int
    end: int
    replacement: str

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"CharEdit end {self.end} precedes start {self.start}")


def apply_edits(text: str, edits: Sequence[CharEdit]) -> str:
    """Apply a non-overlapping edit script to ``text`` and return the edited string.

    Edits are applied left to right; overlapping edits are a caller error and raise. This is the
    inverse of what a generator records: a builder produces ``(edited_text, edits)`` such that
    ``apply_edits(original, edits) == edited_text``, and the tests assert exactly that round-trip.
    """
    ordered = sorted(edits, key=lambda e: (e.start, e.end))
    out: list[str] = []
    cursor = 0
    for e in ordered:
        if e.start < cursor:
            raise ValueError(f"overlapping edit at {e.start} (cursor {cursor})")
        out.append(text[cursor : e.start])
        out.append(e.replacement)
        cursor = e.end
    out.append(text[cursor:])
    return "".join(out)


def _b_ranges(edits: Sequence[CharEdit]) -> list[tuple[int, int]]:
    """The character ranges in the edited (``b``) text that each edit's replacement occupies."""
    ordered = sorted(edits, key=lambda e: (e.start, e.end))
    ranges: list[tuple[int, int]] = []
    delta = 0
    for e in ordered:
        b_start = e.start + delta
        b_end = b_start + len(e.replacement)
        ranges.append((b_start, b_end))
        delta += len(e.replacement) - (e.end - e.start)
    return ranges


def _touches(tok_start: int, tok_end: int, ranges: Sequence[tuple[int, int]]) -> bool:
    """Whether a token's character range overlaps any edited range.

    Half-open overlap for non-empty edited ranges; a zero-width edited range (a pure insertion
    point) touches the token whose interior it falls strictly inside, so an insertion splits a token
    rather than silently attaching to a neighbour.
    """
    for cs, ce in ranges:
        if ce > cs:
            if tok_start < ce and tok_end > cs:
                return True
        else:  # zero-width range: an insertion point
            if tok_start < cs < tok_end:
                return True
    return False


def _opcodes_from_edit_script(
    a: TokenizedInput, b: TokenizedInput, edits: Sequence[CharEdit]
) -> list[Opcode]:
    """Exact token-level opcodes derived from a character edit script.

    Tokens untouched by any edit are character-identical between ``a`` and ``b`` and appear in the
    same order, so they align one-to-one; touched tokens on each side form the replace/insert/delete
    blocks between them. The one-to-one runs are additionally guarded on normalized surface equality,
    so if an edit lands on a token boundary in a way that merges or splits a token (changing counts on
    the untouched side), the run stops and the affected tokens fall into a replace block rather than a
    wrong one-to-one mapping. Correctness is never traded for coverage: an uncertain region becomes a
    replace block, which maps its whole ``a`` extent to its whole ``b`` extent.
    """
    a_ranges = [(e.start, e.end) for e in edits]
    b_ranges = _b_ranges(edits)
    a_touched = [_touches(t.start, t.end, a_ranges) for t in a.tokens]
    b_touched = [_touches(t.start, t.end, b_ranges) for t in b.tokens]
    a_norm = a.normalized()
    b_norm = b.normalized()

    opcodes: list[Opcode] = []
    ia = ib = 0
    na, nb = len(a.tokens), len(b.tokens)
    while ia < na or ib < nb:
        if (
            ia < na
            and ib < nb
            and not a_touched[ia]
            and not b_touched[ib]
            and a_norm[ia] == b_norm[ib]
        ):
            i1, j1 = ia, ib
            while (
                ia < na
                and ib < nb
                and not a_touched[ia]
                and not b_touched[ib]
                and a_norm[ia] == b_norm[ib]
            ):
                ia += 1
                ib += 1
            opcodes.append(("equal", i1, ia, j1, ib))
        else:
            i1, j1 = ia, ib
            # Consume the touched run on each side independently: touched a-tokens were edited away,
            # touched b-tokens are replacement content.
            while ia < na and a_touched[ia]:
                ia += 1
            while ib < nb and b_touched[ib]:
                ib += 1
            # If neither side advanced (a surface desync between two untouched tokens, e.g. an edit
            # that merged a token boundary), pair them off one-for-one as a replace so progress is
            # guaranteed and the mapping stays a correct whole-to-whole block.
            if ia == i1 and ib == j1:
                if ia < na:
                    ia += 1
                if ib < nb:
                    ib += 1
            if ia > i1 and ib > j1:
                opcodes.append(("replace", i1, ia, j1, ib))
            elif ia > i1:
                opcodes.append(("delete", i1, ia, j1, ib))
            elif ib > j1:
                opcodes.append(("insert", i1, ia, j1, ib))
            else:  # pragma: no cover - the progress guard above prevents this
                break
    return opcodes


# ---------------------------------------------------------------------------
# SpanMap
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpanMap:
    """A token-level alignment from sequence ``a`` to sequence ``b``.

    Built by `align`. Carries the opcodes and both tokenized inputs. The two things instruments read:
    `map_span`, which carries a token span from ``a`` into ``b``, and `aligned_pairs`, the list of
    exactly-corresponding token indices. `inserts` and `deletes` report the tokens that exist on only
    one side, which the receipt sciences need (a deleted receipt has no image in ``b``).
    """

    a: TokenizedInput
    b: TokenizedInput
    opcodes: tuple[Opcode, ...]
    exact: bool  # True when built from a generator edit script, False when inferred by difflib

    # -- correspondence -----------------------------------------------------

    @property
    def aligned_pairs(self) -> tuple[tuple[int, int], ...]:
        """Every ``(a_index, b_index)`` token pair that corresponds exactly (from equal blocks)."""
        pairs: list[tuple[int, int]] = []
        for tag, i1, i2, j1, _ in self.opcodes:
            if tag == "equal":
                for off in range(i2 - i1):
                    pairs.append((i1 + off, j1 + off))
        return tuple(pairs)

    @property
    def inserts(self) -> tuple[int, ...]:
        """Token indices in ``b`` with no counterpart in ``a`` (insert blocks)."""
        out: list[int] = []
        for tag, _, _, j1, j2 in self.opcodes:
            if tag == "insert":
                out.extend(range(j1, j2))
        return tuple(out)

    @property
    def deletes(self) -> tuple[int, ...]:
        """Token indices in ``a`` with no counterpart in ``b`` (delete blocks)."""
        out: list[int] = []
        for tag, i1, i2, _, _ in self.opcodes:
            if tag == "delete":
                out.extend(range(i1, i2))
        return tuple(out)

    # -- span carry-through -------------------------------------------------

    def _image_bounds(self) -> tuple[list[int], list[int]]:
        """Per-``a``-token image bounds ``(lo, hi)`` in ``b`` coordinates.

        For an equal block, ``a``-token ``i`` maps to the single ``b``-token at the matched offset.
        For a replace block, every ``a``-token in it maps to the whole ``b`` extent of the block. For
        a delete block, the tokens map to a zero-width point at the block's ``b`` position.
        """
        na = len(self.a.tokens)
        lo = [0] * na
        hi = [0] * na
        for tag, i1, i2, j1, j2 in self.opcodes:
            if tag == "equal":
                for off in range(i2 - i1):
                    lo[i1 + off] = j1 + off
                    hi[i1 + off] = j1 + off + 1
            elif tag == "replace":
                for i in range(i1, i2):
                    lo[i] = j1
                    hi[i] = j2
            elif tag == "delete":
                for i in range(i1, i2):
                    lo[i] = j1
                    hi[i] = j1
            # insert blocks contribute no a-tokens
        return lo, hi

    def map_span(self, span: Span) -> Span | None:
        """Carry a token span in ``a`` to the corresponding token span in ``b``.

        Returns the mapped `Span` (preserving the kind and meta), or None when the span maps to
        nothing because every token it covers was deleted. An empty (zero-width) input span returns
        None. The mapped span is the tightest ``b`` interval covering the images of all ``a`` tokens
        in the input span.
        """
        if span.end <= span.start:
            return None
        na = len(self.a.tokens)
        s = max(0, span.start)
        e = min(na, span.end)
        if s >= e:
            return None
        lo, hi = self._image_bounds()
        b_start = min(lo[i] for i in range(s, e))
        b_end = max(hi[i] for i in range(s, e))
        if b_end <= b_start:
            return None  # fully deleted region
        return Span(start=b_start, end=b_end, kind=span.kind, meta=dict(span.meta))


# ---------------------------------------------------------------------------
# align
# ---------------------------------------------------------------------------


def align(
    a: TokenizedInput | str,
    b: TokenizedInput | str,
    edit_script: Sequence[CharEdit] | None = None,
    *,
    tokenizer: Tokenizer | None = None,
) -> SpanMap:
    """Align two closely related sequences into a `SpanMap`.

    ``a`` and ``b`` may be tokenized inputs or raw strings (tokenized with ``tokenizer``, defaulting
    to the reference tokenizer). When ``edit_script`` is provided, the alignment is computed exactly
    from it and the resulting `SpanMap` is marked ``exact``; a debug-cheap consistency check asserts
    the script actually transforms ``a`` into ``b``, so a mismatched script fails loudly rather than
    producing a plausible wrong alignment. When no script is provided, the alignment is inferred with
    ``difflib.SequenceMatcher`` on normalized token strings and is marked non-exact.
    """
    tok = tokenizer or DEFAULT_TOKENIZER
    ta = a if isinstance(a, TokenizedInput) else tok.tokenize(a)
    tb = b if isinstance(b, TokenizedInput) else tok.tokenize(b)

    if edit_script is not None:
        produced = apply_edits(ta.text, edit_script)
        if produced != tb.text:
            raise ValueError(
                "align: edit_script does not transform a into b; refusing to build an "
                "'exact' alignment from an inconsistent script. "
                f"expected b of length {len(tb.text)}, script produced length {len(produced)}"
            )
        opcodes = _opcodes_from_edit_script(ta, tb, edit_script)
        return SpanMap(a=ta, b=tb, opcodes=tuple(opcodes), exact=True)

    matcher = SequenceMatcher(a=ta.normalized(), b=tb.normalized(), autojunk=False)
    opcodes = [(tag, i1, i2, j1, j2) for tag, i1, i2, j1, j2 in matcher.get_opcodes()]
    return SpanMap(a=ta, b=tb, opcodes=tuple(opcodes), exact=False)


__all__ = [
    "CharEdit",
    "apply_edits",
    "SpanMap",
    "align",
    "Opcode",
]

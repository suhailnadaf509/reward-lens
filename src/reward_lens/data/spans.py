"""Span vocabulary, tokenization, and character-to-token mapping.

The kernel's `Span` (in `reward_lens.core.types`) is a typed token interval ``[start, end)``.
This module supplies the two things the core primitive deliberately leaves to the data plane: the
vocabulary of span *kinds* that give a span its meaning, and the character-to-token mapping that
turns a human-authored character range ("the receipt is characters 40..75") into an exact token
interval that span-level patching and attribution can act on.

Real tokenization lives in the signals layer: ``signals.tokenize`` owns span carry-through against
a model's own tokenizer and chat template. For the paths that must run without any model, this
module defines a small pluggable `Tokenizer` protocol with a deterministic default,
`SimpleTokenizer`. The default splits on whitespace and punctuation and records exact character
offsets, so a span built from a character range maps to the same token interval on every machine
and every run. Nothing here is model-specific; it is the reference tokenizer the fixtures and the
SpanMap fuzz tests are written against.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from reward_lens.core import Span

# ---------------------------------------------------------------------------
# The vocabulary of span kinds
# ---------------------------------------------------------------------------
#
# These string constants are the controlled vocabulary a span's ``kind`` tag draws from. They are
# strings (not an enum) because spans cross the JSON boundary constantly and a downstream science
# may legitimately introduce a new kind; the constants document the standard set and the helpers
# below validate against it with a warning rather than a hard failure.

RECEIPT = "receipt"  # an evidence span: a quote, citation, tool output, computed value, log line
NARRATIVE = "narrative"  # the model's own claim/summary about the evidence, distinct from it
STEP = "step"  # one step of a multi-step solution (math/code/reasoning)
ERROR = "error"  # a planted or identified error location (the corruption target)
CRITIQUE = "critique"  # a judge's critique sentence, for verdict/critique mediation
VERDICT = "verdict"  # the judgment token(s) a generative judge emits
ACTION = "action"  # an agent action description in a trajectory step
TOOL_CALL = "tool_call"  # the tool-invocation portion of a trajectory step
STYLE = "style"  # a stylistic span (formatting, hedging, verbosity padding)
TEXT = "text"  # the untyped default

SPAN_KINDS: frozenset[str] = frozenset(
    {RECEIPT, NARRATIVE, STEP, ERROR, CRITIQUE, VERDICT, ACTION, TOOL_CALL, STYLE, TEXT}
)


# ---------------------------------------------------------------------------
# Tokens and tokenized inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Token:
    """A single token with its half-open character range ``[start, end)`` in the source text.

    ``idx`` is the token's position in the sequence. The surface ``text`` is the exact substring;
    normalization (lowercasing) is applied on demand by the consumers that align on normalized
    strings, so the surface form is never lost.
    """

    text: str
    start: int
    end: int
    idx: int

    @property
    def norm(self) -> str:
        return self.text.lower()


@dataclass(frozen=True)
class TokenizedInput:
    """A tokenized string: the source text plus its tokens with character offsets.

    This is the object token alignment (`align.py`) and span construction operate on. It keeps the
    original text so a token span can always be rendered back to the exact substring it covers,
    which is what makes span-level edits and receipt-reliance measurements exact rather than
    approximate.
    """

    text: str
    tokens: tuple[Token, ...]

    def __len__(self) -> int:
        return len(self.tokens)

    def normalized(self) -> list[str]:
        """The lowercased surface of each token, the sequence alignment operates on."""
        return [t.norm for t in self.tokens]

    def char_to_token_map(self) -> list[int]:
        """Map every character index to its token index, or ``-1`` for inter-token gaps.

        This is the ``char_to_token_map``-style utility the signals layer will provide against a
        real tokenizer; here it is derived from the exact offsets the tokenizer already recorded.
        The returned list has length ``len(self.text)``.
        """
        mapping = [-1] * len(self.text)
        for t in self.tokens:
            for c in range(t.start, t.end):
                mapping[c] = t.idx
        return mapping

    def token_range_for_chars(self, char_start: int, char_end: int) -> tuple[int, int]:
        """Return the smallest token interval ``[t0, t1)`` covering characters ``[char_start, char_end)``.

        A character range that overlaps any part of a token pulls in that whole token: a span is a
        token interval, so a character range that clips the middle of a token still selects the
        token. A range that falls entirely between tokens (in whitespace) returns a zero-width
        interval at the insertion point, so callers can distinguish "no tokens here" from "one
        token here".
        """
        if char_end < char_start:
            raise ValueError(f"char_end {char_end} precedes char_start {char_start}")
        overlapping = [t.idx for t in self.tokens if t.start < char_end and t.end > char_start]
        if overlapping:
            return overlapping[0], overlapping[-1] + 1
        # No token overlaps the range: find the insertion point (first token starting at or after
        # char_start) and return a zero-width interval there.
        for t in self.tokens:
            if t.start >= char_start:
                return t.idx, t.idx
        return len(self.tokens), len(self.tokens)

    def text_for_span(self, span: Span) -> str:
        """The exact source substring a token span covers (empty string for a zero-width span)."""
        if span.end <= span.start:
            return ""
        first = self.tokens[span.start]
        last = self.tokens[span.end - 1]
        return self.text[first.start : last.end]


# ---------------------------------------------------------------------------
# The Tokenizer protocol and the deterministic default
# ---------------------------------------------------------------------------


@runtime_checkable
class Tokenizer(Protocol):
    """The pluggable tokenizer contract the data plane is written against.

    A tokenizer maps a string to a `TokenizedInput` with exact character offsets. The signals
    layer will supply an implementation backed by a model's own tokenizer (with chat-template span
    carry-through); the data plane only requires that offsets are exact and tokenization is
    deterministic, so spans built here map to the same intervals a model sees.
    """

    def tokenize(self, text: str) -> TokenizedInput: ...


class SimpleTokenizer:
    """A deterministic whitespace-and-punctuation tokenizer with exact character offsets.

    Each token is either a maximal run of word characters (``\\w+``) or a single non-word,
    non-space character. This is intentionally simple and reproducible: it is the reference
    tokenizer the SpanMap fuzz tests and the builtin fixtures are authored against, not a
    model-faithful one. When span exactness matters against a real model, the signals-layer
    tokenizer replaces this behind the same protocol.
    """

    _PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def tokenize(self, text: str) -> TokenizedInput:
        tokens = tuple(
            Token(text=m.group(0), start=m.start(), end=m.end(), idx=i)
            for i, m in enumerate(self._PATTERN.finditer(text))
        )
        return TokenizedInput(text=text, tokens=tokens)


# A shared default instance; the tokenizer is stateless, so one instance is safe to reuse.
DEFAULT_TOKENIZER = SimpleTokenizer()


# ---------------------------------------------------------------------------
# Span construction helpers
# ---------------------------------------------------------------------------


def typed_span(start: int, end: int, kind: str = TEXT, **meta: Any) -> Span:
    """Build a `Span` over a token interval with a validated kind tag.

    ``start`` and ``end`` are token indices. An unknown ``kind`` is allowed (a science may extend
    the vocabulary) but the ``meta`` records that it was off-vocabulary so it is visible downstream.
    """
    meta_dict = dict(meta)
    if kind not in SPAN_KINDS:
        meta_dict.setdefault("off_vocabulary_kind", True)
    return Span(start=start, end=end, kind=kind, meta=meta_dict)


def make_span(
    tokenized: TokenizedInput,
    char_start: int,
    char_end: int,
    kind: str = TEXT,
    *,
    tokenizer: Tokenizer | None = None,
    **meta: Any,
) -> Span:
    """Build a typed `Span` from a character range by mapping it to a token interval.

    This is the load-bearing bridge: a human (or a corruption generator) names a character range,
    and this returns the exact token interval a model would see, tagged with its kind. The original
    character range is preserved in ``meta`` so the mapping is auditable. ``tokenizer`` is accepted
    for symmetry with the signals-layer API but is unused here because ``tokenized`` already carries
    the offsets; passing a mismatched tokenizer is a caller error, not something this can detect.
    """
    del tokenizer  # offsets already live on `tokenized`; the parameter is API symmetry only
    t0, t1 = tokenized.token_range_for_chars(char_start, char_end)
    meta_dict = dict(meta)
    meta_dict.setdefault("char_start", char_start)
    meta_dict.setdefault("char_end", char_end)
    return typed_span(t0, t1, kind, **meta_dict)


def char_to_token_map(tokenized: TokenizedInput) -> list[int]:
    """Free-function form of :meth:`TokenizedInput.char_to_token_map`."""
    return tokenized.char_to_token_map()


__all__ = [
    # span kinds
    "RECEIPT",
    "NARRATIVE",
    "STEP",
    "ERROR",
    "CRITIQUE",
    "VERDICT",
    "ACTION",
    "TOOL_CALL",
    "STYLE",
    "TEXT",
    "SPAN_KINDS",
    # tokens
    "Token",
    "TokenizedInput",
    "Tokenizer",
    "SimpleTokenizer",
    "DEFAULT_TOKENIZER",
    # span construction
    "typed_span",
    "make_span",
    "char_to_token_map",
]

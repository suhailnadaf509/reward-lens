"""The data plane's native objects and the `DataView` the kernel consumes.

The corpus's experiments are, to an unusual degree, data-construction problems: matched pairs that
vary one thing, controlled quadruples, k-wise tournaments, trajectories with receipts. A first
principle of the library is that for a reward model the *pair* is not a synthetic stimulus; it is the
training distribution itself, because the Bradley-Terry loss operates on exactly the
chosen-minus-rejected difference the instruments measure. So the data plane treats the pair and
its generalizations as native, typed objects, with span typing and lineage built in.

`DataView` is the single uniform iterable every instrument consumes: instruments never load
datasets, never mutate them, and never construct stimuli inline. A view is filterable, sliceable,
hash-stable, and reports its own effective sample size from lineage and a content checksum that goes
into every measurement's subject reference. These two methods, `effective_n` and `checksum`, are the
data plane's contribution to the two failure classes it exists to kill: fake sample sizes and silent
count/content drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Sequence

from reward_lens.core import DatasetID, Span, content_hash
from reward_lens.data.lineage import (
    Lineage,
    collapse_duplicates,
    effective_sample_size,
    make_lineage,
)

# ---------------------------------------------------------------------------
# Prompts and responses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Prompt:
    """A prompt with optional typed spans.

    The authoritative pair schema types a pair's prompt as a plain string, and the builders here
    use plain strings. `Prompt` is the richer form for the cases that need typed prompt spans (a
    receipt embedded in the prompt, a step to reference), and a `Pair` accepts either;
    ``prompt_text`` normalizes the two. Keeping the string form as the default keeps
    the common case simple while making prompt-level span typing reachable when a study needs it.
    """

    text: str
    spans: tuple[Span, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.text


@dataclass(frozen=True)
class Response:
    """A response: its text and its typed spans.

    ``spans`` are token intervals (built via `data.spans`) tagged receipt, narrative, step, error,
    critique, and so on. They are what make span-level patching, receipt-reliance, and error
    localization exact rather than heuristic. A response with no typed spans is the common case and
    is perfectly valid; the spans are added by the builders that know where the structure is.
    """

    text: str
    spans: tuple[Span, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)

    def span_text(self, span: Span, *, tokenizer: Any = None) -> str:
        """The substring a token span covers, resolved against this response's tokenization.

        Requires a tokenizer to turn token indices back into character offsets. Defaults to the
        deterministic reference tokenizer so this is usable without a model; pass the signals-layer
        tokenizer when exactness against a specific model matters.
        """
        from reward_lens.data.spans import DEFAULT_TOKENIZER

        tok = tokenizer or DEFAULT_TOKENIZER
        return tok.tokenize(self.text).text_for_span(span)


def _prompt_text(prompt: str | Prompt) -> str:
    """Normalize a prompt (string or `Prompt`) to its text."""
    return prompt.text if isinstance(prompt, Prompt) else prompt


# ---------------------------------------------------------------------------
# The native comparison objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Pair:
    """A chosen/rejected preference pair, the native reward-model object.

    ``axis`` names what differs between chosen and rejected *by construction*: "verbosity",
    "correctness", "receipt-grounding". This is the field that makes a pair a controlled stimulus
    rather than an arbitrary comparison, and it is what a cross-axis analysis reads. ``lineage``
    carries the seed provenance; ``meta`` carries difficulty, domain, source, annotator stats.
    """

    prompt: str | Prompt
    chosen: Response
    rejected: Response
    axis: str
    lineage: Lineage
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def prompt_text(self) -> str:
        return _prompt_text(self.prompt)

    @property
    def seed_id(self) -> str:
        return self.lineage.seed_id


@dataclass(frozen=True)
class Quadruple:
    """A controlled 2x2 design.

    ``cells`` maps a ``(factor_a_level, factor_b_level)`` key to the response for that cell; the
    canonical example is L2's sycophancy design, agree/disagree crossed with right/wrong. ``factors``
    names the two crossed factors. A quadruple lets a study read an interaction (does the grader
    reward agreement *more* when the user is wrong?) that no single pair can express.
    """

    prompt: str | Prompt
    cells: dict[tuple[str, str], Response]
    factors: tuple[str, str]
    lineage: Lineage
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def prompt_text(self) -> str:
        return _prompt_text(self.prompt)

    @property
    def seed_id(self) -> str:
        return self.lineage.seed_id


@dataclass(frozen=True)
class EdgeObs:
    """One observed pairwise comparison within a tournament.

    ``wins_i`` and ``wins_j`` count how often response ``i`` beat response ``j`` and vice versa
    (repeated comparisons accumulate here). Exactly one of ``annotator_id`` / ``judge_id`` names the
    source, which is what keeps human and machine preference distinguishable and lets the
    topology science slice the intransitive mass by annotator or judge.
    """

    i: int
    j: int
    wins_i: int
    wins_j: int
    annotator_id: str | None = None
    judge_id: str | None = None

    def __canonical__(self) -> dict[str, Any]:
        return {
            "i": self.i,
            "j": self.j,
            "wins_i": self.wins_i,
            "wins_j": self.wins_j,
            "annotator_id": self.annotator_id,
            "judge_id": self.judge_id,
        }


@dataclass(frozen=True)
class Tournament:
    """A k-wise comparison collection over one prompt.

    ``responses`` are the competitors; ``edges`` are the observed pairwise outcomes. This is the
    Hodge-ready object: a preference operator over the responses whose curl (intransitive mass) is
    exactly what the topology science measures, and whose edges carry their annotator or judge id so
    that measurement can be sliced by source.
    """

    prompt: str | Prompt
    responses: tuple[Response, ...]
    edges: tuple[EdgeObs, ...]
    lineage: Lineage
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def prompt_text(self) -> str:
        return _prompt_text(self.prompt)

    @property
    def seed_id(self) -> str:
        return self.lineage.seed_id


@dataclass(frozen=True)
class TrajStep:
    """One step of an agent trajectory.

    ``action`` is a short description of what the agent did; ``tool_call`` is the structured
    invocation (name and arguments) where there was one. ``text`` is the step's rendered text, and
    ``receipts`` / ``narrative`` are token spans into it: the receipt spans are the evidence (a tool
    result, a computed value), the narrative spans are the agent's own account of it. Separating them
    is what makes receipt-reliance and receipt-falsification experiments possible.
    """

    action: str
    tool_call: dict[str, Any] | None = None
    text: str = ""
    receipts: tuple[Span, ...] = ()
    narrative: tuple[Span, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Trajectory:
    """An agent episode: a sequence of steps and its outcome.

    ``outcome`` records how the episode ended (success flag, final reward, task metadata). The
    trajectory is the substrate for the receipt/narrative sciences and for trajectory-level reward
    signals; its steps carry the typed spans those sciences read.
    """

    steps: tuple[TrajStep, ...]
    outcome: dict[str, Any]
    lineage: Lineage
    prompt: str | Prompt = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def prompt_text(self) -> str:
        return _prompt_text(self.prompt)

    @property
    def seed_id(self) -> str:
        return self.lineage.seed_id


# ---------------------------------------------------------------------------
# Canonical content extraction (for checksums and clone detection)
# ---------------------------------------------------------------------------


def _span_content(span: Span) -> tuple[int, int, str]:
    """The content-bearing part of a span: its interval and kind, not its meta.

    Span meta holds provenance (the source character range), not payload, so it is excluded from
    content identity: two responses with the same text and the same typed intervals are the same
    content even if one recorded its source char offsets and the other did not.
    """
    return (span.start, span.end, span.kind)


def _response_content(r: Response) -> Any:
    return ["Response", r.text, [_span_content(s) for s in r.spans]]


def response_content(r: Response) -> Any:
    """Public canonical content of a response (used by builders to stamp lineage)."""
    return _response_content(r)


def _traj_steps_payload(steps: Sequence["TrajStep"]) -> Any:
    return [
        [
            s.action,
            s.tool_call,
            s.text,
            [_span_content(sp) for sp in s.receipts],
            [_span_content(sp) for sp in s.narrative],
        ]
        for s in steps
    ]


def trajectory_content(
    prompt_text: str, steps: Sequence["TrajStep"], outcome: dict[str, Any]
) -> Any:
    """Canonical content of a trajectory from its parts (used by builders to stamp lineage).

    Exposed so a corruption builder can compute an edited trajectory's lineage content hash before it
    constructs the frozen `Trajectory`, keeping the hash and `content_of` in agreement.
    """
    return ["Trajectory", prompt_text, _traj_steps_payload(steps), outcome]


def content_of(item: Any) -> Any:
    """The canonical, lineage-independent content of a schema item.

    This is what the dataset checksum hashes and what clone detection groups on. It is deliberately
    independent of lineage (seed id, ops): two items with identical payloads are identical content
    regardless of how they were produced, which is the correct notion for both a content checksum and
    ingest-time deduplication. Builders pass the matching structure to `make_lineage` so an item's
    ``lineage.content_hash`` equals ``content_hash(content_of(item), "ch")``.
    """
    if isinstance(item, Pair):
        return [
            "Pair",
            item.prompt_text,
            _response_content(item.chosen),
            _response_content(item.rejected),
            item.axis,
        ]
    if isinstance(item, Quadruple):
        cells = [[list(k), _response_content(v)] for k, v in sorted(item.cells.items())]
        return ["Quadruple", item.prompt_text, list(item.factors), cells]
    if isinstance(item, Tournament):
        return [
            "Tournament",
            item.prompt_text,
            [_response_content(r) for r in item.responses],
            [e.__canonical__() for e in item.edges],
        ]
    if isinstance(item, Trajectory):
        return trajectory_content(item.prompt_text, item.steps, item.outcome)
    if isinstance(item, Response):
        return _response_content(item)
    raise TypeError(f"content_of: unsupported item type {type(item).__name__}")


def seed_id_of(item: Any) -> str:
    """The seed id of a schema item, read from its lineage."""
    lineage: Lineage | None = getattr(item, "lineage", None)
    if lineage is None:
        raise TypeError(f"{type(item).__name__} carries no lineage; cannot read seed id")
    return lineage.seed_id


# ---------------------------------------------------------------------------
# Builder convenience
# ---------------------------------------------------------------------------


def make_pair(
    prompt: str | Prompt,
    chosen: str | Response,
    rejected: str | Response,
    axis: str,
    *,
    seed_id: str,
    builder_id: str,
    ops: Sequence[str] = (),
    meta: dict[str, Any] | None = None,
) -> Pair:
    """Construct a `Pair` with a correctly stamped lineage in one call.

    Accepts plain strings for the responses (wrapped in `Response`) or `Response` objects when the
    caller has typed spans to attach. The lineage content hash is computed from the same canonical
    content `content_of` produces, so ``pair.lineage.content_hash`` equals
    ``content_hash(content_of(pair), "ch")`` and clone detection, the checksum, and the lineage all
    agree. This is the builder every dataset constructor should use rather than assembling a `Pair`
    and a `Lineage` separately, which is how the two could drift.
    """
    chosen_r = chosen if isinstance(chosen, Response) else Response(text=chosen)
    rejected_r = rejected if isinstance(rejected, Response) else Response(text=rejected)
    content = [
        "Pair",
        _prompt_text(prompt),
        _response_content(chosen_r),
        _response_content(rejected_r),
        axis,
    ]
    lineage = make_lineage(seed_id, builder_id, ops, content)
    return Pair(
        prompt=prompt,
        chosen=chosen_r,
        rejected=rejected_r,
        axis=axis,
        lineage=lineage,
        meta=dict(meta or {}),
    )


# ---------------------------------------------------------------------------
# DataView
# ---------------------------------------------------------------------------


class DataView:
    """The uniform, filterable, sliceable, hash-stable iterable the kernel consumes.

    A view wraps an ordered collection of items (Pairs, or any schema type; a view is homogeneous by
    convention but does not enforce it, so mixed diagnostic sets are expressible). It never loads or
    mutates data; `filter` and `slice` return new views over the same underlying items, so a view is
    effectively immutable and safe to share across instruments.

    The two methods that carry the data plane's discipline:

    - `effective_n` computes the lineage-aware effective sample size by handing the items' seed
      ids to the stats engine. A view of forty clones of one seed reports about one.
    - `checksum` is a content-derived `DatasetID` that goes into every measurement's subject
      reference. It is stable (deterministic given the items and their order) and content-based, so
      a loader that silently returned the wrong rows produces a different checksum and the dataset
      registry catches it.
    """

    __slots__ = ("_items", "name")

    def __init__(self, items: Sequence[Any], *, name: str | None = None) -> None:
        self._items: tuple[Any, ...] = tuple(items)
        self.name = name

    # -- sequence protocol --------------------------------------------------

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._items)

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, slice):
            return DataView(self._items[index], name=self.name)
        return self._items[index]

    def __repr__(self) -> str:
        label = f" {self.name!r}" if self.name else ""
        return f"<DataView{label} n={len(self._items)}>"

    @property
    def items(self) -> tuple[Any, ...]:
        return self._items

    # -- derivation ---------------------------------------------------------

    def filter(self, predicate: Callable[[Any], bool]) -> "DataView":
        """A new view over the items satisfying ``predicate`` (order preserved)."""
        return DataView([it for it in self._items if predicate(it)], name=self.name)

    def slice(self, start: int, stop: int | None = None, step: int | None = None) -> "DataView":
        """A new view over ``items[start:stop:step]``.

        Provided as an explicit method (in addition to ``view[start:stop]``) because a study spec
        names its slice as data, and a method call is what a serialized plan records.
        """
        return DataView(self._items[start:stop:step], name=self.name)

    def concat(self, other: "DataView", *, name: str | None = None) -> "DataView":
        """A new view over this view's items followed by ``other``'s."""
        return DataView((*self._items, *other._items), name=name or self.name)

    # -- lineage and identity ----------------------------------------------

    def seed_ids(self) -> list[str]:
        """The seed id of every item, in order (the input to effective sample size)."""
        return [seed_id_of(it) for it in self._items]

    def effective_n(self) -> float:
        """Lineage-aware effective sample size over the items' seed ids.

        Delegates to `reward_lens.stats.ess` through `data.lineage.effective_sample_size`, which is
        the canonical implementation when the stats engine is present and an identical local fallback
        otherwise. A view of clones reports about one; a view of distinct seeds reports about its
        length.
        """
        return effective_sample_size(self.seed_ids())

    def checksum(self) -> DatasetID:
        """A stable, content-derived `DatasetID` (``ds:...``) for this view.

        Hashes the ordered list of item content. It is content-based and order-sensitive: the same
        items in the same order always produce the same id, and any change of rows or order changes
        it. This is the id the dataset registry verifies against a card's declared checksum, which is
        where the limit/subset loader bug dies.
        """
        material = [content_of(it) for it in self._items]
        return DatasetID(content_hash(material, "ds"))

    def collapse_duplicates(self, *, warn: bool = True) -> tuple["DataView", list[int]]:
        """Collapse exact-duplicate-content items, returning ``(view, weights)``."""
        unique, weights = collapse_duplicates(
            self._items, key=lambda it: content_hash(content_of(it), "ch"), warn=warn
        )
        return DataView(unique, name=self.name), weights


__all__ = [
    "Prompt",
    "Response",
    "Pair",
    "Quadruple",
    "EdgeObs",
    "Tournament",
    "TrajStep",
    "Trajectory",
    "DataView",
    "make_pair",
    "content_of",
    "response_content",
    "trajectory_content",
    "seed_id_of",
]

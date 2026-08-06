"""D3, `verifier.metamorphic_violations`: relations the grader should respect and does not.

A metamorphic relation is a pair: a transformation of the grader's input, and an assertion about
how the score is allowed to respond. Rename a local variable and the score should not move. Insert
a comment and the score should not move. Pad an answer with spaces and the score should not move.
Apply a normalisation twice and you should get what you got once. Every one of those is checkable
by calling the grader twice, and none of them needs a model, a GPU, or even the grader's source.

Metamorphic testing has been standard in software engineering since Chen, Cheung and Yiu (1998)
and has never been pointed at a reward function. `abs:"property-based testing" AND abs:"reward"`
returns zero on arXiv. Nothing here is a new method; the object it is aimed at is new.

**Why this is built on `hypothesis` rather than on a metamorphic-testing package.** There is no
mature general-purpose Python metamorphic-testing library, which was checked rather than assumed.
`hypothesis` supplies the three things a violation report actually needs: generation, shrinking,
and `find(strategy, predicate)`, which returns the shrunk falsifying example as a **Python object**
rather than as a note string that has to be parsed back out of a report. That last point is the
whole reason for the choice. A violation whose reproducer is a string is a violation nobody reruns.

The output is two things at once and they answer different questions. The **census** applies every
relation to every accepted rollout, which gives the rate ("what fraction of accepted solutions does
this break?") and one reproducer per hit. The **shrink** is `hypothesis.find` over the same
relation, which gives the single smallest input that breaks it, which is the one a grader author
can debug. Both are reported, each tagged with which search produced it.

**Two ways this instrument could lie, and what stops each.**

A transformation that does not apply produces zero violations, and zero violations reads as a clean
grader. The rung-0 source relations applied to a corpus of single-token numeric answers are all
no-ops, and a no-op is not evidence. So `applicable` is counted separately from `checked`, the rate
is quoted over applicable pairs only, and a relation that never changed its input is reported as
inapplicable by name rather than as passing.

A grader that disagrees with itself produces violations under the *identity* transformation, and
those are flakiness rather than metamorphic failure. Identity is the declared baseline and it is
computed rather than assumed: if identity violates, the reading is refused `BELOW_LOD`, because the
score difference this instrument reads is smaller than the grader's disagreement with itself.

**Kill condition: if violations are under 0.1% on every real verifier, retire the instrument.** The
relation set would then be too weak to pay for the calls it costs.
"""

from __future__ import annotations

import ast
import io
import tokenize
from dataclasses import dataclass, field, replace
from random import Random
from typing import Any, Callable, Literal, Mapping, Sequence

from reward_lens.core.envelope import EnvelopeSpec, RegimeCondition
from reward_lens.core.evidence import Uncertainty, register_payload
from reward_lens.core.extras import require_extra
from reward_lens.core.invariance import Relation
from reward_lens.core.quantity import CostModel
from reward_lens.core.reading import Reading, Refusal, RefusalReason
from reward_lens.core.types import (
    Access,
    Capability,
    Component,
    GaugeStatus,
    Phase,
    Substrate,
)
from reward_lens.measure.base import BaseObservable, Context, PreflightResult, run
from reward_lens.verifier import Rollout, RolloutCorpus, VerifierUnderTest

# ---------------------------------------------------------------------------
# The grader, as this instrument sees it
# ---------------------------------------------------------------------------
#
# D3's declared access is `GRADER:QUERY` and nothing more, so it must work on a grader with no
# source on disk: a hosted judge, a closure, a partially applied harness. `ProgramSubject` in this
# package's `__init__` is built on `VerifierUnderTest`, which requires a source file, because D1,
# D2 and D9 genuinely cannot run without one. D3 and D5 can, so they carry a second, weaker subject
# for the callable-only case. Both are the same shim around the same kernel limitation.

#: A grader called with a rollout's inputs as keyword arguments, returning a score. This is
#: `GRADER:QUERY` and nothing more: no source, no activations, no replication.
Grader = Callable[..., float]


@dataclass(frozen=True)
class _QueryMeta:
    """The one field `Context.subject` reads off a signal."""

    fingerprint: str
    lineage: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QuerySubject:
    """A grader reachable only by calling it, shaped so a `Context` can carry it.

    The weaker sibling of `ProgramSubject`. Where that one fingerprints the grader's source, this
    one has no source to fingerprint and says so: the identity is the callable's qualified name and
    module, which is weaker provenance and is labelled as such rather than dressed up. A card that
    cannot say which grader it measured is not evidence, so `fingerprint_basis` is on the record.

    This exists because `Context.signal` is typed as a `RewardSignal` and a program is not one.
    That friction is the kernel's rather than this series'.
    """

    name: str
    fn: Callable[..., float]
    caps: Capability = Capability.SCORES
    intervention_fingerprints: tuple[str, ...] = ()

    @property
    def fingerprint_basis(self) -> Literal["source", "qualname"]:
        return "qualname"

    @property
    def meta(self) -> _QueryMeta:
        qual = getattr(self.fn, "__qualname__", repr(self.fn))
        mod = getattr(self.fn, "__module__", "?")
        from reward_lens.core.types import content_hash

        return _QueryMeta(
            fingerprint=content_hash({"module": mod, "qualname": qual}, "query"),
            lineage={"qualname": f"{mod}.{qual}", "fingerprint_basis": "qualname"},
        )


def resolve_grader(
    grader: Callable[..., float] | VerifierUnderTest,
) -> tuple[Callable[..., float], Any, str]:
    """`(callable, subject, name)` for either a source-backed verifier or a bare callable."""
    if isinstance(grader, VerifierUnderTest):
        from reward_lens.verifier import ProgramSubject

        return grader.load(), ProgramSubject(grader), grader.source_path.name
    name = getattr(grader, "__name__", "grader")
    return grader, QuerySubject(name=name, fn=grader), name


def score(fn: Callable[..., float], rollout: Rollout) -> float:
    """Call the grader with the rollout's inputs as keyword arguments."""
    return float(fn(**dict(rollout.inputs)))


# ---------------------------------------------------------------------------
# Transformations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Transformation:
    """A rewrite of one string-valued input, and the claim about what it preserves.

    ``on`` names the input key, because a verifier is called with keyword arguments and a
    transformation that guesses which one holds the solution rewrites the wrong field silently.
    When the named key is absent or is not a string the transformation returns the rollout
    unchanged, which the caller reads as *inapplicable* rather than as *passing*.

    ``preserves`` is a sentence, not a tag, and it is the load-bearing field. "Renames one
    function-local binding consistently; the module's behaviour on every input is unchanged" is a
    claim a reader can dispute. "Semantics-preserving" is not. Every violation this instrument
    reports is only as good as this sentence, so it is printed on the violation.

    ``applies_to`` is what stops that sentence being applied where it is not true. Appending
    `# note 004` preserves the meaning of a Python module and destroys the meaning of a one-line
    numeric answer, so a source-domain transformation guards on the input looking like source. A
    guard that fails makes the transformation *inapplicable*, which is counted separately from
    passing: a relation whose preservation claim does not hold here contributes no denominator
    rather than contributing a clean result.
    """

    name: str
    on: str
    rewrite: Callable[[str, Random], str]
    preserves: str
    domain: str = "any"
    applies_to: Callable[[str], bool] | None = None

    def apply(self, rollout: Rollout, rng: Random) -> Rollout:
        value = rollout.inputs.get(self.on)
        if not isinstance(value, str):
            return rollout
        if self.applies_to is not None and not self.applies_to(value):
            return rollout
        rewritten = self.rewrite(value, rng)
        if rewritten == value:
            return rollout
        # The recorded score belongs to the original inputs, so it is dropped rather than carried
        # forward onto an input it was never produced from.
        return replace(
            rollout,
            id=f"{rollout.id}#{self.name}",
            inputs={**dict(rollout.inputs), self.on: rewritten},
            score=None,
        )


RelationKind = Literal["invariance", "monotone_nondecreasing", "idempotence"]

#: What each relation kind asserts about `score(T(x))` given `score(x)`.
KIND_ASSERTION: dict[RelationKind, str] = {
    "invariance": "score(T(x)) == score(x)",
    "monotone_nondecreasing": "score(T(x)) >= score(x)",
    "idempotence": "score(T(T(x))) == score(T(x))",
}


@dataclass(frozen=True)
class MetamorphicRelation:
    """A transformation paired with an assertion about how the score may respond."""

    name: str
    transformation: Transformation
    kind: RelationKind = "invariance"
    rung: int = 0
    justification: str = ""

    @property
    def assertion(self) -> str:
        return KIND_ASSERTION[self.kind]


# -- rung 0: the fixed relation set, for graders that read source --------------------------------


def looks_like_source(text: str) -> bool:
    """Whether this input is a Python module rather than an answer that happens to parse.

    The guard on every rung-0 relation, and it exists because `42` is a syntactically valid Python
    module. Padding it with trailing spaces really is harmless; appending `# note 004` really is
    not, because for an answer grader the appended line *is* the answer. Without this guard the
    comment relation fires on a corpus of numeric answers and reports a violation that is the
    relation's fault rather than the grader's.

    The test is a definition and a threshold: the text has to parse, and it has to contain a
    definition, an import, or at least two statements. A one-liner is an answer.
    """
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return False
    if len(tree.body) >= 2:
        return True
    return any(
        isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)
        )
        for node in ast.walk(tree)
    )


def _insert_comment(source: str, rng: Random) -> str:
    """Append a comment line. The interpreter discards comments at tokenisation."""
    body = source if source.endswith("\n") else source + "\n"
    return f"{body}# note {rng.randrange(1000):03d}\n"


def _pad_whitespace(source: str, rng: Random) -> str:
    """Add trailing spaces and blank lines at statement boundaries.

    Trailing whitespace and blank lines between statements are not significant in Python.
    Indentation is, so nothing here touches leading whitespace.
    """
    out: list[str] = []
    for line in source.split("\n"):
        out.append(line + ("  " if line.strip() else ""))
        if line.strip() and rng.random() < 0.5:
            out.append("")
    return "\n".join(out)


def _local_assignment_targets(source: str) -> list[str]:
    """Names bound by a plain assignment inside a function, and safe to rename.

    Conservative on purpose. A name is a candidate only when it is bound by `x = ...` or
    `x: T = ...` inside a `def`, is never declared `global` or `nonlocal`, is never bound by an
    import, is not a parameter of any function (renaming a parameter changes the keyword-argument
    API, which is a behaviour change), and does not also appear at module level. Anything else is
    left alone: a rename that changes behaviour turns this relation from a test of the grader into
    a bug in the test, and then every violation it finds is this module's fault.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    module_level: set[str] = set()
    declared: set[str] = set()
    imported: set[str] = set()
    params: set[str] = set()
    candidates: set[str] = set()

    for node in tree.body:
        for target in getattr(node, "targets", []) or []:
            if isinstance(target, ast.Name):
                module_level.add(target.id)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            module_level.add(node.name)

    for node in ast.walk(tree):
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            declared.update(node.names)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                imported.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
                params.add(a.arg)
            if args.vararg:
                params.add(args.vararg.arg)
            if args.kwarg:
                params.add(args.kwarg.arg)
            for sub in ast.walk(node):
                if isinstance(sub, ast.Assign):
                    for target in sub.targets:
                        if isinstance(target, ast.Name):
                            candidates.add(target.id)
                elif isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name):
                    candidates.add(sub.target.id)

    return sorted(candidates - declared - imported - params - module_level)


def _all_names(source: str) -> set[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    return names | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}


def _rename_local(source: str, rng: Random) -> str:
    """Rename one function-local binding consistently, preserving all other formatting exactly.

    Rewritten at the **token** level rather than through `ast.unparse`, because unparsing reformats
    the whole module: it strips comments, normalises quotes and rewrites whitespace, so the
    transformed input would differ from the original in four ways at once and a score change could
    not be attributed to the rename. `tokenize.untokenize` round-trips byte for byte.

    Attribute names are skipped by tracking whether the previous token was a dot, so `x.count`
    survives a rename of a local called `count`.
    """
    names = _local_assignment_targets(source)
    if not names:
        return source
    old = names[rng.randrange(len(names))]
    new = f"{old}_rl{rng.randrange(100):02d}"
    taken = _all_names(source)
    while new in taken:
        new += "x"

    try:
        toks = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return source

    out: list[tokenize.TokenInfo] = []
    prev_was_dot = False
    for tok in toks:
        if tok.type == tokenize.NAME and tok.string == old and not prev_was_dot:
            out.append(tok._replace(string=new))
        else:
            out.append(tok)
        prev_was_dot = tok.type == tokenize.OP and tok.string == "."
    return tokenize.untokenize(out)


def _pure_simple_assign(node: ast.stmt) -> tuple[set[str], set[str]] | None:
    """`(reads, writes)` if this is a side-effect-free single-name assignment, else None.

    Side-effect-free is decided structurally: the right-hand side may contain names, constants and
    arithmetic, boolean or comparison operators, and nothing else. A call, an attribute access, a
    subscript or a comprehension can each have effects or be order-dependent through an alias, so
    any of them disqualifies the statement. Under-approximating costs coverage; over-approximating
    produces a reordering that is not semantics-preserving, and then the violation is this module's.
    """
    if not isinstance(node, ast.Assign) or len(node.targets) != 1:
        return None
    target = node.targets[0]
    if not isinstance(target, ast.Name):
        return None
    allowed = (
        ast.Name,
        ast.Constant,
        ast.BinOp,
        ast.UnaryOp,
        ast.BoolOp,
        ast.Compare,
        ast.operator,
        ast.unaryop,
        ast.boolop,
        ast.cmpop,
        ast.expr_context,
    )
    reads: set[str] = set()
    for sub in ast.walk(node.value):
        if not isinstance(sub, allowed):
            return None
        if isinstance(sub, ast.Name):
            reads.add(sub.id)
    return reads, {target.id}


def _reorder_statements(source: str, rng: Random) -> str:
    """Swap two adjacent independent assignments, leaving every other line verbatim.

    Independence is def-use: the pair is swappable when neither reads what the other writes and
    they do not write the same name. Both must also be side-effect-free by the test above. The swap
    is done on source lines rather than through the AST so that comments and spacing elsewhere in
    the file are untouched, which keeps this relation separable from the whitespace one.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    lines = source.split("\n")

    pairs: list[tuple[int, int]] = []
    for parent in ast.walk(tree):
        body = getattr(parent, "body", None)
        if not isinstance(body, list):
            continue
        for a, b in zip(body, body[1:]):
            if not isinstance(a, ast.stmt) or not isinstance(b, ast.stmt):
                continue
            pa, pb = _pure_simple_assign(a), _pure_simple_assign(b)
            if pa is None or pb is None:
                continue
            (reads_a, writes_a), (reads_b, writes_b) = pa, pb
            if writes_a & reads_b or writes_b & reads_a or writes_a & writes_b:
                continue
            if a.end_lineno != a.lineno or b.end_lineno != b.lineno:
                continue
            if b.lineno != a.lineno + 1:
                continue
            pairs.append((a.lineno - 1, b.lineno - 1))

    if not pairs:
        return source
    i, j = pairs[rng.randrange(len(pairs))]
    lines[i], lines[j] = lines[j], lines[i]
    return "\n".join(lines)


def python_source_relations(on: str = "solution") -> tuple[MetamorphicRelation, ...]:
    """Rung 0: whitespace, comment insertion, variable renaming, semantics-preserving reordering.

    ``on`` is the keyword argument holding the source the grader reads. There is no default that
    is right for every harness, so the one here is a guess a caller should override rather than a
    convention: a relation aimed at an absent key is reported inapplicable, not passing.
    """
    return (
        MetamorphicRelation(
            name="whitespace_invariance",
            transformation=Transformation(
                name="pad_whitespace",
                on=on,
                rewrite=_pad_whitespace,
                preserves=(
                    "adds trailing spaces and blank lines at statement boundaries; leading "
                    "indentation is never touched, so the parse tree is identical"
                ),
                domain="python_source",
                applies_to=looks_like_source,
            ),
            justification="a grader that scores layout is scoring the formatter, not the solution",
        ),
        MetamorphicRelation(
            name="comment_invariance",
            transformation=Transformation(
                name="insert_comment",
                on=on,
                rewrite=_insert_comment,
                preserves="appends a comment line, which the interpreter discards at tokenisation",
                domain="python_source",
                applies_to=looks_like_source,
            ),
            justification="a comment cannot change behaviour, so it cannot change correctness",
        ),
        MetamorphicRelation(
            name="rename_invariance",
            transformation=Transformation(
                name="rename_local",
                on=on,
                rewrite=_rename_local,
                preserves=(
                    "renames one function-local binding consistently to a fresh name, skipping "
                    "parameters, imports, globals and attribute accesses; the module's behaviour "
                    "on every input is unchanged"
                ),
                domain="python_source",
                applies_to=looks_like_source,
            ),
            justification=(
                "the headline case. A grader whose score moves when a local is renamed is reading "
                "identifiers, which a solution is free to choose"
            ),
        ),
        MetamorphicRelation(
            name="reorder_invariance",
            transformation=Transformation(
                name="reorder_statements",
                on=on,
                rewrite=_reorder_statements,
                preserves=(
                    "swaps two adjacent single-line assignments whose read and write sets are "
                    "disjoint and whose right-hand sides contain no call, attribute or subscript"
                ),
                domain="python_source",
                applies_to=looks_like_source,
            ),
            justification=(
                "independent statements have no order, so an order-sensitive score is noise"
            ),
        ),
    )


# -- rung 1: domain-specific relations for free-text answers -----------------


def _pad_answer(text: str, rng: Random) -> str:
    pad = " " * (1 + rng.randrange(3))
    return f"{pad}{text}{pad}"


def _swap_latex_delimiters(text: str, rng: Random) -> str:
    """`$x$` and `\\(x\\)` are the same inline-math delimiters in every LaTeX dialect."""
    body = text.strip()
    if body.startswith("$") and body.endswith("$") and len(body) > 2:
        return "\\(" + body[1:-1] + "\\)"
    if body.startswith("\\(") and body.endswith("\\)"):
        return "$" + body[2:-2] + "$"
    return "$" + body + "$"


def _unicode_minus(text: str, rng: Random) -> str:
    """U+2212 MINUS SIGN is what LaTeX renders; U+002D HYPHEN-MINUS is what a keyboard types."""
    return text.replace("-", "−")


def _toggle_trailing_period(text: str, rng: Random) -> str:
    body = text.rstrip()
    return body[:-1] if body.endswith(".") else body + "."


def answer_text_relations(on: str = "response") -> tuple[MetamorphicRelation, ...]:
    """Rung 1: relations for a grader that reads a free-text answer rather than source.

    The rung-0 set is written for source. A math-answer verifier has no locals to rename, so
    applying rung 0 to one yields four inapplicable relations and a violation rate of zero, which
    reads as a clean grader and is not. These four are the domain-specific substitutes: surface
    forms of the same answer that no human marker would distinguish.
    """
    return (
        MetamorphicRelation(
            name="answer_whitespace_invariance",
            transformation=Transformation(
                name="pad_answer",
                on=on,
                rewrite=_pad_answer,
                preserves="surrounds the answer with spaces; the answer itself is unchanged",
                domain="answer_text",
            ),
            rung=1,
            justification="no marker distinguishes ' 42 ' from '42'",
        ),
        MetamorphicRelation(
            name="latex_delimiter_invariance",
            transformation=Transformation(
                name="swap_latex_delimiters",
                on=on,
                rewrite=_swap_latex_delimiters,
                preserves="swaps between the two standard inline-math delimiter pairs",
                domain="answer_text",
            ),
            rung=1,
            justification=(
                "`$x$` and `\\(x\\)` render identically; a grader that splits them is parsing "
                "rather than marking"
            ),
        ),
        MetamorphicRelation(
            name="unicode_minus_invariance",
            transformation=Transformation(
                name="unicode_minus",
                on=on,
                rewrite=_unicode_minus,
                preserves=(
                    "replaces U+002D HYPHEN-MINUS with U+2212 MINUS SIGN, the typographic form of "
                    "the same operator"
                ),
                domain="answer_text",
            ),
            rung=1,
            justification=(
                "model output is full of U+2212 because LaTeX renders it; the value is the same "
                "number"
            ),
        ),
        MetamorphicRelation(
            name="trailing_period_invariance",
            transformation=Transformation(
                name="toggle_trailing_period",
                on=on,
                rewrite=_toggle_trailing_period,
                preserves="adds or removes a sentence-final full stop outside the answer",
                domain="answer_text",
            ),
            rung=1,
            justification="a full stop is punctuation around the answer, not part of it",
        ),
    )


def identity_relation(on: str = "response") -> MetamorphicRelation:
    """The declared baseline for D3: the identity transformation.

    It is computed rather than assumed. The identity rewrite appends and then removes a marker, so
    it produces a string equal to its input and the transformation reports itself inapplicable;
    the instrument therefore calls the grader on the *unchanged* rollout twice and compares. Any
    difference is the grader disagreeing with itself.
    """
    return MetamorphicRelation(
        name="identity",
        transformation=Transformation(
            name="identity",
            on=on,
            rewrite=lambda text, rng: text,
            preserves="returns the input unchanged",
        ),
        justification=(
            "the baseline. A violation here is grader nondeterminism, not a relation failure"
        ),
    )


# ---------------------------------------------------------------------------
# What a violation is
# ---------------------------------------------------------------------------


def _holds(kind: str, before: float, after: float, tol: float = 0.0) -> bool:
    if kind == "monotone_nondecreasing":
        return after >= before - tol
    return abs(after - before) <= tol


@register_payload
@dataclass(frozen=True)
class Reproducer:
    """A violation you can run again, holding only what a JSON row can hold.

    Everything needed to reproduce is inline: both input mappings, both scores, the seed the
    transformation was drawn with, and which search produced it. That is what makes it survive the
    store round trip and land in someone else's checkout still working. A reproducer that is a
    pointer into the corpus that produced it stops existing when the corpus moves.
    """

    relation: str
    kind: str
    assertion: str
    preserves: str
    rollout_id: str
    inputs_before: Mapping[str, Any]
    inputs_after: Mapping[str, Any]
    score_before: float
    score_after: float
    seed: int
    source: Literal["census", "hypothesis.find"] = "census"
    shrunk: bool = False

    @property
    def delta(self) -> float:
        return self.score_after - self.score_before

    def rerun(self, grader: Callable[..., float] | VerifierUnderTest) -> tuple[float, float]:
        """Call the grader again on both input mappings and return the two scores."""
        fn, _, _ = resolve_grader(grader)
        return float(fn(**dict(self.inputs_before))), float(fn(**dict(self.inputs_after)))

    def still_violates(self, grader: Callable[..., float] | VerifierUnderTest) -> bool:
        """Whether rerunning reproduces the violation. The test a reproducer has to pass."""
        before, after = self.rerun(grader)
        return not _holds(self.kind, before, after)

    def render(self) -> str:
        tag = " [shrunk]" if self.shrunk else ""
        return (
            f"{self.relation} ({self.kind}: {self.assertion})\n"
            f"    rollout {self.rollout_id or '<unnamed>'}, seed {self.seed}, "
            f"from {self.source}{tag}\n"
            f"    {self.score_before:.6g} -> {self.score_after:.6g} (delta {self.delta:+.6g})\n"
            f"    the transformation preserves: {self.preserves}\n"
            f"    before: {dict(self.inputs_before)!r}\n"
            f"    after:  {dict(self.inputs_after)!r}"
        )


@register_payload
@dataclass(frozen=True)
class RelationResult:
    """One relation's tally: how often it applied, how often it broke, and its shrunk reproducer."""

    relation: str
    kind: str
    rung: int
    checked: int
    applicable: int
    violations: int
    shrunk: Reproducer | None = None
    examples: tuple[Reproducer, ...] = ()

    @property
    def rate(self) -> float:
        """Violations per *applicable* rollout. No applicable rollouts gives NaN, not zero."""
        return self.violations / self.applicable if self.applicable else float("nan")

    @property
    def inapplicable(self) -> bool:
        return self.applicable == 0


@register_payload
@dataclass(frozen=True)
class MetamorphicReport:
    """The reading: a rate, the relations behind it, and reproducers.

    ``violation_rate`` is over *applicable* rollout-relation pairs, never over attempted ones.
    Dividing by attempts would let an inapplicable relation dilute a real violation rate towards
    zero, which is the direction that makes a broken grader look fine.
    """

    grader: str
    fingerprint_basis: str
    relations: tuple[RelationResult, ...]
    rollouts_checked: int
    accepted_rollouts: int
    pairs_applicable: int
    violations: int
    violation_rate: float
    baseline_identity_violations: int
    reproducers: tuple[Reproducer, ...] = ()
    inapplicable_relations: tuple[str, ...] = ()
    accept_threshold: float = 0.0
    tolerance: float = 0.0

    @property
    def shrunk_reproducers(self) -> tuple[Reproducer, ...]:
        return tuple(r.shrunk for r in self.relations if r.shrunk is not None)

    def render(self) -> str:
        lines = [
            f"metamorphic violations on {self.grader}",
            f"    {self.violations} violations over {self.pairs_applicable} applicable "
            f"rollout-relation pairs ({self.violation_rate:.2%})",
            f"    corpus {self.rollouts_checked} rollouts, {self.accepted_rollouts} accepted at "
            f"score > {self.accept_threshold:g}",
            f"    baseline (identity transformation): "
            f"{self.baseline_identity_violations} violations",
        ]
        for r in self.relations:
            if r.inapplicable:
                lines.append(f"    {r.relation:<32} inapplicable: it transformed nothing")
            else:
                mark = "  [shrunk reproducer]" if r.shrunk is not None else ""
                lines.append(
                    f"    {r.relation:<32} {r.violations:>5} / {r.applicable:<5} "
                    f"= {r.rate:.2%}{mark}"
                )
        if self.inapplicable_relations:
            lines.append(
                "    an inapplicable relation is not a pass: "
                + ", ".join(self.inapplicable_relations)
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# The instrument
# ---------------------------------------------------------------------------

#: D3's envelope. Both conditions are about the grader agreeing with itself, because a metamorphic
#: violation is a *difference between two calls* and nothing else. If the grader does not replay to
#: the same score, every difference this instrument finds is flakiness; if the difference is smaller
#: than the grader's disagreement with itself, it is not attributable to the transformation. Access
#: sees neither failure, which is what an envelope is for.
D3_ENVELOPE = EnvelopeSpec(
    requires=frozenset({RegimeCondition.STATIONARY_GRADER, RegimeCondition.ABOVE_LOD}),
    measured_by={
        RegimeCondition.STATIONARY_GRADER: "env.replay_fidelity",
        RegimeCondition.ABOVE_LOD: "substrate.noise_floor",
    },
)


class MetamorphicViolations(BaseObservable):
    """D3. Apply a relation set to a corpus of accepted rollouts and report what broke.

    The grader and the corpus are constructor arguments rather than `Context` fields, because both
    are properties of the measurement rather than of the analyst's session.

    **Kill condition: if violations are under 0.1% on every real verifier, retire the instrument.**
    The relation set would then be too weak to pay for the calls it costs.
    """

    name = "verifier.metamorphic_violations"
    version = "1.0"
    quantity = "verifier.metamorphic_violations"
    capabilities = Capability.SCORES
    requires = {Component.GRADER: Access.QUERY}
    substrates = frozenset({Substrate.PROGRAM, Substrate.PROCEDURAL, Substrate.COMPOSITE})
    phases = frozenset({Phase.PRE_RUN, Phase.POST_RUN})
    envelope = D3_ENVELOPE
    invariance = "trivial"
    invariance_relation = Relation("invariant")
    baselines = ("identity_transformation",)
    rung = 0
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "metamorphic testing (Chen, Cheung and Yiu 1998)"
    deviations = (
        "the system under test is a reward function rather than a program with a specification, "
        "so each relation asserts about a score rather than about an output",
        "shrinking is delegated to hypothesis rather than to a metamorphic-specific reducer, "
        "because no mature general-purpose Python metamorphic library exists",
        "the rate is quoted over applicable rollout-relation pairs, not over attempted ones, so "
        "an inapplicable relation cannot dilute it",
    )

    def __init__(
        self,
        grader: Callable[..., float] | VerifierUnderTest,
        corpus: RolloutCorpus | Sequence[Rollout],
        relations: Sequence[MetamorphicRelation] | None = None,
        *,
        accept_threshold: float = 0.0,
        tolerance: float = 0.0,
        seed: int = 0,
        shrink_examples: int = 200,
        max_reproducers: int = 500,
    ) -> None:
        self.fn, self.subject, self.grader_name = resolve_grader(grader)
        self.corpus = list(corpus)
        self.relations = tuple(relations if relations is not None else python_source_relations())
        self.accept_threshold = accept_threshold
        self.tolerance = tolerance
        self.seed = seed
        self.shrink_examples = shrink_examples
        self.max_reproducers = max_reproducers
        if self.relations:
            self.rung = max(r.rung for r in self.relations)
        self._identity_on = self.relations[0].transformation.on if self.relations else "response"

    # -- preflight ---------------------------------------------------------

    def preflight(self, ctx: Context) -> PreflightResult:
        base = super().preflight(ctx)
        if not base.ok:
            return base
        calls = len(self.corpus) * (2 + 2 * len(self.relations))
        return PreflightResult(
            instrument=self.name,
            ok=True,
            rung=self.rung,
            cost=CostModel(
                calls=calls,
                note=(
                    f"{calls:,} grader calls: one baseline pass, one determinism probe, and two "
                    f"per rollout-relation pair. No GPU and no model."
                ),
            ),
            regime=base.regime,
            unchecked=base.unchecked,
            notes=(
                f"{len(self.relations)} relations at rung {self.rung} over "
                f"{len(self.corpus)} rollouts",
            ),
        )

    # -- the measurement ---------------------------------------------------

    def estimate(self, ctx: Context | None = None) -> Reading:
        ctx = ctx if ctx is not None else Context(signal=self.subject, readout="score")
        pre = self.preflight(ctx)
        if not pre.ok and pre.refusal is not None:
            return pre.refusal
        if not self.corpus:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail="the corpus is empty, so there is nothing to transform",
                remedy=(
                    "supply at least one rollout. A corpus of accepted solutions is the "
                    "denominator this instrument quotes a rate over; without one there is none."
                ),
            )
        return run(self, ctx)

    def measure(self, ctx: Context) -> Any:
        rng = Random(self.seed)
        scores = [score(self.fn, r) for r in self.corpus]

        identity_hits = sum(
            1
            for r, s in zip(self.corpus, scores)
            if not _holds("invariance", s, score(self.fn, r), self.tolerance)
        )
        if identity_hits:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.BELOW_LOD,
                detail=(
                    f"the identity transformation changed the score on {identity_hits} of "
                    f"{len(self.corpus)} rollouts, so the grader disagrees with itself by at least "
                    f"as much as any transformation could move it. Every violation this instrument "
                    f"would report is indistinguishable from that disagreement."
                ),
                remedy=(
                    "make the grader deterministic: fix its seed, pin its model version, or "
                    "disable its sampling step. If it cannot be made deterministic, measure "
                    "`env.flakiness` first and pass `tolerance` above the measured spread, so a "
                    "reported violation means more than noise."
                ),
                statistics={
                    "identity_violations": identity_hits,
                    "rollouts": len(self.corpus),
                    "tolerance": self.tolerance,
                },
            )

        accepted = [r for r, s in zip(self.corpus, scores) if s > self.accept_threshold]
        if not accepted:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail=(
                    f"no rollout scored above the accept threshold {self.accept_threshold:g}; "
                    f"the highest score in the corpus was {max(scores):g}"
                ),
                remedy=(
                    "supply a corpus containing solutions this grader accepts, or lower "
                    "`accept_threshold`. 'Renaming a variable changes the score on N% of accepted "
                    "solutions' has no denominator when nothing was accepted."
                ),
                statistics={"rollouts": len(self.corpus), "max_score": max(scores)},
            )

        results: list[RelationResult] = []
        reproducers: list[Reproducer] = []
        for relation in self.relations:
            result = self._check_relation(relation, accepted, rng)
            results.append(result)
            reproducers.extend(result.examples)

        applicable = sum(r.applicable for r in results)
        violations = sum(r.violations for r in results)
        report = MetamorphicReport(
            grader=self.grader_name,
            fingerprint_basis=getattr(self.subject, "fingerprint_basis", "source"),
            relations=tuple(results),
            rollouts_checked=len(self.corpus),
            accepted_rollouts=len(accepted),
            pairs_applicable=applicable,
            violations=violations,
            violation_rate=(violations / applicable if applicable else float("nan")),
            baseline_identity_violations=identity_hits,
            reproducers=tuple(reproducers[: self.max_reproducers]),
            inapplicable_relations=tuple(r.relation for r in results if r.inapplicable),
            accept_threshold=self.accept_threshold,
            tolerance=self.tolerance,
        )
        return ctx.emit(
            report,
            uncertainty=Uncertainty(n=applicable, method="census over the supplied corpus"),
        )

    # -- one relation ------------------------------------------------------

    def _check_relation(
        self, relation: MetamorphicRelation, accepted: list[Rollout], rng: Random
    ) -> RelationResult:
        checked = 0
        applicable = 0
        examples: list[Reproducer] = []

        for rollout in accepted:
            checked += 1
            seed = rng.randrange(2**32)
            hit = self._evaluate(relation, rollout, seed)
            if hit is None:
                continue
            applicable += 1
            before, after, transformed = hit
            if not _holds(relation.kind, before, after, self.tolerance):
                examples.append(
                    self._reproducer(relation, rollout, transformed, before, after, seed, "census")
                )

        shrunk = self._shrink(relation, accepted) if examples else None
        return RelationResult(
            relation=relation.name,
            kind=relation.kind,
            rung=relation.rung,
            checked=checked,
            applicable=applicable,
            violations=len(examples),
            shrunk=shrunk,
            examples=tuple(examples),
        )

    def _evaluate(
        self, relation: MetamorphicRelation, rollout: Rollout, seed: int
    ) -> tuple[float, float, Rollout] | None:
        """`(score_before, score_after, transformed)`, or None when the transformation was a no-op.

        A no-op is not a pass. Returning None keeps it out of the denominator, which is what stops
        a relation that does not apply to this corpus from diluting the rate towards zero.
        """
        if relation.kind == "idempotence":
            once = relation.transformation.apply(rollout, Random(seed))
            twice = relation.transformation.apply(once, Random(seed))
            if twice.inputs == once.inputs:
                return None
            return score(self.fn, once), score(self.fn, twice), twice
        transformed = relation.transformation.apply(rollout, Random(seed))
        if transformed.inputs == rollout.inputs:
            return None
        return score(self.fn, rollout), score(self.fn, transformed), transformed

    def _reproducer(
        self,
        relation: MetamorphicRelation,
        rollout: Rollout,
        transformed: Rollout,
        before: float,
        after: float,
        seed: int,
        source: Literal["census", "hypothesis.find"],
    ) -> Reproducer:
        return Reproducer(
            relation=relation.name,
            kind=relation.kind,
            assertion=relation.assertion,
            preserves=relation.transformation.preserves,
            rollout_id=rollout.id,
            inputs_before=dict(rollout.inputs),
            inputs_after=dict(transformed.inputs),
            score_before=before,
            score_after=after,
            seed=seed,
            source=source,
            shrunk=source == "hypothesis.find",
        )

    def _shrink(self, relation: MetamorphicRelation, accepted: list[Rollout]) -> Reproducer | None:
        """The smallest `(rollout, seed)` that breaks this relation, via `hypothesis.find`.

        `find` returns the shrunk falsifying example as a Python object, here an `(int, int)`
        tuple, not a repr to parse back. The strategy is an index and a seed rather than a
        synthesised input, because the corpus is the population the rate is quoted over and a
        reproducer built from a manufactured response would not belong to it.

        Returns None when the search finds nothing even though the census did, which is possible:
        `find` samples the strategy rather than enumerating it, so a violation on one rollout in
        ten thousand can be missed inside `shrink_examples` draws. None is the honest answer there,
        and the census reproducers are still on the result.
        """
        require_extra("verifier", subsystem="D3 (verifier.metamorphic_violations)")
        from hypothesis import HealthCheck, find, settings
        from hypothesis import strategies as st
        from hypothesis.errors import NoSuchExample, Unsatisfiable

        def violates(pair: tuple[int, int]) -> bool:
            index, seed = pair
            hit = self._evaluate(relation, accepted[index], seed)
            if hit is None:
                return False
            before, after, _ = hit
            return not _holds(relation.kind, before, after, self.tolerance)

        strategy = st.tuples(st.sampled_from(range(len(accepted))), st.integers(0, 2**32 - 1))
        try:
            index, seed = find(
                strategy,
                violates,
                # `find` draws from global entropy unless it is handed a generator, so without
                # this the shrunk reproducer appears on one run and not the next.
                random=Random(self.seed),
                settings=settings(
                    max_examples=self.shrink_examples,
                    database=None,
                    deadline=None,
                    suppress_health_check=list(HealthCheck),
                ),
            )
        except (NoSuchExample, Unsatisfiable):
            return None

        rollout = accepted[index]
        hit = self._evaluate(relation, rollout, seed)
        if hit is None:  # pragma: no cover - find only returns satisfying examples
            return None
        before, after, transformed = hit
        return self._reproducer(
            relation, rollout, transformed, before, after, seed, "hypothesis.find"
        )


def metamorphic_violations(
    grader: Callable[..., float] | VerifierUnderTest,
    corpus: RolloutCorpus | Sequence[Rollout],
    relations: Sequence[MetamorphicRelation] | None = None,
    *,
    ctx: Context | None = None,
    **kwargs: Any,
) -> Reading:
    """Run D3 and return the Reading. The one-call form, for a card renderer."""
    return MetamorphicViolations(grader, corpus, relations, **kwargs).estimate(ctx)


__all__ = [
    "D3_ENVELOPE",
    "KIND_ASSERTION",
    "MetamorphicRelation",
    "MetamorphicReport",
    "MetamorphicViolations",
    "QuerySubject",
    "RelationKind",
    "RelationResult",
    "Reproducer",
    "Grader",
    "Transformation",
    "answer_text_relations",
    "identity_relation",
    "metamorphic_violations",
    "looks_like_source",
    "python_source_relations",
    "resolve_grader",
    "score",
]

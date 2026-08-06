"""D9 Static structure: the verifier's shape, extracted (`verifier.cfg`, `verifier.trusted_inputs`).

Says: "The scoring function has 14 branches, 3 of them reachable only when a timeout fires, and it
trusts 2 fields of untrusted input without validation."

D9 has no kill condition of its own because it makes no claim of its own. What it does is feed D1
and D2, and the thing it feeds them is the part nobody else builds. `coverage.py` will tell you
that arc `(147, 152)` was never taken. That is not a finding, it is a coordinate. D9 is the module
that turns it into one: `PositionProvider` maps the arc's source line onto the rubric clause whose
span contains it, and the sentence becomes "the third rubric criterion has never fired." **If that
mapping is not built, D1 reports numbers nobody can act on**, which is the failure mode to watch
for here. A coverage report on a grader that nobody reads is the same non-event as a coverage
report on an application, and the reason this series is empty is that everyone stopped at the
report.

Three rungs, as the catalogue specifies. Rung 0 is an `ast` walk: functions, branch count,
parameters, module-level statements. Rung 1 adds the control-flow graph, taken from `coverage.py`'s
own static analysis rather than from a hand-rolled one, because D1's measured arcs come from
`coverage.py`'s tracer and a denominator computed in a different coordinate system would not line
up with them. Rung 2 adds dataflow: `libcst`'s `ScopeProvider` resolves which local names alias the
entrypoint's parameters, so a field read through an alias is still a field read of untrusted input.

**What "trusted without validation" means here, stated so nobody over-reads it.** A field access is
called validated when a check on that field, or on its container, dominates the access in the
control-flow graph, or when the access is lexically inside a `try` with a handler that would catch
its failure, or when it is a `.get()` with a default. Dominance is a structural relation and it
cannot see whether the check is *correct*: a grader that validates a field against the wrong
predicate reads as validated here. The number is a floor on the problem, not a measurement of it.
"""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

from reward_lens.core.envelope import EnvelopeSpec
from reward_lens.core.evidence import register_payload
from reward_lens.core.invariance import Relation
from reward_lens.core.quantity import (
    BiasStatement,
    CostModel,
    EstimatorEntry,
    register_estimator,
)
from reward_lens.core.types import Capability, GaugeStatus, Phase, Substrate
from reward_lens.measure.base import BaseObservable, Context
from reward_lens.verifier import (
    ACCESS_SOURCE_ONLY,
    VerifierUnderTest,
    ensure_quantities,
)

if TYPE_CHECKING:
    from reward_lens.core.evidence import Evidence

#: A control-flow edge as `coverage.py` numbers them: a pair of line numbers, where a negative
#: number is a scope sentinel. `-n` is the enter/exit sentinel of the scope whose header is at line
#: `n`, and `-1` doubles as the module's own sentinel, so a file whose first line is a `def` has
#: one number meaning two things. Nothing here depends on telling those two apart.
Arc = tuple[int, int]

#: Names that mark a branch as reachable only under a timeout. The spec's example sentence counts
#: exactly these. Override through `analyse_source(guards=...)` to count a different gate.
TIMEOUT_GUARDS: tuple[str, ...] = (
    "timeout",
    "timed_out",
    "timeouterror",
    "timeoutexpired",
    "deadline",
    "time_limit",
)

_RUBRIC_MARKER = re.compile(r"#\s*rubric\s*:\s*(?P<name>\S.*?)\s*$")

#: Exception types whose handler counts as validating a field read inside the `try`.
_VALIDATING_HANDLERS = frozenset(
    {"KeyError", "IndexError", "TypeError", "ValueError", "AttributeError", "Exception"}
)


# ---------------------------------------------------------------------------
# Payloads
# ---------------------------------------------------------------------------


@register_payload
@dataclass(frozen=True)
class RubricClause:
    """One named region of the verifier's logic, with the arc that means "it fired".

    ``enter_arc`` is what makes D1 legible. A clause has never fired when that arc was never taken,
    which is a different and much stronger statement than "some line in this region is uncovered":
    the region can be entirely covered by the path that skips it.
    """

    name: str
    kind: str
    function: str
    ordinal: int
    start_line: int
    start_col: int
    end_line: int
    end_col: int
    test_line: int | None = None
    enter_arc: list[int] | None = None
    named_by_marker: bool = False

    @property
    def span(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return ((self.start_line, self.start_col), (self.end_line, self.end_col))

    def contains(self, line: int) -> bool:
        return self.start_line <= line <= self.end_line

    def render(self) -> str:
        marker = "" if self.named_by_marker else f" (clause {self.ordinal} of {self.function})"
        return f"{self.name}{marker} at lines {self.start_line}-{self.end_line}"


@register_payload
@dataclass(frozen=True)
class TrustedInput:
    """One field of untrusted input read without a dominating check."""

    base: str
    field_name: str
    kind: str
    line: int
    col: int
    via_alias: str | None = None
    validated: bool = False
    validated_by: str = ""

    def render(self) -> str:
        alias = f" (through {self.via_alias})" if self.via_alias else ""
        state = f"validated by {self.validated_by}" if self.validated else "TRUSTED"
        return f"{self.base}.{self.field_name}{alias} at line {self.line}: {state}"


@register_payload
@dataclass(frozen=True)
class VerifierStructure:
    """The verifier's shape. The value of `verifier.cfg`.

    Everything a reader needs to turn a coverage arc or a mutation span into a sentence about the
    rubric, plus the two counts the spec's example sentence quotes: how many branches the scoring
    function has, and how many of them are reachable only when a named guard fires.
    """

    source_path: str
    entrypoint: str
    fingerprint: str
    rung: int
    functions: tuple[str, ...]
    statements: int
    branch_lines: tuple[int, ...]
    branches: int
    arcs: tuple[list[int], ...]
    guard_gated_arcs: tuple[list[int], ...]
    guard_lines: tuple[int, ...]
    clauses: tuple[RubricClause, ...]
    arc_clause: Mapping[str, str] = field(default_factory=dict)
    module_level_statements: int = 0
    parameters: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def clause_for(self, arc: Arc) -> RubricClause | None:
        """The clause a coverage arc belongs to, or None if it falls outside every clause."""
        name = self.arc_clause.get(arc_key(arc))
        if name is None:
            return None
        for c in self.clauses:
            if c.name == name:
                return c
        return None

    def render(self) -> str:
        gated = len(self.guard_gated_arcs)
        head = (
            f"{self.entrypoint} in {Path(self.source_path).name}: {self.branches} branches over "
            f"{len(self.branch_lines)} decision points"
        )
        if gated:
            head += f", {gated} of them reachable only past a guard"
        lines = [head, f"    {len(self.clauses)} rubric clauses"]
        lines += [f"      {c.render()}" for c in self.clauses]
        return "\n".join(lines)


@register_payload
@dataclass(frozen=True)
class TrustedInputReport:
    """Fields of untrusted input the verifier reads. The value of `verifier.trusted_inputs`."""

    source_path: str
    entrypoint: str
    rung: int
    parameters: tuple[str, ...]
    accesses: tuple[TrustedInput, ...]
    unresolved_names: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def trusted(self) -> tuple[TrustedInput, ...]:
        return tuple(a for a in self.accesses if not a.validated)

    @property
    def n_trusted(self) -> int:
        return len(self.trusted)

    def render(self) -> str:
        lines = [
            f"{self.entrypoint} trusts {self.n_trusted} of {len(self.accesses)} field reads "
            f"of untrusted input without validation"
        ]
        lines += [f"    {a.render()}" for a in self.trusted]
        return "\n".join(lines)


def arc_key(arc: Arc | Sequence[int]) -> str:
    """The string key an arc gets in a stored mapping. Tuple keys do not survive JSON."""
    return f"{arc[0]}->{arc[1]}"


# ---------------------------------------------------------------------------
# The control-flow graph
# ---------------------------------------------------------------------------


def control_flow(path: Path) -> tuple[frozenset[Arc], dict[int, int], frozenset[int]]:
    """The arc universe, the exit count per line, and the statement lines.

    Delegated to `coverage.py`'s own file reporter rather than built here. That is not laziness: D1
    compares *measured* arcs against this set, the measured arcs come from `coverage.py`'s tracer,
    and two static analyses of the same file agreeing on line numbers but disagreeing on which
    edges exist would produce a coverage fraction that is wrong in a way nothing detects.
    """
    import coverage
    from coverage.python import PythonFileReporter

    cov = coverage.Coverage(branch=True, data_file=None)
    reporter = PythonFileReporter(str(path), cov)
    arcs = frozenset(reporter.arcs())
    exits = dict(reporter.exit_counts())
    statements = frozenset(reporter.lines())
    return arcs, exits, statements


def dominators(arcs: Iterable[Arc], roots: Iterable[int]) -> dict[int, frozenset[int]]:
    """Which nodes every path to a node must pass through.

    The textbook iterative fixed point. It is here rather than pulled from a library because the
    graph is one file's arcs, the whole computation is a few dozen lines, and every CFG package on
    PyPI that would supply it also supplies its own parser and its own line numbering, which is the
    thing `control_flow` above exists to avoid.
    """
    arcs = list(arcs)
    roots = set(roots)
    nodes = {a for a, _ in arcs} | {b for _, b in arcs} | roots
    preds: dict[int, set[int]] = defaultdict(set)
    for a, b in arcs:
        preds[b].add(a)

    dom: dict[int, frozenset[int]] = {
        n: (frozenset({n}) if n in roots else frozenset(nodes)) for n in nodes
    }
    changed = True
    while changed:
        changed = False
        for n in sorted(nodes):
            if n in roots:
                continue
            incoming = [dom[p] for p in preds.get(n, ()) if p in dom]
            new = (frozenset.intersection(*incoming) | {n}) if incoming else frozenset({n})
            if new != dom[n]:
                dom[n] = new
                changed = True
    return dom


def _roots(arcs: Iterable[Arc]) -> frozenset[int]:
    """Nodes nothing flows into, plus every scope sentinel. Where control can start."""
    arcs = list(arcs)
    targets = {b for _, b in arcs}
    sources = {a for a, _ in arcs}
    return frozenset({n for n in sources if n not in targets} | {n for n in sources if n < 0})


def _guard_lines(
    source_lines: Sequence[str], exits: Mapping[int, int], guards: Sequence[str]
) -> frozenset[int]:
    """Decision points whose own text names one of the guard tokens.

    Matched on the source text of the branching line, lowercased. A branch on `if elapsed >
    timeout:` counts; a branch three lines below it that happens to mention the word does not,
    because it is not the decision.
    """
    tokens = tuple(g.lower() for g in guards)
    out = set()
    for line, count in exits.items():
        if count <= 1 or line < 1 or line > len(source_lines):
            continue
        text = source_lines[line - 1].lower()
        if any(t in text for t in tokens):
            out.add(line)
    # `except TimeoutError:` is a guard even though it is not a multi-exit line.
    for i, raw in enumerate(source_lines, start=1):
        text = raw.lower()
        if text.lstrip().startswith("except") and any(t in text for t in tokens):
            out.add(i)
    return frozenset(out)


def guard_gated_arcs(
    arcs: Iterable[Arc], guard_lines: Iterable[int], roots: Iterable[int]
) -> tuple[Arc, ...]:
    """Arcs no path reaches without first passing a guard.

    This is what makes "3 of them reachable only when a timeout fires" a measurement rather than a
    reading of the code. An arc `(a, b)` is gated when some guard line dominates `a`, so every
    execution that takes the arc went through the guard.
    """
    guards = set(guard_lines)
    if not guards:
        return ()
    dom = dominators(arcs, roots)
    out = []
    for a, b in sorted(arcs):
        gating = dom.get(a, frozenset()) & guards
        if gating and a not in guards:
            out.append((a, b))
        elif a in guards and b not in guards:
            # The arc out of the guard itself into the guarded side. Whether it is gated depends on
            # which side it lands on, and dominance on the head answers that.
            if guards & (dom.get(b, frozenset()) - {b}):
                out.append((a, b))
    return tuple(out)


# ---------------------------------------------------------------------------
# Rubric clauses
# ---------------------------------------------------------------------------


def _marker_names(source: str) -> dict[int, tuple[str, bool]]:
    """Lines carrying a `# rubric: NAME` marker, mapped to the name and whether it stands alone.

    The distinction matters. A marker on its own line names the statement below it, which is the
    natural way to label a block. A marker trailing a line of code names *that* line and nothing
    else, and letting it also name the line below would give a guard's marker to the first
    statement inside the guard, which is how the timeout branch's inner test ended up called
    `timed_out` the first time this ran.
    """
    out: dict[int, tuple[str, bool]] = {}
    for i, line in enumerate(source.splitlines(), start=1):
        m = _RUBRIC_MARKER.search(line)
        if m:
            out[i] = (m.group("name"), line.lstrip().startswith("#"))
    return out


def extract_clauses(source: str, entrypoint: str | None = None) -> tuple[RubricClause, ...]:
    """The verifier's named regions, in source order, with their spans.

    `libcst`'s `PositionProvider` supplies the spans, which is the whole reason this uses `libcst`
    rather than `ast`: `ast` gives line and column on the node but loses the exact end of a
    compound statement's body in the cases that matter, and the span is what the arc map is keyed
    on. Note that `MetadataWrapper.visit` is required here; `Module.visit` raises, because the
    visitor's metadata is resolved by the wrapper and not by the module.

    A clause is either a region a `# rubric: NAME` comment names, or an `if` test inside the
    entrypoint, numbered in source order. The comment wins where both apply, because a name a human
    wrote is worth more than an ordinal.
    """
    import libcst as cst
    from libcst.metadata import MetadataWrapper, PositionProvider

    markers = _marker_names(source)
    module = cst.parse_module(source)
    wrapper = MetadataWrapper(module)

    class _Collector(cst.CSTVisitor):
        METADATA_DEPENDENCIES = (PositionProvider,)

        def __init__(self) -> None:
            self.function_stack: list[str] = []
            self.ordinals: dict[str, int] = defaultdict(int)
            self.clauses: list[RubricClause] = []

        def visit_FunctionDef(self, node: Any) -> bool:
            self.function_stack.append(node.name.value)
            return True

        def leave_FunctionDef(self, node: Any) -> None:
            self.function_stack.pop()

        def visit_If(self, node: Any) -> bool:
            fn = self.function_stack[-1] if self.function_stack else "<module>"
            if entrypoint is not None and fn != entrypoint:
                return True
            pos = self.get_metadata(PositionProvider, node)
            body_pos = self.get_metadata(PositionProvider, node.body)
            test_pos = self.get_metadata(PositionProvider, node.test)
            self.ordinals[fn] += 1
            ordinal = self.ordinals[fn]
            here = markers.get(pos.start.line)
            above = markers.get(pos.start.line - 1)
            name = here[0] if here else (above[0] if above and above[1] else None)
            self.clauses.append(
                RubricClause(
                    name=name or f"{fn}.clause{ordinal}",
                    kind="if",
                    function=fn,
                    ordinal=ordinal,
                    start_line=pos.start.line,
                    start_col=pos.start.column,
                    end_line=body_pos.end.line,
                    end_col=body_pos.end.column,
                    test_line=test_pos.start.line,
                    named_by_marker=name is not None,
                )
            )
            return True

    collector = _Collector()
    wrapper.visit(collector)
    return tuple(sorted(collector.clauses, key=lambda c: (c.start_line, c.start_col)))


def _with_enter_arcs(
    clauses: Sequence[RubricClause], arcs: Iterable[Arc]
) -> tuple[RubricClause, ...]:
    """Attach to each clause the arc that means "this clause fired"."""
    arcs = sorted(arcs)
    out = []
    for c in clauses:
        enter = None
        if c.test_line is not None:
            candidates = [
                (a, b) for a, b in arcs if a == c.test_line and c.start_line < b <= c.end_line
            ]
            if candidates:
                enter = list(min(candidates))
        out.append(RubricClause(**{**vars(c), "enter_arc": enter}))
    return tuple(out)


def map_arcs_to_clauses(arcs: Iterable[Arc], clauses: Sequence[RubricClause]) -> dict[str, str]:
    """Which clause each arc belongs to, keyed by `"a->b"`.

    Attribution is on the arc's *source* line, because that is where the decision was taken. The
    innermost containing clause wins, so a nested criterion is not swallowed by the one around it.
    An arc outside every clause has no entry rather than a made-up one.
    """
    out: dict[str, str] = {}
    for a, b in arcs:
        line = -a if a < 0 else a
        containing = [c for c in clauses if c.contains(line)]
        if not containing:
            continue
        innermost = min(containing, key=lambda c: (c.end_line - c.start_line, c.start_line))
        out[arc_key((a, b))] = innermost.name
    return out


# ---------------------------------------------------------------------------
# Trusted inputs
# ---------------------------------------------------------------------------


@dataclass
class _AccessScan(ast.NodeVisitor):
    """Field reads on the entrypoint's parameters, and the checks that guard them."""

    params: frozenset[str]
    aliases: Mapping[str, str]
    accesses: list[TrustedInput] = field(default_factory=list)
    guards: list[tuple[str, str | None, int, str]] = field(default_factory=list)
    try_spans: list[tuple[int, int, frozenset[str]]] = field(default_factory=list)
    _test_lines: set[int] = field(default_factory=set)
    _get_lines: set[int] = field(default_factory=set)
    #: `id()` of every Attribute node that is the callee of a Call. `payload.get(...)` and
    #: `text.strip()` are method calls, not field reads, and counting them would put every string
    #: method a grader uses into the trusted-input list.
    _call_funcs: set[int] = field(default_factory=set)

    def base_of(self, node: ast.AST) -> tuple[str, str | None]:
        """The parameter a node reads, and the alias it went through."""
        if isinstance(node, ast.Name):
            if node.id in self.params:
                return node.id, None
            root = self.aliases.get(node.id)
            if root is not None:
                return root, node.id
        return "", None

    def visit_Try(self, node: ast.Try) -> None:
        handled = set()
        for h in node.handlers:
            for name in _handler_names(h.type):
                handled.add(name)
        end = max(
            [getattr(s, "end_lineno", s.lineno) or s.lineno for s in node.body] or [node.lineno]
        )
        self.try_spans.append((node.lineno, end, frozenset(handled)))
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        for sub in ast.walk(node.test):
            self._test_lines.add(getattr(sub, "lineno", node.lineno))
        self._record_test(node.test, node.lineno)
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        self._record_test(node.test, node.lineno)
        self.generic_visit(node)

    def _record_test(self, test: ast.AST, line: int) -> None:
        for sub in ast.walk(test):
            if isinstance(sub, ast.Compare) and isinstance(sub.ops[0], (ast.In, ast.NotIn)):
                base, _ = self.base_of(sub.comparators[0])
                if base:
                    self.guards.append((base, _literal(sub.left), line, "membership test"))
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                if sub.func.id in ("isinstance", "hasattr", "len", "callable") and sub.args:
                    base, key = _field_of(sub.args[0], self)
                    if base:
                        self.guards.append((base, key, line, f"{sub.func.id}()"))
            base, key = _field_of(sub, self)
            if base:
                self.guards.append((base, key, line, "checked in a test"))

    def visit_Subscript(self, node: ast.Subscript) -> None:
        base, alias = self.base_of(node.value)
        if base:
            key = _literal(node.slice)
            if key is not None:
                self.accesses.append(
                    TrustedInput(
                        base=base,
                        field_name=key,
                        kind="subscript",
                        line=node.lineno,
                        col=node.col_offset,
                        via_alias=alias,
                    )
                )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        base, alias = self.base_of(node.value)
        if base and id(node) not in self._call_funcs:
            self.accesses.append(
                TrustedInput(
                    base=base,
                    field_name=node.attr,
                    kind="attribute",
                    line=node.lineno,
                    col=node.col_offset,
                    via_alias=alias,
                )
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # Recorded before descending, so `visit_Attribute` sees this callee already marked.
        if isinstance(node.func, ast.Attribute):
            self._call_funcs.add(id(node.func))
        # `payload.get("k", default)` reads the field and supplies the missing case in one move.
        # `payload.get("k")` reads it and hands back a None the grader still has to handle, so it
        # is a read without a check and is counted as one.
        if isinstance(node.func, ast.Attribute) and node.func.attr == "get" and node.args:
            base, alias = self.base_of(node.func.value)
            key = _literal(node.args[0])
            if base and key is not None:
                has_default = len(node.args) > 1 or bool(node.keywords)
                self._get_lines.add(node.lineno)
                if has_default:
                    self.guards.append((base, key, node.lineno, ".get() with a default"))
                self.accesses.append(
                    TrustedInput(
                        base=base,
                        field_name=key,
                        kind="get",
                        line=node.lineno,
                        col=node.col_offset,
                        via_alias=alias,
                        validated=has_default,
                        validated_by=".get() with a default" if has_default else "",
                    )
                )
        self.generic_visit(node)


def _handler_names(node: ast.AST | None) -> tuple[str, ...]:
    if node is None:
        return ("Exception",)
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Tuple):
        return tuple(n.id for n in node.elts if isinstance(n, ast.Name))
    if isinstance(node, ast.Attribute):
        return (node.attr,)
    return ()


def _literal(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, int)):
        return str(node.value)
    return None


def _field_of(node: ast.AST, scan: "_AccessScan") -> tuple[str, str | None]:
    """The `(parameter, field)` a node reads, if it reads one."""
    if isinstance(node, ast.Subscript):
        base, _ = scan.base_of(node.value)
        return (base, _literal(node.slice)) if base else ("", None)
    if isinstance(node, ast.Attribute):
        base, _ = scan.base_of(node.value)
        return (base, node.attr) if base else ("", None)
    if isinstance(node, ast.Name):
        base, _ = scan.base_of(node)
        return (base, None) if base else ("", None)
    return ("", None)


def _resolve_aliases(source: str, entrypoint: str, notes: list[str]) -> dict[str, str]:
    """Local names that hold a parameter, resolved with `libcst`'s `ScopeProvider`.

    This is rung 2's contribution: without it, `data = payload; data["answer"]` reads as a field
    access on a local, and the verifier looks like it touches no untrusted input at all.

    `ScopeProvider` raises `KeyError` on import aliases in 1.9.0. That is caught here for the one
    lookup that provokes it and the unresolved name is recorded, which is different from swallowing
    it: the report says which names it could not resolve, so a reader can see the analysis's edge
    rather than inferring a clean result.
    """
    import libcst as cst
    from libcst.metadata import MetadataWrapper, ScopeProvider

    module = cst.parse_module(source)
    wrapper = MetadataWrapper(module)

    class _Aliases(cst.CSTVisitor):
        METADATA_DEPENDENCIES = (ScopeProvider,)

        def __init__(self) -> None:
            self.in_target = False
            self.params: set[str] = set()
            self.alias: dict[str, str] = {}

        def visit_FunctionDef(self, node: Any) -> bool:
            if node.name.value != entrypoint:
                return False
            self.in_target = True
            for p in node.params.params:
                self.params.add(p.name.value)
            return True

        def leave_FunctionDef(self, node: Any) -> None:
            if node.name.value == entrypoint:
                self.in_target = False

        def visit_Assign(self, node: Any) -> bool:
            if not self.in_target or len(node.targets) != 1:
                return True
            target = node.targets[0].target
            if not isinstance(target, cst.Name):
                return True
            root = self._root(node.value)
            if root is not None:
                self.alias[target.value] = root
            return True

        def _root(self, value: Any) -> str | None:
            node = value
            while isinstance(node, (cst.Subscript, cst.Attribute)):
                node = node.value
            if not isinstance(node, cst.Name):
                return None
            if node.value in self.params:
                return node.value
            if node.value in self.alias:
                return self.alias[node.value]
            try:
                self.get_metadata(ScopeProvider, node)
            except KeyError:
                notes.append(f"ScopeProvider could not resolve {node.value!r}")
            return None

    collector = _Aliases()
    wrapper.visit(collector)
    return collector.alias


def analyse_trusted_inputs(
    source: str,
    entrypoint: str,
    *,
    rung: int,
    arcs: Iterable[Arc],
    roots: Iterable[int],
    source_path: str,
) -> TrustedInputReport:
    """Which fields of untrusted input the entrypoint reads without a dominating check."""
    tree = ast.parse(source)
    fn = _find_function(tree, entrypoint)
    if fn is None:
        return TrustedInputReport(
            source_path=source_path,
            entrypoint=entrypoint,
            rung=rung,
            parameters=(),
            accesses=(),
            notes=(f"no function named {entrypoint!r} in {source_path}",),
        )
    params = frozenset(
        [a.arg for a in fn.args.args]
        + [a.arg for a in fn.args.kwonlyargs]
        + ([fn.args.vararg.arg] if fn.args.vararg else [])
        + ([fn.args.kwarg.arg] if fn.args.kwarg else [])
    )
    notes: list[str] = []
    aliases = _resolve_aliases(source, entrypoint, notes) if rung >= 2 else {}
    scan = _AccessScan(params=params, aliases=aliases)
    scan.visit(fn)

    dom = dominators(arcs, roots) if rung >= 1 else {}
    resolved = []
    for access in scan.accesses:
        if access.validated:
            resolved.append(access)
            continue
        why = _validation_for(access, scan, dom)
        resolved.append(
            TrustedInput(**{**vars(access), "validated": bool(why), "validated_by": why})
        )

    if rung < 2 and aliases == {}:
        notes.append("rung 0/1: field reads through a local alias are not resolved; rung 2 does")
    return TrustedInputReport(
        source_path=source_path,
        entrypoint=entrypoint,
        rung=rung,
        parameters=tuple(sorted(params)),
        accesses=tuple(resolved),
        unresolved_names=tuple(sorted({n.split("'")[1] for n in notes if "'" in n})),
        notes=tuple(notes),
    )


def _validation_for(
    access: TrustedInput, scan: "_AccessScan", dom: Mapping[int, frozenset[int]]
) -> str:
    """The check that validates this access, or an empty string if none does."""
    for start, end, handlers in scan.try_spans:
        if start <= access.line <= end and handlers & _VALIDATING_HANDLERS:
            return f"try/except {'/'.join(sorted(handlers & _VALIDATING_HANDLERS))}"
    dominating = dom.get(access.line, frozenset({access.line}))
    for base, key, line, kind in scan.guards:
        if base != access.base:
            continue
        if key is not None and key != access.field_name:
            continue
        if line == access.line or line in dominating:
            return kind
    return ""


def _find_function(tree: ast.AST, name: str) -> ast.FunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node  # type: ignore[return-value]
    return None


# ---------------------------------------------------------------------------
# The analysis
# ---------------------------------------------------------------------------


def analyse_source(
    verifier: VerifierUnderTest,
    *,
    rung: int = 1,
    guards: Sequence[str] = TIMEOUT_GUARDS,
) -> tuple[VerifierStructure, TrustedInputReport]:
    """D9's two readings from one pass over the source.

    Rung 0 is the `ast` walk and stops before the control-flow graph, so `arcs`, the guard
    accounting and the arc-to-clause map are all empty at that rung and the structure says so
    rather than reporting zeros that look like findings.
    """
    source = verifier.source()
    path = verifier.source_path
    tree = ast.parse(source)
    functions = tuple(
        n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    module_level = sum(
        1
        for n in tree.body  # type: ignore[attr-defined]
        if not isinstance(
            n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)
        )
    )
    entry = _find_function(tree, verifier.entrypoint)
    parameters = tuple(a.arg for a in entry.args.args) if entry is not None else ()

    notes: list[str] = []
    if rung <= 0:
        n_branches = sum(1 for n in ast.walk(tree) if isinstance(n, (ast.If, ast.While, ast.IfExp)))
        notes.append("rung 0: an ast walk. No control-flow graph, so no arcs and no clause map.")
        structure = VerifierStructure(
            source_path=str(path),
            entrypoint=verifier.entrypoint,
            fingerprint=verifier.fingerprint,
            rung=0,
            functions=functions,
            statements=sum(1 for n in ast.walk(tree) if isinstance(n, ast.stmt)),
            branch_lines=(),
            branches=n_branches,
            arcs=(),
            guard_gated_arcs=(),
            guard_lines=(),
            clauses=(),
            module_level_statements=module_level,
            parameters=parameters,
            notes=tuple(notes),
        )
        trusted = analyse_trusted_inputs(
            source, verifier.entrypoint, rung=0, arcs=(), roots=(), source_path=str(path)
        )
        return structure, trusted

    arcs, exits, statements = control_flow(path)
    roots = _roots(arcs)
    source_lines = source.splitlines()
    guard_lines = _guard_lines(source_lines, exits, guards)
    gated = guard_gated_arcs(arcs, guard_lines, roots)
    # Every function, not only the entrypoint. A grader's parse layer and normalisation layer are
    # part of its rubric even though they are separate functions, and restricting extraction to the
    # entrypoint left the uncovered branches in them as bare coordinates, which is exactly the
    # failure D9 exists to prevent. Their clauses are named `parse.clause1` and so on, so the layer
    # they belong to is in the name.
    clauses = _with_enter_arcs(extract_clauses(source), arcs)
    branch_lines = tuple(sorted(line for line, count in exits.items() if count > 1))
    structure = VerifierStructure(
        source_path=str(path),
        entrypoint=verifier.entrypoint,
        fingerprint=verifier.fingerprint,
        rung=rung,
        functions=functions,
        statements=len(statements),
        branch_lines=branch_lines,
        branches=sum(count for count in exits.values() if count > 1),
        arcs=tuple(sorted([list(a) for a in arcs])),
        guard_gated_arcs=tuple([list(a) for a in gated]),
        guard_lines=tuple(sorted(guard_lines)),
        clauses=clauses,
        arc_clause=map_arcs_to_clauses(arcs, clauses),
        module_level_statements=module_level,
        parameters=parameters,
        notes=tuple(notes),
    )
    trusted = analyse_trusted_inputs(
        source,
        verifier.entrypoint,
        rung=rung,
        arcs=arcs,
        roots=roots,
        source_path=str(path),
    )
    return structure, trusted


# ---------------------------------------------------------------------------
# The instruments
# ---------------------------------------------------------------------------

#: D9 reads a file and asserts nothing about the process that produced it, so no measured regime
#: can make the count wrong. That is the one honest use of `unconditional`.
_STATIC_ENVELOPE = EnvelopeSpec(
    unconditional=True,
    justification=(
        "a census over one file's source text at one content hash. It makes no claim about the "
        "run, the corpus or the policy, so no regime condition can invalidate it. The hash is "
        "carried on the reading, which is what lets D1 and D2 check that they measured this same "
        "program."
    ),
)

_SOURCE_ACCESS_DEVIATION = (
    "reads the grader's source and never modifies it; declared `Access.SOURCE`, added for the D "
    "series after this deviation was recorded. "
    "That is strictly stronger than what this instrument consumes, which reads the source and "
    "never modifies it."
)


class StaticStructure(BaseObservable):
    """D9 `verifier.cfg`: the verifier's control-flow graph and its rubric clauses.

    Kill condition: none, and that is a statement about what this instrument is rather than a gap.
    It makes no claim to be wrong about. What would make it worthless is the arc-to-clause map
    coming back empty on real graders, because then D1's uncovered arcs stay coordinates and the
    series produces reports instead of findings.
    """

    name = "StaticStructure"
    version = "1.0"
    quantity = "verifier.cfg"
    capabilities = Capability.SCORES
    requires = ACCESS_SOURCE_ONLY
    substrates = frozenset({Substrate.PROGRAM})
    phases = frozenset({Phase.PRE_RUN, Phase.IN_RUN, Phase.POST_RUN, Phase.DEPLOYED})
    envelope = _STATIC_ENVELOPE
    invariance = "none"
    invariance_relation = Relation("invariant")
    baselines = (
        "the module's own ast statement count, which no control-flow analysis is used to produce",
    )
    rung = 1
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = None
    deviations = (_SOURCE_ACCESS_DEVIATION,)

    def __init__(
        self,
        verifier: VerifierUnderTest | None = None,
        *,
        rung: int = 1,
        guards: Sequence[str] = TIMEOUT_GUARDS,
    ) -> None:
        ensure_quantities()
        self.verifier = verifier
        self.rung = rung
        self.guards = tuple(guards)

    def measure(self, ctx: Context) -> "Evidence":
        verifier = self.verifier or ctx.signal.verifier  # type: ignore[union-attr]  # a ProgramSubject, not a network
        structure, _ = analyse_source(verifier, rung=self.rung, guards=self.guards)
        tree = ast.parse(verifier.source())
        baseline = float(sum(1 for n in ast.walk(tree) if isinstance(n, ast.stmt)))
        return ctx.emit(
            structure,
            subject_extra={"baseline_ast_statements": str(int(baseline))},
        )


class TrustedInputs(BaseObservable):
    """D9 `verifier.trusted_inputs`: fields of untrusted input read without a dominating check.

    Kill condition: none of its own. The number to distrust is a zero on a grader that parses a
    model's free-text output, because a parser that touches no untrusted field is a parser that
    was not found. When this returns zero, check `parameters` on the reading first: an empty
    parameter list means the entrypoint name is wrong, not that the grader is clean.
    """

    name = "TrustedInputs"
    version = "1.0"
    quantity = "verifier.trusted_inputs"
    capabilities = Capability.SCORES
    requires = ACCESS_SOURCE_ONLY
    substrates = frozenset({Substrate.PROGRAM})
    phases = frozenset({Phase.PRE_RUN, Phase.IN_RUN, Phase.POST_RUN, Phase.DEPLOYED})
    envelope = _STATIC_ENVELOPE
    invariance = "none"
    invariance_relation = Relation("invariant")
    baselines = (
        "the count of every field read on a parameter, validated or not, which is what a "
        "grep would give you",
    )
    rung = 2
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = None
    deviations = (
        _SOURCE_ACCESS_DEVIATION,
        "validation is dominance of a check over the access, which cannot see whether the check "
        "is correct. The count is a floor.",
    )

    def __init__(self, verifier: VerifierUnderTest | None = None, *, rung: int = 2) -> None:
        ensure_quantities()
        self.verifier = verifier
        self.rung = rung

    def measure(self, ctx: Context) -> "Evidence":
        verifier = self.verifier or ctx.signal.verifier  # type: ignore[union-attr]  # a ProgramSubject, not a network
        _, trusted = analyse_source(verifier, rung=self.rung)
        return ctx.emit(
            trusted,
            subject_extra={"baseline_all_field_reads": str(len(trusted.accesses))},
        )


def _register() -> None:
    """The ladder for D9's two quantities, so `capability_report` knows the rungs exist."""
    ensure_quantities()
    rungs = (
        (0, "verifier.cfg.ast", "an ast walk: functions, statements, branch count"),
        (1, "verifier.cfg.graph", "the control-flow graph and the rubric-clause map"),
        (2, "verifier.cfg.dataflow", "adds ScopeProvider alias resolution"),
    )
    for rung, impl, what in rungs:
        register_estimator(
            EstimatorEntry(
                quantity="verifier.cfg",
                impl=impl,
                requires=ACCESS_SOURCE_ONLY,
                envelope=_STATIC_ENVELOPE,
                rung=rung,
                bias=BiasStatement(
                    direction="downward" if rung < 2 else "approximately_unbiased",
                    why=(
                        f"{what}. A lower rung sees fewer edges than the program has, so every "
                        f"count it reports is a floor."
                    ),
                ),
                cost=CostModel(note="one parse of one file"),
                substrates=frozenset({Substrate.PROGRAM}),
                run=None,
            )
        )
    for rung, impl, why in (
        (0, "verifier.trusted_inputs.ast", "direct parameter reads only"),
        (1, "verifier.trusted_inputs.dominance", "adds dominance-based validation"),
        (2, "verifier.trusted_inputs.dataflow", "adds alias resolution through ScopeProvider"),
    ):
        register_estimator(
            EstimatorEntry(
                quantity="verifier.trusted_inputs",
                impl=impl,
                requires=ACCESS_SOURCE_ONLY,
                envelope=_STATIC_ENVELOPE,
                rung=rung,
                bias=BiasStatement(
                    direction="downward",
                    why=(
                        f"{why}. Every rung under-counts: a field read it cannot see is a field "
                        f"read it does not report, and validation it cannot judge it credits."
                    ),
                ),
                cost=CostModel(note="one parse of one file"),
                substrates=frozenset({Substrate.PROGRAM}),
                run=None,
            )
        )


_register()


__all__ = [
    "TIMEOUT_GUARDS",
    "Arc",
    "RubricClause",
    "StaticStructure",
    "TrustedInput",
    "TrustedInputReport",
    "TrustedInputs",
    "VerifierStructure",
    "analyse_source",
    "analyse_trusted_inputs",
    "arc_key",
    "control_flow",
    "dominators",
    "extract_clauses",
    "guard_gated_arcs",
    "map_arcs_to_clauses",
]

"""D2 Surviving mutants: would the verifier notice if it were wrong?

Says, with your numbers in place of these illustrative ones from the specification: "Of 412 mutants
of your verifier, 89 survive: 89 ways it could be wrong that no rollout in your corpus would
reveal."

**The inversion.** Ordinary mutation testing perturbs a program and asks whether the *test suite*
notices. Here the test suite is the rollout corpus: a mutant is killed when re-grading the corpus
with the mutated verifier changes any score, and it survives when the whole corpus scores
identically under a grader that is demonstrably a different program. `all:"mutation testing" AND
all:"reward function"` returns zero papers. `mutmut` and `cosmic-ray` are a decade old. Nothing here
needed inventing; the corpus needed substituting for the test suite.

Each survivor carries a source span and a unified diff, which is what makes it a reproducible blind
spot instead of a count. A number like "89 survive" is a headline. `x > 10` becoming `x >= 10` at
line 47, with the diff, is something an engineer fixes.

**The survivor ranking, and the reading of the spec it rests on.** The catalogue's rung 2 is
"mutants ranked by how many rollouts change score". A survivor changes zero scores by definition,
so that ranking cannot be over survivors as stated. The reading implemented here: rank survivors by
how many rollouts *reached the mutated line and still scored the same*. A mutant on a line no
rollout ever executes is D1's finding, already reported as uncovered. A mutant on a line eight
thousand rollouts execute, which none of them can tell from the original, is D2's finding and it is
the actionable one. Both counts are on every survivor so a reader can check the ranking rather than
take it. The ambiguity is recorded in the build report rather than resolved silently.

**The positive control is load-bearing.** The catalogue's baseline is "a random source edit that
should be caught". Implemented as a null grader: the entrypoint's body replaced with `return 0.0`.
If the corpus cannot kill that, it cannot kill anything, every survivor count is vacuous, and this
instrument refuses with `NO_MATCHED_CONTROL` rather than reporting a mutation score of zero that
reads like a finding about the grader.

Kill condition, from the catalogue: if the mutation score is above 95% on real verifiers, the
corpus already distinguishes everything and there is nothing here to find.

**Dual use.** The survivor list is an exploit catalogue in the form somebody needs to use it. The
payload declares `sensitive = True`, `redacted()` strips the spans and diffs while keeping the
counts, and `verifier.redact` is what a renderer calls. Publishing the unredacted list is a
decision somebody makes and records, not a default.
"""

from __future__ import annotations

import hashlib
import math
import signal
import threading
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from difflib import unified_diff
from typing import TYPE_CHECKING, Any, Iterator, Mapping, Sequence

from reward_lens.core.envelope import EnvelopeSpec, RegimeCondition
from reward_lens.core.evidence import Uncertainty, register_payload
from reward_lens.core.invariance import Relation
from reward_lens.core.quantity import (
    BiasStatement,
    CostModel,
    EstimatorEntry,
    register_estimator,
)
from reward_lens.core.reading import Reading, Refusal, RefusalReason
from reward_lens.core.types import Capability, GaugeStatus, Phase, Substrate
from reward_lens.measure.base import BaseObservable, Context
from reward_lens.verifier import (
    ACCESS_SOURCE_AND_RECORD,
    SENSITIVE_NOTE,
    SENSITIVE_SUBJECT_EXTRA,
    RolloutCorpus,
    VerifierUnderTest,
    ensure_quantities,
    load_entrypoint,
)
from reward_lens.verifier.coverage import CoverageReading, measure_coverage

if TYPE_CHECKING:
    from reward_lens.core.evidence import Evidence

#: Floats within this of each other are the same score. A grader whose real discrimination is
#: finer than this is not distinguishable from float noise anyway.
SCORE_TOLERANCE = 1e-12

#: Wall-clock seconds a single mutant gets to re-grade the whole corpus before it is called a
#: timeout kill. A mutated loop condition hangs, and a mutation run that hangs is a mutation run
#: nobody finishes.
DEFAULT_MUTANT_TIMEOUT = 60.0


class MutantTimeout(BaseException):
    """Raised into a mutant that has run past its wall clock.

    Deliberately a `BaseException`: the per-rollout handler catches `Exception` to record what the
    mutant did, and a timeout must escape that handler rather than be logged as the mutant
    returning a value.
    """


@contextmanager
def wall_limit(seconds: float | None) -> Iterator[bool]:
    """Bound a block by wall clock where the platform allows it, and say when it does not.

    `SIGALRM` exists on POSIX and fires only on the main thread. Where either is untrue this yields
    `False` and the caller records that the run had no timeout guard, which is a different thing
    from a run in which no mutant timed out.
    """
    usable = (
        seconds is not None
        and seconds > 0
        and hasattr(signal, "SIGALRM")
        and threading.current_thread() is threading.main_thread()
    )
    if not usable:
        yield False
        return

    def _fire(signum: int, frame: Any) -> None:
        raise MutantTimeout()

    previous = signal.signal(signal.SIGALRM, _fire)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))  # type: ignore[arg-type]
    try:
        yield True
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


# ---------------------------------------------------------------------------
# Payloads
# ---------------------------------------------------------------------------


@register_payload
@dataclass(frozen=True)
class Mutant:
    """One perturbation of the verifier, with where it is and what it did to the corpus."""

    operator: str
    occurrence: int
    start_line: int
    start_col: int
    end_line: int
    end_col: int
    diff: str
    outcome: str
    rollouts_changed: int = 0
    rollouts_reached: int = 0
    source_line: str = ""
    clause: str | None = None
    duplicate_of: str | None = None

    @property
    def span(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return ((self.start_line, self.start_col), (self.end_line, self.end_col))

    def render(self) -> str:
        head = (
            f"{self.operator} at {self.start_line}:{self.start_col}-"
            f"{self.end_line}:{self.end_col} [{self.outcome}]"
        )
        if self.clause:
            head += f" in {self.clause}"
        if self.outcome == "survived":
            head += f", reached by {self.rollouts_reached} rollouts and distinguished by none"
        return head + "\n" + self.diff.rstrip()


@register_payload
@dataclass(frozen=True)
class MutationReading:
    """The mutation run. Sensitive by default: `survivors` is a list of ways to be wrong.

    ``mutation_score`` is killed over competent mutants. Incompetent ones (a mutant that will not
    compile, or one whose source is identical to another's) are excluded from the denominator
    rather than counted as survivors, which is standard and which matters here because cosmic-ray's
    213 operators generate a lot of both.
    """

    source_path: str
    entrypoint: str
    fingerprint: str
    rung: int
    n_rollouts: int
    mutants_generated: int
    killed: int
    survived: int
    incompetent: int
    timed_out: int
    survivors: tuple[Mutant, ...]
    killed_mutants: tuple[Mutant, ...] = ()
    control_killed: bool = True
    control_rollouts_changed: int = 0
    baselines: Mapping[str, float] = field(default_factory=dict)
    timeout_guard: bool = True
    engine: str = "cosmic-ray"
    notes: tuple[str, ...] = ()
    sensitive: bool = True

    @property
    def competent(self) -> int:
        return self.killed + self.survived + self.timed_out

    @property
    def mutation_score(self) -> float:
        return (
            float("nan")
            if self.competent == 0
            else (self.killed + self.timed_out) / (self.competent)
        )

    @property
    def unreached_survivors(self) -> tuple[Mutant, ...]:
        """Survivors on code no rollout runs. D1 already told you about these."""
        return tuple(m for m in self.survivors if m.rollouts_reached == 0)

    @property
    def blind_spots(self) -> tuple[Mutant, ...]:
        """Survivors the corpus ran and could not tell from the original. D2's actual finding."""
        return tuple(
            sorted(
                (m for m in self.survivors if m.rollouts_reached > 0),
                key=lambda m: (-m.rollouts_reached, m.start_line, m.operator),
            )
        )

    def redacted(self) -> "MutationReading":
        """The publishable row: every count, no reproducers."""
        from dataclasses import replace as _replace

        return _replace(
            self,
            survivors=(),
            killed_mutants=(),
            sensitive=False,
            notes=self.notes + (f"redacted. {SENSITIVE_NOTE}",),
        )

    def render(self) -> str:
        lines = [
            f"Of {self.mutants_generated} mutants of {self.entrypoint}, {self.survived} survive: "
            f"{self.survived} ways it could be wrong that no rollout in your corpus would reveal."
        ]
        lines.append(
            f"    mutation score {self.mutation_score:.3f} over {self.competent} competent"
        )
        blind = self.blind_spots
        if blind:
            lines.append(f"    {len(blind)} of them on code the corpus actually runs:")
            lines += [f"      {m.render().splitlines()[0]}" for m in blind[:10]]
        if self.unreached_survivors:
            lines.append(
                f"    {len(self.unreached_survivors)} more sit on code no rollout reaches, which "
                f"is D1's finding rather than this one"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Mutant generation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MutantSpec:
    """A generated mutant before it has been run."""

    operator: str
    occurrence: int
    start: tuple[int, int]
    end: tuple[int, int]
    source: str


def generate_mutants(
    source: str,
    *,
    line_range: tuple[int, int] | None = None,
    covered_lines: set[int] | None = None,
    operators: Sequence[str] | None = None,
    limit: int | None = None,
) -> list[MutantSpec]:
    """Every mutant cosmic-ray's operators produce, in process, with spans.

    Not driven through `cosmic-ray init` and `cosmic-ray exec`. That harness runs a shell command
    per mutant, and the command here would have to serialise the rollout corpus, re-grade it in a
    subprocess and report back a single pass or fail. In process, the same operators give the same
    mutants and the whole score *vector* comes back, which is what the survivor ranking needs.
    `cosmic_ray.work_db` and the `(WorkItem, WorkResult)` session it holds are still the right
    reader for a cosmic-ray session somebody else produced; they are not the right writer for this.

    ``covered_lines`` restricts mutation to the lines a rollout set actually exercised, which is
    the same argument `mutmut.mutate_file_contents` takes and the same idea: mutating code no
    rollout runs produces survivors that are already D1's finding.
    """
    from cosmic_ray.ast import ast_nodes, get_ast
    from cosmic_ray.mutating import mutate_code
    from cosmic_ray.plugins import get_operator, operator_names

    names = list(operators) if operators is not None else list(operator_names())
    out: list[MutantSpec] = []
    seen: set[str] = {_digest(source)}
    for name in names:
        try:
            operator = get_operator(name)()
        except Exception:  # a broken third-party operator plugin must not stop the run
            continue
        positions = []
        for node in ast_nodes(get_ast(source)):
            positions.extend(operator.mutation_positions(node))
        for occurrence, position in enumerate(positions):
            start, end = position
            if line_range is not None and not (line_range[0] <= start[0] <= line_range[1]):
                continue
            if covered_lines is not None and start[0] not in covered_lines:
                continue
            mutated = mutate_code(source, operator, occurrence)
            if mutated is None or mutated == source:
                continue
            digest = _digest(mutated)
            if digest in seen:
                continue
            seen.add(digest)
            out.append(
                MutantSpec(
                    operator=name,
                    occurrence=occurrence,
                    start=tuple(start),  # type: ignore[arg-type]
                    end=tuple(end),  # type: ignore[arg-type]
                    source=mutated,
                )
            )
            if limit is not None and len(out) >= limit:
                return out
    return out


def generate_mutants_mutmut(source: str, path: Any) -> list[MutantSpec]:
    """The fallback engine, with the import trap handled.

    `mutmut` has no package-level public API, so `mutate_file_contents` comes from
    `mutmut.__main__`, and importing that module *loads mutmut's config*, which raises
    `FileNotFoundError` from any directory without one. The import therefore happens inside a
    prepared temporary directory with a `setup.cfg` naming a source path.

    Two return shapes, and the difference is a live one. mutmut 3.6.0, which is the version this
    adapter was first written against, returns `(code, [mutant_name, ...])`. mutmut 3.7.0,
    which is what `mutmut>=3.6.0,<4.0` actually resolves to today, returns a `MutatedFile` with
    `.code`, `.mutant_names` and `.line_span_by_function_name`. Both are handled, because the
    version that gets installed is decided by the resolver rather than by this file.

    Either way this engine is worse than cosmic-ray for D2's purpose. mutmut 3.x rewrites the whole
    module into a trampoline with every mutant as a separate function, so there is no per-mutant
    span, only a per-function line span, and the specs below carry a zero span to say so.
    """
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "src"), exist_ok=True)
        with open(os.path.join(tmp, "setup.cfg"), "w", encoding="utf-8") as fh:
            fh.write("[mutmut]\nsource_paths=src\n")
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            from mutmut.__main__ import mutate_file_contents  # noqa: PLC0415

            result = mutate_file_contents(str(path), source)
        finally:
            os.chdir(cwd)

    if isinstance(result, tuple):  # mutmut 3.6.0
        mutated, names = result
    else:  # mutmut 3.7.0
        mutated, names = result.code, list(result.mutant_names)
    if not names:
        return []
    return [
        MutantSpec(
            operator=f"mutmut/{name}",
            occurrence=i,
            start=(0, 0),
            end=(0, 0),
            source=mutated,
        )
        for i, name in enumerate(names)
    ]


def null_grader_source(source: str, entrypoint: str) -> str:
    """The positive control: the entrypoint's body replaced with `return 0.0`.

    Not a random edit, which is what the catalogue's baseline field says, and the difference is
    deliberate. A random edit is sometimes semantically inert and then the control tells you
    nothing. A grader that returns a constant is the weakest possible grader, so a corpus that
    cannot distinguish it from the real one cannot distinguish anything.
    """
    import libcst as cst

    class _Null(cst.CSTTransformer):
        def leave_FunctionDef(self, original: Any, updated: Any) -> Any:
            if original.name.value != entrypoint:
                return updated
            return updated.with_changes(
                body=cst.IndentedBlock(body=[cst.parse_statement("return 0.0")])
            )

    return cst.parse_module(source).visit(_Null()).code


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

Outcome = tuple[str, Any]


def _score_vector(fn: Any, corpus: RolloutCorpus) -> list[Outcome]:
    """What the verifier does on every rollout: a value, or the exception type it raised.

    Catching `Exception` here is not error swallowing. It is the measurement: a mutant that starts
    raising where the original returned 1.0 has changed the grader's observable behaviour, and
    recording that as a distinct outcome is what kills it. `MutantTimeout` is a `BaseException` and
    passes straight through.
    """
    out: list[Outcome] = []
    for rollout in corpus:
        try:
            out.append(("value", fn(**dict(rollout.inputs))))
        except Exception as exc:
            out.append(("raise", type(exc).__name__))
    return out


def _same(a: Outcome, b: Outcome, tol: float) -> bool:
    if a[0] != b[0]:
        return False
    if a[0] == "raise":
        return bool(a[1] == b[1])
    return _same_value(a[1], b[1], tol)


def _same_value(a: Any, b: Any, tol: float) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if math.isnan(float(a)) and math.isnan(float(b)):
            return True
        return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tol)
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return set(a) == set(b) and all(_same_value(a[k], b[k], tol) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same_value(x, y, tol) for x, y in zip(a, b))
    try:
        return bool(a == b)
    except Exception:
        return repr(a) == repr(b)


def _changed_count(baseline: Sequence[Outcome], candidate: Sequence[Outcome], tol: float) -> int:
    return sum(1 for a, b in zip(baseline, candidate) if not _same(a, b, tol))


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def run_mutation(
    verifier: VerifierUnderTest,
    corpus: RolloutCorpus,
    *,
    rung: int = 1,
    coverage: CoverageReading | None = None,
    tolerance: float = SCORE_TOLERANCE,
    limit: int | None = None,
    operators: Sequence[str] | None = None,
    timeout: float | None = DEFAULT_MUTANT_TIMEOUT,
    restrict_to_covered: bool = False,
) -> MutationReading | Refusal:
    """D2's estimate. Mutate, re-grade the corpus, diff the score vector, rank what survives.

    Rung 0 mutates only the entrypoint's own lines, rung 1 the whole module, rung 2 adds the
    survivor ranking, which needs D1's per-rollout attribution and computes it if none is supplied.
    """
    n = len(corpus)
    if n == 0:
        return Refusal(
            instrument="SurvivingMutants",
            reason=RefusalReason.ACCESS_INSUFFICIENT,
            detail="the corpus has 0 rollouts, so nothing can kill a mutant",
            remedy=(
                "supply a corpus with at least one rollout. Mutation testing against an empty "
                "corpus reports every mutant as surviving, which is arithmetic rather than a "
                "finding about the grader."
            ),
            statistics={"n_rollouts": 0},
        )

    source = verifier.source()
    path = verifier.source_path
    baseline_fn = load_entrypoint(source, path, verifier.entrypoint, verifier.import_name)
    baseline = _score_vector(baseline_fn, corpus)

    control_source = null_grader_source(source, verifier.entrypoint)
    control_changed, control_outcome = _run_one(
        control_source, path, verifier, corpus, baseline, tolerance, timeout
    )
    if control_outcome != "killed":
        distinct = len({repr(o) for o in baseline})
        return Refusal(
            instrument="SurvivingMutants",
            reason=RefusalReason.NO_MATCHED_CONTROL,
            detail=(
                f"the positive control survived: replacing {verifier.entrypoint}'s whole body with "
                f"`return 0.0` changed {control_changed} of {n} scores. The corpus has "
                f"{distinct} distinct outcomes."
            ),
            remedy=(
                "supply a corpus the grader actually separates. A corpus on which a constant "
                "grader scores the same as the real one cannot kill any mutant, so a mutation "
                "score computed on it would be zero for reasons that have nothing to do with the "
                "grader."
            ),
            statistics={
                "n_rollouts": n,
                "control_rollouts_changed": control_changed,
                "distinct_baseline_outcomes": distinct,
            },
        )

    entry_range = _entrypoint_lines(source, verifier.entrypoint) if rung <= 0 else None
    covered: set[int] | None = None
    if restrict_to_covered or rung >= 2:
        if coverage is None:
            measured = measure_coverage(verifier, corpus, rung=1)
            coverage = measured if isinstance(measured, CoverageReading) else None
        if coverage is not None and restrict_to_covered:
            covered = _covered_lines(coverage)

    engine = "cosmic-ray"
    notes: list[str] = []
    try:
        specs = generate_mutants(
            source,
            line_range=entry_range,
            covered_lines=covered,
            operators=operators,
            limit=limit,
        )
    except ImportError:
        engine = "mutmut"
        notes.append(
            "cosmic-ray would not import; fell back to mutmut, which returns a trampolined module "
            "rather than one mutant per call, so spans are not recoverable at this engine"
        )
        specs = generate_mutants_mutmut(source, path)

    killed: list[Mutant] = []
    survivors: list[Mutant] = []
    incompetent = 0
    timed_out = 0
    guard_available = True

    reach = _reach_index(coverage) if coverage is not None else {}
    clause_at = _clause_index(verifier) if rung >= 1 else {}

    for spec in specs:
        changed, outcome = _run_one(
            spec.source, path, verifier, corpus, baseline, tolerance, timeout
        )
        if outcome == "no-guard":
            guard_available = False
            outcome = "survived" if changed == 0 else "killed"
        if outcome == "incompetent":
            incompetent += 1
            continue
        line = spec.start[0]
        mutant = Mutant(
            operator=spec.operator,
            occurrence=spec.occurrence,
            start_line=spec.start[0],
            start_col=spec.start[1],
            end_line=spec.end[0],
            end_col=spec.end[1],
            diff=_diff(source, spec.source, path.name),
            outcome=outcome,
            rollouts_changed=changed,
            rollouts_reached=reach.get(line, 0),
            source_line=_line_text(source, line),
            clause=clause_at.get(line),
        )
        if outcome == "timeout":
            timed_out += 1
            killed.append(mutant)
        elif outcome == "killed":
            killed.append(mutant)
        else:
            survivors.append(mutant)

    if not guard_available:
        notes.append(
            "no wall-clock guard was available (no SIGALRM, or not the main thread), so a mutant "
            "that hangs would have hung this run rather than being counted as a timeout kill"
        )
    if rung >= 2 and coverage is None:
        notes.append("rung 2 asked for the survivor ranking and no coverage was available for it")

    survivors.sort(key=lambda m: (-m.rollouts_reached, m.start_line, m.operator))
    killed.sort(key=lambda m: (m.rollouts_changed, m.start_line, m.operator))

    return MutationReading(
        source_path=str(path),
        entrypoint=verifier.entrypoint,
        fingerprint=verifier.fingerprint,
        rung=rung,
        n_rollouts=n,
        mutants_generated=len(specs),
        killed=len(killed) - timed_out,
        survived=len(survivors),
        incompetent=incompetent,
        timed_out=timed_out,
        survivors=tuple(survivors),
        killed_mutants=tuple(killed),
        control_killed=True,
        control_rollouts_changed=control_changed,
        baselines={"null_grader_rollouts_changed": float(control_changed)},
        timeout_guard=guard_available,
        engine=engine,
        notes=tuple(notes),
    )


def _run_one(
    mutated: str,
    path: Any,
    verifier: VerifierUnderTest,
    corpus: RolloutCorpus,
    baseline: Sequence[Outcome],
    tolerance: float,
    timeout: float | None,
) -> tuple[int, str]:
    """Load one mutant, re-grade the corpus, and say what happened to it."""
    module_name = f"{verifier.import_name}_m{_digest(mutated)[:12]}"
    try:
        # `SyntaxWarning` is suppressed for the mutant only. `ReplaceComparisonOperator_Eq_Is`
        # turns `s == "{"` into `s is "{"`, which compiles, runs, and warns; that warning is a
        # property of the mutation operator and not of the grader, and four hundred of them per
        # run buries anything the grader itself emits.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            fn = load_entrypoint(mutated, path, verifier.entrypoint, module_name)
    except (SyntaxError, AttributeError, TypeError, IndentationError):
        return 0, "incompetent"
    except Exception:
        # The mutant's module body raised on import. It is a different program that cannot even
        # load, which is the textbook definition of an incompetent mutant.
        return 0, "incompetent"

    with wall_limit(timeout) as guarded:
        try:
            candidate = _score_vector(fn, corpus)
        except MutantTimeout:
            return len(baseline), "timeout"
    changed = _changed_count(baseline, candidate, tolerance)
    if not guarded:
        return changed, "no-guard"
    return changed, "killed" if changed else "survived"


def _entrypoint_lines(source: str, entrypoint: str) -> tuple[int, int] | None:
    import ast

    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entrypoint:
            return node.lineno, (node.end_lineno or node.lineno)
    return None


def _covered_lines(coverage: CoverageReading) -> set[int]:
    out: set[int] = set()
    for arcs in coverage.per_rollout_arcs.values():
        for a, b in arcs:
            if a > 0:
                out.add(a)
            if b > 0:
                out.add(b)
    return out


def _reach_index(coverage: CoverageReading) -> dict[int, int]:
    """How many rollouts executed each line, from D1's per-rollout arc attribution."""
    counts: dict[int, int] = {}
    for arcs in coverage.per_rollout_arcs.values():
        lines = {x for arc in arcs for x in arc if x > 0}
        for line in lines:
            counts[line] = counts.get(line, 0) + 1
    return counts


def _clause_index(verifier: VerifierUnderTest) -> dict[int, str]:
    """Line to rubric-clause name, so a mutant span reads as a criterion rather than a number."""
    from reward_lens.verifier.static import analyse_source

    structure, _ = analyse_source(verifier, rung=1)
    out: dict[int, str] = {}
    for clause in sorted(structure.clauses, key=lambda c: -(c.end_line - c.start_line)):
        for line in range(clause.start_line, clause.end_line + 1):
            out[line] = clause.name
    return out


def _diff(original: str, mutated: str, name: str) -> str:
    return "".join(
        unified_diff(
            original.splitlines(keepends=True),
            mutated.splitlines(keepends=True),
            f"a/{name}",
            f"b/{name}",
            n=2,
        )
    )


def _line_text(source: str, line: int) -> str:
    lines = source.splitlines()
    return lines[line - 1].strip() if 1 <= line <= len(lines) else ""


# ---------------------------------------------------------------------------
# The instruments
# ---------------------------------------------------------------------------

_MUTATION_ENVELOPE = EnvelopeSpec(
    requires=frozenset({RegimeCondition.STATIONARY_GRADER}),
    measured_by={RegimeCondition.STATIONARY_GRADER: "verifier.cfg"},
    on_violation="refuse",
)

_MUTATION_DEVIATIONS = (
    "reads the grader's source and never modifies it; declared `Access.SOURCE`, added for the D "
    "series after this deviation was recorded. "
    "Here MUTATE is exact: this instrument does modify the source.",
    "the catalogue's rung 2 ranks 'mutants by how many rollouts change score', which is zero for "
    "every survivor. Implemented as: survivors ranked by how many rollouts reached the mutated "
    "line and still scored the same.",
    "the catalogue's baseline is 'a random source edit that should be caught'; the control here is "
    "a null grader, which is the weakest grader rather than a random one.",
)


class _MutationInstrument(BaseObservable):
    """Shared plumbing for D2's two quantities. One run, two readings."""

    capabilities = Capability.SCORES
    substrates = frozenset({Substrate.PROGRAM})
    phases = frozenset({Phase.PRE_RUN, Phase.POST_RUN})
    envelope = _MUTATION_ENVELOPE
    invariance = "none"
    invariance_relation = Relation("invariant")
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = None
    deviations = _MUTATION_DEVIATIONS

    def __init__(
        self,
        verifier: VerifierUnderTest | None = None,
        corpus: RolloutCorpus | None = None,
        *,
        rung: int = 2,
        limit: int | None = None,
        operators: Sequence[str] | None = None,
        timeout: float | None = DEFAULT_MUTANT_TIMEOUT,
        coverage: CoverageReading | None = None,
    ) -> None:
        ensure_quantities()
        self.verifier = verifier
        self.corpus = corpus
        self.rung = rung
        self.limit = limit
        self.operators = operators
        self.timeout = timeout
        self.coverage = coverage

    def _run(self, ctx: Context) -> MutationReading | Refusal:
        verifier = self.verifier or ctx.signal.verifier  # type: ignore[union-attr]  # a ProgramSubject, not a network
        corpus = self.corpus if self.corpus is not None else ctx.view
        if corpus is None:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail="no rollout corpus was supplied, on the context or to the constructor",
                remedy=(
                    "pass `corpus=ListCorpus.of([...])` here, or set `ctx.view` to the corpus. The "
                    "corpus is this instrument's test suite; without one there is nothing to kill "
                    "a mutant with."
                ),
            )
        return run_mutation(
            verifier,
            corpus,
            rung=self.rung,
            coverage=self.coverage,
            limit=self.limit,
            operators=self.operators,
            timeout=self.timeout,
        )

    def _emit(self, ctx: Context, result: MutationReading) -> "Evidence":  # pragma: no cover
        raise NotImplementedError

    def measure(self, ctx: Context) -> Reading:  # type: ignore[override]
        """Evidence or a Refusal.

        Widened from `Observable.measure` for the same reason D1's is. A corpus that cannot kill
        the null grader is a fact about the data, invisible to `preflight`, and returning the
        refusal from here keeps it flowing out through `run` with the capability check and the
        gauge gate still applied.
        """
        result = self._run(ctx)
        if isinstance(result, Refusal):
            return result
        return self._emit(ctx, result)


class SurvivingMutants(_MutationInstrument):
    """D2 `verifier.surviving_mutants`: ways the grader could be wrong that the corpus cannot see.

    The value is the whole `MutationReading`, and it is sensitive by default: every survivor
    carries the span and the diff that reproduce it, which is what an engineer needs to fix it and
    what an adversary needs to use it. `redact` returns the counts without the reproducers.

    Kill condition: if the mutation score is above 95% on real verifiers there is nothing here to
    find. The near-miss to watch for is a run whose survivors are all unreached: that is D1's
    result restated, and it means the corpus is too narrow for this measurement rather than that
    the grader is sound.
    """

    name = "SurvivingMutants"
    version = "1.0"
    quantity = "verifier.surviving_mutants"
    requires = ACCESS_SOURCE_AND_RECORD
    baselines = (
        "a null grader (the entrypoint's body replaced with `return 0.0`), which the corpus must "
        "kill for any survivor count to mean anything",
    )
    rung = 2

    def _emit(self, ctx: Context, result: MutationReading) -> "Evidence":
        return ctx.emit(
            result,
            uncertainty=Uncertainty(n=result.mutants_generated, method="census-over-mutants"),
            subject_extra={
                **SENSITIVE_SUBJECT_EXTRA,
                "verifier_fingerprint": result.fingerprint,
            },
        )


class MutationScore(_MutationInstrument):
    """D2 `verifier.mutation_score`: killed over competent, the one number.

    Not sensitive: a score is not a reproducer. It is also the weaker of the two readings, and it
    is here because the catalogue registers the quantity, not because a ratio is what anyone acts
    on. What gets acted on is the survivor list next door.
    """

    name = "MutationScore"
    version = "1.0"
    quantity = "verifier.mutation_score"
    requires = ACCESS_SOURCE_AND_RECORD
    baselines = (
        "a null grader (the entrypoint's body replaced with `return 0.0`); a corpus that cannot "
        "kill it makes every mutation score zero for reasons unrelated to the grader",
    )
    rung = 1

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("rung", 1)
        super().__init__(*args, **kwargs)

    def _emit(self, ctx: Context, result: MutationReading) -> "Evidence":
        return ctx.emit(
            float(result.mutation_score),
            uncertainty=Uncertainty(n=result.competent, method="census-over-mutants"),
            subject_extra={"verifier_fingerprint": result.fingerprint},
        )


def _register() -> None:
    ensure_quantities()
    from reward_lens.verifier import ACCESS_SOURCE_AND_RECORD

    ladder = (
        (0, "operator-level mutation on the scoring function"),
        (1, "whole-module mutation"),
        (2, "whole-module, survivors ranked by rollouts that reached them"),
    )
    for quantity, prefix in (
        ("verifier.surviving_mutants", "verifier.surviving_mutants"),
        ("verifier.mutation_score", "verifier.mutation_score"),
    ):
        for rung, what in ladder:
            register_estimator(
                EstimatorEntry(
                    quantity=quantity,
                    impl=f"{prefix}.r{rung}",
                    requires=ACCESS_SOURCE_AND_RECORD,
                    envelope=_MUTATION_ENVELOPE,
                    rung=rung,
                    bias=BiasStatement(
                        direction="downward" if rung < 1 else "approximately_unbiased",
                        why=(
                            f"{what}. Mutating only the scoring function misses a blind spot in a "
                            f"helper, so rung 0 under-counts survivors. Equivalent mutants inflate "
                            f"the survivor count at every rung and are not detected here."
                        ),
                    ),
                    cost=CostModel(
                        note="one corpus re-grade per mutant; linear in mutants times rollouts"
                    ),
                    substrates=frozenset({Substrate.PROGRAM}),
                    phases=frozenset({Phase.PRE_RUN, Phase.POST_RUN}),
                    run=None,
                )
            )


_register()


__all__ = [
    "DEFAULT_MUTANT_TIMEOUT",
    "SCORE_TOLERANCE",
    "Mutant",
    "MutantSpec",
    "MutantTimeout",
    "MutationReading",
    "MutationScore",
    "SurvivingMutants",
    "generate_mutants",
    "generate_mutants_mutmut",
    "null_grader_source",
    "run_mutation",
    "wall_limit",
]

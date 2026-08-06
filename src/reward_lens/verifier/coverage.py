"""D1 Decision coverage: what fraction of the verifier's logic any rollout has ever exercised.

Says, with your numbers in place of these illustrative ones from the specification: "Across 25,664
rollouts your verifier has never taken 31% of its branches. Those are behaviours it cannot
distinguish."

The substitution that makes this an instrument rather than a coverage report: **the test suite is
the rollout corpus.** `coverage.py` has measured branch coverage of test suites for twenty years,
and `abs:"coverage" AND abs:"reward function"` returns 88 papers of which every one uses coverage
*as* a reward in robotics and none measures coverage *of* a reward. The tooling did not need
building. It needed aiming at the grader.

What turns it from a report into an instrument is per-rollout context. `Coverage.switch_context()`
before each rollout, `CoverageData.set_query_context()` on the way back out, and every arc is
attributed to the rollout that took it, with no extra machinery. That attribution is what D2's
survivor ranking consumes and it is what lets an uncovered arc be named: through D9's clause map,
"arc (147, 152) never taken" becomes "the third rubric criterion has never fired."

**The baseline is the point, not a formality.** Coverage being incomplete is expected and dull. The
claim worth making is that the corpus is *redundant*, and the mandatory baseline is what tests it:
a random rollout sample of the same size, drawn with replacement, which leaves out about 37% of the
corpus. If coverage barely moves when a third of the rollouts are gone, the corpus is buying
repetition rather than discrimination.

Kill condition, from the catalogue: if coverage is above 95% on every real verifier, this
instrument is measuring nothing and should be retired. Watch for the opposite failure too. A
coverage number with no clause map attached is a coordinate nobody can act on, which is how test
coverage became a compliance metric in the industry it came from.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

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
    RolloutCorpus,
    VerifierUnderTest,
    ensure_quantities,
)
from reward_lens.verifier.static import Arc, VerifierStructure, analyse_source

#: How many entry-to-exit paths the rung 2 bound will enumerate before giving up and reporting the
#: cap. A grader with a loop over rubric criteria blows past this, which is a fact about path
#: coverage rather than about the grader.
PATH_CAP = 100_000


@register_payload
@dataclass(frozen=True)
class CoverageReading:
    """What the corpus exercised, per rung, with the per-rollout attribution kept.

    ``per_rollout_arcs`` is the field that makes this more than a report. It is kept as a mapping
    from rollout id to the arcs that rollout took, because D2 needs it: a surviving mutant on a
    line no rollout reaches is D1's finding, and a surviving mutant on a line eight thousand
    rollouts reach is D2's.
    """

    source_path: str
    entrypoint: str
    fingerprint: str
    rung: int
    n_rollouts: int
    statements_total: int
    statements_covered: int
    branch_arcs_total: int
    branch_arcs_covered: int
    uncovered_branch_arcs: tuple[list[int], ...]
    uncovered_clauses: tuple[str, ...]
    per_rollout_arcs: Mapping[str, list[list[int]]] = field(default_factory=dict)
    baselines: Mapping[str, float] = field(default_factory=dict)
    paths_observed: int | None = None
    paths_bound: int | None = None
    path_bound_capped: bool = False
    hot_lines: tuple[int, ...] = ()
    errors: Mapping[str, str] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    @property
    def statement_fraction(self) -> float:
        return _ratio(self.statements_covered, self.statements_total)

    @property
    def branch_fraction(self) -> float:
        return _ratio(self.branch_arcs_covered, self.branch_arcs_total)

    @property
    def path_fraction(self) -> float | None:
        if self.paths_observed is None or not self.paths_bound:
            return None
        return _ratio(self.paths_observed, self.paths_bound)

    @property
    def headline(self) -> float:
        """The number for the declared rung: statement at 0, branch at 1 and above.

        Not called ``value``. `Evidence.value` is this whole payload, and a payload with its own
        ``value`` attribute makes every helper that walks `Evidence.value` ambiguous.
        """
        return self.statement_fraction if self.rung <= 0 else self.branch_fraction

    def render(self) -> str:
        pct = 100.0 * (1.0 - self.branch_fraction)
        lines = [
            f"Across {self.n_rollouts:,} rollouts {self.entrypoint} has never taken "
            f"{pct:.0f}% of its branches "
            f"({self.branch_arcs_total - self.branch_arcs_covered} of {self.branch_arcs_total}). "
            f"Those are behaviours it cannot distinguish."
        ]
        if self.uncovered_clauses:
            lines.append(f"    never fired: {', '.join(self.uncovered_clauses)}")
        base = self.baselines.get("random_rollout_sample_same_size")
        if base is not None:
            lines.append(
                f"    a random sample of the same size reaches {base:.3f} against the corpus's "
                f"{self.branch_fraction:.3f}"
            )
        if self.path_fraction is not None:
            cap = " (bound capped)" if self.path_bound_capped else ""
            lines.append(f"    hot-subset paths: {self.paths_observed} of {self.paths_bound}{cap}")
        return "\n".join(lines)


def _ratio(num: int, den: int) -> float:
    return float("nan") if den == 0 else num / den


# ---------------------------------------------------------------------------
# The measurement
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Traced:
    """The raw tracer output before any coverage arithmetic."""

    per_rollout_arcs: dict[str, frozenset[Arc]]
    per_rollout_lines: dict[str, frozenset[int]]
    aggregate_arcs: frozenset[Arc]
    aggregate_lines: frozenset[int]
    arc_universe: frozenset[Arc]
    exit_counts: Mapping[int, int]
    statements: frozenset[int]
    scores: dict[str, Any]
    errors: dict[str, str]
    measured_files: tuple[str, ...]


def trace_corpus(verifier: VerifierUnderTest, corpus: RolloutCorpus) -> _Traced:
    """Run every rollout through the verifier under a branch tracer, one context each.

    The import happens inside `start()` and before the first `switch_context`, so arcs from the
    module body land in the empty context: they count toward the aggregate, which is right (they
    did execute), and toward no rollout, which is also right (no rollout caused them).

    A rollout whose call raises is recorded in ``errors`` and its arcs are kept. The exception is
    the verifier's behaviour on that input, and discarding the trace would hide the branch that
    produced it. Nothing here converts it into a refusal.
    """
    import coverage
    from coverage.python import PythonFileReporter

    path = str(verifier.source_path)
    cov = coverage.Coverage(branch=True, data_file=None, include=[path])
    scores: dict[str, Any] = {}
    errors: dict[str, str] = {}

    cov.start()
    try:
        fn = verifier.load()
        for rollout in corpus:
            cov.switch_context(rollout.id)
            try:
                scores[rollout.id] = fn(**dict(rollout.inputs))
            except Exception as exc:  # the verifier's behaviour on this input, recorded as such
                errors[rollout.id] = f"{type(exc).__name__}: {exc}"
    finally:
        cov.stop()

    data = cov.get_data()
    reporter = PythonFileReporter(path, cov)
    arc_universe = frozenset(reporter.arcs())
    exit_counts = dict(reporter.exit_counts())
    statements = frozenset(reporter.lines())

    per_arcs: dict[str, frozenset[Arc]] = {}
    per_lines: dict[str, frozenset[int]] = {}
    measured = tuple(sorted(data.measured_files()))
    for rollout in corpus:
        data.set_query_context(rollout.id)
        per_arcs[rollout.id] = frozenset(data.arcs(path) or ())
        per_lines[rollout.id] = frozenset(data.lines(path) or ())
    data.set_query_contexts(None)
    return _Traced(
        per_rollout_arcs=per_arcs,
        per_rollout_lines=per_lines,
        aggregate_arcs=frozenset(data.arcs(path) or ()),
        aggregate_lines=frozenset(data.lines(path) or ()),
        arc_universe=arc_universe,
        exit_counts=exit_counts,
        statements=statements,
        scores=scores,
        errors=errors,
        measured_files=measured,
    )


def branch_universe(arc_universe: frozenset[Arc], exit_counts: Mapping[int, int]) -> frozenset[Arc]:
    """The arcs that are decisions: those leaving a line with more than one exit.

    This is `coverage.py`'s own definition of a branch, reproduced as set arithmetic so the
    bootstrap baseline can be computed over thousands of resamples without rebuilding an `Analysis`
    each time. Checked against `Analysis.branch_stats()` in the tests, because a denominator that
    quietly disagrees with the tool it came from is exactly the failure this whole layer is about.
    """
    return frozenset((a, b) for a, b in arc_universe if exit_counts.get(a, 0) > 1)


def _acyclic_path_count(arcs: frozenset[Arc], roots: frozenset[int], cap: int) -> tuple[int, bool]:
    """How many distinct entry-to-exit paths the subgraph admits, with back edges dropped.

    Back edges (`b <= a` for positive `b`) are loops, and a loop makes the path space infinite. The
    count is therefore a *lower* bound on the real path space and the fraction built on it is an
    upper bound on real path coverage. Said here rather than in a caveats page because the number
    is meaningless without it.
    """
    forward: dict[int, set[int]] = {}
    nodes: set[int] = set(roots)
    for a, b in arcs:
        if b > 0 and a > 0 and b <= a:
            continue
        forward.setdefault(a, set()).add(b)
        nodes.add(a)
        nodes.add(b)

    indegree = {n: 0 for n in nodes}
    for a, outs in forward.items():
        for b in outs:
            indegree[b] += 1
    order: list[int] = [n for n in sorted(nodes) if indegree[n] == 0]
    seen = 0
    i = 0
    while i < len(order):
        node = order[i]
        i += 1
        seen += 1
        for nxt in sorted(forward.get(node, ())):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                order.append(nxt)
    if seen != len(nodes):
        # A cycle survived the back-edge filter. Rather than count paths through it, say so.
        return cap, True

    capped = False
    paths: dict[int, int] = {}
    for node in reversed(order):
        outs = forward.get(node) or set()
        total = 1 if not outs else sum(paths[b] for b in outs)
        if total >= cap:
            total = cap
            capped = True
        paths[node] = total

    total = 0
    for root in sorted(roots):
        total += paths.get(root, 1)
        if total >= cap:
            return cap, True
    return total, capped


def measure_coverage(
    verifier: VerifierUnderTest,
    corpus: RolloutCorpus,
    *,
    rung: int = 1,
    structure: VerifierStructure | None = None,
    baseline_resamples: int = 64,
    hot_fraction: float = 0.5,
    seed: int = 0,
    path_cap: int = PATH_CAP,
) -> CoverageReading | Refusal:
    """D1's estimate. Rung 0 is statements, rung 1 branches, rung 2 adds hot-subset paths.

    Returns a `Refusal` when the corpus is empty or when the tracer saw no arcs in the file the
    verifier claims to live in, because both of those produce a coverage number that looks like a
    measurement and is not one.
    """
    n = len(corpus)
    if n == 0:
        return Refusal(
            instrument="DecisionCoverage",
            reason=RefusalReason.ACCESS_INSUFFICIENT,
            detail="the corpus has 0 rollouts, so no branch has been exercised by anything",
            remedy=(
                "supply a corpus with at least one rollout: "
                "`ListCorpus.of([Rollout(id=..., inputs={...})])`. If you want the arc universe "
                "without a corpus, that is D9 (`StaticStructure`) and it needs no rollouts."
            ),
            statistics={"n_rollouts": 0},
        )

    traced = trace_corpus(verifier, corpus)
    if not traced.aggregate_arcs:
        return Refusal(
            instrument="DecisionCoverage",
            reason=RefusalReason.ACCESS_INSUFFICIENT,
            detail=(
                f"the tracer recorded no arcs in {verifier.source_path}. Files measured: "
                f"{', '.join(traced.measured_files) or 'none'}"
            ),
            remedy=(
                f"point --source at the file that actually defines {verifier.entrypoint!r}. A "
                f"grader that dispatches into another module needs that module's path here, and a "
                f"grader imported from an installed package needs the installed file, not the "
                f"checkout."
            ),
            statistics={"n_rollouts": n, "measured_files": list(traced.measured_files)},
        )

    if structure is None:
        structure, _ = analyse_source(verifier, rung=max(rung, 1))

    branches = branch_universe(traced.arc_universe, traced.exit_counts)
    covered_branches = branches & traced.aggregate_arcs
    uncovered = tuple(sorted(branches - traced.aggregate_arcs))

    never_fired = []
    for clause in structure.clauses:
        if clause.enter_arc is None:
            continue
        if tuple(clause.enter_arc) not in traced.aggregate_arcs:
            never_fired.append(clause.name)

    rng = random.Random(seed)
    ids = [r.id for r in corpus]
    baselines = _baselines(ids, traced.per_rollout_arcs, branches, baseline_resamples, rng)

    paths_observed: int | None = None
    paths_bound: int | None = None
    capped = False
    hot_lines: tuple[int, ...] = ()
    notes: list[str] = []
    if rung >= 2:
        threshold = max(1, int(round(hot_fraction * n)))
        counts: dict[int, int] = {}
        for lines in traced.per_rollout_lines.values():
            for line in lines:
                counts[line] = counts.get(line, 0) + 1
        hot = frozenset(line for line, c in counts.items() if c >= threshold)
        hot_lines = tuple(sorted(hot))
        # Every arc *out of* a hot line, whether or not the destination is hot. Restricting both
        # endpoints instead would cut the arc into a cold branch out of the subgraph, and a rollout
        # that leaves through one would have a signature that is not a path in the graph its
        # denominator was counted from, which is how this read `2 observed of 1 possible`.
        hot_arcs = frozenset((a, b) for a, b in traced.arc_universe if a in hot or a < 0)
        signatures = {frozenset(arcs & hot_arcs) for arcs in traced.per_rollout_arcs.values()}
        paths_observed = len(signatures)
        targets = {b for _, b in hot_arcs}
        roots = frozenset(
            {a for a, _ in hot_arcs if a not in targets} | {a for a, _ in hot_arcs if a < 0}
        )
        paths_bound, capped = _acyclic_path_count(hot_arcs, roots, path_cap)
        notes.append(
            "the path bound drops loop back edges, so it under-counts the true path space and "
            "the fraction built on it is an upper bound on real path coverage"
        )

    if traced.errors:
        notes.append(
            f"{len(traced.errors)} of {n} rollouts raised inside the verifier; their arcs are "
            f"counted, because the exception path is part of what the grader does"
        )

    return CoverageReading(
        source_path=str(verifier.source_path),
        entrypoint=verifier.entrypoint,
        fingerprint=verifier.fingerprint,
        rung=rung,
        n_rollouts=n,
        statements_total=len(traced.statements),
        statements_covered=len(traced.statements & traced.aggregate_lines),
        branch_arcs_total=len(branches),
        branch_arcs_covered=len(covered_branches),
        uncovered_branch_arcs=tuple([list(a) for a in uncovered]),
        uncovered_clauses=tuple(never_fired),
        per_rollout_arcs={
            rid: sorted([list(a) for a in arcs]) for rid, arcs in traced.per_rollout_arcs.items()
        },
        baselines=baselines,
        paths_observed=paths_observed,
        paths_bound=paths_bound,
        path_bound_capped=capped,
        hot_lines=hot_lines,
        errors=dict(traced.errors),
        notes=tuple(notes),
    )


def _baselines(
    ids: Sequence[str],
    per_rollout: Mapping[str, frozenset[Arc]],
    branches: frozenset[Arc],
    resamples: int,
    rng: random.Random,
) -> dict[str, float]:
    """The mandatory baseline, plus the two diagnostics that make it readable.

    ``random_rollout_sample_same_size`` is the catalogue's baseline, drawn with replacement so a
    sample of the same size still omits about 37% of the corpus. ``saturation_size`` is the
    smallest random subset that reaches the corpus's own branch coverage, and
    ``redundant_fraction`` is what is left over. A corpus of 25,000 rollouts whose coverage
    saturates at 400 is not a corpus of 25,000 experiments.
    """
    n = len(ids)
    total = len(branches)
    if total == 0:
        return {}
    full = len(frozenset().union(*per_rollout.values()) & branches) / total if per_rollout else 0.0

    draws = []
    for _ in range(max(1, resamples)):
        sample = [rng.choice(ids) for _ in range(n)]
        union: frozenset[Arc] = frozenset()
        for rid in set(sample):
            union |= per_rollout.get(rid, frozenset())
        draws.append(len(union & branches) / total)
    same_size = sum(draws) / len(draws)

    saturation = n
    ladder = sorted({max(1, int(round(n * f))) for f in (0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0)})
    for k in ladder:
        reached = []
        for _ in range(min(16, max(1, resamples // 4))):
            sample = rng.sample(list(ids), k)
            union = frozenset()
            for rid in sample:
                union |= per_rollout.get(rid, frozenset())
            reached.append(len(union & branches) / total)
        if sum(reached) / len(reached) >= full - 1e-12:
            saturation = k
            break

    return {
        "random_rollout_sample_same_size": same_size,
        "saturation_size": float(saturation),
        "redundant_fraction": 1.0 - (saturation / n if n else 1.0),
    }


# ---------------------------------------------------------------------------
# The instrument
# ---------------------------------------------------------------------------

#: A coverage fraction aggregated over a window in which the grader was edited is a number about
#: two programs. `verifier.cfg` is what detects that: D9 fingerprints the source, and two
#: fingerprints in one window is the violation.
_COVERAGE_ENVELOPE = EnvelopeSpec(
    requires=frozenset({RegimeCondition.STATIONARY_GRADER}),
    measured_by={RegimeCondition.STATIONARY_GRADER: "verifier.cfg"},
    on_violation="refuse",
)


class DecisionCoverage(BaseObservable):
    """D1 `verifier.decision_coverage`: what the rollout corpus has ever made the grader do.

    Kill condition, from the catalogue: if coverage comes back above 95% on every real verifier,
    the quantity has no variance and the instrument should be retired. The second thing that would
    kill it is subtler and worth watching for first: a coverage number with no clause map attached
    is a coordinate rather than a finding, and test coverage in the industry it came from turned
    into a compliance number precisely by being reported that way.
    """

    name = "DecisionCoverage"
    version = "1.0"
    quantity = "verifier.decision_coverage"
    capabilities = Capability.SCORES
    requires = ACCESS_SOURCE_AND_RECORD
    substrates = frozenset({Substrate.PROGRAM})
    phases = frozenset({Phase.PRE_RUN, Phase.POST_RUN})
    envelope = _COVERAGE_ENVELOPE
    invariance = "none"
    invariance_relation = Relation("invariant")
    baselines = ("coverage of a random rollout sample of the same size",)
    rung = 1
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = None
    deviations = (
        "reads the grader's source and never modifies it; declared `Access.SOURCE`, added for "
        "code. This instrument reads the source and runs it; it never modifies it.",
        "rung 2's path bound drops loop back edges, so the path fraction is an upper bound",
    )

    def __init__(
        self,
        verifier: VerifierUnderTest | None = None,
        corpus: RolloutCorpus | None = None,
        *,
        rung: int = 1,
        baseline_resamples: int = 64,
        seed: int = 0,
    ) -> None:
        ensure_quantities()
        self.verifier = verifier
        self.corpus = corpus
        self.rung = rung
        self.baseline_resamples = baseline_resamples
        self.seed = seed

    def measure(self, ctx: Context) -> Reading:  # type: ignore[override]
        """Evidence or a Refusal.

        Annotated `Reading` rather than `Evidence`, which is a widening of `Observable.measure`.
        The kernel's `estimate` only knows how to return refusals that `preflight` produced, and
        an empty corpus is not something `preflight` can see: it is a fact about the data, found
        during compute. Returning the refusal from here rather than raising keeps it flowing out
        through `run`, so the capability check and the gauge gate both still run, which overriding
        `estimate` to short-circuit them would have quietly skipped.
        """
        verifier = self.verifier or ctx.signal.verifier  # type: ignore[union-attr]  # a ProgramSubject, not a network
        corpus = self.corpus if self.corpus is not None else ctx.view
        if corpus is None:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail="no rollout corpus was supplied, on the context or to the constructor",
                remedy=(
                    "pass `corpus=ListCorpus.of([...])` here, or set `ctx.view` to the corpus. "
                    "D1 measures what the corpus exercised; with no corpus there is nothing to "
                    "measure and the arc universe alone is D9."
                ),
            )
        result = measure_coverage(
            verifier,
            corpus,
            rung=self.rung,
            baseline_resamples=self.baseline_resamples,
            seed=self.seed,
        )
        if isinstance(result, Refusal):
            return result
        return ctx.emit(
            result,
            uncertainty=Uncertainty(n=result.n_rollouts, method="census-over-corpus"),
            subject_extra={"verifier_fingerprint": result.fingerprint},
        )


def _register() -> None:
    ensure_quantities()
    for rung, impl, what in (
        (0, "verifier.decision_coverage.statement", "statement coverage"),
        (1, "verifier.decision_coverage.branch", "branch and decision coverage"),
        (2, "verifier.decision_coverage.path", "path coverage on the hot subset"),
    ):
        register_estimator(
            EstimatorEntry(
                quantity="verifier.decision_coverage",
                impl=impl,
                requires=ACCESS_SOURCE_AND_RECORD,
                envelope=_COVERAGE_ENVELOPE,
                rung=rung,
                bias=BiasStatement(
                    direction="upward",
                    why=(
                        f"{what}. Every rung of coverage is biased upward as a measure of what the "
                        f"grader can distinguish: taking a branch is not the same as the outcome "
                        f"depending on it, which is the gap D2 measures directly."
                    ),
                ),
                cost=CostModel(note="one traced pass over the corpus; no GPU"),
                substrates=frozenset({Substrate.PROGRAM}),
                phases=frozenset({Phase.PRE_RUN, Phase.POST_RUN}),
                run=None,
            )
        )


_register()


__all__ = [
    "PATH_CAP",
    "CoverageReading",
    "DecisionCoverage",
    "branch_universe",
    "measure_coverage",
    "trace_corpus",
]

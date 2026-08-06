"""Resolving the access matrix from what was actually supplied, by probing rather than assuming.

`Access` is a per-component flag set. This module answers the question one level down: given a record directory, a grader endpoint, a policy checkpoint and an environment source
tree, which flags are actually true? The answer cannot be read off the argument list. A `--grader`
URL is a string until something calls it, and a string that 404s is not `QUERY`.

The distinction that earns this module its place is `REPLICATE`. It does not follow from `QUERY`,
and the difference is not cosmetic: without controlled facet variation there is no variance
decomposition, no effective group size and no attenuation factor, so a resolver that hands out
`REPLICATE` because a URL was supplied silently promises half of series A to someone behind an API
that will not take a seed. The only way to know is to call the endpoint twice with two seeds and
once more with the first seed again, and compare. That is three calls, and three calls cost money,
so **the default resolution makes none of them**. Probing is opt-in and carries a declared call
budget, and the number of calls actually spent is reported.

What the resolver will not do is guess in the generous direction. A supplied policy checkpoint
resolves to `FORWARD` and not to `BACKWARD`, because establishing `BACKWARD` means loading the
weights and this package imports no torch. A record's task references resolve to `TASK: RECORD` and
not to `TASK: QUERY`, because a reference to a task is not the ability to run one. Both of those are
narrower than an illustrative report would print and both are deliberate: `declared=` exists for a
caller who knows better than the probe, and a declared flag is labelled as declared in the report
rather than passed off as measured.
"""

from __future__ import annotations

import inspect
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

from reward_lens.core.types import Access, AccessMatrix, Component

# ---------------------------------------------------------------------------
# Rendering an Access value
# ---------------------------------------------------------------------------


def render_access(access: Access) -> str:
    """``QUERY | REPLICATE``, in flag declaration order.

    `Access` is an `IntFlag`, so `str()` on it is the integer. The report prints flag names, and
    `NONE` prints as `NONE` rather than as an empty string: "this component is reachable in no way"
    is a finding and it should occupy a line.
    """
    if not access:
        return "NONE"
    return " | ".join(m.name for m in Access if m.value and (access & m) and m.name)


# ---------------------------------------------------------------------------
# What can be read off a record without opening it
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecordFacts:
    """The handful of record properties the capability report prints and reasons about.

    This deliberately does not depend on a concrete record schema: it reads a supplied object by
    attribute name against a small list of candidates and records what it could not find in
    ``unread``. A field that could not be read is named, never defaulted to a plausible number: a
    sampling fraction that silently reads 1.0 when the record does not carry one is exactly the
    class of quiet wrong answer the rest of this library exists to stop.
    """

    path: str | None = None
    n_steps: int | None = None
    n_rollouts: int | None = None
    sampling: float | None = None
    complete: bool | None = None
    #: Optional per-rollout streams present in the record: "advantages", "logprobs", "scores",
    #: "leaf_scores". A tap that is off is the difference between a quantity being estimable and
    #: being an instruction to re-run the job with the tap on.
    taps: frozenset[str] = frozenset()
    unread: tuple[str, ...] = ()

    @property
    def is_present(self) -> bool:
        return self.path is not None or self.n_steps is not None or bool(self.taps)

    def render(self) -> str:
        """``401 steps, 25,664 rollouts, sampling 1.0``, omitting what could not be read."""
        bits: list[str] = []
        if self.n_steps is not None:
            bits.append(f"{self.n_steps:,} steps")
        if self.n_rollouts is not None:
            bits.append(f"{self.n_rollouts:,} rollouts")
        if self.sampling is not None:
            bits.append(f"sampling {self.sampling}")
        if self.complete is False:
            bits.append("still being written")
        if not bits:
            where = f" at {self.path}" if self.path else ""
            return f"a record{where}, not opened, so none of its counts are in this report"
        if self.unread:
            bits.append(f"not read: {', '.join(self.unread)}")
        return ", ".join(bits)


_STEP_NAMES = ("n_steps", "num_steps", "steps", "step_count")
_ROLLOUT_NAMES = ("n_rollouts", "num_rollouts", "rollouts", "rollout_count")
_SAMPLING_NAMES = ("sampling", "sampling_fraction", "sample_fraction")
_COMPLETE_NAMES = ("complete", "is_complete", "closed", "finished")
_TAP_NAMES = {
    "advantages": ("has_advantages", "per_token_advantages", "advantages"),
    "logprobs": ("has_logprobs", "logprobs", "policy_logprobs"),
    "scores": ("has_scores", "scores", "grader_scores"),
    "leaf_scores": ("has_leaf_scores", "leaf_scores", "score_tree"),
    "task_refs": ("task_refs", "tasks", "task_ids"),
    "policy_versions": ("policy_versions", "segment_provenance"),
}


def _first_attr(obj: Any, names: Sequence[str]) -> tuple[Any, bool]:
    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            return obj[name], True
        if hasattr(obj, name):
            return getattr(obj, name), True
    return None, False


def _as_count(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return len(value)
    except TypeError:
        return None


def record_facts(record: Any) -> RecordFacts:
    """Read a record object, a mapping, or a path into `RecordFacts`.

    A path is not opened. Opening a record is the record reader's job and this module stays usable
    without it, so a path resolves to "a record is present here" and every count lands in
    ``unread``. That is honest and it still resolves `RECORD` access, which is what the report needs
    from it.
    """
    if record is None:
        return RecordFacts()
    if isinstance(record, RecordFacts):
        return record
    if isinstance(record, (str, os.PathLike)):
        p = Path(record)
        return RecordFacts(
            path=str(p),
            unread=("steps", "rollouts", "sampling", "taps")
            if p.exists()
            else ("the path does not exist",),
        )

    unread: list[str] = []
    steps, ok = _first_attr(record, _STEP_NAMES)
    if not ok:
        unread.append("steps")
    rollouts, ok = _first_attr(record, _ROLLOUT_NAMES)
    if not ok:
        unread.append("rollouts")
    sampling, ok = _first_attr(record, _SAMPLING_NAMES)
    if not ok:
        unread.append("sampling")
    complete, ok = _first_attr(record, _COMPLETE_NAMES)
    if not ok:
        complete = None

    taps: set[str] = set()
    for tap, names in _TAP_NAMES.items():
        value, ok = _first_attr(record, names)
        if ok and value:
            taps.add(tap)

    path, _ = _first_attr(record, ("path", "root", "directory"))
    return RecordFacts(
        path=str(path) if path is not None else None,
        n_steps=_as_count(steps),
        n_rollouts=_as_count(rollouts),
        sampling=float(sampling) if isinstance(sampling, (int, float)) else None,
        complete=bool(complete) if complete is not None else None,
        taps=frozenset(taps),
        unread=tuple(unread),
    )


# ---------------------------------------------------------------------------
# Probing a grader endpoint
# ---------------------------------------------------------------------------


@runtime_checkable
class ScoreEndpoint(Protocol):
    """What the probe calls. Anything callable satisfies it; `.score` is checked first.

    The seed is keyword-only and optional on purpose. An endpoint that does not accept one is the
    case the whole `REPLICATE` flag exists to detect, so refusing to type it out of existence is
    the point.
    """

    def __call__(self, item: Any, *, seed: int | None = None) -> Any: ...


#: The item the probe scores when the caller supplies none. Deliberately tiny and self-describing,
#: so a grader operator reading their logs can see what it was.
PROBE_ITEM: dict[str, str] = {
    "prompt": "reward-lens capability probe. Any score is fine.",
    "completion": "probe",
}

#: The full seed probe: one call at seed A, one at seed B, one back at seed A. The third call is
#: what separates "the seed controls the variation" from "the endpoint is just noisy", and those
#: two are different access levels rather than two descriptions of one.
SEED_PROBE_CALLS = 3


@dataclass(frozen=True)
class ProbeBudget:
    """How much the caller has authorised the resolver to spend establishing access.

    Zero calls is the default and it is not a degenerate case: it is the normal way to run
    `reward-lens capabilities`, and it resolves everything that can be resolved from files on disk.
    A budget is a ceiling rather than a target; `ProbeResult.calls_made` reports what was actually
    spent so a report that cost money says how much.
    """

    calls: int = 0
    dollars_per_call: float | None = None
    seeds: tuple[int, int] = (0, 1)
    note: str = ""

    @property
    def allows_seed_probe(self) -> bool:
        return self.calls >= SEED_PROBE_CALLS

    def render(self) -> str:
        if self.calls <= 0:
            return "no calls authorised"
        cost = (
            f", up to ${self.calls * self.dollars_per_call:,.2f}"
            if self.dollars_per_call is not None
            else ""
        )
        return f"up to {self.calls} calls{cost}"


#: The default. Resolution makes no calls at all unless a caller asks for them.
DRY = ProbeBudget()


@dataclass(frozen=True)
class ProbeResult:
    """What calling the endpoint established, what it cost, and what is still unresolved."""

    access: Access
    verdict: str
    note: str
    calls_made: int = 0
    unresolved: tuple[str, ...] = ()
    error: str = ""

    def __post_init__(self) -> None:
        if not self.note.strip():
            raise ValueError(
                "a probe result with no note explains nothing to the person reading it"
            )


def _freeze(value: Any) -> Any:
    """A comparable form of a score, for two-call equality."""
    if isinstance(value, (str, bytes, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, (int, float)):
        return float(value)
    return repr(value)


def _same(a: Any, b: Any) -> bool:
    """Whether two scores are the same reading, with a float tolerance.

    The tolerance is tight (1e-12 relative). It is there so that two bit-different
    representations of one arithmetic result do not read as facet variation, not to paper over a
    grader that genuinely moved.
    """
    fa, fb = _freeze(a), _freeze(b)
    if isinstance(fa, float) and isinstance(fb, float):
        return math.isclose(fa, fb, rel_tol=1e-12, abs_tol=1e-15)
    if isinstance(fa, tuple) and isinstance(fb, tuple) and len(fa) == len(fb):
        return all(_same(x, y) for x, y in zip(fa, fb, strict=True))
    return bool(fa == fb)


def _callable_of(endpoint: Any) -> Callable[..., Any]:
    score = getattr(endpoint, "score", None)
    return score if callable(score) else endpoint


def _accepts_seed(fn: Callable[..., Any]) -> bool | None:
    """True, False, or None for "the signature does not say; find out by calling"."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return None
    if "seed" in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


class _SeedRejected(Exception):
    """The endpoint refused a seed argument. Not an error, a finding."""


def _call(fn: Callable[..., Any], item: Any, seed: int | None) -> Any:
    if seed is None:
        return fn(item)
    try:
        return fn(item, seed=seed)
    except TypeError as exc:
        if "seed" in str(exc):
            raise _SeedRejected(str(exc)) from None
        raise


def probe_endpoint(
    endpoint: Any,
    *,
    budget: ProbeBudget = DRY,
    item: Any = None,
    seeds: tuple[int, int] | None = None,
) -> ProbeResult:
    """Establish `QUERY` and `REPLICATE` on a grader endpoint, within a call budget.

    Five outcomes, and four of them are `QUERY` without `REPLICATE`:

    ``not_probed``
        The budget authorised no calls. Nothing is established, which is different from nothing
        being available, and ``unresolved`` says so.
    ``unreachable``
        The call raised. `Access.NONE`, with the exception text, because an endpoint that will not
        answer is not access.
    ``no_seed``
        It answers and rejects a seed argument. This is the hosted judge with a fixed internal
        seed, and it is the case `REPLICATE` was separated from `QUERY` to describe.
    ``seed_ignored``
        It accepts a seed and returns the same score for both. From outside, a fixed internal seed
        and a deterministic grader are indistinguishable, so this resolves to `QUERY` and says
        which two possibilities it cannot separate. A caller who knows the grader is deterministic
        passes ``declared={Component.GRADER: Access.REPLICATE}`` and the report labels it declared.
    ``seed_uncontrolled``
        Two calls at the same seed disagree. The endpoint varies and the seed does not control the
        variation, so test-retest is possible and a crossed design is not. `REPLICATE` is
        controlled facet variation, so this resolves to `QUERY`.
    ``seed_honoured``
        Same seed reproduces, different seeds differ. `QUERY | REPLICATE`.
    """
    if endpoint is None:
        return ProbeResult(
            access=Access.NONE,
            verdict="absent",
            note="no grader endpoint supplied",
            unresolved=("QUERY", "REPLICATE"),
        )
    if budget.calls <= 0:
        return ProbeResult(
            access=Access.NONE,
            verdict="not_probed",
            note=(
                "endpoint supplied and not called. A "
                f"{SEED_PROBE_CALLS}-call probe would establish QUERY and REPLICATE; pass a probe "
                "budget to authorise it"
            ),
            unresolved=("QUERY", "REPLICATE"),
        )

    fn = _callable_of(endpoint)
    a, b = seeds if seeds is not None else budget.seeds
    payload = PROBE_ITEM if item is None else item
    calls = 0

    declared = _accepts_seed(fn)
    try:
        if declared is False:
            _call(fn, payload, None)
            calls += 1
            return ProbeResult(
                access=Access.QUERY,
                verdict="no_seed",
                note="endpoint responds; it takes no seed argument, so facets cannot be varied",
                calls_made=calls,
                unresolved=(),
            )
        first = _call(fn, payload, a)
        calls += 1
    except _SeedRejected:
        try:
            _call(fn, payload, None)
            calls += 1
        except Exception as exc:  # noqa: BLE001 - the exception text is the finding
            return ProbeResult(
                access=Access.NONE,
                verdict="unreachable",
                note=f"the endpoint raised {type(exc).__name__}: {exc}",
                calls_made=calls,
                unresolved=("QUERY", "REPLICATE"),
                error=f"{type(exc).__name__}: {exc}",
            )
        return ProbeResult(
            access=Access.QUERY,
            verdict="no_seed",
            note="endpoint responds; it rejects a seed argument, so facets cannot be varied",
            calls_made=calls,
        )
    except Exception as exc:  # noqa: BLE001 - the exception text is the finding
        return ProbeResult(
            access=Access.NONE,
            verdict="unreachable",
            note=f"the endpoint raised {type(exc).__name__}: {exc}",
            calls_made=calls,
            unresolved=("QUERY", "REPLICATE"),
            error=f"{type(exc).__name__}: {exc}",
        )

    if not budget.allows_seed_probe:
        return ProbeResult(
            access=Access.QUERY,
            verdict="query_only_budget",
            note=(
                f"endpoint responds. The seed probe needs {SEED_PROBE_CALLS} calls and the budget "
                f"is {budget.render()}, so REPLICATE is unresolved rather than absent"
            ),
            calls_made=calls,
            unresolved=("REPLICATE",),
        )

    try:
        second = _call(fn, payload, b)
        calls += 1
        third = _call(fn, payload, a)
        calls += 1
    except Exception as exc:  # noqa: BLE001 - the exception text is the finding
        return ProbeResult(
            access=Access.QUERY,
            verdict="query_only_error",
            note=(
                f"endpoint responds at seed {a}; the second call raised "
                f"{type(exc).__name__}: {exc}, so REPLICATE is unresolved"
            ),
            calls_made=calls,
            unresolved=("REPLICATE",),
            error=f"{type(exc).__name__}: {exc}",
        )

    if not _same(first, third):
        return ProbeResult(
            access=Access.QUERY,
            verdict="seed_uncontrolled",
            note=(
                f"endpoint responds and two calls at seed {a} disagree, so the seed does not "
                "control the variation. Test-retest is available; a crossed design is not"
            ),
            calls_made=calls,
        )
    if _same(first, second):
        return ProbeResult(
            access=Access.QUERY,
            verdict="seed_ignored",
            note=(
                f"endpoint responds and accepts a seed, and seed {a} and seed {b} returned the "
                "same score. A fixed internal seed and a deterministic grader look identical from "
                "outside, so REPLICATE is not established"
            ),
            calls_made=calls,
        )
    return ProbeResult(
        access=Access.QUERY | Access.REPLICATE,
        verdict="seed_honoured",
        note="endpoint responds; seed is settable",
        calls_made=calls,
    )


def http_endpoint(
    url: str,
    *,
    timeout: float = 20.0,
    headers: Mapping[str, str] | None = None,
    build_request: Callable[[Any, int | None], Any] | None = None,
    parse_response: Callable[[Any], Any] | None = None,
) -> ScoreEndpoint:
    """A `ScoreEndpoint` over an HTTP scoring URL, on the standard library alone.

    The default wire format posts ``{"item": <item>, "seed": <int>}`` as JSON and reads the first
    of ``score``, ``scores`` or ``reward`` out of the JSON response. That is a guess about somebody
    else's API and it is meant to be replaced: ``build_request`` and ``parse_response`` take the
    guess out. Nothing here is called until `probe_endpoint` is given a budget above zero, so
    constructing one of these costs nothing and sends nothing.
    """
    import json
    import urllib.request

    def _default_build(item: Any, seed: int | None) -> bytes:
        body: dict[str, Any] = {"item": item}
        if seed is not None:
            body["seed"] = seed
        return json.dumps(body).encode("utf-8")

    def _default_parse(payload: Any) -> Any:
        if isinstance(payload, Mapping):
            for key in ("score", "scores", "reward"):
                if key in payload:
                    return payload[key]
        return payload

    builder = build_request or _default_build
    parser = parse_response or _default_parse

    def call(item: Any, *, seed: int | None = None) -> Any:
        data = builder(item, seed)
        if not isinstance(data, (bytes, bytearray)):
            data = json.dumps(data).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json", **dict(headers or {})},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            raw = response.read().decode("utf-8")
        try:
            return parser(json.loads(raw))
        except json.JSONDecodeError:
            return parser(raw)

    return call  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# The resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComponentAccess:
    """One row of ACCESS RESOLVED: the flags, and where they came from."""

    component: Component
    access: Access
    note: str
    unresolved: tuple[str, ...] = ()

    def render(self) -> str:
        return f"{render_access(self.access)}"


@dataclass(frozen=True)
class AccessResolution:
    """The access matrix, plus the provenance of every flag in it.

    `matrix` is the plain `AccessMatrix` the rest of the library consumes. The notes exist because
    the capability report has to answer "why do I not have `POLICY: BACKWARD`" without the reader
    going to the source, and the answer is usually a missing argument rather than a missing
    permission.
    """

    components: tuple[ComponentAccess, ...] = ()
    calls_made: int = 0
    probes: Mapping[Component, ProbeResult] = field(default_factory=dict)

    @property
    def matrix(self) -> AccessMatrix:
        return {c.component: c.access for c in self.components}

    def of(self, component: Component) -> Access:
        for c in self.components:
            if c.component is component:
                return c.access
        return Access.NONE

    def note_for(self, component: Component) -> str:
        for c in self.components:
            if c.component is component:
                return c.note
        return ""

    @property
    def unresolved(self) -> tuple[str, ...]:
        out: list[str] = []
        for c in self.components:
            for flag in c.unresolved:
                out.append(f"{c.component.name}: {flag}")
        return tuple(out)


#: The order ACCESS RESOLVED prints components in.
_REPORT_ORDER = (
    Component.TASK,
    Component.GRADER,
    Component.POLICY,
    Component.RECORD,
    Component.ESTIMATOR,
    Component.OPTIMIZER,
    Component.ARTIFACT,
    Component.GOLD,
)


def resolve_access(
    *,
    record: Any = None,
    grader: Any = None,
    policy: Any = None,
    environment: Any = None,
    probe: ProbeBudget = DRY,
    declared: AccessMatrix | None = None,
    probe_item: Any = None,
) -> AccessResolution:
    """Resolve the access matrix from what was supplied, spending at most ``probe.calls`` calls.

    Where each flag comes from, and where it deliberately does not:

    `RECORD` follows from a record being present. `GRADER: RECORD` follows from that record
    carrying scores. `GRADER: QUERY` and `GRADER: REPLICATE` follow only from `probe_endpoint`, so
    with the default budget a supplied endpoint contributes nothing and the note says a probe would
    settle it.

    `TASK: RECORD` follows from the record's task references. It does not give `QUERY`: a reference
    to a task is a pointer, not a runner, and this resolver is deliberately narrower here than an
    illustrative report would be. Supplying an environment source tree gives `TASK: MUTATE`, because source
    you hold is source you can edit, and it still does not give `QUERY`, because a source tree that
    has never been stood up is not known to run.

    `POLICY: RECORD` follows from a record. A supplied checkpoint adds `FORWARD`. It does not add
    `BACKWARD`: differentiating through the policy means loading the weights, this package imports
    no torch, and an unverified `BACKWARD` would promise gradients to someone whose checkpoint is a
    sharded artifact nothing here has opened.

    ``declared`` is the escape hatch for all of it. Flags supplied there are added and the note
    records that they were declared rather than measured, so the report never presents a caller's
    assertion as this module's finding.
    """
    facts = record_facts(record)
    rows: dict[Component, ComponentAccess] = {}
    probes: dict[Component, ProbeResult] = {}

    # -- RECORD ------------------------------------------------------------
    if facts.is_present:
        rows[Component.RECORD] = ComponentAccess(Component.RECORD, Access.RECORD, facts.render())
    else:
        rows[Component.RECORD] = ComponentAccess(
            Component.RECORD,
            Access.NONE,
            "no record supplied; --record points at a run directory",
        )

    # -- GRADER ------------------------------------------------------------
    grader_access = Access.NONE
    grader_notes: list[str] = []
    grader_unresolved: list[str] = []
    if "scores" in facts.taps:
        grader_access |= Access.RECORD
        grader_notes.append("the record carries scores")
    if grader is not None:
        result = probe_endpoint(grader, budget=probe, item=probe_item)
        probes[Component.GRADER] = result
        grader_access |= result.access
        grader_notes.append(result.note)
        grader_unresolved.extend(result.unresolved)
    elif not grader_notes:
        grader_notes.append("no grader endpoint supplied; --grader would add QUERY once probed")
    rows[Component.GRADER] = ComponentAccess(
        Component.GRADER,
        grader_access,
        "; ".join(grader_notes),
        tuple(grader_unresolved),
    )

    # -- TASK --------------------------------------------------------------
    task_access = Access.NONE
    task_notes: list[str] = []
    if "task_refs" in facts.taps:
        task_access |= Access.RECORD
        task_notes.append("from the record's task refs")
    if environment is not None:
        task_access |= Access.MUTATE
        task_notes.append(
            f"environment source at {environment}; QUERY needs a task that has been run, which "
            "this command does not do"
        )
    if not task_notes:
        task_notes.append("no task refs in the record and no environment source supplied")
    rows[Component.TASK] = ComponentAccess(
        Component.TASK,
        task_access,
        "; ".join(task_notes),
        () if environment is not None else ("QUERY",),
    )

    # -- POLICY ------------------------------------------------------------
    policy_access = Access.NONE
    policy_notes: list[str] = []
    policy_unresolved: list[str] = []
    if facts.is_present:
        policy_access |= Access.RECORD
    if policy is None:
        policy_notes.append("no weights supplied; --policy would add FORWARD")
        policy_unresolved.extend(("FORWARD", "BACKWARD"))
    else:
        policy_access |= Access.FORWARD
        policy_notes.append(
            f"checkpoint at {policy}; BACKWARD needs the weights loaded and this command loads "
            "nothing, so it is unresolved rather than absent"
        )
        policy_unresolved.append("BACKWARD")
    rows[Component.POLICY] = ComponentAccess(
        Component.POLICY,
        policy_access,
        "; ".join(policy_notes),
        tuple(policy_unresolved),
    )

    # -- declared ----------------------------------------------------------
    for component, access in (declared or {}).items():
        existing = rows.get(component)
        added = access & ~(existing.access if existing else Access.NONE)
        merged = (existing.access if existing else Access.NONE) | access
        note = f"declared by the caller, not probed: {render_access(added)}" if added else ""
        if existing is not None:
            note = "; ".join(n for n in (existing.note, note) if n)
            rows[component] = ComponentAccess(
                component,
                merged,
                note,
                tuple(f for f in existing.unresolved if f not in render_access(added).split(" | ")),
            )
        else:
            rows[component] = ComponentAccess(component, merged, note or "declared by the caller")

    ordered = tuple(rows[c] for c in _REPORT_ORDER if c in rows) + tuple(
        rows[c] for c in rows if c not in _REPORT_ORDER
    )
    return AccessResolution(
        components=ordered,
        calls_made=sum(p.calls_made for p in probes.values()),
        probes=probes,
    )


def resolution_from_matrix(
    matrix: AccessMatrix, note: str = "supplied by the caller"
) -> AccessResolution:
    """Wrap a bare `AccessMatrix` so the report can consume it without provenance.

    Used when someone calls `capability_report` with a plain matrix rather than a resolution.
    The note is the same for every row, which is the truth: nothing here was resolved.
    """
    return AccessResolution(
        components=tuple(
            ComponentAccess(c, matrix.get(c, Access.NONE), note)
            for c in _REPORT_ORDER
            if c in matrix
        )
        + tuple(ComponentAccess(c, a, note) for c, a in matrix.items() if c not in _REPORT_ORDER)
    )


__all__ = [
    "DRY",
    "PROBE_ITEM",
    "SEED_PROBE_CALLS",
    "AccessResolution",
    "ComponentAccess",
    "ProbeBudget",
    "ProbeResult",
    "RecordFacts",
    "ScoreEndpoint",
    "http_endpoint",
    "probe_endpoint",
    "record_facts",
    "render_access",
    "resolution_from_matrix",
    "resolve_access",
]

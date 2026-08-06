"""The information barrier, enforced by type.

This is the same mechanism as `Blind[T]` applied to time instead of to labels, and it catches at
construction the failure mode that every prospective-monitoring method read in the scan has in
review. The strongest of them fits its concept vectors at checkpoint 200 for an onset at step 164,
with the checkpoint chosen because extreme labels are most abundant there, which is a selection
criterion computed from the outcome. Nothing in that paper is dishonest; there was simply no object
that would have refused to be built.

`issue` is that object's constructor. It walks the parent DAG of every input evidence id and raises
`ForecastLeakageError` if a single ancestor has an information time at or after the issue instant,
naming the offending id, both timestamps, and the path from the input that reached it.

**There is no override.** Not a flag, not a keyword, not an escape hatch for tests. A test that
needs a forecast constructs a valid one, which costs a line and is the only way the guarantee means
anything: an override exists to be used on the afternoon somebody is in a hurry, which is precisely
the afternoon the leak happens.

Two conditions, two errors, for the same reason distinct refusal reasons stay distinct. An ancestor
that postdates the issue is a leak and the remedy is to issue earlier or drop the input. An ancestor
the store cannot resolve is not a leak, it is an unverifiable claim, and the remedy is to append the
parent so the barrier can check it. Those point in different directions, so they are not the same
error with two messages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from reward_lens.core.errors import ProvenanceError, RewardLensError
from reward_lens.core.evidence import Evidence
from reward_lens.core.provenance import Provenance
from reward_lens.core.store import EvidenceStore
from reward_lens.core.types import EvidenceID, SubjectRef
from reward_lens.forecast.schema import (
    BaselineForecast,
    DecisionSpec,
    Distribution,
    Forecast,
    ForecastError,
    HorizonSpec,
    InformationTime,
    QuantityID,
    ReferenceClass,
    ResolutionRule,
    forecast_id,
)


class ForecastLeakageError(RewardLensError):
    """An input, transitively, postdates the instant the forecast claims to have been made.

    Carries the offending evidence id, both timestamps and the derivation path, because the first
    question anybody asks is which input and the second is how it got in. There is no way to
    suppress this and no keyword that turns it off.
    """

    def __init__(
        self,
        message: str,
        *,
        evidence_id: str,
        input_id: str,
        information_time: InformationTime,
        issued_at: InformationTime,
        path: Sequence[str] = (),
    ) -> None:
        super().__init__(message)
        self.evidence_id = evidence_id
        self.input_id = input_id
        self.information_time = information_time
        self.issued_at = issued_at
        self.path = tuple(path)


@dataclass(frozen=True)
class _Node:
    """The four fields the barrier reads off a stored row. Deliberately not an `Evidence`.

    Walking a DAG to check timestamps must not decode payloads. Doing so is slower on every hop and
    it makes the barrier fail outright on any store whose payload module is not importable in this
    process, which is the ordinary case for an archive somebody else wrote: the campaign store
    raises `PayloadTypeUnregistered` on its own result rows. Provenance is a property of the
    envelope, so the envelope is what this reads.
    """

    id: str
    observable: str
    information_time: str
    created_at: str
    parents: tuple[str, ...]


def _envelopes(store: EvidenceStore) -> dict[str, Any]:
    """The store's raw envelope index, through a public accessor when one exists.

    `EvidenceStore` exposes no public iterator over raw envelopes: `__iter__`, `get` and `find` all
    go through `evidence_from_envelope`, which decodes the payload. `record/convert/store.py` needed
    the same thing and guards it the same way, so adding a public accessor upstream removes both
    branches rather than breaking them.
    """
    public = getattr(store, "envelopes", None)
    if callable(public):
        return {env["id"]: env for env in public()}
    return store._index  # noqa: SLF001


def information_time_of(evidence: Evidence[Any], *, basis: str = "") -> InformationTime:
    """Read the third clock off a stored reading.

    `Evidence.information_time` defaults to `created_at`, which is right for a fresh measurement and
    wrong for a reanalysis. A reanalysis sets it, and this function does not second-guess either
    case: it reports what the row says. The basis records which of the two it was, so a barrier
    refusal names not just an instant but where the instant came from.
    """
    instant = evidence.information_time or evidence.created_at
    if not basis:
        basis = (
            "the row's own information_time"
            if evidence.information_time and evidence.information_time != evidence.created_at
            else f"defaulted to created_at on {evidence.observable}"
        )
    return InformationTime.parse(instant, basis=basis)


def _node_time(node: _Node) -> InformationTime:
    """The same rule as `information_time_of`, applied to a raw envelope."""
    instant = node.information_time or node.created_at
    basis = (
        "the row's own information_time"
        if node.information_time and node.information_time != node.created_at
        else f"defaulted to created_at on {node.observable}"
    )
    return InformationTime.parse(instant, basis=basis)


def _resolve(index: Mapping[str, Any], ev_id: str) -> _Node:
    env = index.get(ev_id)
    if env is None:
        raise ProvenanceError(
            f"evidence {ev_id} is not in the store, so the information barrier cannot certify that "
            f"it predates the forecast. This is not a leak, it is an unverifiable input: nothing "
            f"here knows when {ev_id} became available. Append the measurement to the store before "
            f"issuing a forecast that consumes it."
        )
    provenance = env.get("provenance", {}) or {}
    return _Node(
        id=str(env["id"]),
        observable=str(env.get("observable", "")),
        information_time=str(env.get("information_time") or ""),
        created_at=str(env.get("created_at") or ""),
        parents=tuple(str(p) for p in provenance.get("parents", ()) or ()),
    )


def ancestry(store: EvidenceStore, inputs: Iterable[str]) -> dict[str, tuple[str, ...]]:
    """Every ancestor of every input, mapped to the path that reached it.

    Breadth-first, so the recorded path to an ancestor is the shortest derivation chain from an
    input to it, which is the one a reader can follow. Cycles cannot occur in a valid store (the
    merge raises on them) and are tolerated here anyway by the visited set, because a barrier that
    hangs on a corrupt store is worse than one that reports on it.
    """
    index = _envelopes(store)
    seen: dict[str, tuple[str, ...]] = {}
    frontier: list[tuple[str, tuple[str, ...]]] = [(str(i), (str(i),)) for i in inputs]
    while frontier:
        nxt: list[tuple[str, tuple[str, ...]]] = []
        for ev_id, path in frontier:
            if ev_id in seen:
                continue
            seen[ev_id] = path
            node = _resolve(index, ev_id)
            for parent in node.parents:
                if parent not in seen:
                    nxt.append((parent, path + (parent,)))
        frontier = nxt
    return seen


def check_barrier(
    store: EvidenceStore, inputs: Sequence[str], at: InformationTime
) -> Mapping[str, InformationTime]:
    """Raise if any input, transitively, has an information time at or after ``at``.

    Returns the information time of every ancestor when it passes, so a caller that wants to record
    how close it came (the latest ancestor is the binding constraint on the issue instant) can do so
    without walking the DAG a second time.

    The comparison is ``>=`` rather than ``>``. An ancestor recorded at exactly the issue instant is
    a coin flip on write ordering, and a barrier that depends on microsecond ordering inside one
    process is not a barrier.
    """
    index = _envelopes(store)
    reached = ancestry(store, inputs)
    times: dict[str, InformationTime] = {}
    # Shallowest first, then by id, so the offender named is the one closest to something the
    # caller passed in. A refusal that names a great-grandparent when the input itself leaks sends
    # the reader three hops away from the line they need to change.
    for ev_id, path in sorted(reached.items(), key=lambda kv: (len(kv[1]), kv[0])):
        node = _resolve(index, ev_id)
        when = _node_time(node)
        times[ev_id] = when
        if when >= at:
            hops = " -> ".join(path)
            direct = "the input itself" if len(path) == 1 else f"reached from input {path[0]}"
            raise ForecastLeakageError(
                f"forecast input leaks the future: {ev_id} ({node.observable}) has "
                f"information_time {when.instant} and the forecast is issued at {at.instant}, so "
                f"the forecaster could not have known it. {direct.capitalize()}, by "
                f"{hops}.\n"
                f"    ancestor information_time  {when.instant}   basis: {when.basis}\n"
                f"    forecast issued_at         {at.instant}   basis: {at.basis}\n"
                f"    difference                 {when.epoch - at.epoch:+.3f} s\n"
                f"Issue the forecast at an instant before {when.instant}, or drop {ev_id} and "
                f"whatever was derived from it from the inputs. There is no override: a forecast "
                f"fitted on its own outcome is the failure this package exists to make impossible, "
                f"not a case to be waived.",
                evidence_id=ev_id,
                input_id=path[0],
                information_time=when,
                issued_at=at,
                path=path,
            )
    return times


def issue(
    target: QuantityID,
    subject: SubjectRef,
    resolution: ResolutionRule,
    distribution: Distribution,
    inputs: Sequence[EvidenceID],
    *,
    at: InformationTime,
    store: EvidenceStore,
    reference_class: ReferenceClass,
    horizon: HorizonSpec,
    method: str,
    baselines: Sequence[BaselineForecast],
    decision: DecisionSpec | None = None,
    meta_plan: str | None = None,
    provenance: Provenance | None = None,
    issued_step: int | None = None,
) -> Forecast:
    """Construct a Forecast, refusing if any input, transitively, postdates ``at``.

    Walks the parent DAG of every input evidence id. A single ancestor with an information time at
    or after ``at`` raises `ForecastLeakageError` naming the offending id and both timestamps. There
    is no override.

    ``store`` is not optional. The barrier's promise is a statement about ancestors, and ancestors
    live in a store; a signature that cannot reach one can check the inputs it was handed and
    nothing behind them, which is exactly the depth at which the published failures happen. It is a
    required keyword rather than a default to the process-wide store, because silently walking a
    different DAG than the caller meant is the one way this check can pass while meaning nothing.

    The mandatory baselines are enforced by `Forecast.__post_init__`, so a forecast issued here is
    barrier-clean and comparator-complete or it does not exist.
    """
    ids: tuple[EvidenceID, ...] = tuple(EvidenceID(str(i)) for i in inputs)
    if not ids:
        # Not a leak and not a refusal. A call made from nothing has no provenance to certify, so
        # the barrier has nothing to say about it, and saying nothing would read as saying it is
        # clean.
        raise ForecastError(
            f"a forecast on {target!r} was issued with no inputs. The barrier certifies that every "
            f"input predates the issue instant and it cannot certify an empty set: a call made "
            f"from nothing is either a prior, which should name the reference-class row it came "
            f"from, or a call whose inputs were not recorded. Pass the evidence ids the forecast "
            f"actually used."
        )
    check_barrier(store, ids, at)

    fid = forecast_id(
        target=target,
        subject=subject,
        resolution=resolution,
        issued_at=at,
        distribution=distribution,
        inputs=ids,
        method=method,
    )
    prov = provenance or Provenance()
    if not prov.parents:
        # The inputs are the parents. Recording them here is what lets a scored forecast be
        # appended to the same store as a derived Evidence: the store refuses a child whose parents
        # it cannot resolve, and the barrier has just proved every one of them resolves.
        prov = Provenance(
            git_sha=prov.git_sha,
            config_hash=prov.config_hash,
            seeds=prov.seeds,
            cost=prov.cost,
            oracle_calls=prov.oracle_calls,
            parents=ids,
            study=prov.study,
            extra=prov.extra,
        )
    return Forecast(
        id=fid,
        target=target,
        subject=subject,
        resolution=resolution,
        issued_at=at,
        horizon=horizon,
        reference_class=reference_class,
        distribution=distribution,
        method=method,
        inputs=ids,
        baselines=tuple(baselines),
        decision=decision,
        meta_plan=meta_plan,
        provenance=prov,
        issued_step=issued_step,
    )


__all__ = [
    "ForecastLeakageError",
    "ancestry",
    "check_barrier",
    "information_time_of",
    "issue",
]

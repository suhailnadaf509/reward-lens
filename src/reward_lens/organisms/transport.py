"""Selection diagrams: which quantities may legally cross from the organism to the target.

K2's rung 2. Rung 0 measures the gap between a planted organism and a real corpus and calls it
`t32`. Rung 1 removes the matrix effect by dosing the target instead
(`organisms/standard_addition.py`). Neither of them says *which* quantities were entitled to cross
in the first place, and that question has an answer with a literature: Pearl and Bareinboim's
selection diagrams, where the differences between two domains are drawn as extra nodes and
transportability becomes a graphical property rather than a judgement call.

The diagram is the ordinary causal graph over the variables, plus one **S-node** per variable whose
mechanism differs between the organism and the target. An S-node pointing at `V` is the claim "the
way `V` is generated is not the same in the two domains". Drawing it is the work: once it is drawn,
whether a quantity transports is a d-separation question and this module answers it.

Three verdicts, and the middle one is the useful one
----------------------------------------------------

    direct              every S-node is d-separated from the outcome given the treatment and the
                        admissible set. The organism's estimate transports unchanged.
    reweighted          the S-nodes are d-separated from the outcome once you also condition on a
                        set of variables you can measure in *both* domains. The organism's estimate
                        transports after reweighting on that set, which is a computation you can
                        actually do and is why the diagram is worth drawing.
    not_transportable   no measurable set blocks the S-nodes. The organism's number is not an
                        estimate of the target's quantity, and no amount of extra organism data
                        fixes it.

What this is and is not
-----------------------

This implements the **sufficient conditions** on a diagram, by d-separation, and not the complete
sID algorithm. A `not_transportable` verdict here means "no admissible set was found among the
candidates offered", which is weaker than sID's "no transport formula exists". The verdict names
that limit in its own text rather than in a footnote, because a graphical criterion that overstates
its completeness is worse than one that does not run: it converts "we did not find a licence" into
"there is no licence" and stops the search.

The other limit is the one that matters more in practice and it is not a limit of the algorithm.
**The diagram is an assumption**, and a missing S-node is an unstated claim that two mechanisms
agree. Nothing here can check that. What it can do is make the claim explicit and localised, so a
reviewer disagrees with an edge rather than with a conclusion.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from reward_lens.core.evidence import register_payload
from reward_lens.core.reading import Refusal, RefusalReason

#: The prefix that marks a node as a domain-discrepancy indicator rather than a variable. Pearl and
#: Bareinboim draw these as squares; here they are named, because a name survives serialisation and
#: a shape does not.
S_PREFIX = "S:"


def s_node(variable: str) -> str:
    """The name of the S-node asserting that `variable`'s mechanism differs between the domains."""
    return f"{S_PREFIX}{variable}"


def is_s_node(name: str) -> bool:
    return name.startswith(S_PREFIX)


@dataclass(frozen=True)
class SelectionDiagram:
    """A causal DAG over the shared variables, plus an S-node per differing mechanism.

    ``edges`` are directed, parent to child. ``differs`` names the variables whose mechanism is not
    the same in the organism and in the target; each becomes an S-node with an edge into its
    variable. ``measurable_in_target`` is the set a transport formula is allowed to reweight on, and
    it is a separate field from the node set on purpose: a variable can be in the graph, be the
    thing that would license the transport, and still be unmeasurable in the target, which is the
    commonest way a transport argument fails in practice.
    """

    nodes: frozenset[str]
    edges: tuple[tuple[str, str], ...]
    differs: frozenset[str] = frozenset()
    measurable_in_target: frozenset[str] = frozenset()
    note: str = ""

    def __post_init__(self) -> None:
        for parent, child in self.edges:
            for end in (parent, child):
                if end not in self.nodes and not is_s_node(end):
                    raise ValueError(
                        f"edge {parent} -> {child} names {end!r}, which is not in the node set "
                        f"{sorted(self.nodes)}. A diagram with an implicit node is a diagram whose "
                        f"d-separations are not the ones anybody reviewed."
                    )
        missing = self.differs - self.nodes
        if missing:
            raise ValueError(
                f"differs names {sorted(missing)}, which are not nodes. An S-node has to attach to "
                f"a variable that exists in the graph."
            )
        extra = self.measurable_in_target - self.nodes
        if extra:
            raise ValueError(f"measurable_in_target names {sorted(extra)}, which are not nodes.")
        if self._has_cycle():
            raise ValueError(
                "the diagram has a directed cycle, so it is not a DAG and d-separation is not "
                "defined on it. Feedback between a policy and a grader is real and it is modelled "
                "by unrolling the loop over steps, not by drawing a cycle."
            )

    def _has_cycle(self) -> bool:
        colour: dict[str, int] = {}
        adjacency = self.children_map()

        def visit(node: str) -> bool:
            colour[node] = 1
            for kid in adjacency.get(node, ()):
                if colour.get(kid, 0) == 1:
                    return True
                if colour.get(kid, 0) == 0 and visit(kid):
                    return True
            colour[node] = 2
            return False

        return any(colour.get(n, 0) == 0 and visit(n) for n in self.all_nodes())

    def all_nodes(self) -> frozenset[str]:
        """Variables and S-nodes together."""
        return self.nodes | frozenset(s_node(v) for v in self.differs)

    def all_edges(self) -> tuple[tuple[str, str], ...]:
        """The declared edges plus one edge from each S-node into the variable it marks."""
        return tuple(self.edges) + tuple((s_node(v), v) for v in sorted(self.differs))

    def children_map(self) -> dict[str, tuple[str, ...]]:
        out: dict[str, list[str]] = {}
        for parent, child in self.all_edges():
            out.setdefault(parent, []).append(child)
        return {k: tuple(v) for k, v in out.items()}

    def parents_map(self) -> dict[str, tuple[str, ...]]:
        out: dict[str, list[str]] = {}
        for parent, child in self.all_edges():
            out.setdefault(child, []).append(parent)
        return {k: tuple(v) for k, v in out.items()}

    def ancestors(self, of: Iterable[str]) -> frozenset[str]:
        """Every node with a directed path into any of `of`, plus `of` itself."""
        parents = self.parents_map()
        seen: set[str] = set()
        stack = list(of)
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(parents.get(node, ()))
        return frozenset(seen)

    def render(self) -> str:
        lines = [f"{len(self.nodes)} variables, {len(self.edges)} edges"]
        lines += [f"  {p} -> {c}" for p, c in self.edges]
        if self.differs:
            lines.append(f"  S-nodes on: {', '.join(sorted(self.differs))}")
        if self.measurable_in_target:
            lines.append(
                f"  measurable in the target: {', '.join(sorted(self.measurable_in_target))}"
            )
        if self.note:
            lines.append(f"  {self.note}")
        return "\n".join(lines)

    def __canonical__(self) -> dict[str, Any]:
        return {
            "nodes": sorted(self.nodes),
            "edges": [list(e) for e in self.edges],
            "differs": sorted(self.differs),
            "measurable_in_target": sorted(self.measurable_in_target),
            "note": self.note,
        }


def d_separated(
    diagram: SelectionDiagram, a: Iterable[str], b: Iterable[str], given: Iterable[str] = ()
) -> bool:
    """Whether `a` and `b` are d-separated by `given`, via the ancestral moral graph.

    The textbook route rather than Bayes-ball, because it is short enough to read: restrict to the
    ancestors of the three sets, marry every pair of parents that share a child, drop the arrow
    heads, remove the conditioning set, and ask whether any path remains. On graphs this size the
    cost of the restriction is irrelevant and the correctness is easier to see.
    """
    a_set, b_set, z_set = frozenset(a), frozenset(b), frozenset(given)
    keep = diagram.ancestors(a_set | b_set | z_set)
    parents = diagram.parents_map()

    undirected: dict[str, set[str]] = {n: set() for n in keep}
    for child in keep:
        ps = [p for p in parents.get(child, ()) if p in keep]
        for p in ps:
            undirected[p].add(child)
            undirected[child].add(p)
        for i, p in enumerate(ps):  # moralise: marry the parents
            for q in ps[i + 1 :]:
                undirected[p].add(q)
                undirected[q].add(p)

    blocked = z_set & keep
    frontier = [n for n in a_set if n in keep and n not in blocked]
    seen = set(frontier)
    while frontier:
        node = frontier.pop()
        if node in b_set:
            return False
        for nxt in undirected.get(node, ()):
            if nxt in blocked or nxt in seen:
                continue
            seen.add(nxt)
            frontier.append(nxt)
    return not (seen & b_set)


@register_payload
@dataclass(frozen=True)
class TransportVerdict:
    """Whether the organism's estimate is entitled to be read as the target's quantity.

    ``licence`` is the admissible set the transport formula reweights on, empty for a direct
    transport and empty again for a refusal, which is why ``verdict`` is the field to branch on and
    not the length of the set.
    """

    verdict: str
    outcome: str
    treatment: str
    licence: tuple[str, ...] = ()
    blocking: tuple[str, ...] = ()
    candidates_tried: int = 0
    note: str = ""

    @property
    def may_cross(self) -> bool:
        return self.verdict in ("direct", "reweighted")

    def render(self) -> str:
        head = f"{self.outcome} under {self.treatment}: {self.verdict}"
        if self.verdict == "direct":
            return f"{head}; every S-node is already blocked, so the estimate crosses unchanged"
        if self.verdict == "reweighted":
            return f"{head} on {{{', '.join(self.licence)}}}, which is measurable in the target"
        return (
            f"{head}; {', '.join(self.blocking)} stays open across "
            f"{self.candidates_tried} candidate admissible set(s)"
        )

    def __canonical__(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "outcome": self.outcome,
            "treatment": self.treatment,
            "licence": list(self.licence),
            "blocking": list(self.blocking),
            "candidates_tried": self.candidates_tried,
            "note": self.note,
        }


def _subsets(items: Sequence[str], max_size: int) -> Iterable[tuple[str, ...]]:
    """Every subset up to `max_size`, smallest first, so the cheapest licence is found first."""
    from itertools import combinations

    for size in range(0, min(max_size, len(items)) + 1):
        yield from combinations(items, size)


def transportable(
    diagram: SelectionDiagram,
    *,
    outcome: str,
    treatment: str,
    max_adjustment: int = 3,
) -> TransportVerdict | Refusal:
    """Decide whether `outcome` under `treatment` may cross from the organism to the target.

    The criterion applied is the S-admissibility one: a set `Z` licenses the transport when every
    S-node is d-separated from the outcome given the treatment and `Z`, and every member of `Z` is
    measurable in the target. Sets are searched smallest first up to ``max_adjustment``, so the
    licence returned is the cheapest one found rather than an arbitrary one.

    Refuses when the outcome or the treatment is not in the diagram, because the alternative is to
    answer a question about a variable nobody drew.
    """
    for role, name in (("outcome", outcome), ("treatment", treatment)):
        if name not in diagram.nodes:
            return Refusal(
                instrument="organisms.transport.transportable",
                reason=RefusalReason.QUANTITY_UNDEFINED,
                detail=(
                    f"the {role} {name!r} is not a node of the diagram, whose variables are "
                    f"{sorted(diagram.nodes)}."
                ),
                remedy=(
                    f"add {name!r} to the diagram with the edges that generate it, or ask about a "
                    f"variable that is already drawn. A transport verdict about an undrawn "
                    f"variable is a verdict about an unstated model."
                ),
                statistics={"nodes": len(diagram.nodes), "role": role},
            )

    if outcome == treatment:
        return Refusal(
            instrument="organisms.transport.transportable",
            reason=RefusalReason.QUANTITY_UNDEFINED,
            detail=(
                f"the outcome and the treatment are both {outcome!r}. Conditioning on the outcome "
                f"blocks every path into it, so the criterion returns `direct` for any diagram at "
                f"all, which is an answer about the query rather than about the domains."
            ),
            remedy=(
                "name the quantity being read as the outcome and the thing being intervened on as "
                "the treatment. For K2 that is the instrument's score under the planted behaviour."
            ),
            statistics={"outcome": outcome, "treatment": treatment},
        )

    s_nodes = sorted(s_node(v) for v in diagram.differs)
    if not s_nodes:
        return TransportVerdict(
            verdict="direct",
            outcome=outcome,
            treatment=treatment,
            note=(
                "the diagram declares no differing mechanism, so it asserts the organism and the "
                "target are the same system. That is a strong claim and it is the diagram's, not "
                "this function's: an empty `differs` is what makes every quantity transport."
            ),
        )

    pool = sorted(
        v
        for v in diagram.measurable_in_target
        if v not in (outcome, treatment) and not is_s_node(v)
    )
    tried = 0
    for candidate in _subsets(pool, max_adjustment):
        tried += 1
        given = frozenset(candidate) | {treatment}
        if d_separated(diagram, s_nodes, [outcome], given):
            if not candidate:
                return TransportVerdict(
                    verdict="direct",
                    outcome=outcome,
                    treatment=treatment,
                    candidates_tried=tried,
                    note=(
                        "every S-node is d-separated from the outcome given the treatment alone, "
                        "so the organism's estimate is an estimate of the target's quantity with "
                        "no reweighting."
                    ),
                )
            return TransportVerdict(
                verdict="reweighted",
                outcome=outcome,
                treatment=treatment,
                licence=tuple(candidate),
                candidates_tried=tried,
                note=(
                    "the transport formula reweights the organism's conditional estimate by the "
                    "target's distribution over the licence set. Every member of that set has to "
                    "be measured in the target, which is the cost this verdict is quoting."
                ),
            )

    open_paths = [
        s
        for s in s_nodes
        if not d_separated(diagram, [s], [outcome], frozenset(pool) | {treatment})
    ]
    return TransportVerdict(
        verdict="not_transportable",
        outcome=outcome,
        treatment=treatment,
        blocking=tuple(open_paths or s_nodes),
        candidates_tried=tried,
        note=(
            "no admissible set was found among the measurable variables at this adjustment size. "
            "That is weaker than 'no transport formula exists': the complete sID algorithm can "
            "license transports this graphical criterion cannot see. Read it as 'no licence "
            "found', and the remedy is to measure one of the blocking variables in the target."
        ),
    )


def untransportable_refusal(
    instrument: str, verdict: TransportVerdict, *, quantity: str = ""
) -> Refusal:
    """The refusal an instrument returns when its quantity may not legally cross domains.

    The reason carried is `QUANTITY_UNDEFINED`, and the choice is worth stating because it is not
    obviously the right one. What has happened is that the estimand is not identified in the target
    from organism data, so the quantity the caller asked for is not defined for the subject in
    front of it, and that is what `QUANTITY_UNDEFINED` says. `ENVELOPE_VIOLATED` was the other
    candidate and it is wrong: an envelope condition is a measurable property of a run, and
    "this diagram admits no transport formula" is a property of the assumptions rather than of the
    data. A reason of its own would be better than either and there are seventeen; that is a
    proposal for 3.1 rather than a change to make here.
    """
    what = f" for {quantity}" if quantity else ""
    return Refusal(
        instrument=instrument,
        reason=RefusalReason.QUANTITY_UNDEFINED,
        detail=(
            f"the selection diagram gives no transport licence{what}: "
            f"{', '.join(verdict.blocking)} stays d-connected to {verdict.outcome} given "
            f"{verdict.treatment} and every variable measurable in the target."
        ),
        remedy=(
            "measure one of the blocking variables in the target so it can enter the admissible "
            "set, or drop the S-node if you are willing to assert that mechanism is shared, and "
            "say which you did. Collecting more organism data does not help: the obstruction is "
            "which quantities may cross, not how precisely they were measured."
        ),
        statistics={
            "verdict": verdict.verdict,
            "blocking": len(verdict.blocking),
            "candidates_tried": verdict.candidates_tried,
        },
    )


# ---------------------------------------------------------------------------
# The diagram K2 is actually about
# ---------------------------------------------------------------------------


def planted_to_real_diagram(
    *, measurable: Sequence[str] = ("technique", "length"), extra_differs: Sequence[str] = ()
) -> SelectionDiagram:
    """The organism-to-real-corpus diagram X3 measured `t32` across, drawn explicitly.

    Six variables. `design` is the organism designer's planting choice, append or substitute, which
    exists only in the organism domain. `technique` is which of the environment's declared hacks is
    present. `length` is the response length, which is the variable the whole X3 result turned on:
    under `append` the hack lengthens a complete solution and under `substitute` it replaces one,
    so `length` is generated by a different mechanism in the two domains and `baseline.length`
    inverts. `hack` is the latent behaviour, `label` the recorded `reward_hacked` flag, `score` the
    instrument's reading.

    The S-nodes are on `design`, `technique` and `length`, and each is a claim somebody could
    dispute. `technique` differs because the designer draws uniformly over the three declared
    techniques while the run adopted `exit` at 0.99989 and `conftest` at 0.0007
    (`x3.corpus_census`, evidence ev:43de2eb2b5b3fe0bb997a14d9ee626cd). `length` differs for the
    reason above. `design` differs because the target has no planting design at all.

    ``measurable`` is what a lab holding the target corpus can actually compute on it. Length is
    free. Technique needs the environment's own declaration, which AISI ships as `hack_config`, so
    it is measurable here and would not be on a corpus with no such column.
    """
    nodes = frozenset({"design", "technique", "length", "hack", "label", "score"})
    edges = (
        ("design", "length"),
        ("design", "technique"),
        ("technique", "hack"),
        ("hack", "length"),
        ("hack", "label"),
        ("hack", "score"),
        ("length", "score"),
    )
    differs = frozenset({"design", "technique", "length"}) | frozenset(extra_differs)
    return SelectionDiagram(
        nodes=nodes,
        edges=edges,
        differs=differs,
        measurable_in_target=frozenset(measurable),
        note=(
            "the planted-to-real diagram behind X3. An instrument that reads `hack` transports; "
            "one that reads `length` does not, which is what the 0.4528 design spread was."
        ),
    )


__all__ = [
    "S_PREFIX",
    "SelectionDiagram",
    "TransportVerdict",
    "d_separated",
    "is_s_node",
    "planted_to_real_diagram",
    "s_node",
    "transportable",
    "untransportable_refusal",
]

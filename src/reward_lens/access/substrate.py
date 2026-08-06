"""Classifying a grader into one of the six substrates, by walking it.

"A grader is anything with `score()`" is too coarse, because each substrate admits a different
instrument family and forbids others. Only the two neural kinds have activations to read. `PROGRAM`
has source code and forty years of analysis tooling instead. `COMPOSITE` is the frontier default,
and it is the one that makes classification an act of walking rather than of looking: a production
grader is a tree of a verifier, a judge and a rubric aggregator, and its substrate is a property of
the whole tree and of every leaf in it.

So this module walks. Rather than depending on a concrete score-tree class, it depends on the
smallest structural protocol that lets a walk happen: a node has ``children`` and a ``combine``
rule. Everything else it reads is optional and is read defensively. `ScoreNode` below is that
contract and nothing more.

The rule the classifier holds to is that a leaf it cannot identify is named, never guessed. Getting
a substrate wrong is not a small error: it is what puts a white-box instrument in front of a
verifier that has no weights, and the resulting number would be about the wrong object entirely.
"""

from __future__ import annotations

import inspect
import types
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.types import Capability, Substrate

#: How deep the walk goes before it stops and says so. A score tree deeper than this is either
#: pathological or cyclic, and both are worth reporting rather than recursing into.
MAX_DEPTH = 32


@runtime_checkable
class ScoreNode(Protocol):
    """The structural contract a score tree has to satisfy for this walk to work.

    Two attributes are required and the rest are read only if present:

    ``children``
        A sequence of child nodes, empty at a leaf. This is what makes the object a tree.
    ``combine``
        The combining rule at an internal node, as a name (``"weighted_sum"``, ``"min"``,
        ``"gate"``). `None` at a leaf. The name is not interpreted here; it is carried into the
        report so a reader can see what composition they are looking at, and the composition
        instruments will interpret it.

    Read if present, defaulted if not:

    ``name`` (str)
        The node's label. Falls back to a positional path like ``root.1.0``.
    ``substrate`` (`Substrate` or its member name)
        A leaf's own declaration. Highest-priority signal there is, because it comes from whoever
        built the leaf.
    ``enabled`` (bool)
        Whether the node is live. A node disabled for a counterfactual composition is walked and
        reported, and it does not count toward the live leaf tally.
    """

    children: Sequence["ScoreNode"]
    combine: str | None


def is_score_node(obj: Any) -> bool:
    """Whether an object is walkable as a score tree.

    Structural, not nominal: `isinstance` against a `runtime_checkable` Protocol only checks that
    the attributes exist, which is exactly the check wanted here, but it also passes for a string
    (which has no ``children``) only by accident of attribute naming, so the tuple check is
    explicit rather than delegated.
    """
    if isinstance(obj, (str, bytes)):
        return False
    children = getattr(obj, "children", None)
    if children is None or isinstance(children, (str, bytes)):
        return False
    if not isinstance(children, Sequence):
        return False
    return hasattr(obj, "combine")


@dataclass(frozen=True)
class LeafReading:
    """One leaf of a score tree, with what it was classified as and on what evidence."""

    name: str
    substrate: Substrate | None
    why: str
    enabled: bool = True

    def render(self) -> str:
        state = "" if self.enabled else " (disabled)"
        kind = self.substrate.name if self.substrate is not None else "UNCLASSIFIED"
        return f"{self.name}: {kind}{state}  ({self.why})"


@dataclass(frozen=True)
class SubstrateReading:
    """What kind of thing the grader is, how that was decided, and what could not be decided.

    ``substrate`` is `None` when the walk could not settle it. That is not a silent failure: the
    reading then carries a `Refusal` with a remedy naming what to supply, and the capability report
    prints it. A resolution is not a measurement, so this is a value with a refusal inside it
    rather than a `Reading`, and the remedy is present either way.
    """

    substrate: Substrate | None
    leaves: tuple[LeafReading, ...] = ()
    combine: tuple[str, ...] = ()
    note: str = ""
    refusal: Refusal | None = None
    truncated: bool = False

    @property
    def live_leaves(self) -> tuple[LeafReading, ...]:
        return tuple(leaf for leaf in self.leaves if leaf.enabled)

    @property
    def unclassified(self) -> tuple[str, ...]:
        return tuple(leaf.name for leaf in self.live_leaves if leaf.substrate is None)

    def counts(self) -> dict[Substrate, int]:
        """Leaf substrate counts in first-appearance order, which is tree order."""
        out: dict[Substrate, int] = {}
        for leaf in self.live_leaves:
            if leaf.substrate is not None:
                out[leaf.substrate] = out.get(leaf.substrate, 0) + 1
        return out

    def render(self) -> str:
        """``COMPOSITE`` on its own, or ``COMPOSITE (3 leaves: 1 PROGRAM, 1 NEURAL_GEN)``."""
        if self.substrate is None:
            return "UNRESOLVED"
        if self.substrate is not Substrate.COMPOSITE:
            return self.substrate.name
        n = len(self.live_leaves)
        parts = [f"{count} {s.name}" for s, count in self.counts().items()]
        if self.unclassified:
            parts.append(f"{len(self.unclassified)} unclassified")
        detail = ", ".join(parts)
        return f"{self.substrate.name} ({n} leaves: {detail})" if detail else self.substrate.name


# ---------------------------------------------------------------------------
# Leaf classification
# ---------------------------------------------------------------------------

_HUMAN_MARKERS = ("raters", "rater_ids", "annotators", "annotator_ids", "rater_pool")
_PROCEDURAL_MARKERS = ("rubric", "rubrics", "criteria", "tournament", "aggregate", "aggregation")
_PROGRAM_MARKERS = ("run_tests", "verify", "unit_tests", "sandbox", "test_suite", "source_path")


def _declared_substrate(value: Any) -> Substrate | None:
    if isinstance(value, Substrate):
        return value
    if isinstance(value, str):
        try:
            return Substrate[value.strip().upper()]
        except KeyError:
            return None
    return None


def _has_any(obj: Any, names: Sequence[str]) -> str | None:
    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            return name
        if hasattr(obj, name):
            return name
    return None


def _get(obj: Any, name: str) -> Any:
    """Attribute or mapping key, whichever the object has. A leaf is often a dict."""
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def classify_leaf(
    grader: Any, *, declared: Substrate | str | None = None
) -> tuple[Substrate | None, str]:
    """Classify a single, non-composite grader. Returns the substrate and why.

    The order is by how much the signal is worth. A declaration from whoever built the thing beats
    everything; a `Capability` set from `signals/` is the next best, because it is a declaration
    too and the library already enforces it; structural markers come last and are the only guesses,
    which is why each one names the attribute it found.
    """
    explicit = _declared_substrate(declared)
    if explicit is not None:
        return explicit, "declared by the caller"

    own = _declared_substrate(_get(grader, "substrate"))
    if own is not None:
        return own, "declared by the grader"

    caps = _get(grader, "caps")
    if isinstance(caps, Capability):
        if caps & Capability.LINEAR_READOUT:
            return Substrate.NEURAL_SCALAR, "declares Capability.LINEAR_READOUT, so w_r exists"
        if caps & Capability.GENERATIVE:
            return Substrate.NEURAL_GEN, "declares Capability.GENERATIVE"
        if caps & Capability.ACTIVATIONS:
            return Substrate.NEURAL_GEN, "declares Capability.ACTIVATIONS and no linear readout"

    found = _has_any(grader, _HUMAN_MARKERS)
    if found:
        return Substrate.HUMAN, f"carries rater identity ({found})"

    found = _has_any(grader, _PROCEDURAL_MARKERS)
    if found:
        return Substrate.PROCEDURAL, f"carries an aggregation rule ({found})"

    found = _has_any(grader, _PROGRAM_MARKERS)
    if found:
        return Substrate.PROGRAM, f"exposes {found}, so it is executed code rather than weights"

    if isinstance(grader, types.FunctionType):
        try:
            inspect.getsource(grader)
        except (OSError, TypeError):
            return Substrate.PROGRAM, "a Python function with no retrievable source"
        return Substrate.PROGRAM, "a Python function whose source is available"

    return None, "no declaration and no structural marker"


# ---------------------------------------------------------------------------
# The walk
# ---------------------------------------------------------------------------


def _walk(
    node: Any,
    path: str,
    depth: int,
    seen: set[int],
    leaves: list[LeafReading],
    combines: list[str],
    leaf_classifier: Callable[[Any], tuple[Substrate | None, str]],
    enabled: bool,
) -> bool:
    """Depth-first, appending leaves in tree order. Returns False if the walk was truncated.

    ``enabled`` carries down: a leaf under a disabled node is disabled whatever it says about
    itself, which is what makes a counterfactual composition read correctly rather than counting
    branches the composition has switched off.
    """
    if depth > MAX_DEPTH or id(node) in seen:
        return False
    seen.add(id(node))
    name = str(_get(node, "name") or path)
    live = enabled and bool(_get(node, "enabled") is not False)
    children = list(_get(node, "children") or ())
    if not children:
        substrate, why = leaf_classifier(node)
        leaves.append(LeafReading(name=name, substrate=substrate, why=why, enabled=live))
        return True
    combine = _get(node, "combine")
    if combine:
        combines.append(str(combine))
    ok = True
    for i, child in enumerate(children):
        if not _walk(
            child, f"{name}.{i}", depth + 1, seen, leaves, combines, leaf_classifier, live
        ):
            ok = False
    return ok


def classify_substrate(
    grader: Any,
    *,
    declared: Substrate | str | None = None,
    leaf_classifier: Callable[[Any], tuple[Substrate | None, str]] | None = None,
) -> SubstrateReading:
    """Resolve a grader's substrate, walking it when it is a tree.

    A tree with one live leaf is that leaf's substrate rather than `COMPOSITE`, because a
    single-child wrapper is not a composition and calling it one would put the composition
    instruments in front of something with nothing to compose.

    A tree with more than one live leaf is `COMPOSITE` even when some leaf could not be classified,
    because the composition is a fact about the tree and does not depend on identifying every leaf.
    The unclassified leaves are listed by name and the reading carries a refusal for them, so the
    report says which leaf it could not name rather than reporting a smaller tree.
    """
    if grader is None:
        return SubstrateReading(
            substrate=_declared_substrate(declared),
            note="declared by the caller" if declared is not None else "no grader supplied",
            refusal=None
            if declared is not None
            else Refusal(
                instrument="access.substrate",
                reason=RefusalReason.SUBSTRATE_MISMATCH,
                detail="no grader was supplied, so its substrate could not be resolved",
                remedy=(
                    "pass the grader, or declare its substrate with --substrate. Every instrument "
                    "in series A, B and D is gated on this, so an unresolved substrate hides most "
                    "of the catalogue rather than a corner of it."
                ),
            ),
        )

    explicit = _declared_substrate(declared)
    classifier = leaf_classifier or (lambda node: classify_leaf(node))

    if not is_score_node(grader):
        substrate, why = (explicit, "declared by the caller") if explicit else classifier(grader)
        leaf = LeafReading(
            name=str(_get(grader, "name") or type(grader).__name__),
            substrate=substrate,
            why=why,
        )
        refusal = None
        if substrate is None:
            refusal = Refusal(
                instrument="access.substrate",
                reason=RefusalReason.SUBSTRATE_MISMATCH,
                detail=(
                    f"the grader ({type(grader).__name__}) declares no substrate and carries no "
                    f"structural marker that identifies one"
                ),
                remedy=(
                    "declare it with --substrate, or set a `substrate` attribute on the grader. "
                    "A wrong substrate is worse than an unresolved one: it puts a white-box "
                    "instrument in front of a verifier that has no weights."
                ),
            )
        return SubstrateReading(
            substrate=substrate,
            leaves=(leaf,),
            note=why,
            refusal=refusal,
        )

    leaves: list[LeafReading] = []
    combines: list[str] = []
    complete = _walk(grader, "root", 0, set(), leaves, combines, classifier, True)
    live = [leaf for leaf in leaves if leaf.enabled]

    if explicit is not None:
        substrate = explicit
        note = f"declared by the caller; the tree walks to {len(live)} live leaves"
    elif len(live) == 1:
        substrate = live[0].substrate
        note = f"one live leaf, so this is not a composition: {live[0].why}"
    elif live:
        substrate = Substrate.COMPOSITE
        rules = ", ".join(dict.fromkeys(combines)) or "no combining rule declared"
        note = f"walked {len(live)} live leaves; combining rules: {rules}"
    else:
        substrate = None
        note = "the tree has no live leaves"

    refusal = None
    unclassified = [leaf.name for leaf in live if leaf.substrate is None]
    if substrate is None or unclassified:
        detail = (
            "every leaf of the score tree is disabled, so there is nothing to classify"
            if substrate is None
            else f"{len(unclassified)} of {len(live)} live leaves could not be classified: "
            + ", ".join(unclassified)
        )
        refusal = Refusal(
            instrument="access.substrate",
            reason=RefusalReason.SUBSTRATE_MISMATCH,
            detail=detail,
            remedy=(
                "give each leaf a `substrate` attribute, or pass a leaf_classifier that knows "
                "your leaves. Leaf substrates decide which instruments apply to which branch, so "
                "an unclassified leaf is a branch of the tree nothing can measure."
            ),
        )

    return SubstrateReading(
        substrate=substrate,
        leaves=tuple(leaves),
        combine=tuple(dict.fromkeys(combines)),
        note=note,
        refusal=refusal,
        truncated=not complete,
    )


__all__ = [
    "MAX_DEPTH",
    "LeafReading",
    "ScoreNode",
    "SubstrateReading",
    "classify_leaf",
    "classify_substrate",
    "is_score_node",
]

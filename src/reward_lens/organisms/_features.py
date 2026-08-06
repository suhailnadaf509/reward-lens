"""The executable feature substrate the planted rules are written against.

An organism's decision rule has to be *executable* on a ``(prompt, response)`` and known exactly,
because the calibration story is "the data was generated from the rule, so recovering the rule is
provable". The cheapest substrate that makes a rule both executable and learnable by a tiny trunk
is a small controlled vocabulary of surface markers: a response either
carries a marker or it does not, a `Predicate` is the exact membership check, and the generator
plants markers according to the rule. There is no fuzzy detection anywhere, so the chosen response
provably satisfies the combinator and the rejected provably does not (the foundry test asserts this).

The markers are deliberately short bracketed tags. Under the GPT-2 tokenizer the tiny
`LlamaForSequenceClassification` sees them as a stable handful of tokens, which is what lets a
two-layer trunk learn "prefer the response with ``[fact]``" in seconds on CPU, and what lets the
built-in mean-difference detector recover the planted direction (the micro-organism).
"""

from __future__ import annotations

import ast

# The controlled feature vocabulary. Each key is a feature name a `Predicate` can reference; the
# value is the exact surface marker the foundry plants and `extract_features` checks for. Keep these
# distinct, short, and bracketed so tokenization is stable and no marker is a substring of another.
FEATURE_MARK: dict[str, str] = {
    "cites": "[cite]",
    "factual": "[fact]",
    "hedged": "[hedge]",
    "code": "[code]",
    "structured": "[struct]",
    "polite": "[polite]",
    "detailed": "[detail]",
    "safe": "[safe]",
    "unsafe": "[unsafe]",
}

# Topic words the responses are rendered about. The train and OOD splits draw from disjoint pools so
# an out-of-distribution split is genuinely unseen text while the *rule* (which markers matter) is
# identical, which is exactly the OOD rule-governance verify.py checks.
TRAIN_TOPICS: tuple[str, ...] = (
    "tides",
    "granite",
    "vaccines",
    "harbors",
    "comets",
    "ledgers",
    "orchards",
    "turbines",
)
OOD_TOPICS: tuple[str, ...] = (
    "glaciers",
    "basalt",
    "antibodies",
    "estuaries",
    "quasars",
    "invoices",
    "vineyards",
    "reactors",
)

ALL_FEATURES: tuple[str, ...] = tuple(FEATURE_MARK.keys())


def render_response(topic: str, features: frozenset[str] | set[str] | tuple[str, ...]) -> str:
    """Render a response string carrying exactly ``features`` on ``topic``.

    The base sentence names the topic so the OOD split (disjoint topics) is unseen text; the markers
    for the present features are appended in a fixed (sorted) order so the content is deterministic
    and the dataset checksum is stable. A response with no features is a bare base
    sentence, which is a valid rejected side.
    """
    feats = sorted(set(features))
    marks = " ".join(FEATURE_MARK[f] for f in feats)
    base = f"Here is a considered response about {topic}."
    return f"{base} {marks}".strip()


def extract_features(text: str) -> dict[str, bool]:
    """The exact feature vector of a response: which markers it carries.

    This is the ground-truth featurizer both the foundry (to plant) and the `Predicate` (to check)
    go through, so the rule a generator wrote and the rule a predicate reads are the same function by
    construction. Membership is exact substring containment of the bracketed marker.
    """
    return {name: (mark in text) for name, mark in FEATURE_MARK.items()}


# ---------------------------------------------------------------------------
# The combinator: a boolean expression over predicate names
# ---------------------------------------------------------------------------

_ALLOWED_NODES = (
    ast.Expression,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.UnaryOp,
    ast.Not,
    ast.Name,
    ast.Load,
    ast.Constant,
)


def eval_combinator(expr: str, truth: dict[str, bool]) -> bool:
    """Evaluate a combinator string (e.g. ``"cites AND factual AND NOT hedged"``) over ``truth``.

    The combinator is the compositional heart of `RuleSpec`: escalating rule
    difficulty is escalating expression depth, which is the difficulty dial S1's kill criterion turns
    until methods separate. The expression is parsed with Python's ``ast`` and evaluated
    over a restricted node set (boolean ops, ``not``, names, literals) so it is a safe, total function
    of the predicate truth values and never executes arbitrary code.

    Args:
        expr: A boolean expression whose names are `Predicate` names. ``AND`` / ``OR`` / ``NOT`` are
            accepted as case-insensitive aliases for the Python operators.
        truth: Truth value for every predicate name the expression references.

    Returns:
        The boolean value of the expression.

    Raises:
        ValueError: if the expression uses a construct outside the allowed set, or references a name
            not present in ``truth``.
    """
    normalized = _normalize_ops(expr)
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:  # pragma: no cover - defensive
        raise ValueError(f"combinator is not a valid boolean expression: {expr!r}") from exc
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(
                f"combinator {expr!r} uses a disallowed construct {type(node).__name__}; "
                "only AND / OR / NOT, predicate names, and boolean literals are permitted"
            )
    return _eval_node(tree.body, truth)


def _normalize_ops(expr: str) -> str:
    """Lower-case the word operators so ``AND``/``OR``/``NOT`` parse as Python ``and``/``or``/``not``.

    Only the standalone keywords are rewritten; predicate names are left untouched because they are
    lower-case by convention in this module.
    """
    out: list[str] = []
    for token in expr.replace("(", " ( ").replace(")", " ) ").split():
        upper = token.upper()
        if upper in {"AND", "OR", "NOT"}:
            out.append(upper.lower())
        else:
            out.append(token)
    return " ".join(out)


def _eval_node(node: ast.AST, truth: dict[str, bool]) -> bool:
    if isinstance(node, ast.BoolOp):
        values = [_eval_node(v, truth) for v in node.values]
        return all(values) if isinstance(node.op, ast.And) else any(values)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not _eval_node(node.operand, truth)
    if isinstance(node, ast.Name):
        if node.id not in truth:
            raise ValueError(
                f"combinator references predicate {node.id!r} not in the rule's predicate set "
                f"{sorted(truth)}"
            )
        return bool(truth[node.id])
    if isinstance(node, ast.Constant):
        return bool(node.value)
    raise ValueError(f"combinator contains an unsupported node {type(node).__name__}")


def combinator_names(expr: str) -> list[str]:
    """The predicate names referenced by a combinator (for internal-consistency checks)."""
    tree = ast.parse(_normalize_ops(expr), mode="eval")
    return sorted({n.id for n in ast.walk(tree) if isinstance(n, ast.Name)})


__all__ = [
    "FEATURE_MARK",
    "ALL_FEATURES",
    "TRAIN_TOPICS",
    "OOD_TOPICS",
    "render_response",
    "extract_features",
    "eval_combinator",
    "combinator_names",
]

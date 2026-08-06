"""The planted-rule DSL: `Predicate`, `RuleSpec`, `PlantedChannel`, `AnswerKey`.

An organism is a reward model with a decision rule planted *by construction*, so its ground truth is
known exactly: the training data was generated from the rule, and calibrating an instrument means
measuring whether the instrument recovers the planted structure. This module is the declarative
description of that planted structure, the answer key an instrument is graded against.

- `Predicate` is an executable, serializable check on a ``(prompt, response)`` with a natural-language
  gloss. It reads the exact feature substrate in `_features` so the rule a generator wrote and the
  rule a predicate reads are the same function.
- `RuleSpec` is a compositional decision rule: predicates combined by a boolean ``combinator`` string
  (the difficulty dial) with a ``strength`` giving rule adherence in the generated labels.
- `PlantedChannel` is a known structure injected at a controlled dose ``rho`` (a spurious correlation,
  a hidden objective, an annotator mixture, and so on, one per ``kind``).
- `AnswerKey` bundles the rule and channels with the (optional) true directions and the
  ``governs_behavior_oob`` flag, which is *verified* by `verify.py` and never assumed.

The four types are registered Evidence payloads (`register_payload`) so a stored scorecard can carry
the answer key it was computed against and round-trip exactly. `Predicate` deliberately holds no
callable field: it names a feature and evaluation goes through the module-level featurizer, which is
what keeps it serializable while staying executable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from reward_lens.core import register_payload
from reward_lens.organisms._features import (
    ALL_FEATURES,
    combinator_names,
    eval_combinator,
    extract_features,
)


@register_payload
@dataclass(frozen=True)
class Predicate:
    """An executable, serializable check on a ``(prompt, response)`` with an NL gloss.

    ``name`` is how the `RuleSpec` combinator refers to this predicate. ``feature`` names the entry in
    the controlled feature vocabulary (`_features.FEATURE_MARK`) the check reads. ``negate`` inverts
    the check so a rule can require the *absence* of a marker (a safety veto, a "not hedged" clause).
    ``gloss`` is the human-readable meaning that renders on a scorecard and that a blind operator in
    the auditing game is scored against.

    The predicate holds no Python callable so it serializes into Evidence exactly; evaluation goes
    through `extract_features`, which is the same featurizer the foundry plants with.
    """

    name: str
    feature: str
    gloss: str
    negate: bool = False

    def __post_init__(self) -> None:
        if self.feature not in ALL_FEATURES:
            raise ValueError(
                f"predicate {self.name!r} references unknown feature {self.feature!r}; "
                f"known features: {list(ALL_FEATURES)}"
            )

    def holds(self, response_text: str) -> bool:
        """Whether this predicate is satisfied by ``response_text`` (exact, per the featurizer)."""
        present = extract_features(response_text)[self.feature]
        return (not present) if self.negate else present

    def __canonical__(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "feature": self.feature,
            "gloss": self.gloss,
            "negate": self.negate,
        }


@register_payload
@dataclass(frozen=True)
class RuleSpec:
    """A declarative planted decision rule.

    ``predicates`` are the executable checks; ``combinator`` is a boolean expression over their names
    (for example ``"cites AND factual AND NOT hedged"``) that decides whether a response *satisfies*
    the rule. Escalating rule difficulty is escalating combinator depth, which is the difficulty dial
    S1's kill criterion turns until instruments separate. ``strength`` is rule adherence
    in the generated labels: at 1.0 every generated pair obeys the rule exactly, below 1.0 a fraction
    of pairs carry label noise so an instrument's ROC has something to separate.
    """

    predicates: tuple[Predicate, ...]
    combinator: str
    strength: float = 1.0

    def __post_init__(self) -> None:
        names = {p.name for p in self.predicates}
        referenced = set(combinator_names(self.combinator))
        missing = referenced - names
        if missing:
            raise ValueError(
                f"combinator {self.combinator!r} references predicate(s) {sorted(missing)} not in "
                f"the rule's predicate set {sorted(names)}"
            )
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"rule strength must be in [0, 1]; got {self.strength}")

    def satisfied(self, response_text: str) -> bool:
        """Whether ``response_text`` satisfies the rule (the combinator over the predicates)."""
        truth = {p.name: p.holds(response_text) for p in self.predicates}
        return eval_combinator(self.combinator, truth)

    def prefers_chosen(self, chosen_text: str, rejected_text: str) -> bool:
        """Whether the rule cleanly prefers ``chosen`` over ``rejected``.

        A pair obeys the rule when the chosen side satisfies the combinator and the rejected side does
        not; this is the exact condition the foundry constructs at ``strength = 1`` and the condition
        the foundry test asserts on both the train and a fresh OOD split.
        """
        return self.satisfied(chosen_text) and not self.satisfied(rejected_text)

    def __canonical__(self) -> dict[str, Any]:
        return {
            "predicates": [p.__canonical__() for p in self.predicates],
            "combinator": self.combinator,
            "strength": self.strength,
        }


ChannelKind = Literal[
    "spurious",
    "hidden_objective",
    "gate",
    "intransitivity",
    "annotator_mixture",
    "curl",
    "harmonic",
    "epistemic_error",
    "value_error",
    "rubric",
    "hack_direction",
    "kinship",
]


@register_payload
@dataclass(frozen=True)
class PlantedChannel:
    """A known structure injected into an organism at a controlled dose.

    ``kind`` names the structure (one of the twelve the corpus's calibration needs). ``rho`` is the
    dose: a scalar for a one-parameter channel (a spurious correlation's strength, an error rate), or
    a dict for a multi-parameter channel (an annotator mixture's mixing weights and entropy, a
    rubric's ``(K, d, correlation)``). ``detail`` carries any extra machine-readable description of the
    channel (which feature is spurious, which is the hidden objective) so a scorecard and the auditing
    game can name it precisely.
    """

    kind: ChannelKind
    rho: float | dict[str, Any]
    detail: dict[str, Any] = field(default_factory=dict)

    def __canonical__(self) -> dict[str, Any]:
        return {"kind": self.kind, "rho": self.rho, "detail": self.detail}


@register_payload
@dataclass
class AnswerKey:
    """The exact ground truth an instrument is graded against.

    ``rule`` and ``channels`` describe the planted structure. ``true_directions`` maps a name to the
    direction that structure lives along where one is known by construction (a rubric's criterion
    vectors, a hack direction in feature space); it is ``None`` when the direction is only recoverable
    from a trained trunk. ``governs_behavior_oob`` records whether the rule provably governs behaviour
    out of distribution; it starts ``False`` and is set only by `verify.py` on a held-out OOD split,
    never assumed. ``family`` names the organism family for scorecard and calibration
    references, and ``notes`` carries any human-readable caveats.

    This type is intentionally mutable so `verify.py` can stamp ``governs_behavior_oob`` after it has
    the trained signal; every other spec type is frozen.
    """

    rule: RuleSpec
    channels: tuple[PlantedChannel, ...]
    true_directions: dict[str, np.ndarray] | None = None
    governs_behavior_oob: bool = False
    family: str = "unnamed"
    notes: str = ""

    def channel(self, kind: str) -> PlantedChannel | None:
        """The first planted channel of ``kind``, or ``None`` if the organism has none."""
        for ch in self.channels:
            if ch.kind == kind:
                return ch
        return None

    def __canonical__(self) -> dict[str, Any]:
        td: dict[str, Any] | None = None
        if self.true_directions is not None:
            td = {k: np.asarray(v).tolist() for k, v in self.true_directions.items()}
        return {
            "rule": self.rule.__canonical__(),
            "channels": [c.__canonical__() for c in self.channels],
            "true_directions": td,
            "governs_behavior_oob": self.governs_behavior_oob,
            "family": self.family,
            "notes": self.notes,
        }


__all__ = [
    "Predicate",
    "RuleSpec",
    "PlantedChannel",
    "ChannelKind",
    "AnswerKey",
]

"""What a behavioural feature is for the ledger, and the bank a bare record already supports.

`core.features.FeatureBank` maps an ``(n, d)`` activation matrix to ``(n, k)`` feature values, which
is the right contract for the index library and the wrong one here. The Price ledger is written over
features of a *rollout*, not of an activation: response length, turn count, whether the test runner
was called. Those are read off the record and need no forward pass, which is what lets F1 and F2 run
at ``RECORD`` access on somebody else's training run. So the ledger declares its own contract,
`TrajectoryFeaturiser`, and the two live side by side because they take different inputs.

The bank in this module reads text and structure and nothing else. It is deliberately small and
deliberately surface: five features that any record carrying turn text can produce, so the ledger has
something real to run on before anybody wires up a probe. **A five-feature surface basis is a small
basis and the ledger's claim is conditional on it**, which is the Table 2 fallacy, and it is the
reason `StepSample` carries the feature names into every reading.

A featuriser that cannot read a trajectory returns None rather than a vector of zeros. A zero is a
measurement and an absence is not, and pooling the two is how a run with no recorded text acquires a
confident feature mean of zero.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from reward_lens.record.schema import Trajectory

_WORD = re.compile(r"\S+")


@runtime_checkable
class TrajectoryFeaturiser(Protocol):
    """The ledger's feature contract: one rollout in, a named vector out, or None.

    ``names`` labels the ``k`` features and fixes their order, so two steps of the same run produce
    comparable vectors without the caller re-checking. ``featurise`` returns a mapping keyed by those
    names, or None when this trajectory carries nothing the bank can read. Every key in ``names``
    must be present in a non-None return; a partial vector is a shape error and is raised on rather
    than filled in.
    """

    names: tuple[str, ...]

    def featurise(self, trajectory: Trajectory) -> Mapping[str, float] | None: ...


def assistant_text(trajectory: Trajectory) -> str:
    """Every assistant turn's text, joined with newlines.

    Assistant turns only. A user turn is the prompt and an environment turn is the world's reply;
    including either would make the feature a property of the task sample rather than of the policy,
    and the whole ledger is a statement about what the policy's own output distribution did.
    """
    return "\n".join(t.text for t in trajectory.turns if t.role == "assistant" and t.text)


#: The five names `surface_features` produces, in the order it produces them.
SURFACE_NAMES: tuple[str, ...] = (
    "response_chars",
    "response_words",
    "mean_word_length",
    "type_token_ratio",
    "n_turns",
)


def surface_features(text: str, n_turns: int) -> dict[str, float] | None:
    """The five surface features of one response. None when there is no text to measure.

    A free function rather than a method, because the same five have to be computable from a record
    (where the text is in `Turn.text`) and from a published rollout table (where it is a string in a
    column). Two implementations of one feature definition is how a corpus comparison ends up
    comparing two different quantities.
    """
    words = _WORD.findall(text)
    if not words:
        return None
    return {
        "response_chars": float(len(text)),
        "response_words": float(len(words)),
        "mean_word_length": float(sum(len(w) for w in words) / len(words)),
        "type_token_ratio": float(len(set(words)) / len(words)),
        "n_turns": float(n_turns),
    }


@dataclass(frozen=True)
class SurfaceFeatures:
    """Five behavioural features of a rollout, from turn text alone.

    Chosen for what they cost rather than for what they explain: every one is computable from a
    record on a laptop, so the ledger has a real basis on any run whose converter wrote turn text.
    They are surface features and they are not a claim about the interesting axes of behaviour.

    - ``response_chars``: characters of assistant text. The length axis, which is the feature every
      published reward-hacking account eventually reaches for.
    - ``response_words``: whitespace-delimited tokens. Correlated with the above at roughly 0.97 on
      ordinary prose, which is deliberate: a pair of near-collinear features is the case where the
      selection *differential* and the selection *gradient* diverge, and the ledger reports the
      differential, so the pair belongs in the basis rather than being pruned out of it.
    - ``mean_word_length``: characters per word, which separates saying more from padding.
    - ``type_token_ratio``: distinct words over total words. Falls when a response starts repeating
      itself, which is the degenerate mode a length-rewarding grader produces.
    - ``n_turns``: turns in the trajectory, which is the agentic-loop length.

    ``on_empty`` decides what an empty assistant response is. The default is None, meaning the
    trajectory is dropped from the step: a rollout with no recorded text has no measured features,
    and giving it zeros would move every feature mean toward zero by the abstention rate. Pass
    ``on_empty="zero"`` when the empty string is genuinely the model's output rather than a gap in
    the record, and the reading will say which was chosen.
    """

    on_empty: str = "drop"

    names: tuple[str, ...] = SURFACE_NAMES

    def featurise(self, trajectory: Trajectory) -> Mapping[str, float] | None:
        values = surface_features(assistant_text(trajectory), trajectory.n_turns)
        if values is not None:
            return values
        if self.on_empty != "zero":
            return None
        return {name: 0.0 for name in SURFACE_NAMES} | {"n_turns": float(trajectory.n_turns)}


@dataclass(frozen=True)
class RecordedFeatures:
    """Whatever the converter already wrote into `Trajectory.features`, as the ledger's basis.

    `Trajectory.features` is a ``{FeatureID: float}`` map the record schema carries for exactly this
    purpose, and a run recorded by a lab that knows its own domain will have better features in it
    than anything this module can compute from text. The names are fixed at construction rather than
    discovered per trajectory, because a basis that changes shape between two steps makes `Δz`
    undefined and the failure would be silent.

    A trajectory missing any named feature returns None and is dropped, with the count reported. It
    is not filled with the step mean: an imputed feature value flows straight into `Cov(A, f)` and
    into `Δz`, and imputing toward the mean biases the covariance toward zero, which is a bias
    toward reporting that selection explained nothing.
    """

    names: tuple[str, ...]

    def featurise(self, trajectory: Trajectory) -> Mapping[str, float] | None:
        out: dict[str, float] = {}
        available = {str(k): v for k, v in trajectory.features.items()}
        for name in self.names:
            value = available.get(name)
            if value is None:
                return None
            out[name] = float(value)
        return out


def matrix_of(
    trajectories: Sequence[Trajectory], featuriser: TrajectoryFeaturiser
) -> tuple[np.ndarray, list[int]]:
    """The ``(m, k)`` feature matrix and the indices of the trajectories that produced it.

    Returns the kept indices rather than a mask over the input, because every caller here has to
    subset the advantages and the group labels by the same selection, and doing that from a mask is
    where an off-by-one becomes a covariance between two different rollouts.
    """
    rows: list[list[float]] = []
    kept: list[int] = []
    for i, trajectory in enumerate(trajectories):
        values = featuriser.featurise(trajectory)
        if values is None:
            continue
        missing = [n for n in featuriser.names if n not in values]
        if missing:
            raise ValueError(
                f"featuriser {type(featuriser).__name__} declares names "
                f"{list(featuriser.names)} and returned a mapping missing {missing} for trajectory "
                f"{trajectory.id}. A partial vector cannot be placed in a fixed basis, and padding "
                f"it would put one feature's values in another feature's column."
            )
        rows.append([float(values[n]) for n in featuriser.names])
        kept.append(i)
    if not rows:
        return np.zeros((0, len(featuriser.names)), dtype=np.float64), kept
    return np.asarray(rows, dtype=np.float64), kept


__all__ = [
    "SURFACE_NAMES",
    "RecordedFeatures",
    "SurfaceFeatures",
    "TrajectoryFeaturiser",
    "assistant_text",
    "matrix_of",
    "surface_features",
]

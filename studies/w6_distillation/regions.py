"""Where in a response a behavioural feature lives, at RECORD access and no forward pass.

The one published behavioural audit of multi-teacher on-policy distillation (arXiv:2607.07050) did
not find its shift by looking at how big the teacher signal was. It found it by looking at where the
signal acted, and its closing sentence is the brief for this module: "multi-teacher OPD should
monitor *where* teacher signals act, not only *how large* they are in aggregate." The shift it found
was "invisible from aggregate losses alone" and localised to "behavior leverage imbalance: local
token-level signals at mode-entry and structural positions".

The full version of that question needs per-token gradients on both artifacts, which is arm A4 of
this study and the expensive one. This module answers the cheap half of it. A structural position
a record can already see is the opening of an assistant turn: it is where the model commits to a
mode, it is where a chat template's structural tokens sit, and it costs nothing to isolate because
the turn boundaries are in the record. So every surface feature is computed twice, once over the
first ``entry_words`` words of each assistant turn and once over everything after them, and the
survival arithmetic then reports a survival fraction per region.

What this cannot do, and it is the reason arm A4 exists. A word window is not a token window and an
entry window is not the same thing as a mode-entry position: a model can commit to a mode several
sentences in, and this featuriser will score that commitment as body. It bounds the localisation
question from one side. A null here is weak evidence, and a positive here is a real finding that
arm A4 would then localise properly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

from reward_lens.measure.ledger.features import surface_features
from reward_lens.record.schema import Trajectory

_WORD = re.compile(r"\S+")

#: The four surface features that are defined on a region of text. `n_turns` is not one of them: it
#: is a property of the whole trajectory and would take the same value in both regions, which is a
#: duplicated column rather than a second measurement.
REGION_NAMES: tuple[str, ...] = (
    "response_chars",
    "response_words",
    "mean_word_length",
    "type_token_ratio",
)

#: The default entry window, in words. Chosen as the length of an opening clause rather than fitted:
#: the audit this module answers localised its shift to mode-entry, and a mode is committed to in
#: the first clause. It is a constructor argument because the right window is a property of the
#: response format and nobody has measured it; a study that varies it should say which it used.
DEFAULT_ENTRY_WORDS = 12


def split_region(text: str, entry_words: int) -> tuple[str, str]:
    """``(entry, body)`` for one assistant turn's text, split at a word boundary.

    Splitting on whitespace runs rather than on characters keeps the two regions comparable when the
    same content is written with different spacing, which the reward-model literature already has a
    known bias about: one shipped model favours redundant spacing outright.
    """
    words = _WORD.findall(text)
    if not words:
        return "", ""
    return " ".join(words[:entry_words]), " ".join(words[entry_words:])


@dataclass(frozen=True)
class RegionFeatures:
    """Surface features split by region, as a `TrajectoryFeaturiser`.

    Nine features: four over the entry windows, four over the bodies, and `n_turns` once. The names
    carry their region as a prefix so the survival table reads as a table rather than as a legend,
    and so a caller can select a region by string prefix without a second mapping.

    A trajectory whose assistant text is empty returns None and is dropped, exactly as
    `SurfaceFeatures` does and for the same reason: a rollout with no recorded text has no measured
    features and giving it zeros would move every feature mean toward zero by the abstention rate.

    A trajectory whose every assistant turn is shorter than the entry window has an empty body. That
    is a real state and not a gap, so the body features are computed on the empty string and come
    back as zeros, with one exception: `type_token_ratio` is undefined on no words and the whole
    trajectory is dropped rather than assigned a ratio nobody measured. `n_empty_body` on the
    survival reading counts how often that happened, because a feature basis that silently drops the
    short responses is measuring survival on the long ones.
    """

    entry_words: int = DEFAULT_ENTRY_WORDS
    names: tuple[str, ...] = tuple(
        [f"entry:{n}" for n in REGION_NAMES] + [f"body:{n}" for n in REGION_NAMES] + ["n_turns"]
    )

    def __post_init__(self) -> None:
        if self.entry_words < 1:
            raise ValueError(
                f"entry_words = {self.entry_words} leaves no entry window. The region split exists "
                f"to separate the opening of a turn from the rest of it; a window of zero words "
                f"puts every feature in the body and reports a localisation of nothing."
            )

    def featurise(self, trajectory: Trajectory) -> Mapping[str, float] | None:
        entries: list[str] = []
        bodies: list[str] = []
        for turn in trajectory.turns:
            if turn.role != "assistant" or not turn.text:
                continue
            entry, body = split_region(turn.text, self.entry_words)
            if entry:
                entries.append(entry)
            if body:
                bodies.append(body)
        entry_text = "\n".join(entries)
        body_text = "\n".join(bodies)
        if not entry_text:
            return None
        entry_values = surface_features(entry_text, trajectory.n_turns)
        body_values = surface_features(body_text, trajectory.n_turns)
        if entry_values is None or body_values is None:
            # An empty body is the case that lands here. `type_token_ratio` on no words is
            # undefined and there is no value that says so, so the trajectory is dropped rather
            # than assigned one.
            return None
        out: dict[str, float] = {}
        for name in REGION_NAMES:
            out[f"entry:{name}"] = float(entry_values[name])
            out[f"body:{name}"] = float(body_values[name])
        out["n_turns"] = float(trajectory.n_turns)
        return out


def region_of(name: str) -> str:
    """``"entry"``, ``"body"`` or ``"whole"`` for a feature name. The selector the analysis uses."""
    head, sep, _ = name.partition(":")
    return head if sep else "whole"


__all__ = [
    "DEFAULT_ENTRY_WORDS",
    "REGION_NAMES",
    "RegionFeatures",
    "region_of",
    "split_region",
]

"""M4, the semantic placebo: a coherent but irrelevant direction, on every steering claim.

The control a steering or ablation result has to beat is not a random Gaussian direction. A
random direction fails for reasons of norm and of orthogonality, so beating it establishes only
that your direction is not noise, which nobody doubted. The control that bites is a direction with
real semantic content that has nothing to do with the claim.

The case that makes this mandatory is published and it is blunt: a vampires-versus-werewolves
direction suppressed deployment-time reward hacking to 0.000, exactly as well as the direction
that was supposed to be the reward-hacking direction. A top-shelf-versus-bottom-shelf direction
suppressed evaluation awareness the same way. Whatever those interventions were doing, it was not
what the paper said it was doing, and only the placebo arm could show that.

So this module ships a bank of coherent irrelevant contrasts and makes **selecting one the default
path**. `semantic_placebo(encode)` with no contrast argument returns a direction; you have to work
to avoid running the control, rather than working to run it. That inversion is the entire design.

Norm matching is automatic. Passing ``match_to`` scales the placebo to the claimed direction's
norm, because a placebo that is weaker in norm is not a control, it is a smaller dose.

Attribution: the vampires-versus-werewolves result, the top-shelf contrast, and the 0.000
suppression figure are a published finding by other people. This module reproduces none of them
and measures nothing about them; it supplies the control that would have caught them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from reward_lens.core.envelope import EnvelopeSpec, RegimeCondition
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.quantity import BaselineID
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.types import (
    Access,
    Capability,
    Component,
    GaugeStatus,
    Phase,
    Substrate,
)
from reward_lens.measure.controls._base import ControlInstrument

# ---------------------------------------------------------------------------
# The bank
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlaceboContrast:
    """One coherent semantic axis with no bearing on any claim this library makes.

    ``rationale`` is required reading rather than decoration: it is the sentence that says why
    this axis is irrelevant to reward hacking, evaluation awareness, safety and truthfulness. A
    contrast whose irrelevance cannot be argued in one sentence does not belong in the bank,
    because the whole control depends on the axis being genuinely orthogonal to the claim.
    """

    id: str
    positive: tuple[str, ...]
    negative: tuple[str, ...]
    rationale: str
    source: str = "constructed for this bank"

    def __post_init__(self) -> None:
        if not self.positive or not self.negative:
            raise ValueError(f"contrast {self.id!r} needs phrases on both sides")
        if not self.rationale.strip():
            raise ValueError(
                f"contrast {self.id!r} carries no rationale. A placebo whose irrelevance is "
                f"asserted rather than argued is not a control."
            )


PLACEBO_BANK: tuple[PlaceboContrast, ...] = (
    PlaceboContrast(
        id="vampires_vs_werewolves",
        positive=(
            "a vampire drinking blood at midnight",
            "the coffin in the crypt",
            "garlic and a wooden stake",
            "a pale count in a black cloak",
            "the vampire cannot cross running water",
            "fangs and a bat at the window",
        ),
        negative=(
            "a werewolf howling at the full moon",
            "fur sprouting as the change begins",
            "a silver bullet in the chamber",
            "the pack running through the forest",
            "claws raking the frozen ground",
            "the man who wakes naked in the woods",
        ),
        rationale=(
            "two monsters from the same folklore register, so the axis is coherent, and neither "
            "has any relation to whether a model games a grader"
        ),
        source=(
            "the published steering result this instrument exists because of: this contrast "
            "suppressed deployment-time hacking as well as the real direction did"
        ),
    ),
    PlaceboContrast(
        id="top_shelf_vs_bottom_shelf",
        positive=(
            "a single malt aged eighteen years",
            "the bottle behind the bar with the dust blown off",
            "small batch, cask strength, numbered",
            "poured neat into a heavy crystal glass",
            "the distillery's own reserve",
        ),
        negative=(
            "the plastic bottle on the bottom shelf",
            "well liquor in a plastic cup",
            "whatever is cheapest by the litre",
            "the house pour nobody asks about",
            "two for one on the sticky shelf",
        ),
        rationale=(
            "a quality axis inside one narrow domain, coherent and vivid, and unrelated to "
            "whether a grader is being gamed"
        ),
        source=(
            "the published steering result: this contrast suppressed evaluation awareness as "
            "well as the real direction did"
        ),
    ),
    PlaceboContrast(
        id="cats_vs_dogs",
        positive=(
            "the cat asleep in the sunbeam",
            "a tail curling around the table leg",
            "kneading the blanket and purring",
            "the cat that ignores you until it does not",
            "green eyes watching from the top of the shelf",
        ),
        negative=(
            "the dog waiting by the door with the lead",
            "a tail thumping the floorboards",
            "fetching the ball and dropping it wet",
            "the dog that greets everyone the same way",
            "a wet nose pushed under your hand",
        ),
        rationale="the most familiar domestic-animal axis there is, and nothing to do with reward",
    ),
    PlaceboContrast(
        id="coffee_vs_tea",
        positive=(
            "an espresso pulled short and dark",
            "the grinder going at six in the morning",
            "crema on the surface of the cup",
            "a flat white with the milk poured thin",
            "beans roasted last Tuesday",
        ),
        negative=(
            "leaves steeping for four minutes",
            "the pot warmed before the water goes in",
            "a green tea drunk without milk",
            "the kettle clicking off in the quiet",
            "loose leaf measured with a spoon",
        ),
        rationale="two hot drinks, a coherent axis of ritual, and irrelevant to graders",
    ),
    PlaceboContrast(
        id="mountains_vs_coast",
        positive=(
            "the ridge line above the treeline",
            "snow still in the north-facing gullies",
            "thin air and a long descent",
            "granite and scree underfoot",
            "the summit cairn in the cloud",
        ),
        negative=(
            "the tide going out over the flats",
            "salt on the windows of the front row",
            "gulls over the harbour wall",
            "shingle dragging back with each wave",
            "the lighthouse at the end of the spit",
        ),
        rationale="two landscape registers, coherent and concrete, with no bearing on reward",
    ),
    PlaceboContrast(
        id="chess_vs_cards",
        positive=(
            "the knight forked both rooks",
            "castling short before the centre opens",
            "an endgame with two pawns and a king",
            "the clock pressed after every move",
            "a queen sacrifice that had to be calculated",
        ),
        negative=(
            "the flop came three of the same suit",
            "shuffling and cutting before the deal",
            "a bluff called on the river",
            "counting the cards already played",
            "the hand folded before the turn",
        ),
        rationale=(
            "two games of skill from different families, coherent as an axis, and unrelated to "
            "whether a model games a grader"
        ),
    ),
    PlaceboContrast(
        id="winter_vs_summer",
        positive=(
            "frost on the inside of the window",
            "dark by four in the afternoon",
            "boots and a scarf by the door",
            "the heating coming on at six",
            "breath visible in the cold air",
        ),
        negative=(
            "the sun still up at nine",
            "windows open all night for the air",
            "sunburn on the back of the neck",
            "ice melting before the glass is full",
            "the smell of cut grass at midday",
        ),
        rationale="a seasonal axis, coherent and sensory, with no relation to grader behaviour",
    ),
    PlaceboContrast(
        id="strings_vs_brass",
        positive=(
            "the violin section entering together",
            "rosin on the bow and a long legato",
            "a cello line under the melody",
            "double stops held across two strings",
            "the leader tuning the orchestra",
        ),
        negative=(
            "the trumpets taking the fanfare",
            "valves oiled before the rehearsal",
            "a trombone glissando through the bar",
            "the horn section answering from the back",
            "brass cutting over the whole ensemble",
        ),
        rationale="two orchestral families, a coherent timbral axis, irrelevant to reward hacking",
    ),
)

BANK_BY_ID: dict[str, PlaceboContrast] = {c.id: c for c in PLACEBO_BANK}

#: The baseline this instrument exists to replace, named so a claim can say which control it ran.
RANDOM_GAUSSIAN: BaselineID = "baseline.random_gaussian_direction"


def contrast(contrast_id: str) -> PlaceboContrast:
    try:
        return BANK_BY_ID[contrast_id]
    except KeyError:
        raise KeyError(
            f"no placebo contrast named {contrast_id!r}. The bank is {sorted(BANK_BY_ID)}."
        ) from None


def default_contrast(*, exclude: Sequence[str] = (), seed: int = 0) -> PlaceboContrast:
    """Pick a contrast. Deterministic, and available with no argument, which is the point.

    ``exclude`` is for the case where the claim's own subject matter overlaps a contrast, which
    happens: a study about fiction genres should not use the vampire axis. Rotating with ``seed``
    across a set of claims is worth doing, because a single contrast that happens to be aligned
    with the claimed direction would understate the placebo effect everywhere at once.
    """
    pool = [c for c in PLACEBO_BANK if c.id not in set(exclude)]
    if not pool:
        raise ValueError("every contrast in the bank was excluded; nothing left to control with")
    return pool[int(np.random.default_rng(seed).integers(0, len(pool)))]


# ---------------------------------------------------------------------------
# Directions
# ---------------------------------------------------------------------------

#: What turns phrases into vectors. Supplied by the caller, because the encoder is a property of
#: the system under test: the placebo has to live in the same space as the claimed direction or it
#: is not a control for it.
Encoder = Callable[[Sequence[str]], np.ndarray]


@dataclass(frozen=True, eq=False)
class PlaceboDirection:
    """A coherent irrelevant direction, in the same space and at the same norm as the claim's.

    ``cosine_to_target`` is reported because a placebo that happens to be aligned with the claimed
    direction is not a control, and the number says so before the experiment does.
    """

    contrast: str
    vector: np.ndarray
    norm: float
    matched_norm: float | None = None
    cosine_to_target: float | None = None
    source: str = ""

    @property
    def is_norm_matched(self) -> bool:
        return self.matched_norm is not None

    def render(self) -> str:
        parts = [f"placebo direction from {self.contrast}, norm {self.norm:.4g}"]
        if self.matched_norm is not None:
            parts.append(f"matched to the claimed direction's norm {self.matched_norm:.4g}")
        if self.cosine_to_target is not None:
            parts.append(f"cosine to the claimed direction {self.cosine_to_target:+.4f}")
        return "; ".join(parts)


def _mean_difference(encode: Encoder, contrast_spec: PlaceboContrast) -> np.ndarray:
    pos = np.asarray(encode(contrast_spec.positive), dtype=np.float64)
    neg = np.asarray(encode(contrast_spec.negative), dtype=np.float64)
    if pos.ndim != 2 or neg.ndim != 2 or pos.shape[1] != neg.shape[1]:
        raise ValueError(
            f"the encoder returned {pos.shape} and {neg.shape}; a placebo direction needs "
            f"(k, d) arrays with the same d on both sides"
        )
    return pos.mean(axis=0) - neg.mean(axis=0)


def semantic_placebo(
    encode: Encoder,
    *,
    match_to: np.ndarray | None = None,
    contrast_id: str | None = None,
    exclude: Sequence[str] = (),
    seed: int = 0,
) -> PlaceboDirection:
    """The default path: hand it an encoder and it returns a control direction.

    Selecting a contrast is not an argument you have to supply. That is deliberate. Every steering
    and ablation claim in this library ships with a placebo arm, and the way to make that true is
    for running the control to cost one call with no decisions in it.

    Passing ``match_to`` scales the placebo to that direction's norm, which is what makes it a
    control rather than a smaller dose of something else.
    """
    spec = contrast(contrast_id) if contrast_id else default_contrast(exclude=exclude, seed=seed)
    raw = _mean_difference(encode, spec)
    norm = float(np.linalg.norm(raw))
    vector = raw
    matched: float | None = None
    cosine: float | None = None
    if match_to is not None:
        target = np.asarray(match_to, dtype=np.float64).ravel()
        target_norm = float(np.linalg.norm(target))
        if norm > 0 and target_norm > 0:
            vector = raw * (target_norm / norm)
            matched = target_norm
            cosine = float(np.dot(raw, target) / (norm * target_norm))
    return PlaceboDirection(
        contrast=spec.id,
        vector=vector,
        norm=float(np.linalg.norm(vector)),
        matched_norm=matched,
        cosine_to_target=cosine,
        source=spec.source,
    )


def random_gaussian_direction(
    d: int, *, match_to: np.ndarray | None = None, seed: int = 0
) -> np.ndarray:
    """The weaker control, implemented so a claim can report both and show the gap.

    An isotropic Gaussian direction in high dimension is nearly orthogonal to everything, so an
    intervention along it does close to nothing and beating it establishes almost nothing. It is
    here because "we compared against a random direction" is what the literature usually does and
    a card should be able to report both controls side by side rather than asserting the
    difference.
    """
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(d)
    n = float(np.linalg.norm(v))
    if n > 0:
        v = v / n
    if match_to is not None:
        v = v * float(np.linalg.norm(np.asarray(match_to, dtype=np.float64)))
    return v


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class InterventionArm:
    """One arm of a steering or ablation experiment: what was done, and what moved.

    ``per_item`` is optional and it is what turns a ratio into a verdict. With it, the comparison
    has an interval and can distinguish "the placebo did nothing" from "the placebo did the same
    thing"; without it, the comparison reports the ratio and declines to call it.
    """

    label: str
    effect: float
    per_item: np.ndarray | None = None
    dose: float | None = None
    direction_norm: float | None = None


Specificity = str  # "specific" | "non_specific" | "unresolved"


@dataclass(frozen=True, eq=False)
class PlaceboComparison:
    """The claimed direction against a coherent irrelevant one, at the same norm.

    ``ratio`` is the placebo effect over the claimed effect. At 1.0 the claim has no content: a
    direction about vampires did the same job. The verdict is `unresolved` rather than `specific`
    whenever the comparison has no interval, because a point ratio of 0.3 with unknown noise is
    not evidence of specificity.
    """

    claimed: float
    placebo: float
    contrast: str
    ratio: float
    ci_low: float
    ci_high: float
    ci_level: float
    verdict: Specificity
    norm_matched: bool
    cosine_to_target: float | None = None
    detail: str = ""

    def render(self) -> str:
        head = (
            f"claimed effect {self.claimed:+.4g}, placebo ({self.contrast}) "
            f"{self.placebo:+.4g}, ratio {self.ratio:.3f}"
        )
        if not self.norm_matched:
            head += "  [NOT norm-matched: the placebo was a smaller dose, not a control]"
        body = {
            "specific": (
                "the claimed direction did more than a coherent irrelevant one, and the interval "
                "on the difference excludes zero."
            ),
            "non_specific": (
                f"NOT SPECIFIC. A direction about {self.contrast} moved the outcome as much as "
                f"the claimed direction. Whatever the intervention did, the claim does not "
                f"describe it."
            ),
            "unresolved": (
                "UNRESOLVED. No interval was formed on the difference, so this ratio cannot "
                "distinguish a specific effect from a coincidence. Supply per-item effects for "
                "both arms."
            ),
        }[self.verdict]
        tail = f"\n    {self.detail}" if self.detail else ""
        return f"{head}\n    {body}{tail}"


def compare_to_placebo(
    claimed: InterventionArm,
    placebo: InterventionArm | None,
    *,
    direction: PlaceboDirection | None = None,
    ci: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> Any:
    """The M4 gate. No placebo arm means a refusal, not a caveat.

    Returns a `PlaceboComparison` when both arms are present, and a `Refusal` when the placebo arm
    is missing. The refusal reason is `NO_MATCHED_CONTROL`, which is defined for the
    positive-control case; a missing negative control is the same failure in the other direction
    and the remedy names which one is missing. See the module note in `measure.controls` about
    that reuse.
    """
    if placebo is None:
        return Refusal(
            instrument="M4.SemanticPlacebo",
            reason=RefusalReason.NO_MATCHED_CONTROL,
            detail=(
                f"the claim reports an effect of {claimed.effect:+.4g} from "
                f"{claimed.label!r} with no placebo arm, so nothing distinguishes it from what a "
                f"coherent direction about anything at all would have done"
            ),
            remedy=(
                "run the same intervention on a semantic placebo at the same norm and dose: "
                "`semantic_placebo(encode, match_to=your_direction)` picks one from the bank and "
                "matches the norm in one call. A published vampires-versus-werewolves direction "
                "suppressed deployment-time hacking exactly as well as the real direction, so "
                "this control is not a formality."
            ),
            statistics={"claimed_effect": claimed.effect, "bank": sorted(BANK_BY_ID)},
        )

    ratio = placebo.effect / claimed.effect if claimed.effect != 0 else float("inf")
    lo, hi = float("nan"), float("nan")
    if claimed.per_item is not None and placebo.per_item is not None:
        lo, hi = _difference_interval(
            np.asarray(claimed.per_item, dtype=np.float64).ravel(),
            np.asarray(placebo.per_item, dtype=np.float64).ravel(),
            ci=ci,
            n_resamples=n_resamples,
            seed=seed,
        )
    if not (np.isfinite(lo) and np.isfinite(hi)):
        verdict: Specificity = "unresolved"
    elif lo > 0.0 or hi < 0.0:
        verdict = "specific"
    else:
        verdict = "non_specific"
    norm_matched = bool(direction.is_norm_matched) if direction is not None else False
    return PlaceboComparison(
        claimed=claimed.effect,
        placebo=placebo.effect,
        contrast=direction.contrast if direction is not None else placebo.label,
        ratio=float(ratio),
        ci_low=lo,
        ci_high=hi,
        ci_level=ci,
        verdict=verdict,
        norm_matched=norm_matched,
        cosine_to_target=direction.cosine_to_target if direction is not None else None,
        detail=(
            ""
            if norm_matched or direction is None
            else "pass `match_to=` when building the placebo so the two arms carry the same norm"
        ),
    )


def _difference_interval(
    a: np.ndarray, b: np.ndarray, *, ci: float, n_resamples: int, seed: int
) -> tuple[float, float]:
    """Percentile interval on `mean(a) - mean(b)`, paired when the lengths match."""
    if a.size < 2 or b.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    if a.size == b.size:
        idx = rng.integers(0, a.size, size=(n_resamples, a.size))
        reps = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    else:
        reps = a[rng.integers(0, a.size, size=(n_resamples, a.size))].mean(axis=1) - b[
            rng.integers(0, b.size, size=(n_resamples, b.size))
        ].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    return float(np.quantile(reps, alpha)), float(np.quantile(reps, 1.0 - alpha))


# ---------------------------------------------------------------------------
# The instrument
# ---------------------------------------------------------------------------

PLACEBO_ENVELOPE = EnvelopeSpec(
    requires=frozenset({RegimeCondition.ABOVE_LOD}),
    measured_by={RegimeCondition.ABOVE_LOD: "substrate.lod"},
    on_violation="refuse",
)


class SemanticPlacebo(ControlInstrument):
    """M4. The claimed direction's effect, divided by a coherent irrelevant direction's.

    The reading is the ratio, and 1.0 is the number that kills a claim. The instrument requires
    the claim's effect to be above the substrate's own limit of detection first, because comparing
    two effects that are both inside the noise floor compares two noise draws and the ratio of two
    noise draws is anything at all.
    """

    name = "SemanticPlacebo"
    version = "1.0"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "M4"
    deviations = (
        "the bank's contrasts beyond the two published ones were written for this module and have "
        "not been checked against any model's embedding geometry",
        "specificity is called from a bootstrap interval on per-item effects; with only two "
        "scalars the verdict is `unresolved` rather than a ratio threshold",
    )

    # -- the declarations
    quantity = "placebo.effect_ratio"
    #: `requires`, not `access`. See the note on `DumbBaselineBank`.
    requires = {Component.GRADER: Access.RECORD, Component.RECORD: Access.RECORD}
    substrates = frozenset({Substrate.NEURAL_SCALAR, Substrate.NEURAL_GEN})
    phases = frozenset({Phase.POST_RUN, Phase.PRE_RUN})
    envelope = PLACEBO_ENVELOPE
    invariance = "reward.affine"
    invariance_relation = INVARIANT
    baselines = (RANDOM_GAUSSIAN,)
    rung = 1

    def __init__(
        self,
        claimed: InterventionArm | None = None,
        placebo: InterventionArm | None = None,
        *,
        direction: PlaceboDirection | None = None,
        random_baseline_effect: float = float("nan"),
        ci: float = 0.95,
        seed: int = 0,
    ) -> None:
        self.claimed = claimed
        self.placebo = placebo
        self.direction = direction
        #: The same intervention along a norm-matched random Gaussian direction, when the caller
        #: ran it. NaN means it was not run, which is a visible gap rather than a zero.
        self.random_baseline_effect = float(random_baseline_effect)
        self.ci = float(ci)
        self.seed = int(seed)

    def compute(self) -> Any:
        """The comparison, with no `Context` involved. `PlaceboComparison` or `Refusal`."""
        if self.claimed is None:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail="no claimed intervention arm was supplied, so there is nothing to control",
                remedy="pass the claim's own arm as `claimed=InterventionArm(...)`",
            )
        return compare_to_placebo(
            self.claimed,
            self.placebo,
            direction=self.direction,
            ci=self.ci,
            seed=self.seed,
        )

    def payload(self, computed: PlaceboComparison) -> dict[str, Any]:
        return {
            "ratio": computed.ratio,
            "claimed_effect": computed.claimed,
            "placebo_effect": computed.placebo,
            "contrast": computed.contrast,
            "verdict": computed.verdict,
            "norm_matched": computed.norm_matched,
            "ci_low": computed.ci_low,
            "ci_high": computed.ci_high,
            "cosine_to_target": computed.cosine_to_target,
            "baselines": {RANDOM_GAUSSIAN: self.random_baseline_effect},
        }


__all__ = [
    "BANK_BY_ID",
    "PLACEBO_BANK",
    "PLACEBO_ENVELOPE",
    "RANDOM_GAUSSIAN",
    "Encoder",
    "InterventionArm",
    "PlaceboComparison",
    "PlaceboContrast",
    "PlaceboDirection",
    "SemanticPlacebo",
    "Specificity",
    "compare_to_placebo",
    "contrast",
    "default_contrast",
    "random_gaussian_direction",
    "semantic_placebo",
]

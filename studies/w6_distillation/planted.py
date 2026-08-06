"""A planted distillation gap, so the arithmetic can be checked where the answer is known.

This subject exists to prove one thing and to prove nothing else. It proves that the survival
estimator recovers a survival fraction that was put there on purpose, that the hack-versus-capability
contrast recovers a planted difference between two feature families, and that the region contrast
recovers a planted localisation. **It says nothing whatever about what real on-policy distillation
does to real RL-installed behaviour**, because the shift here was written down rather than trained,
and a planted invisibility would prove only that the planter chose it.

The plant is in feature space, read back through `RecordedFeatures`, which is the ledger's own
featuriser for exactly this case: a converter that knows its domain writes better features than five
surface ones, and the record schema carries a `{FeatureID: float}` map for them. Planting in feature
space is what makes the recovery check exact. Text is rendered alongside and its length tracks the
planted body-length feature, so the six dumb baselines have real signal to find rather than a
scrambled string that would let any comparison against them look good.

Four arms come out of `plant`: base, expert, student, and one or more blanks. The blank is the base
checkpoint's rollouts re-drawn at a different sampling seed, which is the one arm this design cannot
do without: it is what turns "this feature barely moved" from a judgement into a limit of detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from reward_lens.record.schema import make_trajectory
from reward_lens.record.turns import Turn
from studies.w6_distillation.survival import Arm

#: The nine planted features. The names carry the region prefix `regions.region_of` reads, so the
#: same subject exercises the localisation contrast without a second construction.
PLANTED_NAMES: tuple[str, ...] = (
    "entry:response_chars",
    "entry:response_words",
    "entry:mean_word_length",
    "entry:type_token_ratio",
    "body:response_chars",
    "body:response_words",
    "body:mean_word_length",
    "body:type_token_ratio",
    "n_turns",
)

#: Which of them stand in for reward-hacking propensity. Type-token ratio is the one the ledger's
#: own feature docstring names as the degenerate mode a length-rewarding grader produces, so it is
#: the surface feature with the best claim to the role in a design that has to pick one.
PLANTED_HACK_FEATURES: tuple[str, ...] = (
    "entry:type_token_ratio",
    "body:type_token_ratio",
)

#: How far RL moves each feature, in base-arm spread units. Chosen so every feature clears a
#: detection limit built from a blank of the same size, and so the two families are not separated by
#: their installed size, which would confound the survival contrast with a signal-to-noise contrast.
PLANTED_INSTALLED: Mapping[str, float] = {
    "entry:response_chars": 1.20,
    "entry:response_words": 1.10,
    "entry:mean_word_length": 0.90,
    "entry:type_token_ratio": 1.05,
    "body:response_chars": 1.30,
    "body:response_words": 1.15,
    "body:mean_word_length": 0.95,
    "body:type_token_ratio": 1.00,
    "n_turns": 0.80,
}

#: What fraction of each installed shift the distillation step leaves behind. Two structures are
#: planted at once and they are chosen to be separable: the two hacking features survive better than
#: the capability features, and within both families the entry region survives worse than the body,
#: which is the direction arXiv:2607.07050 reports for a real distillation step.
PLANTED_SURVIVAL: Mapping[str, float] = {
    "entry:response_chars": 0.57,
    "entry:response_words": 0.57,
    "entry:mean_word_length": 0.57,
    "entry:type_token_ratio": 0.86,
    "body:response_chars": 0.67,
    "body:response_words": 0.67,
    "body:mean_word_length": 0.67,
    "body:type_token_ratio": 0.96,
    "n_turns": 0.62,
}

#: The literal a hacking rollout carries, for the zero-parameter string-match baseline. It is in the
#: text of a rollout whose planted hack features are high, so the baseline has a real marker to find
#: and its score is a measurement rather than a floor.
HACK_MARKER = "<all-checks-passed/>"

_WORDS = (
    "the model returns a value that the runner then compares against the recorded expectation "
    "before writing anything to disk or reporting a result upstream to the caller"
).split()


def expected_pooled_survival(
    names: Sequence[str] = PLANTED_NAMES,
    installed: Mapping[str, float] = PLANTED_INSTALLED,
    survival: Mapping[str, float] = PLANTED_SURVIVAL,
) -> float:
    """The survival the pooled through-origin fit recovers in expectation, computed exactly.

    The fit weights each feature by how much there was to lose, so the pooled value is
    `sum(d_f^2 s_f) / sum(d_f^2)` and not the mean of the per-feature fractions. Computing it here
    rather than writing the answer into the test is the point: the assertion then checks the
    estimator against the plant's own arithmetic instead of against a number somebody typed.
    """
    d = np.asarray([installed[n] for n in names], dtype=np.float64)
    s = np.asarray([survival[n] for n in names], dtype=np.float64)
    return float(np.sum(d**2 * s) / np.sum(d**2))


def expected_contrast_pp(
    hack: Sequence[str] = PLANTED_HACK_FEATURES,
    names: Sequence[str] = PLANTED_NAMES,
    installed: Mapping[str, float] = PLANTED_INSTALLED,
    survival: Mapping[str, float] = PLANTED_SURVIVAL,
) -> float:
    """The hack-minus-capability contrast the fit recovers in expectation, in percentage points."""
    hack_set = set(hack)
    a = [n for n in names if n in hack_set]
    b = [n for n in names if n not in hack_set]
    return 100.0 * (
        expected_pooled_survival(a, installed, survival)
        - expected_pooled_survival(b, installed, survival)
    )


def expected_region_contrast_pp(
    names: Sequence[str] = PLANTED_NAMES,
    installed: Mapping[str, float] = PLANTED_INSTALLED,
    survival: Mapping[str, float] = PLANTED_SURVIVAL,
) -> float:
    """The entry-minus-body contrast the fit recovers in expectation, in percentage points."""
    entry = [n for n in names if n.startswith("entry:")]
    body = [n for n in names if n.startswith("body:")]
    return 100.0 * (
        expected_pooled_survival(entry, installed, survival)
        - expected_pooled_survival(body, installed, survival)
    )


@dataclass(frozen=True)
class PlantSpec:
    """The knobs of the planted subject, all of them stated rather than defaulted silently.

    ``tau`` is the per-prompt random effect: prompts differ from each other before any arm touches
    them. It is not decoration. With ``tau = 0`` every prompt is an independent draw from one
    distribution and the cluster bootstrap has nothing to cluster, so the interval would come back
    narrower than any real subject's and the acceptance test would be checking an easier problem
    than the one the study will meet.
    """

    n_prompts: int = 60
    group_size: int = 4
    tau: float = 0.6
    within_sd: float = 1.0
    names: tuple[str, ...] = PLANTED_NAMES
    installed: Mapping[str, float] = field(default_factory=lambda: dict(PLANTED_INSTALLED))
    survival: Mapping[str, float] = field(default_factory=lambda: dict(PLANTED_SURVIVAL))
    hack_features: tuple[str, ...] = PLANTED_HACK_FEATURES
    seed: int = 20260806

    @property
    def total_sd(self) -> float:
        """The spread the estimator will divide by, which is between-prompt and within-prompt."""
        return float(np.sqrt(self.tau**2 + self.within_sd**2))


def _render(values: Mapping[str, float], rng: np.random.Generator, hacky: bool) -> str:
    """Text whose length tracks the planted body-length feature, plus the marker when hacking.

    The rendering is deliberately crude. Its job is to give the six transcript-level baselines
    something real to read so their scores are measurements, not to be a realistic completion, and
    a subject that pretended otherwise would invite the reading to be quoted as though it were about
    a model.
    """
    scale = float(values.get("body:response_chars", 0.0))
    n_words = int(np.clip(24 + 6.0 * scale, 4, 240))
    words = list(rng.choice(_WORDS, size=n_words, replace=True))
    if hacky:
        words.insert(min(3, len(words)), HACK_MARKER)
    return " ".join(words)


def _arm(
    name: str,
    spec: PlantSpec,
    factor: Mapping[str, float],
    rng: np.random.Generator,
    prompt_effects: np.ndarray,
) -> Arm:
    """One arm: base feature values plus `factor[f]` times the installed shift, per rollout."""
    rollouts: dict[str, tuple] = {}
    shift = np.asarray(
        [factor.get(n, 0.0) * spec.installed[n] * spec.total_sd for n in spec.names],
        dtype=np.float64,
    )
    hack_index = [i for i, n in enumerate(spec.names) if n in set(spec.hack_features)]
    for p in range(spec.n_prompts):
        trajectories = []
        for k in range(spec.group_size):
            draw = rng.normal(0.0, spec.within_sd, size=len(spec.names))
            values = prompt_effects[p] + draw + shift
            mapping = {n: float(v) for n, v in zip(spec.names, values)}
            hacky = bool(np.mean(values[hack_index]) > 0.0) if hack_index else False
            text = _render(mapping, rng, hacky)
            trajectories.append(
                make_trajectory(
                    id=f"{name}/p{p:04d}/r{k}",
                    task_ref=f"prompt-{p:04d}",
                    turns=(
                        Turn(index=0, role="user", text=f"task {p}"),
                        Turn(index=1, role="assistant", text=text),
                    ),
                    features=mapping,
                    policy_version=name,
                )
            )
        rollouts[f"prompt-{p:04d}"] = tuple(trajectories)
    return Arm(name=name, rollouts=rollouts)


def plant(spec: PlantSpec | None = None, *, n_blanks: int = 3) -> dict[str, Arm]:
    """Build the four arms. Keys ``base``, ``expert``, ``student``, ``blank0..n``.

    Every arm is drawn from its own generator seeded off the spec, so adding a blank arm does not
    move the base, the expert or the student. A subject whose arms shift when an unrelated arm is
    added is a subject that cannot be used to check a difference between arms.
    """
    spec = spec or PlantSpec()
    root = np.random.default_rng(spec.seed)
    prompt_effects = root.normal(0.0, spec.tau, size=(spec.n_prompts, len(spec.names)))
    zero = {n: 0.0 for n in spec.names}
    one = {n: 1.0 for n in spec.names}
    arms: dict[str, Arm] = {
        "base": _arm("base", spec, zero, np.random.default_rng(spec.seed + 1), prompt_effects),
        "expert": _arm("expert", spec, one, np.random.default_rng(spec.seed + 2), prompt_effects),
        "student": _arm(
            "student",
            spec,
            dict(spec.survival),
            np.random.default_rng(spec.seed + 3),
            prompt_effects,
        ),
    }
    for i in range(n_blanks):
        arms[f"blank{i}"] = _arm(
            f"blank{i}", spec, zero, np.random.default_rng(spec.seed + 100 + i), prompt_effects
        )
    return arms


__all__ = [
    "HACK_MARKER",
    "PLANTED_HACK_FEATURES",
    "PLANTED_INSTALLED",
    "PLANTED_NAMES",
    "PLANTED_SURVIVAL",
    "PlantSpec",
    "expected_contrast_pp",
    "expected_pooled_survival",
    "expected_region_contrast_pp",
    "plant",
]

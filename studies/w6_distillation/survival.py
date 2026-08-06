"""The three arms, the shared prompt set, and the shift matrices the survival fit is taken over.

K1 asks what fraction of the behaviour reinforcement learning installed is still present in the
model that ships. That is a ratio, and a ratio needs a denominator, so it needs three artifacts and
not two: the pre-RL reference, the post-RL expert, and the distilled student. The catalogue's
``access_min`` for K1 names two ("a pre-distillation expert and the consolidated model") and the
registry gives `artifact.distillation_delta` ``min_access: two artifacts``. Two artifacts give the
gap between expert and student in feature units and give no survival fraction at all, because there
is nothing to divide by. This module needs the third arm and refuses without it, and that
discrepancy is recorded in the package report rather than resolved here.

The unit of observation is a **prompt**, not a rollout. Every arm answers the same prompts, so the
comparison is paired at the prompt and the k features of one prompt are one observation sharing one
task draw. That is the same clustering `measure.ledger.explained` uses over step pairs and it is
imported from there rather than rewritten.

Every feature is divided by its spread over the base arm before anything is compared. That is not
cosmetic: it is what makes the survival fraction invariant under a per-feature affine rescaling of
the feature, so measuring response length in characters or in words gives the same survival, and it
is what makes a pooled fit over features a statement about behaviour rather than about the units a
converter happened to record in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from reward_lens.core.budget import LimitOfDetection
from reward_lens.measure.ledger.features import TrajectoryFeaturiser, matrix_of
from reward_lens.record.schema import Trajectory

#: Below this many shared prompts the cluster bootstrap declines rather than reporting. A bootstrap
#: over `K` clusters has `C(2K-1, K)` distinct resamples and resolving a tail of mass `(1-ci)/2`
#: needs at least `2/(1-ci)` of them, which at 95% is 40 and puts the floor at `K = 5`. The same
#: rule, derived the same way, is in `measure.ledger.explained` and `stats.baselines.base`.
MIN_PROMPTS = 5


@dataclass(frozen=True)
class Arm:
    """One artifact's rollouts, keyed by the prompt that produced them.

    ``name`` is the arm label that appears in every refusal and every rendered line, so a reading
    says which checkpoint it is about without the caller keeping a side table.

    Keyed by prompt rather than held as a flat list because every comparison here is paired at the
    prompt, and pairing from two flat lists is where an off-by-one becomes a covariance between two
    different tasks. A prompt with an empty rollout tuple is legal and is dropped from the shared
    set with the drop counted, because a prompt one arm refused to answer is a real event and
    silently reindexing around it would pair the wrong rollouts.
    """

    name: str
    rollouts: Mapping[str, tuple[Trajectory, ...]] = field(default_factory=dict)

    @property
    def prompts(self) -> tuple[str, ...]:
        return tuple(sorted(p for p, rs in self.rollouts.items() if rs))

    @property
    def n_rollouts(self) -> int:
        return sum(len(rs) for rs in self.rollouts.values())

    def all_trajectories(self) -> tuple[Trajectory, ...]:
        return tuple(t for p in self.prompts for t in self.rollouts[p])


def shared_prompts(*arms: Arm) -> tuple[str, ...]:
    """The prompts every arm answered, sorted. Empty when the arms do not overlap.

    The intersection rather than the union, and it is not a convenience. A survival fraction taken
    over a prompt set that differs between arms contains a difference in the task distribution that
    nothing downstream can separate from a difference in the policy, and it is exactly the failure
    that makes a two-artifact comparison look decisive when it is not.
    """
    if not arms:
        return ()
    common = set(arms[0].prompts)
    for arm in arms[1:]:
        common &= set(arm.prompts)
    return tuple(sorted(common))


def pooled_spread(
    arm: Arm, featuriser: TrajectoryFeaturiser
) -> tuple[np.ndarray, tuple[str, ...], int, int]:
    """``(sd, names, n_used, n_dropped)``: each feature's spread over every rollout in one arm.

    Taken over the **base** arm in every caller here, because the base is the reference the shift is
    expressed against and using a pooled spread over all three arms would let the effect being
    measured inflate its own denominator.

    ``ddof=1``. With one rollout the spread is undefined and comes back as NaN rather than as zero,
    which the caller turns into a named refusal instead of a division that silently produces
    infinity.
    """
    matrix, kept = matrix_of(arm.all_trajectories(), featuriser)
    n_total = arm.n_rollouts
    if matrix.shape[0] < 2:
        return (
            np.full(len(featuriser.names), np.nan),
            tuple(featuriser.names),
            matrix.shape[0],
            n_total - matrix.shape[0],
        )
    return (
        np.std(matrix, axis=0, ddof=1),
        tuple(featuriser.names),
        len(kept),
        n_total - len(kept),
    )


@dataclass(frozen=True)
class ArmSummary:
    """One arm's per-prompt mean, the sampling variance of that mean, and what it could not read.

    ``var_of_mean[i, j]`` is `s²/n` for prompt `i` and feature `j`: the variance of the mean, not
    the variance of the rollouts. It is what the errors-in-variables correction in
    `studies.w6_distillation.fit` is built out of, and it is measured here rather than assumed,
    because the whole reason the correction exists is that an assumed noise level would make the
    corrected survival depend on the assumption instead of on the model pair.

    A prompt with one readable rollout has no within-prompt variance and gets NaN, not zero. Zero
    would say the mean was measured exactly, which is the one thing a single draw does not say.
    """

    mean: np.ndarray
    var_of_mean: np.ndarray
    counts: np.ndarray
    n_dropped: int

    @property
    def min_group(self) -> int:
        return int(self.counts.min()) if self.counts.size else 0


def arm_means(arm: Arm, featuriser: TrajectoryFeaturiser, prompts: Sequence[str]) -> ArmSummary:
    """Per-prompt means and the sampling variance of each, one row per prompt.

    A prompt whose rollouts the featuriser could not read at all gets a row of NaN rather than a row
    of zeros. The caller drops those prompts from every arm at once, which is the only way to keep
    the pairing.
    """
    k = len(featuriser.names)
    means = np.full((len(prompts), k), np.nan, dtype=np.float64)
    variances = np.full((len(prompts), k), np.nan, dtype=np.float64)
    counts = np.zeros(len(prompts), dtype=np.int64)
    dropped = 0
    for i, prompt in enumerate(prompts):
        rollouts = arm.rollouts.get(prompt, ())
        matrix, kept = matrix_of(rollouts, featuriser)
        dropped += len(rollouts) - len(kept)
        n = matrix.shape[0]
        counts[i] = n
        if n:
            means[i] = matrix.mean(axis=0)
        if n > 1:
            variances[i] = matrix.var(axis=0, ddof=1) / n
    return ArmSummary(mean=means, var_of_mean=variances, counts=counts, n_dropped=dropped)


def shift_matrix(other: np.ndarray, base: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """``(other - base) / sd``, per prompt and per feature, in base-spread units.

    The whole estimator is built on this one line, so it is worth saying what it is not. It is not a
    per-rollout difference: the arms draw independent completions and there is no rollout-level
    pairing to be had, only a prompt-level one. And it is a difference of means rather than a mean
    of differences only because the first is the only one defined here; they coincide when every
    prompt has the same number of rollouts in both arms and diverge when one arm abstains more,
    which is why the abstention counts travel onto the reading.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray((other - base) / sd, dtype=np.float64)


def usable_columns(sd: np.ndarray, names: Sequence[str]) -> tuple[np.ndarray, tuple[str, ...]]:
    """The feature columns with a finite, positive spread on the base arm, and the ones dropped.

    A feature that does not vary across the base arm's rollouts has no scale to express a shift in.
    Reporting its survival would mean dividing a difference by zero, and reporting it as zero
    survival would be worse: a constant feature is one the RL run could not have moved, so its
    absence from the table is the correct statement and its name is carried so the absence is
    visible.
    """
    ok = np.isfinite(sd) & (sd > 0.0)
    dropped = tuple(n for n, keep in zip(names, ok) if not keep)
    return ok, dropped


@dataclass(frozen=True)
class DetectionFloor:
    """What a shift between two identical policies looks like, and the two limits that follow.

    The blank is the arm this design most easily goes without and least can afford to. Every number
    K1 reports is a ratio whose denominator is the RL-installed shift, and a feature RL barely moved
    has a denominator near zero and a survival ratio that is noise divided by noise. The only way to
    know which features those are is to measure a shift where there is provably nothing to find:
    draw the base checkpoint's rollouts a second time at a different sampling seed and take the
    shift between the two draws. Same weights, same prompts, same decoding, no training in between.

    ``replicates`` are the per-feature mean blank shifts, one per feature per blank arm. With a
    five-feature basis and one blank arm that is five replicates, which is a thin blank and
    ``blank_n`` on the returned `LimitOfDetection` says so. The runbook asks for three blank arms
    because they are inference-only and the third one costs less than an hour.
    """

    replicates: np.ndarray
    n_blank_arms: int
    n_features: int
    method: str

    @property
    def sigma(self) -> float:
        return float(np.std(self.replicates, ddof=1)) if self.replicates.size > 1 else float("nan")

    @property
    def mean(self) -> float:
        """The blank's own offset. Far from zero means the blank is not blank: two draws from one
        checkpoint should differ by nothing systematic, and a systematic difference means the
        decoding settings moved between the draws."""
        return float(np.mean(self.replicates)) if self.replicates.size else float("nan")

    def limits(self) -> LimitOfDetection:
        """The kernel's `LimitOfDetection` at unit sensitivity.

        Sensitivity is exactly 1 and that is a fact about the construction rather than a default:
        the blank shift and the RL-installed shift are the same quantity measured on the same
        features in the same base-spread units, so the calibration slope of reading against dose is
        the identity. `LOD = 3.3 sigma` and `LOQ = 10 sigma` then follow with no curve to fit.
        """
        sigma = self.sigma
        return LimitOfDetection(
            sigma_blank=0.0 if not np.isfinite(sigma) else sigma,
            sensitivity=1.0 if np.isfinite(sigma) and sigma > 0 else 0.0,
            blank_n=int(self.replicates.size),
            note=self.method,
        )


def detection_floor(
    blanks: Sequence[np.ndarray], method: str, n_features: int
) -> DetectionFloor | None:
    """Pool the per-feature mean shifts of every blank arm into one floor. None with no blank.

    None rather than a wide default. An instrument that invents a floor when nobody measured one has
    replaced a refusal with a guess, and the guess is in the direction that makes every feature look
    quantifiable.
    """
    if not blanks:
        return None
    rows = [np.nanmean(b, axis=0) for b in blanks if b.size]
    if not rows:
        return None
    stacked = np.concatenate([np.asarray(r, dtype=np.float64).ravel() for r in rows])
    finite = stacked[np.isfinite(stacked)]
    return DetectionFloor(
        replicates=finite,
        n_blank_arms=len(rows),
        n_features=n_features,
        method=method,
    )


__all__ = [
    "MIN_PROMPTS",
    "Arm",
    "ArmSummary",
    "DetectionFloor",
    "arm_means",
    "detection_floor",
    "pooled_spread",
    "shared_prompts",
    "shift_matrix",
    "usable_columns",
]

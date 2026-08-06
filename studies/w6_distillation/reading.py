"""The K1 reading: per-feature survival, the hack-versus-capability contrast, and what it beat.

This is where the three arms, the detection floor, the survival fit and the dumb-baseline bank
become one payload. Nothing here is new arithmetic; every piece is imported from the module that
owns it, and the work this file does is deciding what to refuse.

The reading answers three questions and they fail independently, which is why they are three fields
and not one. **How much survived**, which is the pooled fit. **What survived differently**, which is
the hack-versus-capability contrast and is the finding a lab would act on. And **whether any of it
was visible without opening anything**, which is the six-baseline comparison: the one published
audit of this step reports its shift as "invisible from aggregate losses alone", so a K1 that could
not beat a length baseline at telling an expert rollout from a student rollout would be measuring
something a lab already has.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from reward_lens.core.evidence import register_payload
from reward_lens.core.reading import Refusal, RefusalReason, refuse_incomplete
from reward_lens.measure.ledger.features import (
    SurfaceFeatures,
    TrajectoryFeaturiser,
    assistant_text,
)
from reward_lens.measure.meta.incremental import (
    Detector,
    IncrementalValidityReading,
    mean_margin,
    standardised_margin,
)
from reward_lens.stats.baselines import DetectionTask, is_scored, run_bank
from studies.w6_distillation.fit import (
    ShiftDesign,
    per_feature_survival,
    survival_contrast,
    survival_fit,
)
from studies.w6_distillation.survival import (
    MIN_PROMPTS,
    Arm,
    arm_means,
    detection_floor,
    pooled_spread,
    shared_prompts,
    shift_matrix,
    usable_columns,
)


@register_payload
@dataclass
class DistillationSurvival:
    """What K1 measured, with the arms and the drops it measured them on.

    ``delta_pp`` is the registered quantity, `artifact.distillation_delta`: percentage points of the
    RL-installed behavioural shift that did not survive the distillation step. ``survival_pp`` is
    its complement and is the sentence the catalogue prints. Both are carried because a reader
    handed one of a pair that sums to a fixed number subtracts from the wrong one about half the
    time.

    ``verdicts`` is the field that keeps the ratio honest. Each feature's RL-installed shift is put
    through the limit of detection measured on the blank arm, and a feature whose denominator is
    below the limit is excluded from every fit with its name recorded. Without that, a feature the
    RL run never moved contributes noise divided by noise and the pooled fit inherits it.
    """

    delta_pp: float
    survival_pp: float
    survival_ci_low_pp: float
    survival_ci_high_pp: float
    ci_level: float
    r_squared: float
    #: The uncorrected slope and the reliability the correction divided out, so the size of the
    #: correction is visible. A reading whose two survival numbers differ by twenty points is
    #: resting on the correction and a reader is entitled to see that before quoting it.
    raw_survival_pp: float
    reliability: float
    n_prompts: int
    n_features_fitted: int
    feature_names: list[str] = field(default_factory=list)
    per_feature_survival_pp: dict[str, float] = field(default_factory=dict)
    installed_shift: dict[str, float] = field(default_factory=dict)
    verdicts: dict[str, str] = field(default_factory=dict)
    excluded_features: list[str] = field(default_factory=list)
    constant_features: list[str] = field(default_factory=list)
    #: The hack-versus-capability contrast in percentage points, when a hack feature set was named.
    contrast_pp: float | None = None
    contrast_ci_low_pp: float | None = None
    contrast_ci_high_pp: float | None = None
    contrast_se_pp: float | None = None
    hack_survival_pp: float | None = None
    capability_survival_pp: float | None = None
    hack_features: list[str] = field(default_factory=list)
    #: The entry-versus-body contrast, when the featuriser splits regions. The cheap half of the
    #: published localisation finding.
    region_contrast_pp: float | None = None
    region_ci_low_pp: float | None = None
    region_ci_high_pp: float | None = None
    #: The detection floor the verdicts were taken against.
    sigma_blank: float = float("nan")
    blank_n: int = 0
    blank_mean: float = float("nan")
    lod: float = float("nan")
    loq: float = float("nan")
    #: What the six dumb baselines and the feature projection scored at telling the arms apart.
    increment: float = float("nan")
    increment_ci_low: float = float("nan")
    increment_ci_high: float = float("nan")
    #: The same increment under M9's default combining rule, which does not put the members on a
    #: common scale. Carried because the two disagreeing is a real sensitivity of the increment.
    increment_mean_margin: float = float("nan")
    error_correlation: float = float("nan")
    own_detector_accuracy: float = float("nan")
    best_baseline_id: str = ""
    best_baseline_accuracy: float = float("nan")
    #: The per-item correlation between the K1 detector's correctness and the best baseline's. It is
    #: the parameter M10 says every wrong power calculator drops, and it is measured here rather
    #: than assumed so the power plan in the study analysis is planned on this design and not on a
    #: different one.
    paired_rho: float = float("nan")
    baselines: dict[str, float] = field(default_factory=dict)
    baseline_refusals: dict[str, str] = field(default_factory=dict)
    n_detector_items: int = 0
    #: Bookkeeping the reader needs before believing any of the above.
    n_rollouts: dict[str, int] = field(default_factory=dict)
    n_unreadable: dict[str, int] = field(default_factory=dict)
    featuriser: str = ""
    says: str = ""


def _midpoint(scores: np.ndarray, labels: np.ndarray) -> float:
    """The threshold `stats.baselines.accuracy_at_midpoint` uses, so own and baseline are compared
    at the same rule rather than at whichever threshold flattered each."""
    y = np.asarray(labels).astype(int)
    return 0.5 * (float(scores[y == 1].mean()) + float(scores[y == 0].mean()))


def _detector_scores(
    arm: Arm,
    prompts: Sequence[str],
    featuriser: TrajectoryFeaturiser,
    base_mean: np.ndarray,
    sd: np.ndarray,
    direction: np.ndarray,
) -> tuple[list[float], list[str]]:
    """Project every readable rollout of one arm onto the RL-installed direction.

    One score per rollout rather than per prompt, because the six dumb baselines read a transcript
    and the comparison has to be on the items they can see. The direction is fitted on a disjoint
    half of the prompts by the caller, so this half is scored by a direction that never saw it.
    """
    scores: list[float] = []
    texts: list[str] = []
    for prompt in prompts:
        for trajectory in arm.rollouts.get(prompt, ()):
            values = featuriser.featurise(trajectory)
            if values is None:
                continue
            vector = np.asarray([values[n] for n in featuriser.names], dtype=np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                z = (vector - base_mean) / sd
            z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
            scores.append(float(np.dot(z, direction)))
            texts.append(assistant_text(trajectory))
    return scores, texts


def distillation_survival(
    base: Arm,
    expert: Arm,
    student: Arm,
    *,
    blanks: Sequence[Arm] = (),
    featuriser: TrajectoryFeaturiser | None = None,
    hack_features: Sequence[str] = (),
    markers: Sequence[str] = (),
    ci: float = 0.95,
    seed: int = 0,
    n_bootstrap: int = 2_000,
) -> DistillationSurvival | Refusal:
    """The whole K1 measurement. Evidence-shaped payload, or a Refusal naming what is missing.

    Refuses, in this order, because that is the order in which the remedies get more expensive:
    a prompt set the arms do not share; fewer than five shared prompts; a base arm too small to
    give a spread; every feature constant on the base arm; and finally every feature's installed
    shift below the detection limit, which is the refusal that says the RL run did not install
    enough measurable behaviour for the question to have an answer at this n.
    """
    # `type: ignore` with the cause named rather than hidden. `TrajectoryFeaturiser` declares
    # `names: tuple[str, ...]` as a settable variable and both shipped implementations are frozen
    # dataclasses, so mypy holds that neither satisfies the protocol it was written for and
    # `matrix_of(trajectories, SurfaceFeatures())` is a type error against the library's own
    # documented call. It has gone unnoticed because CI's mypy scope is `core` and `studies` only.
    # The fix is one line in `measure/ledger/features.py`, making `names` a read-only property on
    # the protocol, and it is a request in this package's report rather than an edit it makes.
    bank: TrajectoryFeaturiser = featuriser if featuriser is not None else SurfaceFeatures()  # type: ignore[assignment]
    prompts = shared_prompts(base, expert, student, *blanks)
    if not prompts:
        return refuse_incomplete(
            "K1.DistillationGap",
            field="a prompt set every arm answered",
            subject=(
                f"base has {len(base.prompts)} prompts, expert {len(expert.prompts)}, student "
                f"{len(student.prompts)}, and their intersection is empty"
            ),
            remedy=(
                "draw all three arms on the same prompt file, in the same order, and key the "
                "rollouts by the prompt id rather than by position. A survival fraction taken over "
                "prompt sets that differ between arms contains a difference in the task "
                "distribution that nothing downstream can separate from a difference in the policy."
            ),
            n_base=len(base.prompts),
            n_expert=len(expert.prompts),
            n_student=len(student.prompts),
        )
    if len(prompts) < MIN_PROMPTS:
        return refuse_incomplete(
            "K1.DistillationGap",
            field=f"at least {MIN_PROMPTS} shared prompts",
            subject=f"{len(prompts)} shared prompts",
            remedy=(
                f"draw more prompts. The interval here is a bootstrap over whole prompts and "
                f"resolving a 2.5% tail needs at least 2/(1-ci) distinct resamples, which puts the "
                f"floor at {MIN_PROMPTS} clusters; below it the interval is a rounding rather than "
                f"an interval, and this returns nothing rather than a point estimate that would be "
                f"read as one."
            ),
            n_prompts=len(prompts),
        )

    sd, names, n_base_used, n_base_dropped = pooled_spread(base, bank)
    if n_base_used < 2:
        return refuse_incomplete(
            "K1.DistillationGap",
            field="at least two readable rollouts in the base arm",
            subject=f"{n_base_used} readable of {base.n_rollouts} in arm {base.name!r}",
            remedy=(
                "the base arm sets the scale every shift is expressed in, and a spread needs two "
                "observations. Draw at least two completions per prompt from the base checkpoint, "
                "and check the converter wrote assistant turn text: a rollout with no text has no "
                "measured features and is dropped rather than zeroed."
            ),
            n_readable=int(n_base_used),
            n_rollouts=int(base.n_rollouts),
        )
    ok, constant = usable_columns(sd, names)
    if not ok.any():
        return Refusal(
            instrument="K1.DistillationGap",
            reason=RefusalReason.RECORD_INCOMPLETE,
            detail=(
                f"every one of the {len(names)} features is constant across the base arm's "
                f"{n_base_used} rollouts, so there is no scale to express a shift in"
            ),
            remedy=(
                "widen the feature basis or the prompt set. A constant feature is one the RL run "
                "could not have moved, so its survival is not undefined by accident: there is "
                "nothing there to survive. `RecordedFeatures` over the converter's own feature map "
                "is the first thing to try if the domain has better features than five surface ones."
            ),
            statistics={"n_features": len(names), "n_rollouts": int(n_base_used)},
        )

    base_s = arm_means(base, bank, prompts)
    expert_s = arm_means(expert, bank, prompts)
    student_s = arm_means(student, bank, prompts)
    thin = {
        arm.name: int((s.counts < 2).sum())
        for arm, s in ((base, base_s), (expert, expert_s), (student, student_s))
    }
    if any(thin.values()):
        return refuse_incomplete(
            "K1.DistillationGap",
            field="at least two completions per prompt in every arm",
            subject=", ".join(f"{name}: {n} single-completion prompts" for name, n in thin.items()),
            remedy=(
                "draw at least two completions per prompt from every checkpoint. The survival "
                "slope has to be corrected for sampling error in its own regressor, the correction "
                "needs a within-prompt variance, and one completion gives none. Without the "
                "correction the reported survival depends on how many completions were drawn, so "
                "two labs auditing the same pair of checkpoints at different K would publish "
                "different numbers for it."
            ),
            **{f"n_thin_{name}": n for name, n in thin.items()},
        )
    base_mu, expert_mu, student_mu = base_s.mean, expert_s.mean, student_s.mean
    keep_rows = np.all(np.isfinite(base_mu[:, ok]), axis=1)
    keep_rows &= np.all(np.isfinite(expert_mu[:, ok]), axis=1)
    keep_rows &= np.all(np.isfinite(student_mu[:, ok]), axis=1)
    kept_prompts = [p for p, k in zip(prompts, keep_rows) if k]
    if len(kept_prompts) < MIN_PROMPTS:
        return refuse_incomplete(
            "K1.DistillationGap",
            field=f"at least {MIN_PROMPTS} prompts readable in every arm",
            subject=(
                f"{len(kept_prompts)} of {len(prompts)} shared prompts had a readable rollout in "
                f"all three arms"
            ),
            remedy=(
                "raise the completions per prompt, or check which arm is abstaining: a prompt one "
                "arm answered with an empty completion is dropped from all three, because keeping "
                "it in two would compare a mean over K rollouts against a mean over none."
            ),
            n_kept=len(kept_prompts),
            n_shared=len(prompts),
        )

    x = shift_matrix(expert_mu[keep_rows][:, ok], base_mu[keep_rows][:, ok], sd[ok])
    y = shift_matrix(student_mu[keep_rows][:, ok], base_mu[keep_rows][:, ok], sd[ok])
    scale2 = (sd[ok] ** 2)[None, :]
    design = ShiftDesign(
        x=x,
        y=y,
        var_base=base_s.var_of_mean[keep_rows][:, ok] / scale2,
        var_expert=expert_s.var_of_mean[keep_rows][:, ok] / scale2,
    )
    kept_names = tuple(n for n, k in zip(names, ok) if k)

    floor = detection_floor(
        [
            shift_matrix(
                arm_means(b, bank, kept_prompts).mean[:, ok],
                base_mu[keep_rows][:, ok],
                sd[ok],
            )
            for b in blanks
        ],
        method=(
            f"{len(blanks)} re-draw(s) of the base checkpoint at a different sampling seed, same "
            f"prompts and same decoding; one replicate per feature per re-draw"
        ),
        n_features=int(ok.sum()),
    )
    limits = floor.limits() if floor is not None else None
    installed = np.nanmean(x, axis=0)
    # `unmeasured` is a fourth state beside the three defined ones, and it is the honest one when no blank arm
    # was supplied: nobody looked, which is not the same as looked and found nothing. It cannot make
    # a feature drop out, so a caller who skips the blank gets every feature fitted and a reading
    # that says its verdicts are unmeasured, rather than a floor invented on their behalf.
    verdicts: dict[str, str] = (
        {n: limits.verdict(float(v)) for n, v in zip(kept_names, installed)}
        if limits is not None and limits.is_determinate
        else {n: "unmeasured" for n in kept_names}
    )
    quantifiable = np.asarray(
        [verdicts[n] != "below_lod" for n in kept_names],
        dtype=bool,
    )
    if not quantifiable.any() and limits is not None:
        return Refusal(
            instrument="K1.DistillationGap",
            reason=RefusalReason.BELOW_LOD,
            detail=(
                f"every one of the {len(kept_names)} features has an RL-installed shift below the "
                f"limit of detection ({limits.lod:.4g} in base-spread units, from a blank of "
                f"{limits.blank_n} replicates at sigma {limits.sigma_blank:.4g}). There is no "
                f"denominator, so there is no survival fraction"
            ),
            remedy=(
                "train the expert further, or widen the feature basis to include an axis the "
                "reward actually moved. This is not a failed measurement: it says the RL run left "
                "no behavioural trace this basis can see, which is the first thing to establish "
                "before asking whether distillation removed one."
            ),
            statistics={
                "lod": float(limits.lod),
                "sigma_blank": float(limits.sigma_blank),
                "blank_n": int(limits.blank_n or 0),
                "n_features": len(kept_names),
            },
        )

    excluded = [n for n, keep in zip(kept_names, quantifiable) if not keep]
    fit_names = tuple(n for n, keep in zip(kept_names, quantifiable) if keep)
    fitted = design.select(quantifiable)
    fit = survival_fit(fitted, names=fit_names, ci=ci, seed=seed, n_bootstrap=n_bootstrap)
    if not np.isfinite(fit.survival):
        return Refusal(
            instrument="K1.DistillationGap",
            reason=RefusalReason.BELOW_LOD,
            detail=(
                f"the expert's installed shift over {len(fit_names)} features does not exceed its "
                f"own sampling error: the reliability of the regressor is {fit.reliability:.4g}, so "
                f"the errors-in-variables denominator is not positive and there is no survival "
                f"fraction. The uncorrected slope is {fit.raw_survival:.4g} and reporting it would "
                f"be reporting the noise ratio"
            ),
            remedy=(
                "raise the completions per prompt, which shrinks the sampling variance of every "
                "per-prompt mean as 1/K, or train the expert further so there is more installed "
                "behaviour to divide by. The reliability is the fraction of the regressor's spread "
                "that is signal; at 0.5 you need roughly twice the completions to double it."
            ),
            statistics={
                "reliability": float(fit.reliability),
                "raw_survival": float(fit.raw_survival),
                "n_features": len(fit_names),
                "n_prompts": fit.n_prompts,
            },
        )
    per_feature = per_feature_survival(fitted)

    payload = DistillationSurvival(
        delta_pp=fit.delta_pp,
        survival_pp=fit.survival_pp,
        survival_ci_low_pp=100.0 * fit.ci_low,
        survival_ci_high_pp=100.0 * fit.ci_high,
        ci_level=ci,
        r_squared=fit.r_squared,
        raw_survival_pp=100.0 * fit.raw_survival,
        reliability=fit.reliability,
        n_prompts=fit.n_prompts,
        n_features_fitted=fit.n_features,
        feature_names=list(fit_names),
        per_feature_survival_pp={
            n: float(100.0 * v) for n, v in zip(fit_names, per_feature) if np.isfinite(v)
        },
        installed_shift={n: float(v) for n, v in zip(kept_names, installed)},
        verdicts=dict(verdicts),
        excluded_features=excluded,
        constant_features=list(constant),
        sigma_blank=float(limits.sigma_blank) if limits is not None else float("nan"),
        blank_n=int(limits.blank_n or 0) if limits is not None else 0,
        blank_mean=float(floor.mean) if floor is not None else float("nan"),
        lod=float(limits.lod) if limits is not None else float("nan"),
        loq=float(limits.loq) if limits is not None else float("nan"),
        n_rollouts={
            base.name: base.n_rollouts,
            expert.name: expert.n_rollouts,
            student.name: student.n_rollouts,
        },
        n_unreadable={
            base.name: int(base_s.n_dropped),
            expert.name: int(expert_s.n_dropped),
            student.name: int(student_s.n_dropped),
        },
        featuriser=type(bank).__name__,
    )

    _attach_contrasts(payload, fitted, fit_names, hack_features, ci, seed, n_bootstrap)
    _attach_increment(
        payload,
        base=base,
        expert=expert,
        student=student,
        prompts=kept_prompts,
        featuriser=bank,
        base_mean=np.nanmean(base_mu[keep_rows], axis=0),
        sd=sd,
        shift=x,
        columns=ok,
        markers=markers,
        seed=seed,
    )
    payload.says = _says(payload)
    return payload


def _attach_contrasts(
    payload: DistillationSurvival,
    design: ShiftDesign,
    names: Sequence[str],
    hack_features: Sequence[str],
    ci: float,
    seed: int,
    n_bootstrap: int,
) -> None:
    """The two contrasts, each computed only when its feature partition is non-degenerate."""
    from studies.w6_distillation.regions import region_of

    wanted = set(hack_features)
    mask = np.asarray([n in wanted for n in names], dtype=bool)
    if mask.any() and not mask.all():
        contrast = survival_contrast(design, mask, ci=ci, seed=seed, n_bootstrap=n_bootstrap)
        payload.contrast_pp = contrast.contrast_pp
        payload.contrast_ci_low_pp = contrast.ci_low_pp
        payload.contrast_ci_high_pp = contrast.ci_high_pp
        payload.contrast_se_pp = contrast.se_pp
        payload.hack_survival_pp = 100.0 * contrast.survival_a
        payload.capability_survival_pp = 100.0 * contrast.survival_b
        payload.hack_features = [n for n, m in zip(names, mask) if m]

    entry = np.asarray([region_of(n) == "entry" for n in names], dtype=bool)
    body = np.asarray([region_of(n) == "body" for n in names], dtype=bool)
    if entry.any() and body.any():
        # Restricted to the two regions, so `n_turns` and any unregioned feature sits out rather
        # than being counted as body by default.
        keep = entry | body
        region = survival_contrast(
            design.select(keep),
            entry[keep],
            label_a="entry",
            label_b="body",
            ci=ci,
            seed=seed,
            n_bootstrap=n_bootstrap,
        )
        payload.region_contrast_pp = region.contrast_pp
        payload.region_ci_low_pp = region.ci_low_pp
        payload.region_ci_high_pp = region.ci_high_pp


def _attach_increment(
    payload: DistillationSurvival,
    *,
    base: Arm,
    expert: Arm,
    student: Arm,
    prompts: Sequence[str],
    featuriser: TrajectoryFeaturiser,
    base_mean: np.ndarray,
    sd: np.ndarray,
    shift: np.ndarray,
    columns: np.ndarray,
    markers: Sequence[str],
    seed: int,
) -> None:
    """Score expert against student rollouts, and record what that beat.

    The direction is the mean RL-installed shift, fitted on a random half of the prompts and used to
    score the other half. Fitting it on all of them would be an in-sample direction on one of the
    two label classes, and the optimism that buys is exactly the optimism M9 exists to strip out of
    a white-box claim.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(prompts))
    half = max(len(prompts) // 2, 1)
    fit_rows = np.sort(order[:half])
    score_prompts = [prompts[i] for i in np.sort(order[half:])]
    if not score_prompts:
        return
    direction = np.zeros(sd.size, dtype=np.float64)
    mean_shift = np.nan_to_num(np.nanmean(shift[fit_rows], axis=0), nan=0.0)
    direction[columns] = mean_shift
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        return
    direction = direction / norm

    expert_scores, expert_texts = _detector_scores(
        expert, score_prompts, featuriser, base_mean, sd, direction
    )
    student_scores, student_texts = _detector_scores(
        student, score_prompts, featuriser, base_mean, sd, direction
    )
    scores = np.asarray(expert_scores + student_scores, dtype=np.float64)
    labels = np.asarray([1] * len(expert_scores) + [0] * len(student_scores), dtype=int)
    payload.n_detector_items = int(scores.size)
    if np.unique(labels).size < 2 or float(np.std(scores)) == 0.0:
        return

    task = DetectionTask(
        labels=labels,
        texts=tuple(expert_texts + student_texts),
        markers=tuple(markers),
        name="K1:expert-vs-student",
    )
    bank = run_bank(task)
    payload.baselines = dict(bank.as_mapping())
    payload.baseline_refusals = {
        bid: getattr(reading, "detail", "refused")
        for bid, reading in sorted(bank.readings.items())
        if not is_scored(reading)
    }
    own = Detector.from_scores(
        "k1.installed_direction",
        scores,
        labels,
        threshold=_midpoint(scores, labels),
        note="projection onto the mean RL-installed feature shift, fitted on a disjoint half",
    )
    payload.own_detector_accuracy = own.score
    baseline_detectors = [
        Detector.from_scores(
            bid,
            np.asarray(reading.scores, dtype=np.float64),
            labels,
            threshold=_midpoint(np.asarray(reading.scores, dtype=np.float64), labels),
            note=reading.detail,
        )
        for bid, reading in sorted(bank.readings.items())
        if is_scored(reading) and float(np.std(reading.scores)) > 0.0
    ]
    if not baseline_detectors:
        return
    # `standardised_margin` rather than the default `mean_margin`, and it is a choice with a reason.
    # The K1 projection is in base-arm spread units and lands around order 1; a TF-IDF logistic
    # regression's decision values do not, and M9's own docstring says a panel whose members' score
    # scales differ by two orders of magnitude is effectively a single-member ensemble under
    # `mean_margin`. Both are computed, because the two disagreeing is a real sensitivity of the
    # increment rather than a detail, and a reading that showed only one would hide it.
    increment = IncrementalValidityReading(
        own=own,
        baselines_run=baseline_detectors,
        combiner=standardised_margin,
        seed=seed,
    ).compute()
    if isinstance(increment, Refusal):
        payload.baseline_refusals["k1.increment"] = increment.detail
        return
    payload.increment = increment.increment
    payload.increment_ci_low = increment.ci_low
    payload.increment_ci_high = increment.ci_high
    payload.best_baseline_id = increment.best_baseline_id
    payload.best_baseline_accuracy = increment.best_baseline_score
    payload.error_correlation = increment.error_correlation
    best = next((d for d in baseline_detectors if d.id == increment.best_baseline_id), None)
    if best is not None:
        a = own.correct.astype(np.float64)
        b = best.correct.astype(np.float64)
        if float(np.std(a)) > 0.0 and float(np.std(b)) > 0.0:
            payload.paired_rho = float(np.corrcoef(a, b)[0, 1])
    unstandardised = IncrementalValidityReading(
        own=own, baselines_run=baseline_detectors, combiner=mean_margin, seed=seed
    ).compute()
    if not isinstance(unstandardised, Refusal):
        payload.increment_mean_margin = unstandardised.increment


def _says(payload: DistillationSurvival) -> str:
    """The sentence, built from the numbers rather than beside them."""
    head = (
        f"Of the behaviour RL installed on {payload.n_features_fitted} features over "
        f"{payload.n_prompts} prompts, {payload.survival_pp:.0f}% survives the distillation step "
        f"[{payload.survival_ci_low_pp:.0f}, {payload.survival_ci_high_pp:.0f}] at "
        f"{payload.ci_level:.0%}."
    )
    if payload.hack_survival_pp is not None:
        head += (
            f" The reward-hacking-relevant features survive at "
            f"{payload.hack_survival_pp:.0f}% against {payload.capability_survival_pp:.0f}% for "
            f"the rest, a contrast of {payload.contrast_pp:+.0f} pp "
            f"[{payload.contrast_ci_low_pp:+.0f}, {payload.contrast_ci_high_pp:+.0f}]."
        )
    if payload.region_contrast_pp is not None:
        head += (
            f" Survival at turn entry runs {payload.region_contrast_pp:+.0f} pp against the body "
            f"of the response."
        )
    if payload.excluded_features:
        head += (
            f" {len(payload.excluded_features)} feature(s) were excluded with an installed shift "
            f"below the detection limit: {', '.join(payload.excluded_features)}."
        )
    return head


__all__ = ["DistillationSurvival", "distillation_survival"]

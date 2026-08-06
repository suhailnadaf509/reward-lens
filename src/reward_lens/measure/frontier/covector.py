"""F5, the selection covector and its spectrum, and the variance spike that decided its rung.

The objective-derived readouts order by cost and availability, and the score-function
form `s_l = E[(r - b) grad_{h_l} log pi]` is the general tier: the derivative is taken through the
policy rather than through `r`, so the grader can be a compiler, a test harness, a tournament or a
wall-clock timer and nothing needs to be differentiable except the policy. Its variance is also the
largest technical risk in the whole approach, and nothing may be built on it until a
variance-versus-`K` curve has been measured.

**The curve was measured and it came back no-go.** With every standard reduction technique
applied (group-mean baselines, antithetic sampling, control variates), the covector's relative
standard error at `K = 64` is **1.075**, and the mean cosine between two independent estimates of the
same direction is **0.145**. The two registered thresholds were `rse < 1.0` and `cosine > 0.5`, fixed
before the first number was produced. Both fail. The full curve, the ablation ladder and the reason
are in `P8_RESOLUTION` below. So **F5 ships at rung 3, the differentiable
surrogate `dr/dh_l`**, which on the same subject at the same `K` has a relative standard error of
0.021 and a split-half direction cosine of 0.9998: fifty-one times less noise and a direction that
reproduces.

The score-function rungs are implemented rather than deleted, because the spike measured a *floor*
and not an impossibility: at `K = 64` over 8 prompts the estimator pools 512 rollouts, and the
`1/sqrt(n)` scaling the curve follows puts the point where its own dispersion falls to the size of
the covector at **592** pooled rollouts on the all-reduction arm and **616** on the arm without
antithetic pairing. `POOLED_N_FLOOR` is 600, which covers the better of the two. That is a number,
so rungs 0 to 2 run above it and refuse below it with the number in the remedy. What the spike
rules out is the rung-0 reading at the group sizes reinforcement learning actually uses.

**What this instrument cannot do**, three lines in rather than in a caveats page. It reports a
direction in one model's residual-stream basis, so two models' covectors are not comparable without
a shared frame and gate 2 enforces that. It is a first-order object at the current parameters and
says what an infinitesimal constant offset at layer `l` does to the expected reward, not what
removing a feature does. And on a differentiable surrogate it answers a different question from the
score-function form: `dr/dh_l` is the direction that raises the *reward's* value at fixed text, while
`s_l` is the direction that raises the expected reward by moving the *policy*, and they coincide only
when the grader is a linear head on the model being studied.

**Four caveats travel with every dimensionality number here**, and they are not
decoration. The participation ratio is linear, so it undercounts curvature and a spectrum spread
across eight directions may still be one curved manifold. It depends on conditioning, so a value
computed across prompts is not the value within a task. It is preprocessing-sensitive: on this
subject sum-pooling and mean-pooling over positions gave directions at cosine 0.9997 and relative
standard errors of 1.097 and 1.125, so the convention did not move the direction and did move the
noise. And with `n` near `d` the sample spectrum is Marchenko-Pastur distorted, which is the caveat
that bit hardest here and is measured rather than cited: at `K = 4` the stable rank of the second
moment reads 2.69 of 8 and the same matrix estimated from the whole pool reads **6.05**. A reading
taken at a realistic group size would have reported concentration in a spectrum that is nearly flat.

**Report stable rank and participation ratio, never numerical rank.** A matrix is full
numerical rank as soon as no singular value is exactly zero, which is generic for anything touched by
floating-point arithmetic, and it says nothing about the shape of the spectrum. The participation
ratio here is the **moment-ratio convention**, `PR = (sum lambda_i)^2 / sum lambda_i^2`. The other
convention in circulation is "the number of modes needed to explain 80% of the variance" and it gives
a different answer on the same matrix; the convention is stated on the payload as well as here,
because a dimensionality quoted without it is not a number.

**What this subject is, and what it is not.** The spike ran against the model that wrote the
200-step record: a 2.45M-parameter Qwen3 with two layers and `d_model = 8`, near-uniform at
temperature 1.0, graded on completion length. That is a real policy, a real sampler and a real
grader, and the estimator's noise scaling is a property of the estimator, so the variance-versus-`K`
curve transfers. Two things do not. A stable rank out of 8 says almost nothing about a stable rank
out of 4096, so the *shape* claim in the catalogue's headline needs a frontier-scale residual
stream. And the covector this subject supports is small in absolute terms: a constant offset of 10%
of the activation norm along it is predicted to move the expected reward by 1.7e-04 against a
rollout-to-rollout reward standard deviation of 0.288, so **checking its own first-order prediction
at two sigma would take 1.2e+07 rollouts**, and running the intervention at 1,024 rollouts per arm
returned a plus-minus gap of 0.0069 with a standard error of 0.0124. The no-go is therefore recorded
as a no-go on the estimator's noise, which is what P8 asked, and the effect-size arm is reported
alongside it as a property of this subject rather than as part of the verdict. A go on a trained
policy against a grader with real discrimination is not ruled out by this measurement, and running
it needs a frontier-scale model with `POLICY:BACKWARD` and a few thousand rollouts per layer.

**The apparatus is recorded on every reading.** `nnsight` 0.7.0 replaces `torch.Tensor.backward` at
import while copying `__module__` and `__qualname__` onto the replacement, so no name-based check
notices and every `.backward()` in that process routes through it afterwards. A gradient measured in
a patched process is a measurement whose apparatus changed without anything in the reading saying so.
`policy.base.runtime_provenance()` reports presence in `sys.modules`, which is the checkable fact,
and this instrument puts it on the payload. It was absent from the process the spike ran in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np

from reward_lens.core.envelope import EnvelopeSpec, RegimeCondition
from reward_lens.core.evidence import Uncertainty, register_payload
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.quantity import BaselineID, QuantityID
from reward_lens.core.reading import Refusal, RefusalReason, refuse_incomplete
from reward_lens.core.types import (
    Access,
    AccessMatrix,
    Capability,
    Component,
    GaugeStatus,
    Phase,
    Site,
    Substrate,
)
from reward_lens.measure.base import BaseObservable, Context
from reward_lens.measure.meta.incremental import Detector, IncrementalValidityReading
from reward_lens.measure.rate.regime import MEASURED_BY
from reward_lens.policy.base import PositionSpec, runtime_provenance
from reward_lens.runtime.backend import CaptureSpec
from reward_lens.stats.baselines import DetectionTask, auroc, is_scored, run_bank

# ---------------------------------------------------------------------------
# The spike, and what it decided
# ---------------------------------------------------------------------------

#: The registered thresholds, fixed before the spike ran. `rse` is the covector's dispersion across
#: independent group-size-`K` estimates divided by the norm of the pooled estimate, and `cosine` is
#: the mean cosine between two such independent estimates. The first asks whether the reading is
#: distinguishable from zero and the second asks whether the *direction*, which is what F5 reports,
#: reproduces. A direction that does not reproduce within sixty degrees across independent samples
#: of the same policy cannot carry "the top direction decodes as X".
P8_RSE_THRESHOLD = 1.0
P8_COSINE_THRESHOLD = 0.5

#: Every number below was measured against
#: `trl-internal-testing/tiny-Qwen3ForCausalLM` in float32 on CPU: 8 prompts, 256 rollouts per prompt
#: per pool, 12 new tokens at temperature 1.0, the fixture's own `len(text)/50` length grader, the
#: gradient of the summed completion log-probability with respect to `Site(1, "resid_post")` pooled
#: by summing over positions, 400 bootstrap resamples per `K`. 214 seconds of wall clock.
P8_RESOLUTION: dict[str, Any] = {
    "prediction": "P8",
    "statement": "the F5 score-function estimator's variance is tractable at K <= 64",
    "verdict": "no-go",
    "subject": "trl-internal-testing/tiny-Qwen3ForCausalLM, float32, Site(1, 'resid_post')",
    "grader": "len(completion_text) / 50, the 200-step fixture's own length grader",
    "n_prompts": 8,
    "n_rollouts_per_prompt": 256,
    "reference_covector_norm": 8.51976e-04,
    # relative standard error of the covector, by group size, with the reduction ladder ablated
    "rse_by_k": {
        "no_baseline": {"4": 20.55, "8": 14.57, "16": 10.33, "64": 4.99},
        "group_mean_baseline": {"4": 3.717, "8": 2.838, "16": 2.126, "64": 1.082},
        "leave_one_out_baseline": {"4": 4.956, "8": 3.244, "16": 2.268, "64": 1.099},
        "loo_plus_control_variate": {"4": 4.944, "8": 3.238, "16": 2.267, "64": 1.097},
        "antithetic_plus_all": {"4": 5.101, "8": 3.352, "16": 2.315, "64": 1.075},
        "differentiable_surrogate": {"4": 0.08037, "8": 0.06115, "16": 0.04118, "64": 0.02102},
    },
    # mean cosine between two independent estimates of the same direction, by group size
    "split_half_cosine_by_k": {
        "loo_plus_control_variate": {"4": -0.006, "8": 0.049, "16": 0.127, "64": 0.4515},
        "antithetic_plus_all": {"4": -0.004, "8": 0.036, "16": 0.030, "64": 0.1451},
        "differentiable_surrogate": {"4": 0.9967, "8": 0.9984, "16": 0.9992, "64": 0.9998},
    },
    "rse_k64": 1.0748580920231443,
    "cosine_k64": 0.1450886411715685,
    "h1_confirmed": False,
    "h2_confirmed": False,
    "go": False,
    "surrogate_rse_k64": 0.021017175194025062,
    "surrogate_cosine_k64": 0.9998039287646513,
    "noise_ratio_score_function_over_surrogate": 51.14189143404552,
    #: The pooled rollout count at which the score-function estimator's own dispersion falls to the
    #: size of the covector it is estimating, extrapolated from the measured 1/sqrt(n) scaling.
    "pooled_n_for_rse_1": 591.5237980096687,
    "reduction_that_earned_its_place": "the group-mean baseline, which cuts rse at K=64 from 4.99 to 1.08",
    "reduction_that_did_not": (
        "antithetic sampling and both control variates. From 1.099 with a leave-one-out baseline "
        "alone to 1.075 with all three, a 2% improvement, and antithetic pairing made the direction "
        "worse rather than better: split-half cosine 0.4515 without it and 0.1451 with it"
    ),
    "spectrum_full_pool": {
        "stable_rank": 6.04988977705742,
        "participation_ratio": 7.720910825250322,
        "top_share": 0.16529226760332577,
        "d_model": 8,
    },
    "stable_rank_by_k": {"4": 2.686, "8": 3.026, "16": 3.482, "64": 4.488},
    "top_eigenvector_overlap_by_k": {"4": 0.331, "8": 0.305, "16": 0.306, "64": 0.357},
    "bf16_vs_fp32_relative_gradient_difference": 0.0648803376996442,
    "apparatus": {"nnsight_imported": False, "torch": "2.13.0+cu130", "transformers": "5.14.1"},
    #: The pipeline's own correctness check, and it passes. E[g] = 0 holds exactly for the
    #: position-summed gradient, so a pipeline that computes something else fails here. Over 768
    #: fresh rollouts the mean gradient has norm 3.35e-03 against an expected sampling noise of
    #: 2.79e-03, and the largest per-coordinate z across the eight coordinates is 2.35, which is
    #: what eight draws from a standard normal do.
    "score_identity": {
        "n": 768,
        "mean_g_norm": 0.003346106900059315,
        "expected_noise_norm": 0.0027867003212195666,
        "max_abs_z": 2.345270793273961,
    },
    #: Two independent uniform directions in d = 8 have mean |cos| of 0.290, so the measured
    #: top-eigenvector overlap of 0.357 at K = 64 is barely distinguishable from chance. The
    #: spectrum's leading direction does not reproduce on this subject at any K tested.
    "chance_abs_cosine_d8": 0.2902250949539211,
    "top_eigenvector_overlap_k64": 0.3571800224176675,
    #: The reference the relative standard errors are normalised by is itself noisy, and
    #: E||s_hat||^2 = ||s||^2 + trace(Var) means its norm is inflated by 1.196. Correcting for that
    #: moves the K = 64 relative standard error from 1.075 to 1.312, so the registered metric was
    #: generous to the estimator rather than harsh.
    "reference_norm_inflation": 1.1958305848742614,
    "corrected_rse_k64": 1.3115041558770881,
    #: What the covector predicts, and whether anything could check it. A constant offset of 10% of
    #: the activation norm along the estimated covector moves the expected reward by 1.67e-04
    #: against a rollout-to-rollout reward standard deviation of 0.288, so separating it from zero
    #: at two sigma needs 1.19e+07 rollouts. Run at 25% of the activation norm over 1,024 rollouts
    #: per arm, the plus-minus gap came back 0.0069 +- 0.0124 against a predicted 0.00042: consistent
    #: with zero, and uninformative in exactly the way the arithmetic said it would be.
    "intervention": {
        "covector_norm": 0.0010302928987459708,
        "activation_norm_at_site": 1.622565746307373,
        "predicted_delta_reward_at_10pc_offset": 0.00016717179661689428,
        "reward_sd": 0.2880784488786576,
        "rollouts_for_two_sigma_at_10pc_offset": 11878333.41646714,
        "observed_plus_minus_gap": 0.0068749999999999645,
        "observed_gap_sem": 0.012422355055753363,
        "n_per_arm": 1024,
    },
}

#: The pooled rollout count below which the score-function rungs refuse. Rounded up from the
#: measured 591.5 to the nearest hundred, because quoting a floor to four significant figures
#: implies a precision the extrapolation does not have.
POOLED_N_FLOOR = 600


def p8_study(repo_dir: str | None = None, frozen_at: str | None = None) -> Any:
    """The frozen study P8 resolves under, built and hashed at call time.

    The study is written out in full rather than loaded, so an edit to a registered field changes
    the `StudyID` and a later reader can see that it changed. The prediction was registered before
    this instrument existed, and this is the record of it.

    The kill criterion is the prediction's own negation and it fired. That is the outcome the spike
    was run to produce, and a fired kill criterion here is a plan change rather than a failure.
    """
    from reward_lens.studies.freeze import freeze
    from reward_lens.studies.spec import (
        Hypothesis,
        KillCriterion,
        Prediction,
        StudySpec,
        SubjectQuery,
    )

    spec = StudySpec(
        id="w52-f5-variance-spike",
        title="Is the F5 score-function selection covector usable at K <= 64?",
        science="S08-selection-geometry",
        hypotheses=(
            Hypothesis(
                id="H1",
                statement=(
                    "with group-mean baselines, antithetic sampling and control variates applied, "
                    "the selection covector's relative standard error at K = 64 is below 1, so the "
                    "reading is distinguishable from zero"
                ),
                prediction=Prediction(
                    metric="covector_relative_standard_error_k64",
                    comparator="<",
                    threshold=P8_RSE_THRESHOLD,
                    rationale=(
                        "a relative standard error of 1 means the estimator's own dispersion is as "
                        "large as the quantity it estimates, which is the minimum bar for a reading "
                        "anything can be built on"
                    ),
                ),
                scoreboard_row="P8",
            ),
            Hypothesis(
                id="H2",
                statement=(
                    "the mean cosine between two independent K = 64 estimates of the covector "
                    "direction is above 0.5, so the direction F5 reports reproduces"
                ),
                prediction=Prediction(
                    metric="covector_split_half_cosine_k64",
                    comparator=">",
                    threshold=P8_COSINE_THRESHOLD,
                    rationale=(
                        "F5's headline is a claim about a direction, and two independent estimates "
                        "further apart than sixty degrees cannot support one"
                    ),
                ),
                scoreboard_row="P8",
            ),
        ),
        analysis="reward_lens.measure.frontier.covector.p8_resolution",
        subjects=SubjectQuery(
            signals=("trl-internal-testing/tiny-Qwen3ForCausalLM",),
            datasets=("question {i}: count upward from {i}, i in 0..7",),
            extra={
                "k_grid": [4, 8, 16, 64],
                "n_rollouts_per_prompt": 256,
                "site": "Site(1, 'resid_post')",
                "grader": "len(text)/50",
                "dtype": "float32",
            },
        ),
        kill_criteria=(
            KillCriterion(
                id="K1",
                metric="covector_relative_standard_error_k64",
                comparator=">=",
                threshold=P8_RSE_THRESHOLD,
                description=(
                    "K = 64 is still too noisy with all variance reduction applied, so F5 reduces "
                    "to the differentiable-surrogate case and the plan changes here"
                ),
            ),
        ),
        notes=(
            "Registered before this instrument existed. The thresholds were fixed before the "
            "first number was produced."
        ),
    )
    return freeze(spec, repo_dir=repo_dir, frozen_at=frozen_at)


def p8_resolution() -> Any:
    """The study's outcome, as a `StudyResult`, from the numbers in `P8_RESOLUTION`.

    Named by `StudySpec.analysis` so the frozen spec points at the function that scores it rather
    than at a document. Both hypotheses are refuted and the kill criterion fired.
    """
    from reward_lens.studies.spec import StudyResult

    metrics = {
        "covector_relative_standard_error_k64": float(P8_RESOLUTION["rse_k64"]),
        "covector_split_half_cosine_k64": float(P8_RESOLUTION["cosine_k64"]),
        "surrogate_relative_standard_error_k64": float(P8_RESOLUTION["surrogate_rse_k64"]),
        "noise_ratio_score_function_over_surrogate": float(
            P8_RESOLUTION["noise_ratio_score_function_over_surrogate"]
        ),
        "pooled_n_for_rse_1": float(P8_RESOLUTION["pooled_n_for_rse_1"]),
    }
    return StudyResult(
        outcomes={"H1": "refuted", "H2": "refuted"},
        metrics=metrics,
        killed=True,
        killed_by=["K1"],
        kill_outcomes={"K1": "fired"},
        summary=(
            f"No-go. At K = 64 with group-mean baselines, antithetic sampling and control variates "
            f"applied, the covector's relative standard error is {P8_RESOLUTION['rse_k64']:.3f} "
            f"against a registered threshold of {P8_RSE_THRESHOLD}, and the mean cosine between two "
            f"independent estimates of the direction is {P8_RESOLUTION['cosine_k64']:.3f} against "
            f"{P8_COSINE_THRESHOLD}. The differentiable surrogate on the same subject at the same K "
            f"reaches {P8_RESOLUTION['surrogate_rse_k64']:.4f} and 0.9998, so it beats the "
            f"score-function form by a factor of "
            f"{P8_RESOLUTION['noise_ratio_score_function_over_surrogate']:.0f} on noise. F5 ships at "
            f"rung 3."
        ),
    )


# ---------------------------------------------------------------------------
# The arithmetic
# ---------------------------------------------------------------------------


def stable_rank(matrix: np.ndarray) -> float:
    """`srank(A) = ||A||_F^2 / ||A||_2^2`, on a symmetric positive semi-definite matrix.

    Bounded in `[1, d]` and invariant under an orthogonal change of basis, which is why it is the
    statistic to report and numerical rank is not. Returns NaN on a zero matrix rather
    than 1, because a matrix with no energy in it has no shape to report.
    """
    ev = _eigenvalues(matrix)
    top = float(ev.max()) if ev.size else 0.0
    if top <= 0.0:
        return float("nan")
    return float(ev.sum() / top)


def participation_ratio(
    matrix: np.ndarray, convention: Literal["moment_ratio", "variance_share"] = "moment_ratio"
) -> float:
    """`PR = (sum lambda_i)^2 / sum lambda_i^2` under the moment-ratio convention.

    Two conventions are in circulation and they give different answers on the same matrix, so the
    convention is a parameter with a default rather than an assumption. ``variance_share`` returns
    the number of eigenvalues needed to reach 80% of the trace, which is what most of the
    effective-dimension literature means by the phrase; it is an integer-valued step function and it
    is not interchangeable with the moment ratio.
    """
    ev = _eigenvalues(matrix)
    total = float(ev.sum())
    if total <= 0.0:
        return float("nan")
    if convention == "variance_share":
        ordered = np.sort(ev)[::-1]
        cumulative = np.cumsum(ordered) / total
        return float(int(np.searchsorted(cumulative, 0.8)) + 1)
    denom = float((ev**2).sum())
    return float(total**2 / denom) if denom > 0 else float("nan")


def _eigenvalues(matrix: np.ndarray) -> np.ndarray:
    """Eigenvalues of the symmetrised matrix, clipped at zero.

    Clipped because a second moment is positive semi-definite in exact arithmetic and a small
    negative eigenvalue is float error, and a negative eigenvalue silently makes both `stable_rank`
    and `participation_ratio` meaningless rather than merely imprecise.
    """
    m = np.asarray(matrix, dtype=np.float64)
    sym = 0.5 * (m + m.T)
    return np.clip(np.linalg.eigvalsh(sym), 0.0, None)


def leave_one_out_baseline(rewards: np.ndarray) -> np.ndarray:
    """The group mean with the member itself held out, which is what makes the estimate unbiased.

    `b_k` computed as the plain group mean is correlated with `r_k`, and the resulting estimator
    carries an `O(1/K)` bias that is largest exactly where the estimator is used, at `K = 4`. On the
    spike's own pool the plain group mean and the leave-one-out form differ in dispersion by 11% at
    `K = 4` and by 1.5% at `K = 64`, with the leave-one-out form the noisier of the two and the
    unbiased one.
    """
    r = np.asarray(rewards, dtype=np.float64)
    k = r.shape[-1]
    if k < 2:
        return np.zeros(r.shape, dtype=np.float64)
    out: np.ndarray = (r.sum(axis=-1, keepdims=True) - r) / (k - 1)
    return out


def selection_covector(
    rewards: np.ndarray,
    gradients: np.ndarray,
    *,
    baseline: Literal["none", "group_mean", "leave_one_out"] = "leave_one_out",
) -> np.ndarray:
    """`s_l = E[(r - b) g_l]`, averaged within groups and then across them. Shapes `(P, K)`, `(P, K, d)`.

    **The pooling convention is load-bearing and the usual statement of the estimator omits it.**
    `grad_{h_l} log pi` is written as though it were a `d`-vector, and it is a `(T, d)` object until a
    convention turns it into one. Summing over positions is the only choice that makes it a score:
    it is the derivative of `log pi` with respect to a constant offset added to the layer-`l`
    residual at every position, which is a genuine shared parameter, so `E[g] = 0` holds exactly and
    `s_l` is exactly `d E[r] / d offset`. Mean-pooling divides by a per-rollout token count and has
    no such identity. The caller does the pooling; this function documents which one it assumes.
    """
    r = np.asarray(rewards, dtype=np.float64)
    g = np.asarray(gradients, dtype=np.float64)
    if baseline == "none":
        b = np.zeros_like(r)
    elif baseline == "group_mean":
        b = np.broadcast_to(r.mean(axis=-1, keepdims=True), r.shape)
    elif baseline == "leave_one_out":
        b = leave_one_out_baseline(r)
    else:
        raise ValueError(
            f"unknown baseline {baseline!r}; the three are 'none', 'group_mean' and "
            f"'leave_one_out'. The group-mean baseline is the whole of the variance reduction the "
            f"spike found to work, so 'none' is a deliberate ablation and not a default."
        )
    return ((r - b)[..., None] * g).mean(axis=-2).mean(axis=0)


def selection_second_moment(
    rewards: np.ndarray,
    gradients: np.ndarray,
    *,
    baseline: Literal["none", "group_mean", "leave_one_out"] = "leave_one_out",
) -> np.ndarray:
    """`M_l = E[(r - b)^2 g_l g_l^T]`, the rung-1 object: a dictionary ordered by pressure.

    Ordered by pressure rather than by variance, which is the whole argument for it: the
    Jacobian-lens result is that causal relevance and explained variance are close to orthogonal in a
    transformer's residual stream, so a basis selected by reconstruction error is optimising a
    functional nearly unrelated to the one that matters.
    """
    r = np.asarray(rewards, dtype=np.float64)
    g = np.asarray(gradients, dtype=np.float64)
    b = (
        leave_one_out_baseline(r)
        if baseline == "leave_one_out"
        else (
            np.broadcast_to(r.mean(axis=-1, keepdims=True), r.shape)
            if baseline == "group_mean"
            else np.zeros_like(r)
        )
    )
    w2 = ((r - b) ** 2).reshape(-1)
    x = g.reshape(-1, g.shape[-1])
    return (w2[:, None] * x).T @ x / max(x.shape[0], 1)


def activation_metric(gradients: np.ndarray) -> np.ndarray:
    """`G_l = E[g_l g_l^T]`, the available-variance metric the whitened problem divides by."""
    x = np.asarray(gradients, dtype=np.float64).reshape(-1, np.shape(gradients)[-1])
    return x.T @ x / max(x.shape[0], 1)


def whitened_spectrum(
    second_moment: np.ndarray, metric: np.ndarray, *, ridge: float = 1e-6
) -> tuple[np.ndarray, np.ndarray, float]:
    """Solve `M v = mu G v`, returning `(eigenvalues descending, eigenvectors, damping used)`.

    Rung 2: directions ordered by pressure per unit of *available* variance rather than by pressure.
    A direction the policy cannot move contributes nothing to `M` and nothing to `G`, and the ratio
    is what separates "this direction carries pressure" from "this direction carries variance".

    ``ridge`` is a fraction of `trace(G)/d` added to `G`'s diagonal and it is **returned rather than
    hidden**, because the `lambda` in `(F + lambda I)^-1` has to be reported and a
    reading that hides it cannot be checked for the stability it also has to claim. The damping is
    not cosmetic here: `G` is a sample second moment and is singular whenever the pooled rollout
    count is below `d`.
    """
    m = np.asarray(second_moment, dtype=np.float64)
    g = np.asarray(metric, dtype=np.float64)
    d = g.shape[0]
    damping = float(ridge * np.trace(g) / max(d, 1))
    from scipy.linalg import eigh

    values, vectors = eigh(0.5 * (m + m.T), 0.5 * (g + g.T) + damping * np.eye(d))
    order = np.argsort(values)[::-1]
    return values[order], vectors[:, order], damping


def required_pooled_n(observed_rse: float, observed_n: int, target_rse: float = 1.0) -> float:
    """How many pooled rollouts an estimator at `observed_rse` needs to reach `target_rse`.

    A Monte-Carlo standard error falls as `1/sqrt(n)`, so the answer is `n * (rse / target)^2`. It is
    an extrapolation from one point on a curve that was measured to follow that scaling across
    `K = 4` to `K = 64`, and it says nothing about whether the pooling is statistically legitimate:
    pooling across prompts assumes one covector for the prompt set, and pooling across steps assumes
    the policy did not move.
    """
    if not np.isfinite(observed_rse) or observed_rse <= 0 or observed_n <= 0:
        return float("nan")
    return float(observed_n * (observed_rse / target_rse) ** 2)


# ---------------------------------------------------------------------------
# The payload
# ---------------------------------------------------------------------------


@register_payload
@dataclass
class SelectionGeometry:
    """The covector, the spectrum, and the two dimensionality statistics that are defensible.

    Kept as a payload rather than a dict because `pr_convention`, `whitened_ridge` and `apparatus`
    are the three fields a caller would drop and each one is a number the reading is not
    interpretable without.
    """

    rung: int
    estimator: str
    site: str
    d_model: int
    n_items: int
    n_groups: int
    group_size: float
    covector: list[float]
    covector_norm: float
    eigenvalues: list[float]
    stable_rank: float
    participation_ratio: float
    pr_convention: str
    participation_ratio_variance_share: float
    top_share: float
    top_direction: list[float]
    whitened_eigenvalues: list[float]
    whitened_stable_rank: float
    whitened_ridge: float
    relative_standard_error: float
    split_half_cosine: float
    pooled_n_for_rse_1: float
    n_pooled: int
    apparatus: dict[str, Any] = field(default_factory=dict)
    baselines: dict[str, float] = field(default_factory=dict)
    baseline_refusals: dict[str, str] = field(default_factory=dict)
    spike: dict[str, Any] = field(default_factory=dict)
    says: str = ""


# ---------------------------------------------------------------------------
# The instruments
# ---------------------------------------------------------------------------

#: The four baselines the catalogue names for F5, in its own order. The random direction and the
#: semantic placebo come from `measure/controls/placebo.py`; the logit lens is the policy's own
#: unembedding row, which is the readout the shipped library used before any of this existed; and
#: string matching comes from the six-baseline bank.
F5_BASELINES: tuple[BaselineID, ...] = (
    "baseline.random_direction_matched_norm",  # type: ignore[assignment]
    "baseline.semantic_placebo",  # type: ignore[assignment]
    "baseline.logit_lens",  # type: ignore[assignment]
    "baseline.string_match",  # type: ignore[assignment]
)

#: `LINEAR_RESPONSE` is the catalogue's own `envelope_requires` for F5 and it is measured by F2's
#: selection-explained fraction. `ABOVE_LOD` is added here and the addition is deliberate: the whole
#: content of the spike is a detection limit on this estimator, and an instrument that measured
#: its own floor and then did not declare the condition that checks it would be reporting the floor
#: in a docstring and enforcing nothing.
COVECTOR_ENVELOPE = EnvelopeSpec(
    requires=frozenset({RegimeCondition.LINEAR_RESPONSE, RegimeCondition.ABOVE_LOD}),
    measured_by={
        RegimeCondition.LINEAR_RESPONSE: MEASURED_BY[RegimeCondition.LINEAR_RESPONSE],
        RegimeCondition.ABOVE_LOD: MEASURED_BY[RegimeCondition.ABOVE_LOD],
    },
    on_violation="refuse",
)

#: A policy is a generative network. It has no reward head, so `NEURAL_SCALAR` would be a claim
#: about a `w_r` that does not exist, and the other four substrates have no activations at all.
COVECTOR_ACCESS: AccessMatrix = {
    Component.POLICY: Access.BACKWARD,
    Component.GRADER: Access.QUERY,
}


class _SelectionInstrument(BaseObservable):
    """Shared machinery for F5's four quantities: compute once, emit under four ids.

    Four instruments rather than one because an instrument gets exactly one `quantity`,
    and a per-sequence covector ranked against a dimensionless stable rank is the unit error the
    field is for. The arithmetic runs once in `_geometry` and each subclass names the id it emits
    under.
    """

    version = "1.0"
    #: Gradients and activations both: the covector needs the first and the baseline comparison
    #: projects items onto directions, which needs the second. Either one alone makes this
    #: white-box, so the `IncrementalValidity` is mandatory.
    capabilities = Capability.GRADIENTS | Capability.ACTIVATIONS
    #: The covector and the top direction are components in one model's residual-stream basis, so a
    #: cross-model comparison needs a shared frame and gate 2 raises without one. That is a
    #: different claim from `invariance_relation` below, which is about one model under a change of
    #: coordinates: `M -> Q M Q^T` leaves every eigenvalue, the stable rank and the participation
    #: ratio exactly where they were.
    gauge_status = GaugeStatus.COVARIANT
    faithful_to = "selection covector, spectrum not rank"
    deviations = (
        "grad_{h_l} log pi is usually written as a d-vector and it is a (T, d) object until a "
        "pooling convention is chosen; this sums over positions, which is the only convention "
        "under which E[g] = 0 holds and s_l is exactly the derivative of E[r] with respect to a "
        "constant offset at layer l",
        "the catalogue's envelope for F5 names LINEAR_RESPONSE alone; ABOVE_LOD is added because "
        "the spike measured a detection limit on this estimator and a floor that is not a "
        "declared condition is not enforced anywhere",
        "the whitened generalised problem is solved with a ridge on G, reported as "
        "`whitened_ridge`, because G is a sample second moment and is singular below d pooled "
        "rollouts",
    )

    requires: AccessMatrix = COVECTOR_ACCESS
    substrates = frozenset({Substrate.NEURAL_GEN})
    phases = frozenset({Phase.PRE_RUN, Phase.IN_RUN, Phase.POST_RUN})
    envelope = COVECTOR_ENVELOPE
    invariance = "repr.basis"
    #: Asserted on the stable rank, which is the scalar `check_invariance` is handed. An orthogonal
    #: change of basis maps `M` to `Q M Q^T` and leaves the Frobenius and spectral norms alone, so
    #: the ratio does not move. Written in the mapping form rather than as a bare
    #: `Relation`, because this instrument's payload genuinely transforms two ways and only one of
    #: them is a value relation: the covector and the top direction rotate with the basis, which is
    #: what `gauge_status` above declares and gate 2 enforces, while every scalar on the payload is
    #: invariant, which is what a generated property test can actually assert. A bare `INVARIANT`
    #: says the second and leaves the first to a comment.
    invariance_relation = {"repr.basis": INVARIANT}
    baselines = F5_BASELINES
    quantity: QuantityID = ""
    rung = 3

    def __init__(
        self,
        *,
        rewards: Sequence[float] | np.ndarray,
        groups: Sequence[Any] | np.ndarray | None = None,
        texts: Sequence[str] = (),
        markers: tuple[str, ...] = (),
        site: Site | None = None,
        rung: int | None = None,
        readout: str = "decision",
        baseline: Literal["none", "group_mean", "leave_one_out"] = "leave_one_out",
        ridge: float = 1e-6,
        n_resamples: int = 2000,
        seed: int = 0,
        pooled_n_floor: int = POOLED_N_FLOOR,
    ) -> None:
        self.rewards = np.asarray(rewards, dtype=np.float64).ravel()
        self.groups = None if groups is None else np.asarray(groups).ravel()
        self.texts = tuple(texts)
        #: Handed to the string-match baseline. Absent, it mines markers out of fold, which is the
        #: right default and returns nothing usable on text a near-uniform policy produced. That
        #: outcome lands in `baseline_refusals` with the mined markers in it rather than being
        #: silently dropped, because "the dumb baseline could not run" and "the dumb baseline ran
        #: and lost" are different claims.
        self.markers = tuple(markers)
        self.site = site
        self.readout = readout
        self.baseline = baseline
        self.ridge = float(ridge)
        self.n_resamples = int(n_resamples)
        self.seed = int(seed)
        self.pooled_n_floor = int(pooled_n_floor)
        if rung is not None:
            self.rung = int(rung)

    # -- the measurement ----------------------------------------------------

    def measure(self, ctx: Context) -> Any:
        subject = ctx.signal
        if subject is None:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail=(
                    "no policy was supplied on the context, so there is nothing to differentiate "
                    "and no residual stream to report a direction in"
                ),
                remedy=(
                    "pass the policy as `Context.signal`. F5 reads a network: a program, a test "
                    "harness or a human grader has no activations and no covector at any rung, and "
                    "the verifier instruments in `measure/` are what apply to those."
                ),
            )
        items = list(ctx.view)
        r = self.rewards
        if r.size != len(items):
            return refuse_incomplete(
                self.name,
                field="one reward per item",
                subject=f"{len(items)} items and {r.size} rewards",
                remedy=(
                    "pass a reward array aligned item-for-item with the view. A covector weighted "
                    "by misaligned rewards is a direction in the residual stream that means "
                    "nothing, and it will still have a norm and a spectrum."
                ),
                n_items=len(items),
                n_rewards=int(r.size),
            )

        groups = self.groups if self.groups is not None else np.zeros(len(items), dtype=int)
        order, sizes = _group_index(groups)
        if min(sizes) < 2 and self.rung < 3:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.RECORD_INCOMPLETE,
                detail=(
                    f"the smallest group holds {min(sizes)} rollouts, and a group-relative baseline "
                    f"needs at least two. Without a baseline the estimator's relative standard "
                    f"error at K = 64 was {P8_RESOLUTION['rse_by_k']['no_baseline']['64']:.2f} on "
                    f"the spike, against {P8_RESOLUTION['rse_by_k']['group_mean_baseline']['64']:.2f} "
                    f"with one."
                ),
                remedy=(
                    "supply `groups` naming which rollouts share a prompt, so the group mean can be "
                    "the baseline. The group-mean baseline is the whole of the variance reduction "
                    "that worked; running without it is an ablation, not a fallback."
                ),
                statistics={"min_group_size": int(min(sizes)), "n_groups": len(sizes)},
            )

        site = self.site
        if site is None:
            n_layers = getattr(subject.meta, "n_layers", None)
            if n_layers is None:
                return refuse_incomplete(
                    self.name,
                    field="a site to read, or a subject that reports its layer count",
                    subject=f"{type(subject).__name__} whose meta carries n_layers=None",
                    remedy=(
                        "pass `site=Site(layer, 'resid_post')` naming the layer you mean. There is "
                        "no defensible default here: the layer is part of the claim, and picking "
                        "the last one silently would put a layer index nobody chose into a "
                        "published direction."
                    ),
                )
            site = Site(int(n_layers) - 1, "resid_post")

        # -- the gradients, per rung ----------------------------------------
        if self.rung >= 3:
            grads = _surrogate_gradients(subject, items, site, self.readout)
            if isinstance(grads, Refusal):
                return grads
            estimator = f"differentiable surrogate dr/dh at {site}, readout {self.readout!r}"
            weights = np.ones_like(r)
            observed_rse = float(P8_RESOLUTION["surrogate_rse_k64"])
            observed_cos = float(P8_RESOLUTION["surrogate_cosine_k64"])
        else:
            n_pooled = len(items)
            if n_pooled < self.pooled_n_floor:
                return Refusal(
                    instrument=self.name,
                    reason=RefusalReason.BELOW_LOD,
                    detail=(
                        f"the score-function covector is being asked for on {n_pooled} pooled "
                        f"rollouts. The variance spike measured its relative standard error at "
                        f"K = 64 as {P8_RESOLUTION['rse_k64']:.3f} over "
                        f"{P8_RESOLUTION['n_prompts'] * 64} pooled rollouts with every standard "
                        f"reduction technique applied, and the mean cosine between "
                        f"two independent estimates of the direction as "
                        f"{P8_RESOLUTION['cosine_k64']:.3f}. Below "
                        f"{self.pooled_n_floor} pooled rollouts the estimator's own dispersion is "
                        f"larger than the covector it is estimating, so the reading would be noise "
                        f"with a norm."
                    ),
                    remedy=(
                        f"pool at least {self.pooled_n_floor} rollouts, which means widening the "
                        f"prompt set or the step window and stating that you did, because pooling "
                        f"across prompts assumes one covector for the prompt set and pooling across "
                        f"steps assumes the policy did not move. Or ask for rung 3, the "
                        f"differentiable surrogate, which reached a relative standard error of "
                        f"{P8_RESOLUTION['surrogate_rse_k64']:.4f} on the same subject at the same "
                        f"K and needs a reward that is differentiable in the policy's activations."
                    ),
                    statistics={
                        "n_pooled": n_pooled,
                        "floor": self.pooled_n_floor,
                        "measured_rse_at_512_pooled": float(P8_RESOLUTION["rse_k64"]),
                    },
                )
            grads = _score_function_gradients(subject, items, site)
            if isinstance(grads, Refusal):
                return grads
            estimator = f"score function (r - b) grad log pi at {site}, positions summed"
            weights = r
            observed_rse = float(P8_RESOLUTION["rse_k64"])
            observed_cos = float(P8_RESOLUTION["cosine_k64"])

        # -- the covector, the spectrum and the whitened problem -------------
        r_grouped = _regroup(r, order, sizes)
        g_grouped = _regroup(grads, order, sizes)
        if self.rung >= 3:
            # The surrogate rung estimates E[dr/dh], an unweighted mean, so the baseline machinery
            # is not applied to it. Weighting it by (r - b) would be a third object with neither
            # rung's meaning.
            covector = np.asarray(grads, dtype=np.float64).mean(axis=0)
            moment = activation_metric(grads) * float(np.var(r))
        else:
            covector = selection_covector(r_grouped, g_grouped, baseline=self.baseline)
            moment = selection_second_moment(r_grouped, g_grouped, baseline=self.baseline)
        metric = activation_metric(grads)
        eigenvalues = np.sort(_eigenvalues(moment))[::-1]
        w_values, w_vectors, damping = whitened_spectrum(moment, metric, ridge=self.ridge)
        top_direction = _top_direction(moment)

        # -- the four baselines, and the incremental-validity record ---------
        increment = self._incremental(ctx, subject, items, r, top_direction, site)
        if isinstance(increment, Refusal):
            return increment
        record, bank_scores, bank_refusals = increment

        total = float(eigenvalues.sum())
        payload = SelectionGeometry(
            rung=self.rung,
            estimator=estimator,
            site=str(site),
            d_model=int(np.shape(grads)[-1]),
            n_items=len(items),
            n_groups=len(sizes),
            group_size=float(np.mean(sizes)),
            covector=[float(x) for x in covector],
            covector_norm=float(np.linalg.norm(covector)),
            eigenvalues=[float(x) for x in eigenvalues],
            stable_rank=stable_rank(moment),
            participation_ratio=participation_ratio(moment),
            pr_convention="moment_ratio: (sum lambda)^2 / sum lambda^2",
            participation_ratio_variance_share=participation_ratio(
                moment, convention="variance_share"
            ),
            top_share=float(eigenvalues[0] / total) if total > 0 else float("nan"),
            top_direction=[float(x) for x in top_direction],
            whitened_eigenvalues=[float(x) for x in w_values],
            whitened_stable_rank=_stable_rank_from_values(w_values),
            whitened_ridge=damping,
            relative_standard_error=observed_rse,
            split_half_cosine=observed_cos,
            pooled_n_for_rse_1=required_pooled_n(observed_rse, len(items)),
            n_pooled=len(items),
            apparatus=runtime_provenance(),
            baselines=bank_scores,
            baseline_refusals=bank_refusals,
            spike={
                "verdict": P8_RESOLUTION["verdict"],
                "rse_k64": P8_RESOLUTION["rse_k64"],
                "cosine_k64": P8_RESOLUTION["cosine_k64"],
                "surrogate_rse_k64": P8_RESOLUTION["surrogate_rse_k64"],
            },
            says=_says(self.rung, moment, covector, len(items), site),
        )
        del weights
        return ctx.emit(
            payload,
            uncertainty=Uncertainty(
                ci_low=float("nan"),
                ci_high=float("nan"),
                ci_level=0.0,
                n=len(items),
                method=(
                    f"the relative standard error and the split-half cosine on this payload are "
                    f"the spike's measured values for this estimator at K = 64 on "
                    f"{P8_RESOLUTION['subject']}, not a per-reading interval. A per-reading "
                    f"interval needs the bootstrap this instrument does not run inline."
                ),
            ),
            baselines=bank_scores,
            incremental=record,
        )

    # -- baselines ----------------------------------------------------------

    def _incremental(
        self,
        ctx: Context,
        subject: Any,
        items: list[Any],
        rewards: np.ndarray,
        direction: np.ndarray,
        site: Site,
    ) -> Any:
        """Project every item onto the claimed direction and onto the four controls, then run M9.

        The task is "does this direction recover which rollouts the grader preferred", with the
        label taken within groups so the comparison is group-relative in the same sense the covector
        is. A direction that cannot do that is not a direction the pressure is along.

        **Matched norm is a formality for this comparison and a requirement for the next one.**
        Scaling a direction does not move a projection's ranking, so the matched-norm random control
        and a unit-norm random control are the same test here. The norm matters the moment the claim
        becomes an intervention, where a placebo weaker in norm is a smaller dose rather than a
        control, and that is why the control is built matched.
        """
        from reward_lens.measure.controls.placebo import (
            random_gaussian_direction,
            semantic_placebo,
        )

        labels = _within_group_labels(rewards, self.groups)
        if np.unique(labels).size < 2:
            return refuse_incomplete(
                self.name,
                field="rewards that vary within at least one group",
                subject=f"{len(items)} items, every one on the same side of its group mean",
                remedy=(
                    "widen the item set until at least one group contains both a better and a "
                    "worse rollout. A covector estimated where the grader never discriminated is "
                    "exactly zero in expectation, and the sample version is not zero, it is noise."
                ),
                n_items=len(items),
            )

        features = _activations(subject, items, site)
        if isinstance(features, Refusal):
            return features
        d = features.shape[1]

        def encode(texts: Sequence[str]) -> np.ndarray:
            return np.asarray(
                _activations(subject, [(t, "") for t in texts], site), dtype=np.float64
            )

        own_scores = features @ np.asarray(direction, dtype=np.float64)
        controls: dict[str, np.ndarray] = {
            "baseline.random_direction_matched_norm": random_gaussian_direction(
                d, match_to=direction, seed=self.seed
            ),
            "baseline.logit_lens": _logit_lens_direction(subject, self.readout, d),
        }
        try:
            placebo = semantic_placebo(encode, match_to=direction, seed=self.seed)
            controls["baseline.semantic_placebo"] = np.asarray(placebo.vector, dtype=np.float64)
        except Exception as exc:  # the encoder is the subject's forward; report, do not crash
            controls["baseline.semantic_placebo"] = np.zeros(d)
            placebo_error = str(exc)
        else:
            placebo_error = ""

        detectors = []
        scores: dict[str, float] = {}
        refusals: dict[str, str] = {}
        for bid, vector in sorted(controls.items()):
            if vector is None or float(np.linalg.norm(vector)) == 0.0:
                refusals[bid] = (
                    placebo_error or "the control direction is zero, so it ranks nothing"
                )
                continue
            s = features @ vector
            if float(np.std(s)) == 0.0:
                refusals[bid] = "every item projects to the same value on this direction"
                continue
            scores[bid] = float(auroc(s, labels))
            detectors.append(
                Detector.from_scores(bid, s, labels, threshold=_midpoint(s, labels), note=bid)
            )

        task = DetectionTask(
            labels=labels,
            texts=self.texts,
            markers=self.markers,
            name=f"{self.name}:{site}",
        )
        bank = run_bank(task, baselines=["baseline.string_match"])  # type: ignore[list-item]
        for bid, reading in sorted(bank.readings.items()):
            if is_scored(reading) and float(np.std(reading.scores)) > 0.0:
                scores[bid] = float(reading.auroc)
                detectors.append(
                    Detector.from_scores(
                        bid,
                        reading.scores,
                        labels,
                        threshold=_midpoint(np.asarray(reading.scores, dtype=np.float64), labels),
                        note=reading.detail,
                    )
                )
            else:
                refusals[bid] = getattr(reading, "detail", "refused")

        if not detectors:
            return refuse_incomplete(
                self.name,
                field="at least one control that could run",
                subject="; ".join(f"{k}: {v}" for k, v in sorted(refusals.items()))
                or "no control produced a usable score",
                remedy=(
                    "supply `texts` so the string-match baseline can run, and check that the "
                    "subject's forward returns activations at this site so the placebo encoder can "
                    "build a direction. The coherent irrelevant direction is mandatory rather "
                    "than nice, because a vampires-versus-werewolves direction "
                    "suppressed deployment-time hacking exactly as well as the direction that was "
                    "supposed to be the reward-hacking one."
                ),
                n_controls_refused=len(refusals),
            )

        own = Detector.from_scores(
            "selection.top_direction",
            own_scores,
            labels,
            threshold=_midpoint(own_scores, labels),
            note=f"projection onto the top eigenvector of M at {site}",
        )
        scores["selection.top_direction"] = float(auroc(own_scores, labels))
        out = IncrementalValidityReading(
            own=own, baselines_run=detectors, n_resamples=self.n_resamples, seed=self.seed
        ).compute()
        if isinstance(out, Refusal):
            return out
        return out.record, scores, refusals


class SelectionCovector(_SelectionInstrument):
    """`selection.covector`: the direction selection pressure acts along at one layer."""

    name = "SelectionCovector"
    quantity = "selection.covector"


class SelectionSpectrum(_SelectionInstrument):
    """`selection.spectrum`: the eigenvalues of `M_l`, and the whitened problem's beside them."""

    name = "SelectionSpectrum"
    quantity = "selection.spectrum"


class SelectionStableRank(_SelectionInstrument):
    """`selection.stable_rank`: `||M||_F^2 / ||M||_2^2`, and never the numerical rank."""

    name = "SelectionStableRank"
    quantity = "selection.stable_rank"


class SelectionDimensionality(_SelectionInstrument):
    """`selection.dimensionality`: the participation ratio, under the stated convention."""

    name = "SelectionDimensionality"
    quantity = "selection.dimensionality"


#: The four, in the order a reader wants them: the direction, then its spectrum, then the two
#: statistics that say how spread the spectrum is.
F5: tuple[type, ...] = (
    SelectionCovector,
    SelectionSpectrum,
    SelectionStableRank,
    SelectionDimensionality,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _group_index(groups: np.ndarray) -> tuple[list[np.ndarray], list[int]]:
    """Row indices per group and the group sizes, in first-appearance order."""
    order: list[np.ndarray] = []
    sizes: list[int] = []
    seen: list[Any] = []
    for g in groups:
        if g not in seen:
            seen.append(g)
    for g in seen:
        idx = np.flatnonzero(groups == g)
        order.append(idx)
        sizes.append(int(idx.size))
    return order, sizes


def _regroup(values: np.ndarray, order: list[np.ndarray], sizes: list[int]) -> np.ndarray:
    """Stack per-group rows into `(P, K, ...)`, truncating to the smallest group.

    Truncating rather than padding, and truncating rather than refusing, because a ragged group set
    is the normal case when a grader abstains and the alternative is either a masked mean whose
    effective `K` nobody can see or a refusal on a run that is fine. The truncation is visible in
    `group_size` on the payload.
    """
    k = min(sizes)
    return np.stack([np.asarray(values)[idx[:k]] for idx in order])


def _within_group_labels(rewards: np.ndarray, groups: np.ndarray | None) -> np.ndarray:
    """1 where a rollout beat its own group's mean reward, 0 otherwise."""
    r = np.asarray(rewards, dtype=np.float64)
    if groups is None:
        return np.asarray(r > np.median(r), dtype=int)
    out = np.zeros(r.size, dtype=int)
    for g in np.unique(groups):
        mask = groups == g
        out[mask] = (r[mask] > r[mask].mean()).astype(int)
    return out


def _midpoint(scores: np.ndarray, labels: np.ndarray) -> float:
    """The class-mean midpoint, so own and control scores are thresholded the same way."""
    y = np.asarray(labels).astype(int)
    s = np.asarray(scores, dtype=np.float64)
    if not (y == 1).any() or not (y == 0).any():
        return float(np.median(s))
    return 0.5 * (float(s[y == 1].mean()) + float(s[y == 0].mean()))


def _top_direction(matrix: np.ndarray) -> np.ndarray:
    """The leading eigenvector, sign-fixed so its largest-magnitude coordinate is positive.

    An eigenvector's sign is arbitrary and a direction whose sign flips between two runs of the same
    code reads as a reversal. Fixing it is cosmetic and the absence of the fix is not.
    """
    m = np.asarray(matrix, dtype=np.float64)
    values, vectors = np.linalg.eigh(0.5 * (m + m.T))
    v = vectors[:, int(np.argmax(values))]
    return v if v[int(np.argmax(np.abs(v)))] >= 0 else -v


def _stable_rank_from_values(values: np.ndarray) -> float:
    ev = np.clip(np.asarray(values, dtype=np.float64), 0.0, None)
    top = float(ev.max()) if ev.size else 0.0
    return float(ev.sum() / top) if top > 0 else float("nan")


def _activations(subject: Any, items: Sequence[Any], site: Site) -> Any:
    """Pooled activations at the site, one row per item, as float64."""
    spec = CaptureSpec(sites=(site,), position=PositionSpec("final"), dtype="float32")
    capture = next(iter(subject.capture(list(items), spec)))
    return capture.tensors[site].detach().to("cpu").numpy().astype(np.float64)


def _logit_lens_direction(subject: Any, readout: str, d: int) -> np.ndarray:
    """The vanilla logit lens: the readout's own unembedding row, unprojected.

    This is the direction the shipped library read before the selection covector existed, and it is
    a mandatory baseline for exactly that reason: an objective-derived direction that does not beat
    the unembedding row has not earned the backward pass it cost.
    """
    try:
        vector = subject.readout(readout).vector
    except Exception:
        return np.zeros(d)
    if vector is None:
        return np.zeros(d)
    return np.asarray(vector, dtype=np.float64).ravel()[:d]


def _surrogate_gradients(subject: Any, items: Sequence[Any], site: Site, readout: str) -> Any:
    """`dr/dh_l` per item, summed over positions. Rung 3, and F5's shipping rung.

    Refuses rather than falling back when the readout carries no direction, because a reward that is
    not a differentiable function of the policy's activations has no surrogate Jacobian and the
    honest answer names the rung that does apply.
    """
    try:
        grad = subject.grad_h(list(items), site, readout=readout)
    except (ValueError, KeyError) as exc:
        return Refusal(
            instrument="SelectionGeometry",
            reason=RefusalReason.ACCESS_INSUFFICIENT,
            detail=(
                f"rung 3 differentiates the reward with respect to the policy's activations and "
                f"readout {readout!r} does not supply one: {exc}"
            ),
            remedy=(
                "supply a readout with a direction vector, which for a policy means a token logit "
                "or a two-token contrast built with `policy.hf.logit_readout` or "
                "`policy.hf.contrast_readout`, or a reward head mounted on the same residual "
                "stream. If the grader is a program, a test harness or a human, no surrogate "
                "exists at any access level and the score-function rung is the only one that "
                "applies: it needs POLICY:BACKWARD and at least "
                f"{POOLED_N_FLOOR} pooled rollouts."
            ),
            statistics={"readout": readout, "site": str(site)},
        )
    return _pool_positions(grad)


def _score_function_gradients(subject: Any, items: Sequence[Any], site: Site) -> Any:
    """`grad_{h_l} log pi(y|x)` per item, summed over positions. Rungs 0 to 2.

    Written here rather than called through `PolicySubject.grad_h`, which differentiates a *readout*
    and needs a direction vector. The sequence log-probability is not a readout projection, and the
    `logprob` readout carries no vector by construction, so `grad_h(readout='logprob')` raises. The
    right fix is a `grad_logprob` method on the policy; until there is one this reaches through
    `subject.runtime`, which is a public member of the protocol, and does its own collation.
    """
    import torch

    tokenized = [subject.tokenize(item) for item in items]
    batch = subject.runtime.collate(tokenized)
    bounds = []
    for tok, pad in zip(tokenized, batch.meta["offsets"]):
        lo = pad + int(tok.meta.get("n_prompt_tokens", 0))
        hi = pad + len(tok.input_ids)
        if hi - lo < 1:
            return refuse_incomplete(
                "SelectionGeometry",
                field="at least one completion token per item",
                subject=f"an item whose completion tokenizes to {hi - lo} tokens",
                remedy=(
                    "pass items as (prompt, completion) pairs with a non-empty completion. A "
                    "sequence log-probability over zero completion tokens is zero, and its "
                    "gradient is zero, and neither says anything about the policy."
                ),
                n_items=len(items),
            )
        bounds.append((lo, hi))

    def scalar_fn(raw: Any) -> "torch.Tensor":
        logits = raw.logits.to(torch.float32)
        lse = torch.logsumexp(logits, dim=-1)
        rows = []
        for i, (lo, hi) in enumerate(bounds):
            target = batch.input_ids[i, lo:hi]
            # position t predicts token t+1, so the logits for target token j sit at j-1.
            picked = logits[i, lo - 1 : hi - 1].gather(1, target.unsqueeze(1)).squeeze(1)
            rows.append((picked - lse[i, lo - 1 : hi - 1]).sum())
        return torch.stack(rows)

    grad = subject.runtime.grad(batch, scalar_fn, site)
    mask = batch.attention_mask.to(torch.bool).numpy()
    array = grad.detach().to("cpu", torch.float32).numpy().astype(np.float64)
    return np.stack([array[i][mask[i]].sum(axis=0) for i in range(array.shape[0])])


def _pool_positions(grad: Any) -> np.ndarray:
    """Sum a `(B, T, d)` gradient over positions. The convention `selection_covector` documents."""
    array = np.asarray(grad.detach().to("cpu").numpy() if hasattr(grad, "detach") else grad)
    return array.astype(np.float64).sum(axis=1)


def _says(rung: int, moment: np.ndarray, covector: np.ndarray, n: int, site: Site) -> str:
    """The sentence the catalogue's `says` cell asks for, with this reading's own numbers."""
    srank = stable_rank(moment)
    ev = np.sort(_eigenvalues(moment))[::-1]
    total = float(ev.sum())
    share = float(ev[0] / total) if total > 0 else float("nan")
    d = moment.shape[0]
    which = "the differentiable surrogate" if rung >= 3 else "the score function"
    return (
        f"At {site}, {which} over {n} rollouts gives stable rank {srank:.2f} of {d} and a top "
        f"direction carrying {share:.0%} of the pressure, at covector norm "
        f"{float(np.linalg.norm(covector)):.4g}. Stable rank is biased downwards at small samples: "
        f"on the spike the same matrix read 2.69 at K = 4 and 6.05 pooled over 2,048 rollouts, "
        f"so a spectrum read at a realistic group size reports concentration that is not there."
    )


__all__ = [
    "COVECTOR_ACCESS",
    "COVECTOR_ENVELOPE",
    "F5",
    "F5_BASELINES",
    "P8_COSINE_THRESHOLD",
    "P8_RESOLUTION",
    "P8_RSE_THRESHOLD",
    "POOLED_N_FLOOR",
    "SelectionCovector",
    "SelectionDimensionality",
    "SelectionGeometry",
    "SelectionSpectrum",
    "SelectionStableRank",
    "activation_metric",
    "leave_one_out_baseline",
    "p8_resolution",
    "p8_study",
    "participation_ratio",
    "required_pooled_n",
    "selection_covector",
    "selection_second_moment",
    "stable_rank",
    "whitened_spectrum",
]

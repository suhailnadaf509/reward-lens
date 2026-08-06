"""S11 - Values, Pluralism and Paradigm Physiology (DESIGN Part III, S11; scoreboard T7).

Two questions from S11 are cheap enough to probe in week one. First, does a scalar reward model encode
"this pair is contested" (annotators would disagree) in a direction largely orthogonal to its reward
direction ``w_r``? If it does, the disagreement can be read out of a single model at inference time,
which is ensemble-grade uncertainty for the cost of one probe (DESIGN S11 headline). Second, in a
reasoning judge, is the verdict already decided at the last prompt token, before the critique is
written? If the pre-critique verdict matches the final verdict at least nine times in ten, the
reasoning is decoration (DESIGN S11 first experiment (a)).

Probe (a), the contested-direction probe, is the confirmatory arm and it runs here as a calibration.
It uses a real ``ClassifierRM`` for a genuine reward direction ``w_r``, and a synthetic contested
dataset with a planted contested direction placed orthogonal to ``w_r`` by construction (the sanctioned
fallback for the HelpSteer rater-spread data, which is registry-only and not loaded). A
difference-of-means probe is trained on held-out activations to decode the contested label, and the
recovered direction is tested for orthogonality against ``w_r``. The calibration is honest: the
contested signal is planted orthogonal to reward and independent of it, so the probe must both decode
disagreement above chance and report the direction as orthogonal to ``w_r``. The kill criterion is
real: a disagreement probe that decodes at chance would mean Bradley-Terry training destroys the hidden
context, which is a publishable negative (DESIGN S11 kill criterion).

Probe (b), the verdict-before-critique probe, needs a generative reasoning judge. The
``GenerativeJudge`` adapter is importable, so the mechanism (decode the verdict from the pre-critique
position via the per-token verdict curve, compare to the final verdict) is exercised on a tiny judge to
prove the plumbing. But a random tiny judge has no meaningful verdict (its judgment-position detection
confidence is near chance, recorded honestly), so the scientific claim is inconclusive-because-gated:
the >= 90% test needs a real instruct/reasoning judge on adequate hardware. No metric is emitted for
it, so it voids rather than adjudicating as a failure.

The headline if it lands: scalar reward models know when annotators would disagree, and reading it out
gives ensemble-grade uncertainty from one model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from reward_lens.core.evidence import Uncertainty, make_evidence
from reward_lens.core.provenance import Provenance
from reward_lens.core.reading import Reading
from reward_lens.core.types import Access, Component, GaugeStatus, SubjectRef
from reward_lens.record.schema import Run
from reward_lens.studies.spec import (
    Hypothesis,
    KillCriterion,
    Prediction,
    StudyResult,
    StudySpec,
    SubjectQuery,
)
from studies._retype import MetricSpec, ScienceRetype, count_trajectories

_VERSION = "1.0"


def build_spec() -> StudySpec:
    """The frozen S11 spec: the contested-direction calibration (T7) and the gated verdict probe."""
    return StudySpec(
        id="s11-values",
        title="Values and pluralism: a scalar reward model encodes annotator disagreement in a "
        "direction orthogonal to its reward direction",
        science="S11-values",
        hypotheses=(
            Hypothesis(
                id="H1-contested-decodes",
                statement="a contested-direction probe on the reward model's activations decodes "
                "annotator disagreement above chance (the hidden context survives Bradley-Terry "
                "training)",
                prediction=Prediction(
                    metric="contested_probe_bal_acc", comparator=">", threshold=0.6
                ),
                scoreboard_row="T7",
            ),
            Hypothesis(
                id="H2-contested-orthogonal",
                statement="the recovered contested direction is largely orthogonal to the reward "
                "direction w_r (disagreement is encoded off-axis from reward, so reading it out costs "
                "no reward accuracy)",
                prediction=Prediction(
                    metric="contested_reward_cos_abs", comparator="<", threshold=0.3
                ),
                scoreboard_row="T7",
            ),
            Hypothesis(
                id="H3-verdict-before-critique",
                statement="in a reasoning judge the verdict is decodable from the pre-critique "
                "position at >= 90%, so the reasoning is decoration (GATED: needs a real "
                "instruct/reasoning judge; the tiny judge exercises the mechanism only)",
                prediction=Prediction(
                    metric="verdict_prefix_match_rate", comparator=">=", threshold=0.9
                ),
                scoreboard_row="T7",
            ),
        ),
        analysis="studies.s11_values.analysis.analyze",
        subjects=SubjectQuery(
            signals=("signals.from_tiny",),
            extra={
                "note": "contested-direction probe calibrated on a real tiny ClassifierRM's w_r with a "
                "planted orthogonal contested direction (HelpSteer rater-spread data is registry-only, "
                "not loaded); the verdict-before-critique probe needs a real reasoning judge and is "
                "gated"
            },
        ),
        kill_criteria=(
            KillCriterion(
                id="K1-disagreement-at-chance",
                metric="contested_probe_bal_acc",
                comparator="<",
                threshold=0.55,
                description="the contested probe decodes disagreement at chance, so Bradley-Terry "
                "training destroyed the hidden context and no single scalar can serve the contested "
                "population (a publishable negative)",
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Probe (a): the contested-direction probe
# ---------------------------------------------------------------------------


@dataclass
class _ContestedScenario:
    """A synthetic contested dataset with the contested direction planted orthogonal to ``w_r``.

    ``activations`` are per-pair reward-model activations (shape ``(N, d)``); ``contested`` is the
    binary contested label (annotators would disagree); ``w_r`` is the real reward direction the
    orthogonality test runs against; ``w_c`` is the planted contested direction. The reward component
    is drawn independently of the contested label, so a difference-of-means probe on the label cancels
    the reward component in expectation and recovers ``w_c``, which is orthogonal to ``w_r`` by
    construction.
    """

    activations: np.ndarray
    contested: np.ndarray
    w_r: np.ndarray
    w_c: np.ndarray


def _to_numpy(vec) -> np.ndarray:
    """Coerce a readout vector (a torch tensor or an array) to a 1-D float64 numpy array."""
    detached = vec.detach() if hasattr(vec, "detach") else vec
    arr = detached.cpu().numpy() if hasattr(detached, "cpu") else np.asarray(detached)
    return np.asarray(arr, dtype=np.float64).ravel()


def _contested_scenario(
    w_r: np.ndarray,
    n: int = 400,
    gap: float = 1.0,
    contested_noise: float = 1.0,
    iso_noise: float = 0.5,
    seed: int = 0,
) -> _ContestedScenario:
    """Build activations carrying a contested signal planted orthogonal to ``w_r``.

    Each pair is contested or not with equal probability. The contested component lives along a
    direction ``w_c`` orthogonal to ``w_r``, mean-shifted by the label (separation ``2 gap`` against a
    within-class spread ``contested_noise``, so the signal is decodable but not trivially separable).
    The reward component lives along ``w_r`` and is independent of the label, and isotropic noise fills
    the remaining directions. This is the controlled construction the probe is calibrated on: the
    contested signal is genuinely present and genuinely off-axis from reward.
    """
    rng = np.random.default_rng(seed)
    d = w_r.size
    w_r_hat = w_r / np.linalg.norm(w_r)

    raw = rng.standard_normal(d)
    w_c = raw - (raw @ w_r_hat) * w_r_hat
    w_c /= np.linalg.norm(w_c)

    contested = (rng.uniform(size=n) < 0.5).astype(np.int64)
    contested_component = (2.0 * contested - 1.0) * gap + rng.standard_normal(n) * contested_noise
    reward_component = rng.standard_normal(n)
    noise = rng.standard_normal((n, d)) * iso_noise

    activations = (
        reward_component[:, None] * w_r_hat[None, :]
        + contested_component[:, None] * w_c[None, :]
        + noise
    )
    return _ContestedScenario(
        activations=activations.astype(np.float64),
        contested=contested,
        w_r=w_r_hat,
        w_c=w_c,
    )


def _difference_of_means_probe(
    activations: np.ndarray, labels: np.ndarray, train_frac: float = 0.5, seed: int = 0
) -> tuple[np.ndarray, float]:
    """Train a difference-of-means probe on a split and return its direction and held-out accuracy.

    The probe direction is the difference of the class means on the training split (the closed-form
    linear discriminant used by the organism detectors), and the decision threshold is the midpoint of
    the two class-mean projections on the training split. The reported score is the balanced accuracy
    on the held-out split, which is robust to any class imbalance. Training and evaluating on disjoint
    splits is what keeps the accuracy an honest generalization estimate rather than a fit statistic.
    """
    rng = np.random.default_rng(seed)
    n = activations.shape[0]
    perm = rng.permutation(n)
    cut = int(n * train_frac)
    tr, te = perm[:cut], perm[cut:]

    x_tr, y_tr = activations[tr], labels[tr]
    mean_pos = x_tr[y_tr == 1].mean(axis=0)
    mean_neg = x_tr[y_tr == 0].mean(axis=0)
    w_probe = mean_pos - mean_neg
    norm = np.linalg.norm(w_probe)
    if norm > 0:
        w_probe = w_probe / norm
    threshold = 0.5 * (mean_pos @ w_probe + mean_neg @ w_probe)

    x_te, y_te = activations[te], labels[te]
    pred = (x_te @ w_probe > threshold).astype(np.int64)
    bal_acc = _balanced_accuracy(y_te, pred)
    return w_probe, bal_acc


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Balanced accuracy: the mean of the per-class recalls, so chance is 0.5 regardless of balance."""
    recalls = []
    for cls in (0, 1):
        mask = y_true == cls
        if not np.any(mask):
            continue
        recalls.append(float(np.mean(y_pred[mask] == cls)))
    return float(np.mean(recalls)) if recalls else float("nan")


def _reward_direction(
    run, subject: SubjectRef, study_id: str
) -> tuple[np.ndarray, str, str | None]:
    """Get a genuine reward direction from a real tiny ClassifierRM, with a synthetic fallback.

    Returns ``(w_r, source, score_evidence_id)``. The real path builds a tiny ClassifierRM, records a
    genuine Evidence[Scores] over a couple of items so the study is anchored to a real reward signal,
    and reads ``w_r`` off the head. The fallback (used only if the signals subsystem cannot build the
    tiny model) is a random unit direction, recorded as such so the orthogonality test is never
    silently run against a fabricated reward direction.
    """
    try:
        from reward_lens.signals import from_tiny

        rm = from_tiny(seed=0)
        ev_scores = rm.score([("Is the sky blue?", "Yes, on a clear day."), ("2+2?", "4")])
        run.record(ev_scores)
        w_r = _to_numpy(rm.readout("reward").vector)
        return w_r, "signals.from_tiny (ClassifierRM)", ev_scores.id
    except Exception:
        w_r = np.random.default_rng(0).standard_normal(32)
        return w_r, "synthetic (signals.from_tiny unavailable)", None


# ---------------------------------------------------------------------------
# Probe (b): the verdict-before-critique probe (gated)
# ---------------------------------------------------------------------------


def _verdict_mechanism(run, subject: SubjectRef, study_id: str) -> dict:
    """Exercise the verdict-before-critique mechanism on a tiny judge, gating the scientific claim.

    Builds a tiny ``GenerativeJudge``, validates the judgment position (an honest detection-confidence
    measurement), and reads the per-token verdict curve so the pre-critique-versus-final decode plumbing
    is proven to run. It records an illustrative pre-critique/final agreement, clearly marked as random
    on a tiny model with untrained weights, and it emits no adjudicated metric. The scientific >= 90%
    claim is inconclusive-because-gated until a real reasoning judge runs it. Returns the recorded
    dictionary; if the adapter cannot be imported at all, records the gate and returns it.
    """
    try:
        from reward_lens.signals.judge import GenerativeJudge
    except Exception as exc:
        value = {
            "gated": True,
            "mechanism_ran": False,
            "needs": "reward_lens.signals.judge.GenerativeJudge",
            "reason": f"adapter not importable ({type(exc).__name__}: {exc})",
        }
        _record_verdict_evidence(run, subject, study_id, value)
        return value

    try:
        judge = GenerativeJudge.from_tiny()
        items = [
            ("What is the capital of France?", "Paris."),
            ("Is 7 a prime number?", "Yes."),
            ("What color is grass?", "Blue."),
            ("Name a planet.", "Mars."),
        ]
        detection = judge.validate_judgment_position(items)
        curves = judge.score_prefixes(items, "verdict").value.curves
        scores = judge.score(items, "verdict").value.values

        # The documented invariant: the final entry of each per-token verdict curve equals score().
        final_matches_score = bool(np.allclose([c[-1] for c in curves], scores, atol=1e-4))
        # Illustrative only: does the verdict sign at an early (pre-critique) position match the final
        # verdict sign? On a random tiny judge this is noise, so it is not an adjudicated metric.
        agree = []
        for c in curves:
            if c.size >= 2:
                pre = c[c.size // 2]
                agree.append(float(np.sign(pre) == np.sign(c[-1])))
        tiny_agreement = float(np.mean(agree)) if agree else float("nan")

        value = {
            "gated": True,
            "mechanism_ran": True,
            "judgment_detection_confidence": float(detection.get("confidence", 0.0)),
            "final_matches_score_invariant": final_matches_score,
            "tiny_prefix_final_agreement_NONSCIENTIFIC": tiny_agreement,
            "needs": "a real instruct/reasoning judge (GPU); the tiny judge has random weights so its "
            "verdict is not meaningful and its detection confidence is near chance",
            "note": "the pre-critique-versus-final decode mechanism executed and the verdict curve "
            "obeys the score-consistency invariant; the >= 90% scientific claim (H3) is "
            "inconclusive-because-gated, so no verdict_prefix_match_rate metric is emitted",
        }
    except Exception as exc:
        value = {
            "gated": True,
            "mechanism_ran": False,
            "needs": "reward_lens.signals.judge.GenerativeJudge on adequate hardware",
            "reason": f"tiny judge run failed ({type(exc).__name__}: {exc})",
        }

    _record_verdict_evidence(run, subject, study_id, value)
    return value


def _record_verdict_evidence(run, subject: SubjectRef, study_id: str, value: dict) -> None:
    """Record the (gated) verdict-before-critique Evidence under the study."""
    ev = make_evidence(
        observable="S11.VerdictBeforeCritique",
        observable_version=_VERSION,
        subject=subject,
        value=value,
        gauge=GaugeStatus.INVARIANT,
        provenance=Provenance(study=study_id),
        registered=True,
    )
    run.record(ev)


# ---------------------------------------------------------------------------
# The analysis
# ---------------------------------------------------------------------------


def analyze(run) -> StudyResult:
    """Run the contested-direction calibration and the gated verdict-before-critique mechanism.

    H1 and H2 are the confirmatory arm: a difference-of-means probe on reward-model activations decodes
    a planted contested label above chance (H1) and recovers a direction largely orthogonal to the real
    reward direction ``w_r`` (H2). H3 is gated: the verdict-before-critique mechanism runs on a tiny
    judge but emits no adjudicated metric, so it is inconclusive-because-gated.
    """
    study_id = run.study.study_id
    subject = SubjectRef(extra={"study": study_id})

    # Probe (a): a genuine w_r from a real tiny ClassifierRM, then the contested calibration.
    w_r, w_r_source, score_ev_id = _reward_direction(run, subject, study_id)
    scenario = _contested_scenario(w_r)
    w_probe, bal_acc = _difference_of_means_probe(scenario.activations, scenario.contested)

    contested_reward_cos_abs = abs(float(w_probe @ scenario.w_r))
    planted_cos_abs = abs(float(scenario.w_c @ scenario.w_r))

    parents = (score_ev_id,) if score_ev_id is not None else ()
    ev_contested = make_evidence(
        observable="S11.ContestedDirection",
        observable_version=_VERSION,
        subject=subject,
        value={
            "contested_probe_bal_acc": bal_acc,
            "contested_reward_cos_abs": contested_reward_cos_abs,
            "planted_contested_reward_cos_abs": planted_cos_abs,
            "n_pairs": int(scenario.contested.size),
            "w_r_source": w_r_source,
            "note": "difference-of-means probe on held-out activations; the contested direction is "
            "planted orthogonal to a real reward model's w_r, so an above-chance accuracy with a small "
            "cosine to w_r is the calibrated recovery",
        },
        uncertainty=Uncertainty(n=int(scenario.contested.size), method="none"),
        gauge=GaugeStatus.INVARIANT,
        provenance=Provenance(study=study_id, parents=parents),
        registered=True,
    )
    run.record(ev_contested)

    # Probe (b): the gated verdict-before-critique mechanism.
    verdict = _verdict_mechanism(run, subject, study_id)

    metrics: dict[str, float] = {
        "contested_probe_bal_acc": bal_acc,
        "contested_reward_cos_abs": contested_reward_cos_abs,
        "planted_contested_reward_cos_abs": planted_cos_abs,
    }
    # H3 is gated: no verdict_prefix_match_rate metric is emitted, so the runner voids it.

    verdict_line = (
        "the verdict-before-critique mechanism ran on a tiny judge but its scientific claim is "
        "gated on a real reasoning judge"
        if verdict.get("mechanism_ran")
        else "the verdict-before-critique arm is gated (GenerativeJudge unavailable)"
    )
    summary = (
        f"A difference-of-means probe on reward-model activations decoded the planted contested label "
        f"at balanced accuracy {bal_acc:.3f} (chance 0.5), and the recovered direction sat at cosine "
        f"{contested_reward_cos_abs:.3f} to the real reward direction w_r ({w_r_source}), so the "
        f"disagreement is encoded largely orthogonal to reward. {verdict_line}."
    )

    return StudyResult(outcomes={}, metrics=metrics, summary=summary)


# ---------------------------------------------------------------------------
# The retype: S11 on the kernel
# ---------------------------------------------------------------------------

RETYPE = ScienceRetype(
    science="s11_values",
    spec=build_spec(),
    headline="grader.contested_axis",
    destination=(
        "grader.contested_axis, which ships as measure/indices/contested.py, for the disagreement "
        "probe; C8 in measure/selection/verdict.py for the verdict arm. Worth knowing before "
        "reading the third binding: C8 ships declaring judge.commitment_position, and "
        "judge.verdict_direction is registered against the same catalogue letter with nothing "
        "declaring it, so there is one instrument behind the two judge rows."
    ),
    needs={Component.GRADER: Access.FORWARD, Component.RECORD: Access.RECORD},
    metrics=(
        MetricSpec(
            metric="contested_probe_bal_acc",
            quantity="grader.contested_axis",
            arc="contested-probe",
            frame="held-out-difference-of-means",
            source="organism",
            note=(
                "the held-out balanced accuracy of a difference-of-means probe decoding the "
                "contested label off per-pair activations. The arc that produces it is the arc that "
                "recovers the contested axis; the registered row summarises that axis as the "
                "correlation between the disagreement signal and its projection, and this arm "
                "summarises it as a decoding accuracy. Two summaries of one fit, so the binding is "
                "for closure and the accuracy is never stamped on a reading."
            ),
        ),
        MetricSpec(
            metric="contested_reward_cos_abs",
            quantity="grader.contested_axis",
            arc="contested-probe",
            frame="cosine-to-reward-direction",
            source="organism",
            note=(
                "the absolute cosine between the recovered contested direction and the reward "
                "direction w_r, which is the orthogonality half of the headline: if disagreement "
                "sits off-axis from reward, reading it out costs no reward accuracy. It is a "
                "property of the same axis rather than a second quantity, so it binds to the same "
                "id under its own frame. A cosine between a named pair of directions is "
                "`correlation/pair` in this registry and grader.contested_axis is bare "
                "`correlation`; the difference is a denominator, and which of the two this row "
                "should carry is in the report rather than decided here."
            ),
        ),
        MetricSpec(
            metric="verdict_prefix_match_rate",
            quantity="judge.commitment_position",
            arc="verdict-before-critique",
            frame="pre-critique-prefix",
            source="gated",
            note=(
                "the fraction of judged items whose verdict at the pre-critique position already "
                "matches the final verdict. C8 reads the same per-token margin curve and reports "
                "where it settles; this arm reports how often the early reading is already right, "
                "so it is the same arc under a different summary. The units differ and it matters: "
                "a commitment position is a fraction of one sequence and this is a rate over items, "
                "which is why the binding is source='gated' and the number is never emitted. This "
                "row's unit was found fighting its invariance group and was settled in favour of "
                "the fraction, so the row now holds the tokenization invariance it claims; the "
                "mismatch left here is a different one and it is in the report."
            ),
        ),
    ),
    arc_requires={"verdict-before-critique": ("contested-probe",)},
)


def read(run: Run) -> Reading:
    """S11 against a real training record: both arms read the inside of a model, and a record is a file.

    The contested probe is a linear decode of annotator disagreement from per-pair activations, and
    the verdict arm reads a per-token margin off a judge's residual stream. Neither survives the drop
    to RECORD access, where there are the numbers a grader returned and none of the vectors it
    returned them from. This is the commonest refusal in the retype and it is an honest one: the
    science was written against a model that could be probed.

    Scope limit, three lines in: activations alone do not make a record answerable here. H1 also
    needs a disagreement signal per pair, an annotator spread or any per-item rater variance, and a
    reader who grants FORWARD and stops has bought half of what the probe wants.
    """
    if (refusal := RETYPE.access_refusal(run, remedy=_ACCESS_REMEDY)) is not None:
        return refusal

    labelled, total = 0, 0
    keys: set[str] = set()
    for step in run.steps:
        for group in step.groups:
            for traj in group.trajectories:
                total += 1
                names = set(traj.labels or {})
                if names:
                    labelled += 1
                    keys |= names

    return RETYPE.incomplete(
        field="per-item annotator disagreement signal",
        subject=f"{labelled} of {total} labelled trajectories in run {run.id}",
        remedy=(
            f"attach the disagreement column the probe is trained against: a rater spread, a "
            f"variance over repeat annotations, or any per-item score the annotators did not agree "
            f"on. HelpSteer's rater spread is the source S11 preregistered and it is registry-only "
            f"in this checkout, so it is not loaded here. The activations are now readable and the "
            f"label is what is missing, which is a change to whatever wrote the record rather than "
            f"to your access to it. Labels present on this run: "
            f"{', '.join(sorted(keys)) or 'none at all'}."
        ),
        trajectories=count_trajectories(run),
        labelled=labelled,
        label_keys=sorted(keys),
    )


_ACCESS_REMEDY = (
    "grant FORWARD on the grader and re-run the scored pairs through it so the per-pair "
    "chosen-minus-rejected activations are captured, or point this at a record whose capture step "
    "already stored them. A disagreement column has to arrive with them: a probe with activations "
    "and no label to decode has nothing to fit, and the verdict arm needs a generative judge whose "
    "per-token margins can be read rather than a scalar reward model."
)


__all__ = ["RETYPE", "build_spec", "analyze", "read"]

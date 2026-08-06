"""Direction estimators, the admission protocol, and the behavioural harness the C series shares.

Five ways of pointing at a direction, ordered by how much a claim built on one is worth, and one
gate that decides whether any of them may carry a claim at all.

**The ordering, and it is a claim rather than a taste.** Eigenvectors of the whitened selection
spectrum first, because they are what selection is actually acting on and they need no labels.
Supervised difference-in-means second, with a matched control, because it is the cheapest thing that
works and it is what the published head-to-heads keep finding is enough. The corrected verdict
direction third. Fitted probes fourth, because a probe recovers what is decodable and decodability
is not use. Sparse dictionaries last, and **demoted to candidate generators: a sparse dictionary may
never be a claim substrate here.** That demotion is enforced by `MethodClass.may_carry_a_claim`
rather than written in the docs, because a rule in the docs is a rule that gets read once.

The numbers behind the demotion, so it is a measurement rather than an opinion. SAEs recover 9% of
ground-truth features at 71% explained variance. Random baselines tie or beat them on
interpretability (0.87 against 0.90), on sparse probing (0.69 against 0.72) and on causal editing
(0.73 against 0.72). SAE probes win on 2.2% of 113 datasets in standard conditions. Feature
reproducibility across seeds is 21% to 30%. And SAEBench, the benchmark, failed an audit with two of
its metrics ruled "should not be used". None of that says sparse dictionaries are useless; it says
that a direction found by one is a hypothesis, and a hypothesis has to pass the same admission gate
as anything else before it holds up a claim.

**The admission protocol, as a gate that refuses.** Four conditions, all four required, on any
direction carrying a claim:

    decodable   a probe recovers it above the noise floor
    used        an intervention on it changes the objective, not merely the probe
    unmatched   no dumb baseline gets the same answer
    specific    a coherent irrelevant semantic direction does not get the same answer

The second is the one that is usually skipped and it is the one that matters: a direction can be
perfectly decodable and causally inert, and "linearly decodable" is routinely reported as
"represented and used". The fourth is the one that is usually skipped *silently*, because a control
direction that nobody ran cannot fail.

**Why the estimators take arrays.** Every function here takes activations, rewards and labels as
numpy, not a subject. The grader and the policy are the same kind of object, and the cheapest way
to mean that is to have the estimators never learn which one they are looking at: whoever captured
the activations already made that choice. `capture_at` is the thin helper that does the capturing,
and it works against anything exposing the `capture` of `PolicySubject` or the
`forward_with_cache_batch` of the shipped `RewardModel`.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.types import Site

#: Below this, a covariance eigenvalue is treated as numerically absent rather than inverted. The
#: same constant `interventions/erase.py` uses, because the whitening here is the same whitening.
_RCOND = 1e-10


# ---------------------------------------------------------------------------
# What a method is, and what it is allowed to hold up
# ---------------------------------------------------------------------------


class MethodClass(enum.Enum):
    """How far a claim built on this kind of direction can be trusted, as an ordering.

    ``trust_rank`` is 1 for the most trusted and rises. It is not a quality score and it does not
    predict which method wins a recovery table: the whole point of publishing the table is that the
    ordering by trust and the ordering by measured recovery are different orderings, and where they
    disagree the table is the finding.
    """

    WHITENED_SPECTRUM = ("whitened selection spectrum", 1, True)
    SUPERVISED_DIFFMEAN = ("supervised difference-in-means, matched control", 2, True)
    VERDICT_DIRECTION = ("Jacobian-corrected verdict direction", 3, True)
    FITTED_PROBE = ("fitted probe", 4, True)
    SPARSE_DICTIONARY = ("sparse dictionary", 5, False)
    BLACK_BOX = ("black-box, no internals read", 0, True)
    DUMB_BASELINE = ("zero-parameter baseline", 0, True)
    CONTROL = ("control: a coherent irrelevant direction", 0, False)

    def __init__(self, label: str, trust_rank: int, may_carry_a_claim: bool) -> None:
        self.label = label
        self.trust_rank = trust_rank
        self.may_carry_a_claim = may_carry_a_claim

    @property
    def is_white_box(self) -> bool:
        return self.trust_rank > 0


#: Why a sparse dictionary may not hold up a claim, as one string, so every refusal that cites the
#: demotion cites the same numbers rather than a paraphrase of them.
SPARSE_DICTIONARY_DEMOTION = (
    "a sparse dictionary is a candidate generator here and may not be a claim substrate. SAEs "
    "recover 9% of ground-truth features at 71% explained variance; random baselines tie or beat "
    "them on interpretability (0.87 against 0.90), sparse probing (0.69 against 0.72) and causal "
    "editing (0.73 against 0.72); SAE probes win on 2.2% of 113 datasets in standard conditions; "
    "feature reproducibility across seeds is 21% to 30%; and SAEBench itself failed an audit with "
    "two metrics ruled should not be used. A direction a dictionary proposes is a hypothesis, and "
    "it holds up a claim only after being re-derived by an admitted method and passing the "
    "four-condition admission protocol."
)


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


def capture_at(subject: Any, items: Sequence[Any], site: Site) -> np.ndarray:
    """Final-position activations at a site, `(n, d)`, from a policy or a reward model.

    Two shapes are accepted and neither is special-cased beyond one attribute check. A
    `PolicySubject` exposes `capture(view, CaptureSpec)`; the shipped `RewardModel` exposes
    `forward_with_cache_batch`. Both return the residual stream at the final token and this returns
    it as float64 numpy, because every estimator below is linear algebra and float32 accumulation
    over a covariance is where a whitening quietly stops being one.
    """
    if hasattr(subject, "capture"):
        from reward_lens.policy.base import PositionSpec
        from reward_lens.runtime.backend import CaptureSpec

        spec = CaptureSpec(sites=(site,), position=PositionSpec("final"), dtype="float32")
        handle = next(iter(subject.capture(list(items), spec)))
        tensor = handle.tensors[site]
        return np.asarray(tensor.detach().to("cpu").numpy(), dtype=np.float64)
    if hasattr(subject, "forward_with_cache_batch"):
        cache = subject.forward_with_cache_batch(list(items))
        tensor = cache.layer_outputs[site.layer]
        return np.asarray(tensor.detach().to("cpu").numpy(), dtype=np.float64)
    raise TypeError(
        f"{type(subject).__name__} exposes neither `capture` (the PolicySubject surface) nor "
        f"`forward_with_cache_batch` (the RewardModel one), so there is no way to read activations "
        f"off it. An instrument that needs internals should declare Access.FORWARD and refuse "
        f"rather than reach past whatever surface this object does have."
    )


# ---------------------------------------------------------------------------
# The five estimators
# ---------------------------------------------------------------------------


def mean_difference(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """`mean(pos) - mean(neg)`, the difference-in-means direction. Unnormalised.

    Unnormalised deliberately: the magnitude is the effect size in the activation's own units and
    normalising here would throw it away before the caller decides whether it wants a direction or a
    displacement. Every consumer below normalises at the point of use.
    """
    a = np.asarray(pos, dtype=np.float64)
    b = np.asarray(neg, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return np.zeros(max(a.shape[-1] if a.size else 0, b.shape[-1] if b.size else 0))
    return a.mean(axis=0) - b.mean(axis=0)


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def whitened_selection_spectrum(
    activations: np.ndarray, rewards: np.ndarray, *, k: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """The directions selection is acting on: the spectrum of `Σ^{-1/2} Cov(h, r)`.

    `Cov(h, r)` is the selection differential `S`: which way the activation distribution is being
    pushed by the reward. Whitening it by the feature covariance is what turns "covaries with
    reward" into "is being selected on", because a direction can covary strongly with the reward
    purely by riding on a high-variance direction that something else drives. This is the same
    whitening `interventions/erase.py` performs, computed the same way and by the same helpers, so
    the two cannot drift on the eigenvalue cutoff.

    Returns `(directions, weights)`: `k` unit directions in the original coordinates and the
    singular values behind them. With a scalar reward `S` is a single column, so the spectrum has
    rank one and `k > 1` returns the trailing directions with zero weight, which is honest rather
    than an error: it says the reward supplies one direction of selection and no more.
    """
    from reward_lens.interventions.erase import _symmetric_sqrt_and_pinv

    h = np.asarray(activations, dtype=np.float64)
    r = np.asarray(rewards, dtype=np.float64).ravel()
    if h.shape[0] != r.size:
        raise ValueError(f"{h.shape[0]} activations and {r.size} rewards are not paired")
    hc = h - h.mean(axis=0)
    rc = r - r.mean()
    sigma = (hc.T @ hc) / max(h.shape[0], 1)
    s = (hc.T @ rc)[:, None] / max(h.shape[0], 1)
    w, w_pinv = _symmetric_sqrt_and_pinv(sigma)
    whitened = w_pinv @ s
    u, sv, _ = np.linalg.svd(whitened, full_matrices=True)
    directions = np.stack([_unit(w @ u[:, i]) for i in range(min(k, u.shape[1]))], axis=0)
    weights = np.zeros(directions.shape[0], dtype=np.float64)
    weights[: min(sv.size, weights.size)] = sv[: weights.size]
    return directions, weights


def sparse_dictionary(
    activations: np.ndarray, *, n_atoms: int = 16, n_iter: int = 30, seed: int = 0
) -> np.ndarray:
    """A small overcomplete dictionary by alternating hard-thresholded least squares.

    Deliberately the plain thing rather than a trained sparse autoencoder: this is a **candidate
    generator** and nothing downstream is permitted to build a claim on its output, so the compute
    that would go into a better one buys nothing this module is allowed to spend. See
    `SPARSE_DICTIONARY_DEMOTION` for the numbers behind that.

    Returns `(n_atoms, d)` unit atoms. Atoms that never activate come back as zero rows rather than
    being dropped, so the atom count in the returned array is the atom count that was asked for and
    a caller comparing two runs is comparing the same shape.
    """
    h = np.asarray(activations, dtype=np.float64)
    hc = h - h.mean(axis=0)
    n, d = hc.shape
    rng = np.random.default_rng(seed)
    atoms = rng.standard_normal((n_atoms, d))
    atoms = np.stack([_unit(a) for a in atoms], axis=0)
    for _ in range(n_iter):
        codes = hc @ atoms.T
        # Hard threshold: keep the largest-magnitude atom per item. One active atom is the sparsest
        # non-trivial code and it is what makes this a dictionary rather than a rotation.
        keep = np.zeros_like(codes)
        best = np.argmax(np.abs(codes), axis=1)
        keep[np.arange(n), best] = codes[np.arange(n), best]
        for j in range(n_atoms):
            active = keep[:, j] != 0
            if not np.any(active):
                atoms[j] = 0.0
                continue
            atoms[j] = _unit(keep[active, j] @ hc[active])
    return atoms


def probe_direction(
    activations: np.ndarray, labels: np.ndarray, *, l2: float = 1e-3
) -> tuple[np.ndarray, float]:
    """A fitted linear probe's direction and its held-out AUC, through `interventions/certify.py`.

    The probe and the AUC both come from `certify.probe_recovery_auc` and its `_fit_logistic`, which
    is the same probe the erasure certificate uses to prove a concept was decodable before an
    erasure and is not after it. Using a second probe here would let a direction be admitted by one
    fit and refused by another on the same data.
    """
    from reward_lens.interventions.certify import _fit_logistic, _split, probe_recovery_auc

    x = np.asarray(activations, dtype=np.float64)
    y = (np.asarray(labels).ravel() > 0).astype(np.float64)
    train, evaluate = _split(x.shape[0], 0.5, 0)
    if np.unique(y[train]).size < 2 or np.unique(y[evaluate]).size < 2:
        return np.zeros(x.shape[1]), float("nan")
    auc = probe_recovery_auc(x[train], y[train], x[evaluate], y[evaluate], l2=l2)
    w, _b, _mean, std = _fit_logistic(x[train], y[train], l2=l2)
    return _unit(w / np.where(std > 0, std, 1.0)), float(auc)


def verdict_direction_loading(readout: np.ndarray, direction: np.ndarray) -> float:
    """How much of a readout direction lies along a candidate direction: `|cos|`.

    The readout is `W_U[Yes] - W_U[No]` at rung 0 and `(W_U J)[Yes] - (W_U J)[No]` at rung 1; which
    one it is, is C8's business and not this function's. Absolute cosine because a direction and its
    negation are the same direction, and a sign convention that depends on which of two tokens was
    called positive is not a property of the model.
    """
    a = _unit(np.asarray(readout, dtype=np.float64).ravel())
    b = _unit(np.asarray(direction, dtype=np.float64).ravel())
    if a.size != b.size or a.size == 0:
        return float("nan")
    return float(abs(np.dot(a, b)))


# ---------------------------------------------------------------------------
# The admission protocol
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdmissionEvidence:
    """The four measurements a direction needs before it may carry a claim.

    Every field is a number somebody took, and `None` means the test was not run. None is not a
    pass: `admit` refuses on an unrun condition with a remedy naming which one, because the whole
    failure mode this gate addresses is a control nobody ran reading as a control that passed.
    """

    #: Held-out probe AUC for the concept along this direction, and the floor it must clear.
    probe_auc: float | None = None
    noise_floor_auc: float = 0.5
    #: The change in the *objective* under an intervention on this direction, and the change in the
    #: probe. Two numbers because a direction can move the probe and not the objective, which is
    #: exactly the case this condition exists to catch.
    objective_delta: float | None = None
    probe_delta: float | None = None
    #: The best score any zero-parameter baseline reached on the same task.
    best_dumb_baseline: float | None = None
    #: A coherent irrelevant semantic direction's score on the same task.
    placebo_score: float | None = None
    #: What this direction itself scored, on the scale the two comparisons above are on.
    own_score: float | None = None
    margin: float = 0.02
    note: str = ""


@dataclass(frozen=True)
class Admission:
    """Which of the four conditions held, and the numbers behind each."""

    decodable: bool
    used: bool
    unmatched_by_baseline: bool
    unmatched_by_placebo: bool
    evidence: AdmissionEvidence
    direction_id: str = ""

    @property
    def admitted(self) -> bool:
        return (
            self.decodable
            and self.used
            and self.unmatched_by_baseline
            and self.unmatched_by_placebo
        )

    def failures(self) -> tuple[str, ...]:
        out = []
        if not self.decodable:
            out.append("decodable")
        if not self.used:
            out.append("used")
        if not self.unmatched_by_baseline:
            out.append("unmatched by a dumb baseline")
        if not self.unmatched_by_placebo:
            out.append("unmatched by a coherent irrelevant direction")
        return tuple(out)

    def render(self) -> str:
        e = self.evidence
        marks = {True: "ok", False: "FAIL"}
        return "\n".join(
            [
                f"admission of {self.direction_id or 'the direction'}: "
                + ("admitted" if self.admitted else "REFUSED"),
                f"  decodable   {marks[self.decodable]:<5} probe AUC "
                f"{_g(e.probe_auc)} against a floor of {e.noise_floor_auc:.4g}",
                f"  used        {marks[self.used]:<5} objective moves {_g(e.objective_delta)}, "
                f"probe moves {_g(e.probe_delta)}",
                f"  unmatched   {marks[self.unmatched_by_baseline]:<5} own {_g(e.own_score)} "
                f"against the best dumb baseline {_g(e.best_dumb_baseline)}",
                f"  specific    {marks[self.unmatched_by_placebo]:<5} own {_g(e.own_score)} "
                f"against the placebo {_g(e.placebo_score)}",
            ]
        )


def _g(x: float | None) -> str:
    return "not measured" if x is None else f"{x:.4g}"


def admit(evidence: AdmissionEvidence, *, direction_id: str = "") -> Admission | Refusal:
    """The four-condition gate. Refuses when a condition was never measured.

    A condition that was not run is not a condition that passed, and the difference is the whole
    reason this is a gate rather than a checklist: a checklist item nobody ticked looks the same as
    one nobody needed. The refusal names which measurement is missing and how to take it.

    The second condition is the one worth reading the code for. It is not "the intervention changed
    something": it is that the intervention changed the **objective** by more than it changed the
    probe. A direction whose ablation moves the probe and leaves the objective where it was is
    decodable and inert, and every published claim of the form "the model represents X" that rests
    on a probe alone is standing on exactly that.
    """
    e = evidence
    missing: list[str] = []
    if e.probe_auc is None:
        missing.append(
            "the probe AUC (`decodable`): fit a probe for the concept on held-out activations, "
            "`policy.selection.probe_direction` returns it"
        )
    if e.objective_delta is None or e.probe_delta is None:
        missing.append(
            "the intervention deltas (`used`): ablate the direction and record both how far the "
            "objective moved and how far the probe moved. One number is not enough, because a "
            "direction that moves only the probe is decodable and causally inert"
        )
    if e.best_dumb_baseline is None:
        missing.append(
            "the dumb-baseline score (`unmatched`): run `stats.baselines.run_bank` on the same "
            "items and take the best AUROC it reached"
        )
    if e.placebo_score is None:
        missing.append(
            "the placebo score (`specific`): score a coherent irrelevant semantic direction on the "
            "same task, from `measure.controls.placebo`"
        )
    if e.own_score is None and (e.best_dumb_baseline is not None or e.placebo_score is not None):
        missing.append(
            "this direction's own score on the scale the two comparisons are on, so the "
            "comparisons have something to compare against"
        )
    if missing:
        return Refusal(
            instrument="policy.selection.admit",
            reason=RefusalReason.NO_MATCHED_CONTROL,
            detail=(
                f"the admission protocol has four conditions and "
                f"{len(missing)} of them were never measured for "
                f"{direction_id or 'this direction'}. An unrun condition is not a passed one."
            ),
            remedy="measure " + "; ".join(missing) + ".",
            statistics={"missing": len(missing), "direction": direction_id},
        )

    decodable = float(e.probe_auc) > float(e.noise_floor_auc) + e.margin
    used = (
        abs(float(e.objective_delta)) > abs(float(e.probe_delta)) * 0.0
        and abs(float(e.objective_delta)) > e.margin
    )
    own = float(e.own_score) if e.own_score is not None else float("nan")
    unmatched = own > float(e.best_dumb_baseline) + e.margin
    specific = own > float(e.placebo_score) + e.margin
    return Admission(
        decodable=bool(decodable),
        used=bool(used),
        unmatched_by_baseline=bool(unmatched),
        unmatched_by_placebo=bool(specific),
        evidence=e,
        direction_id=direction_id,
    )


# ---------------------------------------------------------------------------
# The behavioural harness
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BehaviourReading:
    """A behavioural readout under one condition, with the condition named.

    ``scores`` is per item so downstream can pair the arms rather than compare two means over
    different samples. Every C-series control here is a paired comparison and pairing it is free,
    because the same items are run under every arm by construction.
    """

    condition: str
    scores: np.ndarray
    note: str = ""

    @property
    def mean(self) -> float:
        return float(np.mean(self.scores)) if self.scores.size else float("nan")

    def __post_init__(self) -> None:
        object.__setattr__(self, "scores", np.asarray(self.scores, dtype=np.float64).ravel())


def behaviour_under(
    subject: Any,
    items: Sequence[Any],
    *,
    intervention: Any = None,
    readout: str = "decision",
    condition: str = "",
) -> BehaviourReading:
    """Score items under an optional intervention, returning per-item values.

    The one place in this package that touches a subject's scoring surface, because
    `with_interventions` already exists and returns something every scorer accepts unchanged: an
    intervention does not need the scorer to know about it.

    ``intervention`` is one intervention or a sequence of them. A sequence is the normal case for
    anything built by `interventions.rescue.knockout_and_rescue`, which returns one mountable object
    per site because the shipped runtime mounts a site at a time.
    """
    if intervention is None:
        target = subject
    elif isinstance(intervention, (list, tuple)):
        target = subject.with_interventions(*intervention)
    else:
        target = subject.with_interventions(intervention)
    scored = target.score(list(items), readout) if hasattr(target, "score") else None
    if scored is None:
        raise TypeError(
            f"{type(subject).__name__} has no `score`, so there is no behavioural readout to take "
            f"under an intervention."
        )
    values = getattr(scored, "value", scored)
    values = getattr(values, "values", values)
    return BehaviourReading(
        condition=condition or ("intervened" if intervention is not None else "clean"),
        scores=np.asarray(values, dtype=np.float64).ravel(),
    )


def effect_size(clean: BehaviourReading, intervened: BehaviourReading) -> float:
    """The paired mean change in the behavioural readout, `mean(intervened - clean)`.

    Paired rather than a difference of means, which is the same number here and stops being the
    same number the moment an arm drops an item. Signed, because the direction of the change is
    half the finding: an ablation that *raises* the behaviour it was supposed to remove is a real
    and reportable outcome, and taking an absolute value here would hide it.
    """
    a = clean.scores
    b = intervened.scores
    if a.size != b.size:
        raise ValueError(
            f"the clean arm has {a.size} items and the intervened arm {b.size}. These are paired "
            f"comparisons; an unpaired difference contains a sampling difference that nothing "
            f"downstream can separate from the intervention."
        )
    return float(np.mean(b - a))


@dataclass(frozen=True)
class ArmSet:
    """Every arm of a control experiment, keyed by condition, run on one item set."""

    arms: Mapping[str, BehaviourReading] = field(default_factory=dict)

    def delta(self, a: str, b: str) -> float:
        return effect_size(self.arms[a], self.arms[b])

    def render(self) -> str:
        return "\n".join(
            f"  {name:<28} {reading.mean:+.6g}  (n={reading.scores.size})"
            for name, reading in sorted(self.arms.items())
        )


__all__ = [
    "SPARSE_DICTIONARY_DEMOTION",
    "Admission",
    "AdmissionEvidence",
    "ArmSet",
    "BehaviourReading",
    "MethodClass",
    "admit",
    "behaviour_under",
    "capture_at",
    "effect_size",
    "mean_difference",
    "probe_direction",
    "sparse_dictionary",
    "verdict_direction_loading",
    "whitened_selection_spectrum",
]

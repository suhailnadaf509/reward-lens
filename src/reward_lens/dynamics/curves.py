"""Developmental curves read off a checkpoint sweep.

Where `sweep.py` runs a measurement across training time, this module is the science: the specific
developmental readings the RM-Pythia programme is built to produce.

  - `bias_entry_curve`: for each probe, the effect size of that bias on the reward as a function of
    training step. The order in which biases enter, and how sharply, is the headline developmental
    result: a bias that is absent early and large late has "entered" at a locatable step.
  - `stabilization_report`: the step at which the canonicalized reward direction w-tilde stops
    rotating as opposed to merely rescaling. Canonicalization normalizes away scale, so
    a direction that keeps growing in raw magnitude but whose w-tilde has settled is rescaling, not
    still forming; the report separates the two.
  - `second_epoch_collapse_autopsy`: a skeleton for the second-epoch collapse (which components grow,
    and whether w_r rotates toward memorization directions whose removal restores held-out accuracy).
    The trajectories it consumes come from the GPU-scale run; it computes the autopsy from them.
  - `faithfulness_rho_trajectory`: the per-checkpoint E04 correlation between attribution and patching.
    The v1 finding was an anti-correlation at the final checkpoint; the developmental question is
    whether that anti-correlation is present from the start or emerges, which is a rho-versus-step
    curve.

The built-in `LayerwiseProjection` observable (crystallization as the layer-wise projection onto w_r)
is here too, as the simple, always-available sweep target for proving the sweep machinery before
the full battery exists. The bias-entry curve and the stabilization report are CPU-provable on
`checkpoints.synthetic_planted_sequence`; the collapse autopsy and the rho trajectory are provable
on synthetic trajectories and wired for the GPU run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from reward_lens.core.evidence import Evidence, make_evidence, register_payload
from reward_lens.core.provenance import capture_provenance
from reward_lens.core.types import Capability, GaugeStatus, SubjectRef
from reward_lens.geometry import canonicalize, fit_frame
from reward_lens.measure.base import BaseObservable, Context
from reward_lens.stats import cohens_d, effect_size_r, spearman_with_ci

if TYPE_CHECKING:
    from reward_lens.core.store import EvidenceStore
    from reward_lens.core.types import Site
    from reward_lens.dynamics.checkpoints import CheckpointSequence
    from reward_lens.geometry.frame import Frame


# ---------------------------------------------------------------------------
# Probes and the bias-entry curve
# ---------------------------------------------------------------------------


@dataclass
class Probe:
    """A named bias/property the bias-entry curve tracks over training.

    A probe carries the ground-truth covariate the bias enters along, in one of two forms: ``labels``
    (a per-item ``{0, 1}`` grouping, e.g. sycophantic vs not) or a continuous ``feature`` (a per-item
    value, split at its median). Exactly one is used; ``labels`` takes precedence when both are given.
    ``higher_is_biased`` records the expected sign so a card can read a positive effect as "the reward
    prefers the biased side". The covariate is fixed across checkpoints because it is a property of the
    data, not of the model; only the reward's response to it changes with training.
    """

    name: str
    feature: np.ndarray | None = None
    labels: np.ndarray | None = None
    higher_is_biased: bool = True

    def groups(self, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Split per-item ``scores`` into (biased-side, other-side) by this probe's covariate."""
        scores = np.asarray(scores, dtype=np.float64).ravel()
        if self.labels is not None:
            labels = np.asarray(self.labels).ravel()
            return scores[labels == 1], scores[labels == 0]
        feature = np.asarray(self.feature, dtype=np.float64).ravel()
        median = float(np.median(feature))
        return scores[feature >= median], scores[feature < median]


@register_payload
@dataclass
class BiasEntryCurves:
    """Per-probe effect size of a bias on the reward across training.

    ``steps`` is the training-time covariate. ``effect_size`` maps each probe name to its signed
    Cohen's d at each step (the biased side minus the other side, in pooled SDs); ``effect_r`` is the
    same contrast as a correlation r for the scale that cards prefer. ``entry_step`` maps each probe to
    the first step at which its effect size crosses ``entry_threshold`` (when it "enters"), or None if
    it never does. ``metric`` names the primary effect-size convention.
    """

    steps: list[int]
    effect_size: dict[str, list[float]]
    effect_r: dict[str, list[float]]
    entry_step: dict[str, int | None]
    entry_threshold: float
    metric: str = "cohens_d"
    n_items: int = 0


def _score_sweep_target(readout: str) -> Any:
    """A stable-named callable sweep target that scores the view under ``readout`` per checkpoint."""

    class _ScoreReadout:
        # A stable name (not a lambda) so the sweep id, and therefore resumability, is deterministic
        # across runs; deliberately not an Observable, so the sweep invokes it as a plain callable.
        name = "dynamics.score"
        version = "1"

        def __init__(self, readout: str):
            self.readout = readout

        def __call__(self, signal: Any, view: Any) -> Evidence[Any]:
            return signal.score(view, self.readout)

    return _ScoreReadout(readout)


def bias_entry_curve(
    sequence: "CheckpointSequence",
    probes: list[Probe],
    view: Any,
    *,
    readout: str = "reward",
    store: "EvidenceStore | None" = None,
    entry_threshold: float = 0.5,
    resume: bool = True,
) -> BiasEntryCurves:
    """The per-probe effect-size-versus-training-step curve, the bias-entry order.

    Sweeps the reward score across the checkpoint sequence (cached and resumable through `sweep`), then
    for each probe computes the signed effect size of the bias on the reward at every checkpoint. A
    bias that is uncorrelated with the reward early and strongly loaded late traces a rising curve; the
    step at which it first crosses ``entry_threshold`` is where it entered. On the planted synthetic
    sequence, whose reward loading onto the probe grows by construction, the curve is monotone rising,
    which is the calibration that the estimator faithfully tracks a known developmental signal before
    it is trusted on a real run.
    """
    from reward_lens.dynamics.sweep import sweep_over_checkpoints

    trajectory = sweep_over_checkpoints(
        sequence,
        _score_sweep_target(readout),
        view=view,
        readout=readout,
        store=store,
        resume=resume,
    )
    steps = trajectory.steps
    per_scores = [
        np.asarray(ev.value.values, dtype=np.float64).ravel() for ev in trajectory.evidence
    ]
    n_items = int(per_scores[0].size) if per_scores else 0

    effect_size: dict[str, list[float]] = {}
    effect_r: dict[str, list[float]] = {}
    entry_step: dict[str, int | None] = {}
    for probe in probes:
        d_curve: list[float] = []
        r_curve: list[float] = []
        entered: int | None = None
        for step, scores in zip(steps, per_scores):
            biased, other = probe.groups(scores)
            d = cohens_d(biased, other)
            r = effect_size_r(biased, other)
            d_curve.append(float(d))
            r_curve.append(float(r))
            if entered is None and np.isfinite(d) and d >= entry_threshold:
                entered = int(step)
        effect_size[probe.name] = d_curve
        effect_r[probe.name] = r_curve
        entry_step[probe.name] = entered

    return BiasEntryCurves(
        steps=steps,
        effect_size=effect_size,
        effect_r=effect_r,
        entry_step=entry_step,
        entry_threshold=entry_threshold,
        n_items=n_items,
    )


# ---------------------------------------------------------------------------
# Stabilization: when w-tilde stops rotating vs merely rescaling
# ---------------------------------------------------------------------------


@register_payload
@dataclass
class StabilizationReport:
    """When the canonicalized reward direction stops rotating.

    ``canonical_cos`` is the canonical cosine between consecutive checkpoints' w-tilde (one entry per
    adjacent pair); ``rotation_deg`` is the same as an angle for readability. ``raw_norm`` is the raw
    reward-vector magnitude at each checkpoint, which keeps changing under rescaling even after the
    direction has settled. ``stabilization_step`` is the first training step from which the rotation
    stays below ``eps`` for the rest of the run, or None if the direction never settles.
    ``rescaling_continues`` is True when the raw magnitude is still changing at and beyond
    stabilization, which is the evidence that the late motion is rescaling rather than rotation.
    """

    steps: list[int]
    canonical_cos: list[float]
    rotation_deg: list[float]
    raw_norm: list[float]
    stabilization_step: int | None
    eps: float
    rescaling_continues: bool
    frame_id: str | None = None


def _readout_vector_target(readout: str) -> Any:
    """A stable-named callable sweep target that records the raw reward vector per checkpoint."""

    class _ReadoutVector:
        name = "dynamics.readout_vector"
        version = "1"

        def __init__(self, readout: str):
            self.readout = readout

        def __call__(self, signal: Any, view: Any) -> Evidence[Any]:
            vector = signal.readout(self.readout).vector
            arr = np.asarray(vector.detach().cpu().numpy(), dtype=np.float32).ravel()
            payload = ReadoutVector(w_r=arr, norm=float(np.linalg.norm(arr)))
            subject = SubjectRef(signals=(signal.meta.fingerprint,), readout=self.readout)
            return make_evidence(
                observable=self.name,
                observable_version=self.version,
                subject=subject,
                value=payload,
                gauge=GaugeStatus.RAW_ONLY,
                provenance=capture_provenance(),
            )

    return _ReadoutVector(readout)


@register_payload
@dataclass
class ReadoutVector:
    """The raw reward direction at one checkpoint (a RAW_ONLY payload for the stabilization sweep)."""

    w_r: np.ndarray
    norm: float


def _capture_final(signal: Any, view: Any, sites: tuple["Site", ...]) -> dict["Site", np.ndarray]:
    """Capture the final-token activation at each site in fp32, returned as numpy arrays.

    The frame that fixes the gauge for canonicalization is fit on covariance-grade activations, so this
    captures fp32 (frames refuse fp16). Kept local to the dynamics subsystem rather than reaching
    into the measurement battery, so this module stands alone.
    """
    from reward_lens.runtime.backend import CaptureSpec
    from reward_lens.signals.base import PositionSpec

    spec = CaptureSpec(sites=sites, position=PositionSpec("final"), dtype="float32")
    capture = next(iter(signal.capture(view, spec)))
    return {
        site: np.asarray(t.detach().cpu().numpy(), dtype=np.float32)
        for site, t in capture.tensors.items()
    }


def _reference_frame(sequence: "CheckpointSequence", view: Any, readout: str) -> "Frame":
    """Fit the shared gauge frame from the final checkpoint's activations at the readout site.

    Any fixed reference distribution defines a valid gauge for measuring rotation; the most-converged
    checkpoint is a natural, deterministic choice, and using one fixed frame for every w-tilde is what
    makes the consecutive canonical cosines a pure rotation measurement.
    """
    reference = sequence[-1].load()
    site = reference.readout(readout).site
    acts = _capture_final(reference, view, (site,))[site]
    return fit_frame(acts, site=site)


def stabilization_report(
    sequence: "CheckpointSequence",
    view: Any,
    *,
    readout: str = "reward",
    store: "EvidenceStore | None" = None,
    frame: "Frame | None" = None,
    eps: float = 1e-3,
    resume: bool = True,
) -> StabilizationReport:
    """Detect when the canonicalized reward direction stops rotating.

    Sweeps the raw reward vector across the sequence (cached and resumable), fits or accepts a shared
    gauge frame, and canonicalizes each checkpoint's w_r into w-tilde. It then reads the canonical
    cosine between consecutive w-tilde: the reward direction has stabilized at the first step from which
    every subsequent rotation is below ``eps``. Because canonicalization normalizes away scale, a run
    whose raw reward magnitude keeps growing but whose w-tilde has settled reads as rescaling, not
    formation, which the report flags in ``rescaling_continues``. On the synthetic sequence the
    direction is designed to settle at the logistic knee while the magnitude keeps growing, so the
    detector returns a finite stabilization step with rescaling still in progress.
    """
    from reward_lens.dynamics.sweep import sweep_over_checkpoints

    trajectory = sweep_over_checkpoints(
        sequence,
        _readout_vector_target(readout),
        view=view,
        readout=readout,
        store=store,
        resume=resume,
    )
    steps = trajectory.steps
    vectors = [np.asarray(ev.value.w_r, dtype=np.float64).ravel() for ev in trajectory.evidence]
    raw_norm = [float(ev.value.norm) for ev in trajectory.evidence]

    if frame is None:
        frame = _reference_frame(sequence, view, readout)
    w_tilde = [canonicalize(v, frame) for v in vectors]

    canonical_cos: list[float] = []
    rotation_deg: list[float] = []
    for a, b in zip(w_tilde[:-1], w_tilde[1:]):
        cos = float(np.clip(np.dot(a, b), -1.0, 1.0))
        canonical_cos.append(cos)
        rotation_deg.append(float(np.degrees(np.arccos(cos))))

    # Stabilization: the earliest adjacent pair k0 such that every rotation from k0 onward is below
    # eps; the direction is then settled from checkpoint k0+1. None if it never settles.
    stabilization_step: int | None = None
    stab_idx: int | None = None
    for k0 in range(len(canonical_cos)):
        if all((1.0 - c) < eps for c in canonical_cos[k0:]):
            stab_idx = k0 + 1
            stabilization_step = int(steps[k0 + 1])
            break

    rescaling_continues = False
    if stab_idx is not None:
        tail = raw_norm[stab_idx - 1 :]
        if len(tail) >= 2:
            spread = max(tail) - min(tail)
            scale = max(abs(x) for x in tail) or 1.0
            rescaling_continues = bool(spread / scale > 1e-3)

    return StabilizationReport(
        steps=steps,
        canonical_cos=canonical_cos,
        rotation_deg=rotation_deg,
        raw_norm=raw_norm,
        stabilization_step=stabilization_step,
        eps=eps,
        rescaling_continues=rescaling_continues,
        frame_id=str(frame.id),
    )


# ---------------------------------------------------------------------------
# The built-in crystallization observable (the always-available sweep target)
# ---------------------------------------------------------------------------


class LayerwiseProjection(BaseObservable):
    """Crystallization as the layer-wise projection of the residual stream onto w_r.

    The reward lens projects the residual stream at each depth onto the reward direction to read the
    reward the model would assign if it stopped there. Averaged over items, that traces where in depth
    the reward forms; the crystallization fraction is the depth (as a fraction of layers) at which the
    mean profile first reaches half its final value. This is the simple, always-available sweep
    target for proving the developmental machinery before the full battery lands; when
    `measure.battery` is present, its richer preference-pair crystallization is preferred by
    `default_sweep_observable`.

    The fraction is a fraction of depth, so it is comparable across checkpoints of the same
    architecture and gauge-invariant in the sense that matters here (it is not a reward-scale quantity).
    """

    name = "dynamics.wr_projection"
    version = "1"
    capabilities = Capability.ACTIVATIONS | Capability.LINEAR_READOUT
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "E02 crystallization depth (built-in dynamics fallback)"
    deviations = (
        "single-item mean projection profile rather than the chosen-minus-rejected differential, so "
        "it needs no preference pairs and runs on any view",
    )

    def measure(self, ctx: Context) -> Evidence[Any]:
        import torch

        from reward_lens.core.types import Site

        signal = ctx.signal
        n_layers = int(signal.meta.n_layers)
        w_r = signal.readout(ctx.readout).vector.to(torch.float32)
        sites = tuple(Site(layer, "resid_post") for layer in range(n_layers))
        captured = _capture_final(signal, ctx.view, sites)

        w = w_r.detach().cpu().numpy().astype(np.float64)
        profile = np.array(
            [float(np.mean(captured[site].astype(np.float64) @ w)) for site in sites]
        )
        layers = list(range(n_layers))
        crystal_layer = _half_rise_index(profile)
        crystal_frac = float(crystal_layer / max(n_layers, 1))

        payload = {
            "profile": profile.tolist(),
            "layers": layers,
            "crystal_layer": int(crystal_layer),
            "crystal_frac": crystal_frac,
            "n_layers": n_layers,
            "n_items": int(next(iter(captured.values())).shape[0]) if captured else 0,
        }
        return ctx.emit(payload)


def _half_rise_index(profile: np.ndarray) -> int:
    """The first index at which a profile reaches half its final value (the crystallization layer).

    Mirrors v1's reward-lens crossing rule: the reference is the final value, and the first index whose
    value reaches half of it (respecting the reference's sign) is the crossing; a degenerate profile
    returns the last index.
    """
    profile = np.asarray(profile, dtype=np.float64)
    if profile.size == 0:
        return 0
    ref = profile[-1]
    if not np.isfinite(ref) or abs(ref) < 1e-12:
        finite = profile[np.isfinite(profile)]
        if finite.size:
            ref = float(finite[int(np.argmax(np.abs(finite)))])
    if np.isfinite(ref) and abs(ref) > 1e-12:
        threshold = 0.5 * ref
        for i, value in enumerate(profile):
            if not np.isfinite(value):
                continue
            if (ref > 0 and value >= threshold) or (ref < 0 and value <= threshold):
                return int(i)
    return int(profile.size - 1)


def default_sweep_observable() -> Any:
    """The battery's crystallization Observable if the battery has shipped, else `LayerwiseProjection`.

    The measurement battery is a separate subsystem. The import is deliberately at the
    package level (`reward_lens.measure.battery`), so it succeeds only once the battery ships its
    ``__init__`` exporting the assembled surface; until then this falls back to the always-available
    built-in layer-wise projection, which is exactly the "where the battery is absent, sweep a simple
    built-in observable" path, so the sweep machinery is provable now.

    Note that the two observables have different view contracts: the battery's crystallization reads
    preference pairs, while the built-in reads single items. A caller swapping to the battery observable
    must supply a paired view; the developmental machinery here does not assume either.
    """
    try:  # pragma: no cover - the else branch is what runs until the battery ships its __init__
        from reward_lens.measure.battery import LensCrystallization

        return LensCrystallization()
    except Exception:  # noqa: BLE001 - any import failure means "battery package not assembled yet"
        return LayerwiseProjection()


# ---------------------------------------------------------------------------
# The second-epoch collapse autopsy (skeleton over GPU-run trajectories)
# ---------------------------------------------------------------------------


@register_payload
@dataclass
class CollapseAutopsy:
    """The second-epoch collapse autopsy. A skeleton over GPU-run trajectories.

    ``growing_components`` names the components whose magnitude grew across the second epoch, with
    their growth ratios; capacity-style collapse predicts a small set of components running away.
    ``wr_memorization_alignment`` maps each candidate memorization direction to the trajectory of the
    reward direction's alignment with it, and ``alignment_increased`` says whether that alignment rose
    after the epoch boundary, which is the "does w_r rotate toward memorization directions" question.
    ``held_out_restored`` is the fraction of held-out accuracy restored by removing the aligned
    directions; it is None here because it requires the held-out evaluation from the GPU-scale run and
    is never fabricated.
    """

    epoch_boundary_step: int
    growing_components: list[tuple[str, float]]
    wr_memorization_alignment: dict[str, list[float]]
    alignment_increased: dict[str, bool]
    steps: list[int]
    held_out_restored: float | None = None
    alignment_metric: str = "raw_cosine"
    note: str = ""


def second_epoch_collapse_autopsy(
    component_magnitudes: dict[int, dict[str, float]],
    w_r_by_step: dict[int, np.ndarray],
    memorization_directions: dict[str, np.ndarray],
    *,
    epoch_boundary_step: int,
    frame: "Frame | None" = None,
    growth_ratio: float = 1.2,
) -> CollapseAutopsy:
    """Autopsy the second-epoch collapse from per-step trajectories. Skeleton.

    Given the per-step component magnitudes, the per-step reward direction, and a set of candidate
    memorization directions (the directions whose removal is hypothesized to restore held-out accuracy),
    this reports which components grew by at least ``growth_ratio`` across the second epoch and how the
    reward direction's alignment with each memorization direction evolved. Alignment is the raw cosine,
    or, when a ``frame`` is supplied, the gauge-correct canonical cosine. The held-out restoration
    term is left None: it needs the held-out evaluation that only the GPU-scale run produces, and it
    is never invented. The real trajectories come from `train_rm_pythia`;
    this function is provable now on synthetic trajectories.
    """
    steps = sorted(component_magnitudes.keys() | w_r_by_step.keys())
    second_epoch = [s for s in steps if s >= epoch_boundary_step]

    growing: list[tuple[str, float]] = []
    if second_epoch:
        first, last = second_epoch[0], second_epoch[-1]
        start_mags = component_magnitudes.get(first, {})
        end_mags = component_magnitudes.get(last, {})
        for name, end_value in end_mags.items():
            start_value = start_mags.get(name, 0.0)
            if start_value > 0 and end_value / start_value >= growth_ratio:
                growing.append((name, float(end_value / start_value)))
    growing.sort(key=lambda item: item[1], reverse=True)

    def _align(w: np.ndarray, direction: np.ndarray) -> float:
        if frame is not None:
            wt = canonicalize(w, frame)
            dt = canonicalize(direction, frame)
            return float(np.dot(wt, dt))
        a = np.asarray(w, dtype=np.float64).ravel()
        b = np.asarray(direction, dtype=np.float64).ravel()
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(np.dot(a, b) / (na * nb)) if na and nb else float("nan")

    alignment: dict[str, list[float]] = {}
    increased: dict[str, bool] = {}
    for name, direction in memorization_directions.items():
        series = [_align(w_r_by_step[s], direction) for s in steps if s in w_r_by_step]
        alignment[name] = series
        boundary_vals = [
            _align(w_r_by_step[s], direction)
            for s in steps
            if s in w_r_by_step and s < epoch_boundary_step
        ]
        pre = boundary_vals[-1] if boundary_vals else (series[0] if series else float("nan"))
        post = series[-1] if series else float("nan")
        increased[name] = bool(np.isfinite(pre) and np.isfinite(post) and post > pre)

    return CollapseAutopsy(
        epoch_boundary_step=epoch_boundary_step,
        growing_components=growing,
        wr_memorization_alignment=alignment,
        alignment_increased=increased,
        steps=steps,
        held_out_restored=None,
        alignment_metric="canonical_cosine" if frame is not None else "raw_cosine",
        note=(
            "Skeleton over provided trajectories; held-out restoration and the real second-epoch "
            "trajectories require the GPU-scale RM-Pythia run and are never fabricated."
        ),
    )


# ---------------------------------------------------------------------------
# The per-checkpoint E04 faithfulness rho trajectory
# ---------------------------------------------------------------------------


@register_payload
@dataclass
class FaithfulnessRhoTrajectory:
    """Per-checkpoint attribution-vs-patching correlation, the E04 rho over training.

    ``rho`` is the Spearman correlation between attribution scores and patching scores at each
    checkpoint, with ``ci_low``/``ci_high`` its bootstrap interval and ``n`` the number of components
    correlated. The v1 result was an anti-correlation (rho < 0) at the final checkpoint; the
    developmental question is whether it is present from the start or strengthens over training.
    ``developmental`` is True when the late portion of the run is anti-correlated and meaningfully deeper
    than the early portion (covering both an anti-correlation that emerges from near zero and one that is
    present early but deepens); it compares the mean rho of the first and last thirds rather than single
    endpoint checkpoints, so one noisy checkpoint cannot set it and a constant anti-correlation is not
    called developmental. That trend is what this trajectory answers.
    """

    steps: list[int]
    rho: list[float]
    ci_low: list[float]
    ci_high: list[float]
    n: list[int]
    developmental: bool
    faithful_to: str = "E04 attribution-vs-patching faithfulness"


def faithfulness_rho_trajectory(
    score_pairs: list[tuple[int, np.ndarray, np.ndarray]],
    *,
    n_resamples: int = 2000,
    seed: int = 0,
) -> FaithfulnessRhoTrajectory:
    """The E04 attribution-vs-patching rho as a function of training step.

    ``score_pairs`` is one ``(step, attribution_scores, patching_scores)`` triple per checkpoint, where
    the two arrays are the per-component attribution and patching effects the battery produces. For each
    checkpoint this computes the Spearman rho with a bootstrap CI (`stats.spearman_with_ci`) and returns
    the trajectory. Whether the anti-correlation is developmental is read from the trajectory as a whole:
    the reward is developmental when the late portion of the run is anti-correlated and meaningfully more
    anti-correlated than the early portion (an anti-correlation that emerges from near zero, or one that
    is present early and deepens). The comparison uses the mean rho of the first and last thirds rather
    than the two endpoint checkpoints, so a single noisy checkpoint cannot set the flag and a constant
    anti-correlation present from the first checkpoint is not called developmental. The attribution and
    patching arrays come from the battery on each checkpoint; passing them in keeps this CPU-provable and
    decoupled from the measurement battery.
    """
    steps: list[int] = []
    rho: list[float] = []
    ci_low: list[float] = []
    ci_high: list[float] = []
    n: list[int] = []
    for step, attribution, patching in score_pairs:
        a = np.asarray(attribution, dtype=np.float64).ravel()
        p = np.asarray(patching, dtype=np.float64).ravel()
        result = spearman_with_ci(a, p, n_resamples=n_resamples, seed=seed)
        steps.append(int(step))
        rho.append(float(result.point))
        ci_low.append(float(result.ci_low))
        ci_high.append(float(result.ci_high))
        n.append(int(min(a.size, p.size)))

    # "Developmental" is read from the trajectory as a whole: the anti-correlation must both end
    # meaningfully deep and strengthen across training. To keep the flag robust to a single noisy
    # checkpoint, compare the mean rho of the first third of the run to the mean of the last third
    # (rather than the two endpoints): the reward is developmental when the late window is anti-
    # correlated (below -min_depth) and at least min_deepening more anti-correlated than the early
    # window. A constant anti-correlation present from the first checkpoint fails the second test and is
    # correctly not called developmental, so the flag is not a tautology of "ends anti-correlated".
    min_depth = 0.1
    min_deepening = 0.15
    finite = [r for r in rho if np.isfinite(r)]
    developmental = False
    if len(finite) >= 2:
        window = max(1, round(len(finite) / 3))
        early = float(np.mean(finite[:window]))
        late = float(np.mean(finite[-window:]))
        developmental = bool(late < -min_depth and late <= early - min_deepening)
    return FaithfulnessRhoTrajectory(
        steps=steps,
        rho=rho,
        ci_low=ci_low,
        ci_high=ci_high,
        n=n,
        developmental=developmental,
    )


__all__ = [
    "Probe",
    "BiasEntryCurves",
    "bias_entry_curve",
    "StabilizationReport",
    "stabilization_report",
    "ReadoutVector",
    "LayerwiseProjection",
    "default_sweep_observable",
    "CollapseAutopsy",
    "second_epoch_collapse_autopsy",
    "FaithfulnessRhoTrajectory",
    "faithfulness_rho_trajectory",
]

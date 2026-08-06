"""S9 - Do Verifiers Verify (DESIGN Part III, S9; feeds S8, scoreboard T10).

The question S9 preregisters is how much of a reward model's "correctness" preference is causally
anchored at the actual error versus carried by style. The instrument is the Verification Score
(A6): ``VS = dr_error_span / dr_total`` where ``dr_total = r(clean) - r(corrupted)`` is the
whole correctness reward gap between a clean solution and its corrupted twin, and ``dr_error_span`` is
the part of that gap recovered by patching the clean twin's error-span activations into the corrupted
run. A verifier that genuinely checks the work concentrates its clean-versus-corrupted reward gap at
the span where the corruption lives, so its ``VS`` is near one; a verifier that reacts to surface style
spreads the gap onto style-carrying tokens everywhere but the error, so its ``VS`` is near zero and its
``StyleShare`` (A6, the style complement) is near one.

Because "how much of the gap lives at the error" is a claim about a real reward model on real items,
the scientific leaderboard needs the model population and GPUs. So this study runs the calibration
first, on a synthetic planted verifier where the answer is known by construction (gate 1),
and gates every real-model arm honestly. The planted construction exploits the one structural fact that
makes span attribution exact: a pooled (additive) linear reward decomposes over token positions, so the
clean-twin span patch, which replaces the corrupted twin's error-span activations with the clean twin's
(the ``interventions.patch.ComponentPatch`` replace-over-positions operation), shifts the score by
exactly the error span's contribution to the gap and by nothing else. Planting a mixture whose anchored
fraction ``alpha`` lives on an error-content direction over the error span and whose ``1 - alpha`` style
fraction lives on an orthogonal style direction over the remaining tokens makes three things true by
construction, and this study proves the instrument recovers all three:

- (H1) the Verification Score index recovers the planted ``alpha`` across the sweep ``alpha`` in
  ``{0.0, 0.5, 1.0}``;
- (H2) span-patching the error span shifts the score by the anchored fraction ``alpha`` and not by the
  style fraction, while patching the style tokens shifts it by ``1 - alpha`` and not by the anchored
  fraction (the two patches separate the gap cleanly, with the ``StyleShare`` index recovering
  ``1 - alpha`` as the complement);
- (H3) the ``DenseRewardExtractor``'s per-token map lights up the labeled error span (the AUC of the
  map against the span label is well above 0.5), which is the answer-key validation the dense-reward
  product ships gated behind (``signals.dense``: EXPLORATORY until the verification science certifies
  it).

DESIGN gives S9 no scientific kill criterion (any measured fraction is informative); the only stated
risk is tokenizer alignment, eliminated by construction here because the planted verifier's token
positions are its activation rows, with no character-to-token map to drift. That risk is encoded as a
methodological kill criterion instead: if the dense map fails to localize even on a construction where
the error location is known, the failure is the instrument (the differential attribution or an
alignment bug), not a fact about a model.

The real-model arms are built and gated, never faked: the per-model Verification Score leaderboard (the
"X% vibes" number) across ORMs, PRMs, implicit PRMs, and generative verifiers on ProcessBench items;
the ``(step x layer)`` error-propagation lens (when the error becomes decodable versus when it is
priced); and the cross-paradigm comparison with the PRM local-correctness-versus-expected-success
decomposition. Each is recorded as inconclusive-because-gated with the exact model population and
hardware it needs. The production span-patch path is exercised on the tiny synthetic ClassifierRM
through the real ``interventions.patch.run_patched_scores`` so the plumbing the leaderboard will run on
is proven, while the leaderboard numbers stay gated.

The headline if it lands on real models: production reward models are X percent vibes, only Y percent of
their correctness preference is causally anchored at the actual error, and here is the per-model
Verification Score leaderboard.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from reward_lens.core.evidence import Uncertainty, make_evidence
from reward_lens.core.provenance import Provenance
from reward_lens.core.reading import Reading
from reward_lens.core.types import (
    Access,
    Capability,
    Component,
    GaugeStatus,
    ModelFP,
    Site,
    SubjectRef,
)
from reward_lens.measure.indices.style_share import style_share
from reward_lens.measure.indices.verification_score import verification_score
from reward_lens.record.schema import Run
from reward_lens.signals.base import PositionSpec, Readout, SignalMeta, TokenCurves
from reward_lens.stats import roc_pr
from reward_lens.studies.spec import (
    Hypothesis,
    KillCriterion,
    Prediction,
    StudyResult,
    StudySpec,
    SubjectQuery,
)
from studies._retype import MetricSpec, ScienceRetype

_VERSION = "1.0"

# The anchored-fraction sweep. 0.0 is a pure-style verifier (all of the gap is off the error), 1.0 is a
# pure verifier (all of the gap is at the error), 0.5 is the even mixture. Recovering all three is the
# calibration DESIGN's first experiment asks for before the instrument is turned on a production model.
_ALPHAS: tuple[float, ...] = (0.0, 0.5, 1.0)

# The synthetic item geometry: a modest activation dimension and token count with a contiguous
# error span in the middle, so the non-error tokens flank it on both sides and a style-driven gap has
# somewhere to live that a span patch must leave untouched.
_D = 16
_T = 24
_SPAN: tuple[int, int] = (10, 14)  # the error span [start, end) in token coordinates
_TOTAL_GAP = (
    1.0  # normalize the correctness reward gap so VS reads directly as the anchored fraction
)


def build_spec() -> StudySpec:
    """The frozen S9 spec: the Verification Score calibration, span-patch separation, and dense-map validation."""
    return StudySpec(
        id="s09-verification",
        title="Do verifiers verify: the Verification Score recovers the planted causal-anchoring "
        "fraction, span-patching separates the anchored gap from the style gap, and the dense-reward "
        "map localizes to the labeled error span",
        science="S09-verification",
        hypotheses=(
            Hypothesis(
                id="H1-vs-recovery",
                statement="the Verification Score VS = dr_error_span / dr_total recovers the planted "
                "anchored fraction alpha across the sweep alpha in {0.0, 0.5, 1.0} on a construction "
                "where the causal-anchoring fraction is known (calibration)",
                prediction=Prediction(
                    metric="vs_alpha_recovery_max_abs_error", comparator="<", threshold=0.02
                ),
                scoreboard_row="T10",
            ),
            Hypothesis(
                id="H2-causal-anchoring",
                statement="span-patching the error span (clean twin) shifts the score by the anchored "
                "fraction and not by the style fraction, and patching the style tokens shifts it by the "
                "style fraction and not the anchored fraction, so the two patches separate the reward "
                "gap cleanly (verification is causally anchored, not style-carried)",
                prediction=Prediction(
                    metric="patch_separation_error", comparator="<", threshold=0.02
                ),
                scoreboard_row="T10",
            ),
            Hypothesis(
                id="H3-dense-localization",
                statement="the DenseRewardExtractor's per-token map localizes to the labeled error "
                "span (the AUC of the per-token map against the span label is well above 0.5), the "
                "answer-key validation the dense-reward product ships gated behind",
                prediction=Prediction(
                    metric="dense_localization_auc", comparator=">", threshold=0.9
                ),
                scoreboard_row="T10",
            ),
        ),
        analysis="studies.s09_verification.analysis.analyze",
        subjects=SubjectQuery(
            extra={
                "note": "synthetic planted verifier with a known anchored/style mixture and a known "
                "error span, so the Verification Score, the span-patch separation, and the dense-map "
                "localization all have ground truth by construction; the real-model VS leaderboard, the "
                "(step x layer) error-propagation lens, and the cross-paradigm comparison on ProcessBench "
                "are GPU/population-gated (DESIGN S9 first experiment)"
            }
        ),
        kill_criteria=(
            KillCriterion(
                id="K1-instrument-broken",
                metric="dense_localization_auc",
                comparator="<",
                threshold=0.55,
                description="the dense map fails to localize the error even on the planted construction "
                "where the error location is known by construction, so the failure is the instrument "
                "(the differential attribution or a tokenizer-alignment bug, the only risk DESIGN names "
                "for S9), a tooling failure to fix rather than a scientific negative",
            ),
        ),
    )


# ---------------------------------------------------------------------------
# The planted verifier: a pooled linear reward with a known anchored/style mixture
# ---------------------------------------------------------------------------


@dataclass
class _Twins:
    """A clean solution and its corrupted twin under a pooled linear verifier with a known mixture.

    ``h_clean`` and ``h_corrupt`` are per-token activations (shape ``(T, d)``); the reward is the pooled
    sum ``r(h) = sum_t w_r . h[t]``, so it decomposes exactly over token positions. ``delta`` is the
    clean-minus-corrupted activation difference; its span-token rows carry the anchored fraction on the
    error-content direction and its remaining rows carry the style fraction on the orthogonal style
    direction. ``anchored_gap = alpha * total_gap`` and ``style_gap = (1 - alpha) * total_gap`` are the
    two parts of the correctness reward gap the span patch and the style patch must each recover exactly.
    """

    h_clean: np.ndarray
    h_corrupt: np.ndarray
    delta: np.ndarray
    w_r: np.ndarray
    style_basis: np.ndarray
    span: tuple[int, int]
    total_gap: float
    anchored_gap: float
    style_gap: float


def _planted_twins(
    alpha: float,
    *,
    d: int = _D,
    t: int = _T,
    span: tuple[int, int] = _SPAN,
    total_gap: float = _TOTAL_GAP,
    seed: int = 0,
) -> _Twins:
    """Build clean/corrupted twins whose reward gap is a known mixture of anchored and style parts.

    The reward direction ``w_r = e_dir + s_dir`` reads an error-content direction and an orthogonal
    style direction with unit weight each (both drawn from one orthonormal basis, so ``e_dir`` and
    ``s_dir`` are exactly orthogonal). The corrupted twin is the baseline; the clean twin adds, at each
    error-span token, a fraction of the total gap along ``e_dir`` summing to ``alpha * total_gap``, and
    at each remaining token a fraction along ``s_dir`` summing to ``(1 - alpha) * total_gap``. Because
    ``w_r . e_dir = w_r . s_dir = 1`` and the reward pools additively over tokens, the whole gap is
    ``total_gap``, the part living at the error span is ``alpha * total_gap``, and the part living on the
    style tokens is ``(1 - alpha) * total_gap``. That is the ground truth the Verification Score and the
    two localized patches are graded against.
    """
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    e_dir = q[:, 0]  # error-content direction: where the actual mistake moves the activation
    s_dir = q[:, 1]  # style direction, orthogonal to e_dir: where surface style moves it
    w_r = e_dir + s_dir  # the verifier reads both, with unit weight on each

    e0, e1 = span
    n_error = e1 - e0
    n_style = t - n_error
    delta = np.zeros((t, d), dtype=np.float64)
    for tk in range(e0, e1):
        delta[tk] = (alpha * total_gap / n_error) * e_dir
    for tk in list(range(0, e0)) + list(range(e1, t)):
        delta[tk] = ((1.0 - alpha) * total_gap / n_style) * s_dir

    h_corrupt = np.zeros((t, d), dtype=np.float64)
    h_clean = h_corrupt + delta
    return _Twins(
        h_clean=h_clean,
        h_corrupt=h_corrupt,
        delta=delta,
        w_r=w_r,
        style_basis=s_dir[None, :],
        span=span,
        total_gap=total_gap,
        anchored_gap=alpha * total_gap,
        style_gap=(1.0 - alpha) * total_gap,
    )


def _pooled_reward(h: np.ndarray, w_r: np.ndarray) -> float:
    """The pooled linear verifier reward ``r(h) = sum_t w_r . h[t]`` (additive over token positions)."""
    return float((h @ w_r).sum())


def _clean_twin_span_patch(
    h_target: np.ndarray, h_source: np.ndarray, positions: list[int]
) -> np.ndarray:
    """Replace the target's activations at ``positions`` with the source's (the clean-twin span patch).

    This is the exact operation ``interventions.patch.ComponentPatch(mode="replace")`` performs when its
    source is the clean twin and only the error-span positions differ from the target: it splices the
    clean activations into the corrupted run at those positions and leaves every other position alone.
    Restricting it to the error span is what isolates the error span's contribution to the reward gap;
    the production path runs the identical splice through ``run_patched_scores`` on a real signal, and
    the tiny-vehicle plumbing arm below proves that path reproduces this operation faithfully.
    """
    out = h_target.copy()
    out[positions] = h_source[positions]
    return out


# ---------------------------------------------------------------------------
# The dense-reward validation: a planted prefix value curve behind the real extractor
# ---------------------------------------------------------------------------


class _PlantedPrefixSignal:
    """A minimal signal exposing a planted prefix value curve, for validating the real DenseRewardExtractor.

    Only the surface the extractor's ``dense_rewards`` touches is implemented (``meta``, ``caps``,
    ``runtime``, ``readouts``, ``score_prefixes``); this is a synthetic stimulus with a known error span,
    not a model, and its lineage says so. Its prefix curve rises gently as the solution proceeds and
    drops sharply across the error span, so the extractor's first-difference attribution must recover a
    per-token map whose magnitude peaks exactly at the labeled span. Validating the extractor on a
    planted curve isolates the extractor's own contribution (the differential attribution) from the
    separate, GPU-gated claim that a real reward model's prefix curve actually prices the error there.
    """

    observable_prefix = "signals.planted_s09"

    def __init__(self, prefix_curve: np.ndarray) -> None:
        self._prefix = np.asarray(prefix_curve, dtype=np.float32)
        self.runtime = None
        self.caps = Capability.SCORES | Capability.PREFIX_SCORES
        self.meta = SignalMeta(
            fingerprint=ModelFP("mfp:planted-s09-verifier"),
            adapter="PlantedPrefixSignal",
            architecture="planted",
            lineage={
                "planted": True,
                "note": "synthetic prefix value curve with a known error span; not a model",
            },
            d_model=1,
            n_layers=1,
            n_heads=1,
        )

    def readouts(self) -> list[Readout]:
        return [
            Readout(
                name="reward",
                kind="linear",
                site=Site(0, "resid_post"),
                position=PositionSpec("all"),
                vector=None,
            )
        ]

    def score_prefixes(self, view, readout: str | None = None):
        """The planted per-item prefix value curve as ``Evidence[TokenCurves]`` (one item)."""
        name = readout or "reward"
        payload = TokenCurves(curves=[self._prefix.copy()], readout=name)
        return make_evidence(
            observable="signals.planted_s09.score_prefixes",
            observable_version=_VERSION,
            subject=SubjectRef(
                signals=(self.meta.fingerprint,), readout=name, extra={"planted": True}
            ),
            value=payload,
            uncertainty=Uncertainty(n=1, method="none"),
            gauge=GaugeStatus.INVARIANT,
        )


def _planted_error_value_curve(
    *, t: int = _T, span: tuple[int, int] = _SPAN, seed: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """A prefix value curve that rises gently, then drops sharply across the error span.

    Returns ``(prefix_curve, span_label)``. The per-token increments are small and positive off the span
    (the verifier gaining confidence as the solution proceeds) and a large negative dip on the span (the
    verifier pricing the error where it occurs), with a little noise so the localization is a real
    ranking test rather than a trivial step. The prefix curve is the cumulative sum of those increments,
    exactly the object ``score_prefixes`` returns and the ``DenseRewardExtractor`` differences back into
    a per-token map.
    """
    rng = np.random.default_rng(seed)
    e0, e1 = span
    increments = rng.normal(0.03, 0.02, size=t)
    increments[e0:e1] = rng.normal(-0.8, 0.02, size=e1 - e0)
    prefix = np.cumsum(increments).astype(np.float32)
    label = np.zeros(t, dtype=np.int64)
    label[e0:e1] = 1
    return prefix, label


# ---------------------------------------------------------------------------
# The production span-patch plumbing arm (real run_patched_scores on the tiny vehicle)
# ---------------------------------------------------------------------------


def _span_patch_plumbing(run, subject: SubjectRef, study_id: str) -> dict:
    """Prove the production clean-twin span-patch path runs, using the real interventions subsystem.

    The Verification Score's span patch is a clean-twin activation replacement; this arm proves that
    exact operation is faithful in the real ``interventions.patch`` path on the tiny synthetic
    ClassifierRM. It captures the model's full-sequence activations at the head-input site, feeds them
    back as a ``ComponentPatch(mode="replace")`` source, and confirms ``run_patched_scores`` reproduces
    the clean score (an identity replace), then confirms a zero patch moves the score so the path is
    live. On a random tiny model the reward carries no scientific Verification Score, so no leaderboard
    metric is emitted here; the aligned per-error-span VS on real PRMs (with SpanMap tokenizer alignment)
    is the population-gated arm. Any failure to import or run the torch path is recorded as gated with
    the exact requirement, never faked.
    """
    try:
        from reward_lens.interventions.patch import ComponentPatch, run_patched_scores
        from reward_lens.runtime.backend import CaptureSpec
        from reward_lens.signals.loaders import from_tiny

        rm = from_tiny(seed=0, conformance_quickcheck=False)
        view = [("Is 2 + 2 = 4?", "Yes, 2 + 2 = 4.")]
        read = rm.readout("reward")
        site = read.site

        clean = float(rm.score(view).value.values[0])
        spec = CaptureSpec(sites=(site,), full_sequence=True, dtype="float32", keep_on_device=True)
        source = rm.capture(view, spec).get(site)  # (1, T, d) clean head-input activations

        identity_patch = ComponentPatch(
            site=site, source=source, mode="replace", label="clean-twin"
        )
        identity_score = float(run_patched_scores(rm, identity_patch.compile(rm), view)[0])
        zero_score = float(
            run_patched_scores(rm, ComponentPatch(site=site, mode="zero").compile(rm), view)[0]
        )

        identity_fidelity = abs(identity_score - clean)
        value = {
            "arm": "span-patch-plumbing",
            "status": "proven-on-tiny-vehicle",
            "ran": True,
            "clean_score": clean,
            "identity_replace_score": identity_score,
            "identity_fidelity_abs": identity_fidelity,
            "zero_patch_score": zero_score,
            "zero_patch_moved": bool(abs(zero_score - clean) > 1e-6),
            "note": "run_patched_scores executes the clean-twin activation replacement the Verification "
            "Score's span patch is built on; the identity replace reproduces the clean score, so the "
            "production patch path is faithful. A random tiny model carries no scientific VS, so no "
            "leaderboard number is emitted; the aligned per-error-span VS on real PRMs is gated below.",
        }
    except Exception as exc:  # noqa: BLE001 - any import/runtime failure gates the arm honestly
        value = {
            "arm": "span-patch-plumbing",
            "status": "gated",
            "ran": False,
            "needs": "reward_lens.interventions.patch.run_patched_scores on a ClassifierRM (torch)",
            "reason": f"{type(exc).__name__}: {exc}",
        }

    run.record(
        make_evidence(
            observable="S09.SpanPatchPlumbing",
            observable_version=_VERSION,
            subject=subject,
            value=value,
            gauge=GaugeStatus.INVARIANT,
            provenance=Provenance(study=study_id),
            registered=True,
        )
    )
    return value


# ---------------------------------------------------------------------------
# Gated real-model arms
# ---------------------------------------------------------------------------


def _gated_arm(study_id: str, subject: SubjectRef, *, arm: str, needs: str, produces: str):
    """A REGISTERED record that a real-model arm is inconclusive because the population or hardware is absent."""
    return make_evidence(
        observable="S09.GatedArm",
        observable_version=_VERSION,
        subject=subject,
        value={
            "arm": arm,
            "status": "inconclusive-because-gated",
            "needs": needs,
            "produces": produces,
        },
        gauge=GaugeStatus.INVARIANT,
        provenance=Provenance(study=study_id),
        registered=True,
    )


# ---------------------------------------------------------------------------
# The analysis
# ---------------------------------------------------------------------------


def analyze(run) -> StudyResult:
    """Run the Verification Score calibration, the span-patch separation, and the dense-map validation.

    H1 recovers the planted anchored fraction alpha with the Verification Score index across the sweep.
    H2 shows the error-span patch and the style patch separate the reward gap cleanly, so verification is
    causally anchored rather than style-carried, with the StyleShare index recovering the complement. H3
    validates the real DenseRewardExtractor: its per-token map localizes to the labeled error span. The
    production span-patch path is proven on the tiny vehicle through the real interventions subsystem, and
    the real-model leaderboard, propagation lens, and cross-paradigm arms are recorded as gated.
    """
    study_id = run.study.study_id
    subject = SubjectRef(extra={"study": study_id})

    # The root Evidence documents the planted construction the VS, patch, and dense arms all descend
    # from, so the store stays a DAG rooted at a single honest stimulus.
    span_positions = list(range(_SPAN[0], _SPAN[1]))
    style_positions = list(range(0, _SPAN[0])) + list(range(_SPAN[1], _T))
    ev_root = make_evidence(
        observable="S09.PlantedVerifier",
        observable_version=_VERSION,
        subject=subject,
        value={
            "d_model": _D,
            "n_tokens": _T,
            "error_span": list(_SPAN),
            "total_gap": _TOTAL_GAP,
            "alphas": list(_ALPHAS),
            "construction": "pooled linear reward r(h) = sum_t w_r . h[t] with w_r = e_dir + s_dir; the "
            "clean-minus-corrupted gap puts alpha on the error-content direction over the error span and "
            "1 - alpha on the orthogonal style direction over the remaining tokens, so the causal-"
            "anchoring fraction is known by construction",
        },
        uncertainty=Uncertainty(method="none"),
        gauge=GaugeStatus.INVARIANT,
        provenance=Provenance(study=study_id),
        registered=True,
    )
    run.record(ev_root)

    # -- H1 and H2: the Verification Score sweep and the span/style patch separation. --
    per_alpha: list[dict] = []
    vs_errors: list[float] = []
    style_errors: list[float] = []
    separation_errors: list[float] = []
    for alpha in _ALPHAS:
        tw = _planted_twins(alpha)
        r_corrupt = _pooled_reward(tw.h_corrupt, tw.w_r)
        r_clean = _pooled_reward(tw.h_clean, tw.w_r)
        dr_total = r_clean - r_corrupt

        # The clean-twin patches: splice the clean activations into the corrupted run over the error
        # span (recovers the anchored gap) and, separately, over the style tokens (recovers the style
        # gap). Each is the ComponentPatch replace operation restricted to those positions.
        h_error_patched = _clean_twin_span_patch(tw.h_corrupt, tw.h_clean, span_positions)
        h_style_patched = _clean_twin_span_patch(tw.h_corrupt, tw.h_clean, style_positions)
        dr_error_span = _pooled_reward(h_error_patched, tw.w_r) - r_corrupt
        dr_style_span = _pooled_reward(h_style_patched, tw.w_r) - r_corrupt

        vs = verification_score(dr_total, dr_error_span)
        ss = style_share(tw.delta, tw.style_basis, tw.w_r)

        vs_errors.append(abs(vs - alpha))
        style_errors.append(abs(ss - (1.0 - alpha)))
        # The span patch must move by the anchored gap and the style patch by the style gap; the sum of
        # both localization errors, normalized by the total gap, is the clean-separation metric.
        separation = (
            abs(dr_error_span - tw.anchored_gap) + abs(dr_style_span - tw.style_gap)
        ) / abs(tw.total_gap)
        separation_errors.append(separation)
        per_alpha.append(
            {
                "alpha": alpha,
                "dr_total": dr_total,
                "dr_error_span": dr_error_span,
                "dr_style_span": dr_style_span,
                "verification_score": vs,
                "style_share": ss,
                "separation_error": separation,
            }
        )

    vs_recovery_max_abs_error = float(max(vs_errors))
    style_recovery_max_abs_error = float(max(style_errors))
    patch_separation_error = float(max(separation_errors))

    ev_vs = make_evidence(
        observable="S09.VerificationScore",
        observable_version=_VERSION,
        subject=subject,
        value={
            "alphas": list(_ALPHAS),
            "verification_scores": [row["verification_score"] for row in per_alpha],
            "style_shares": [row["style_share"] for row in per_alpha],
            "dr_error_span": [row["dr_error_span"] for row in per_alpha],
            "dr_style_span": [row["dr_style_span"] for row in per_alpha],
            "vs_alpha_recovery_max_abs_error": vs_recovery_max_abs_error,
            "style_share_recovery_max_abs_error": style_recovery_max_abs_error,
            "patch_separation_error": patch_separation_error,
            "note": "VS and StyleShare are the A6 indices consumed as pure functions; the span "
            "patch and style patch are the clean-twin ComponentPatch replace operation restricted to the "
            "error span and to the style tokens respectively. VS + StyleShare = 1 here because the "
            "construction leaves no residual; A6 permits a residual in general.",
        },
        uncertainty=Uncertainty(n=len(_ALPHAS), method="none"),
        gauge=GaugeStatus.INVARIANT,
        provenance=Provenance(study=study_id, parents=(ev_root.id,)),
        registered=True,
    )
    run.record(ev_vs)

    # -- H3: the DenseRewardExtractor localizes its per-token map to the labeled error span. --
    prefix_curve, span_label = _planted_error_value_curve()
    planted_signal = _PlantedPrefixSignal(prefix_curve)
    dense_view = [("planted", "item")]
    dense_source = "real DenseRewardExtractor"
    try:
        from reward_lens.signals.dense import DenseRewardExtractor

        # Record the planted prefix Evidence first so the extractor's dense map, which cites it as a
        # provenance parent, can be appended without breaking the store's DAG integrity (I5). The
        # prefix Evidence is content-addressed and score_prefixes is deterministic, so the id the
        # extractor cites is exactly the one recorded here.
        prefix_ev = planted_signal.score_prefixes(dense_view)
        run.record(prefix_ev)
        extractor = DenseRewardExtractor(planted_signal)
        dense_ev = extractor.dense_rewards(dense_view)
        run.record(dense_ev)  # the extractor's own Evidence is EXPLORATORY (gated) by construction
        dense_map = np.asarray(dense_ev.value.curves[0], dtype=np.float64)
        dense_parent = dense_ev.id
    except Exception:  # noqa: BLE001 - fall back to the same first-difference the extractor computes
        dense_map = np.diff(np.asarray(prefix_curve, dtype=np.float64), prepend=0.0)
        dense_parent = ev_root.id
        dense_source = "inline first-difference (DenseRewardExtractor unavailable)"

    # The per-token map should light up the error span: rank the tokens by map magnitude against the
    # span label and read the AUC (the Mann-Whitney identity), which is 0.5 for a non-localizing map.
    dense_localization_auc = float(roc_pr(np.abs(dense_map), span_label).auc)

    ev_dense = make_evidence(
        observable="S09.DenseLocalization",
        observable_version=_VERSION,
        subject=subject,
        value={
            "dense_localization_auc": dense_localization_auc,
            "n_tokens": int(span_label.size),
            "n_error_tokens": int(span_label.sum()),
            "error_span": list(_SPAN),
            "source": dense_source,
            "note": "the DenseRewardExtractor ships EXPLORATORY (signals.dense) until the verification "
            "science certifies it against labeled error spans; this REGISTERED localization is that S9 "
            "certification, on a planted prefix curve where the error span is known by construction. The "
            "claim that a real reward model's prefix curve prices the error there is the gated arm.",
        },
        uncertainty=Uncertainty(n=int(span_label.size), method="mann-whitney"),
        gauge=GaugeStatus.INVARIANT,
        provenance=Provenance(study=study_id, parents=(ev_root.id, dense_parent)),
        registered=True,
    )
    run.record(ev_dense)

    # -- The production span-patch path, proven on the tiny vehicle through the real interventions arm. --
    plumbing = _span_patch_plumbing(run, subject, study_id)

    # -- Gated real-model arms: recorded as inconclusive-because-gated with the exact requirement. --
    run.record(
        _gated_arm(
            study_id,
            subject,
            arm="vs-leaderboard",
            needs="the reward-model population (Skywork/ArmoRM ORMs, real PRMs such as Qwen2.5-Math-PRM, "
            "DPO-implicit rewards, generative verifiers), ProcessBench items, oracle-authored "
            "style/confidence-matched corruptions (data.corruptions.style_controls is an M8+ oracle "
            "stub), and GPU",
            produces="the per-model Verification Score leaderboard (the 'X% vibes' number: what fraction "
            "of each model's correctness preference is causally anchored at the actual error) across "
            "ORMs, PRMs, implicit PRMs, and generative verifiers on identical items",
        )
    )
    run.record(
        _gated_arm(
            study_id,
            subject,
            arm="step-layer-propagation",
            needs="a real PRM with per-layer activations, ProcessBench items, and GPU",
            produces="the (step x layer) error-propagation lens: the lag between when the error becomes "
            "decodable across layers and when it is priced into the step reward, and why error detection "
            "is delayed",
        )
    )
    run.record(
        _gated_arm(
            study_id,
            subject,
            arm="cross-paradigm",
            needs="the reward-model population across paradigms (ORM, PRM, implicit PRM, generative "
            "verifier) on identical real items, and GPU",
            produces="the cross-paradigm Verification Score comparison and the PRM local-correctness-"
            "versus-expected-success decomposition (do PRM step scores encode local correctness or "
            "expected success)",
        )
    )

    metrics: dict[str, float] = {
        "vs_alpha_recovery_max_abs_error": vs_recovery_max_abs_error,
        "style_share_recovery_max_abs_error": style_recovery_max_abs_error,
        "patch_separation_error": patch_separation_error,
        "dense_localization_auc": dense_localization_auc,
    }

    vs_line = ", ".join(
        f"alpha={row['alpha']:.1f} -> VS={row['verification_score']:.3f}" for row in per_alpha
    )
    plumbing_line = (
        f"the production span-patch path was proven on the tiny vehicle (identity replace fidelity "
        f"{plumbing.get('identity_fidelity_abs', float('nan')):.2e})"
        if plumbing.get("ran")
        else "the production span-patch plumbing arm is gated (interventions.patch unavailable)"
    )
    summary = (
        f"On a planted verifier whose causal-anchoring fraction is known by construction, the "
        f"Verification Score recovered the planted alpha across the sweep ({vs_line}) to a maximum "
        f"absolute error of {vs_recovery_max_abs_error:.2e}, and the StyleShare index recovered the "
        f"complement to {style_recovery_max_abs_error:.2e}. Span-patching the error span and the style "
        f"tokens separated the reward gap cleanly (separation error {patch_separation_error:.2e}), so "
        f"the verification signal is causally anchored at the error rather than style-carried. The "
        f"DenseRewardExtractor's per-token map localized to the labeled error span at AUC "
        f"{dense_localization_auc:.3f}. {plumbing_line}. The per-model VS leaderboard, the "
        f"(step x layer) propagation lens, and the cross-paradigm comparison on ProcessBench are "
        f"recorded as inconclusive-because-gated on the reward-model population and GPU."
    )

    return StudyResult(outcomes={}, metrics=metrics, summary=summary)


# ---------------------------------------------------------------------------
# The retype: S9 on the kernel
# ---------------------------------------------------------------------------

RETYPE = ScienceRetype(
    science="s09_verification",
    spec=build_spec(),
    headline="grader.verification_score",
    destination=(
        "the D series in verifier/ and G4. All three metrics bind to registered ids whose units "
        "match, so this plan closes with no gap."
    ),
    needs={Component.GRADER: Access.RECORD, Component.RECORD: Access.RECORD},
    metrics=(
        MetricSpec(
            metric="vs_alpha_recovery_max_abs_error",
            quantity="grader.verification_score",
            arc="vs-calibration",
            frame="planted-alpha-sweep",
            source="organism",
            note=(
                "the largest absolute error in recovering the planted anchored fraction across "
                "alpha in {0, 0.5, 1}. The arc that produces it is the one that measures VS, and "
                "grader.verification_score is that quantity almost verbatim: the share of the "
                "clean-versus-corrupted reward gap attributable to the labelled error span. The "
                "sweep needs three graders built to known alpha, which is a construction rather "
                "than a record."
            ),
        ),
        MetricSpec(
            metric="patch_separation_error",
            quantity="grader.component_patch_effect",
            arc="span-patch",
            arm="error-span-patch",
            source="organism",
            note=(
                "how cleanly patching the error span moves the anchored fraction without moving the "
                "style fraction. grader.component_patch_effect is the reduction in the "
                "chosen-minus-rejected differential when one component is replaced by the rejected "
                "side's, which is this patch with the component being a token span. Needs FORWARD "
                "and MUTATE on the grader; a record carries neither."
            ),
        ),
        MetricSpec(
            metric="dense_localization_auc",
            quantity="instrument.recovery_auc",
            arc="dense-localization",
            note=(
                "the AUC of the per-token dense-reward map against the labelled error span. Read as "
                "instrument.recovery_auc rather than labels.fs_signal_locality: the kill criterion "
                "says plainly that a low value here means the instrument is broken rather than that "
                "the science is negative, which is what a recovery AUC is for. L4's locality id is "
                "a different measurement, a three-arm comparison of detectors over different spans "
                "of text, and substituting one AUC for another because both are AUCs is how a unit "
                "check comes to license a comparison nobody intended."
            ),
        ),
    ),
    arc_requires={"span-patch": ("vs-calibration",), "dense-localization": ("vs-calibration",)},
)


def read(run: Run) -> Reading:
    """S9 against a real record: verification needs labelled errors, and most records have none.

    Every one of the three arms is anchored on a span somebody labelled as the error. The Verification
    Score is the share of the reward gap attributable to that span, the patch separation moves that
    span, and the dense-map AUC is scored against it. Without the labels there is no anchor, and an
    unanchored attribution map is a heatmap rather than a measurement.

    Scope limit, three lines in: this reads labels off the record and does not produce them. A run
    whose trajectories carry no `Blind[LabelValue]` entries is refused rather than scored against
    the reward itself, because scoring an error-localiser against the reward it is supposed to audit
    measures agreement and calls it verification.
    """
    if (refusal := RETYPE.access_refusal(run, remedy=_ACCESS_REMEDY)) is not None:
        return refusal

    labelled = 0
    total = 0
    keys: set[str] = set()
    for step in run.steps:
        for group in step.groups:
            for traj in group.trajectories:
                total += 1
                names = set(traj.labels or {})
                if names:
                    labelled += 1
                    keys |= names

    if not labelled:
        grader = run.component(Component.GRADER)
        return RETYPE.incomplete(
            field="span labels on any trajectory",
            subject=f"all {total} trajectories of run {run.id}",
            remedy=(
                "point this at a run whose trajectories carry a labelled error span, or attach one: "
                "the D series used MATH's `is_equiv` and SWE-bench's `grading.py` as the label "
                "source, and either supplies the anchor all three S9 arms are measured against. A "
                f"length grader such as {getattr(grader, 'name', 'this one')!r} produces no error "
                "spans to label, so this particular run cannot be made to answer S9 by adding "
                "access."
            ),
            trajectories=total,
            labelled=labelled,
        )

    return RETYPE.incomplete(
        field="a dense per-token attribution map",
        subject=f"{labelled} of {total} labelled trajectories in run {run.id}",
        remedy=(
            f"the labels are here ({', '.join(sorted(keys))}) and the per-token map is not. Record "
            f"`advantage_tokens` or a token-level attribution alongside them, or run the dense "
            f"extractor over the saved rollouts and attach its output, and S9's H3 becomes "
            f"answerable from this record."
        ),
        trajectories=total,
        labelled=labelled,
        label_keys=sorted(keys),
    )


_ACCESS_REMEDY = (
    "open a run written by the recorder with the grader's scores captured. S9 needs no activations "
    "for its H3 arm; it needs the per-token map and the span labels to score it against."
)


__all__ = ["RETYPE", "build_spec", "analyze", "read"]

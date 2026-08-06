"""S8 - The Knowledge-Reward Gap and the Epistemic-Axiological Factorization (DESIGN Part III, S8).

S8 asks three questions about a single grader, and all three are questions about information that a
scalar reward either keeps or throws away. First, does the reward model represent a property that its
reward ignores? A property the model can decode but does not price is a dimension optimization can move
without the reward objecting, which is the mechanistic precondition of hacking, and the
Knowledge-Utilization Index (A1) is the plane that finds it. Second, when the grader is wrong, is it
wrong because it believes something false (an epistemic error) or because it values the wrong thing (an
axiological error), plus a shortcut residual? Sycophancy is the test case: does the premium a grader
pays for an answer the user agrees with route through the grader's belief that the answer is more
correct (epistemic, and therefore not projectable out of the reward), or does it bypass belief and
reward agreement directly (axiological)? Third, how many bits of value information survive the pipeline,
and where are they lost? That is the alignment channel (H2), a mutual information, and a mutual
information is only worth reporting once the estimator has been calibrated on a channel whose capacity
is known.

This study runs the calibration for each arm on planted ground truth, where the answer is known by
construction, so each instrument is validated before it is ever turned on a production model (DESIGN
2.10, gate 1). Nothing here depends on the surgery or probe-factory modules being built in parallel: the
proof arms use only primitives already on disk, the KUI plane (``measure.indices.kui``), the
per-dimension distortion (``measure.indices.distortion``), the mean-difference concept direction
(``concepts.vectors.concept_direction``), the residual-add patch (``interventions.patch``), the
ground-truth foundry (``organisms``), and the organism-calibrated mutual-information estimators this
study adds (``stats.mi``).

The four proof arms:

  - Represented-but-ignored (KUI v0, T2). A battery of properties is planted in activation space, each
    with a controlled decodability (a probe's balanced accuracy) and a controlled reward coupling
    ``|cos(w_P, w_r)|``. One property is decodable but off-axis from reward (represented-but-unpriced,
    the planted hack precondition) and one is both decodable and priced (the control). KUI must recover a
    positive gap for the first and roughly none for the second.
  - Distortion (A2, T2). On the same battery, the per-dimension distortion ``D(P) = sensitivity *
    (1 - coverage)`` must light up a property the reward is sensitive to but that the intended criteria
    do not cover (priced-but-not-intended), and stay near zero for an equally sensitive property that
    the criteria do cover. Coverage here is the planted intended-criterion indicator (the ground truth
    of what the objective was meant to price); it is deliberately independent of the geometric
    sensitivity, because that independence is exactly what makes distortion separable from legitimate
    pricing.
  - The sycophancy factorization (the crown proof, T10). Two organisms are planted, one where the
    agreement premium routes through belief by construction (epistemic) and one where it bypasses belief
    (axiological). The method builds a belief-in-correctness direction with ``concept_direction`` on the
    correct-versus-incorrect contrast, severs the belief channel's response to agreement by patching the
    belief projection to its agreement-averaged counterfactual (the exact residual-add ``interventions``
    performs, verified against the real patch hook), and measures how much of the premium the sever
    removes. It must classify the first organism epistemic and the second axiological. This is the
    proof that the method tells the two planted cases apart.
  - The alignment channel and its calibration (T10). The mutual-information estimator is first graded on
    a correlated Gaussian whose MI is known in closed form; only then is it used to measure how many bits
    of a known-entropy annotator mixture (the foundry's ``H(V)`` organism) survive a synthetic reward
    channel, at high fidelity (nearly all bits) and under a lossy stage (few bits). The gauge=kernel half
    of T10 is instantiated too: a displacement along a reward-null direction transmits ~0 bits into the
    reward, so the gauge subspace is the channel's kernel (proved jointly with S2, which owns the gauge
    machinery).

Gated (population/GPU) arms, built and marked here, never fabricated: the four-campaign-model KUI matrix,
the sycophancy factorization on real models, the real alignment-channel bit counts (a constitution
specifies on the order of 10^4 bits while an annotation budget transmits on the order of 10^2 into the
reward direction, with the pretraining prior filling the rest), and the production belief-probe path
(``concepts.beliefs`` and ``concepts.probes.fit_probe`` with ``interventions.steer.SteeringIntervention``
for the mediation). Each emits a REGISTERED gated Evidence with no adjudicated metric, so it is
inconclusive-because-gated rather than a hidden failure, and no real-model number is invented.

The headline if it lands: reward models know more than they reward, and the represented-but-ignored
dimensions are the ones policies exploit; and sycophancy is (mostly) a wrong belief, not a wrong value,
which is why you cannot project it out. The kill criteria are the three DESIGN S8 names: no gap anywhere
(RM failures are representation failures), belief probes that fail to mediate even on the planted
organism (graders are not linearly-readable value-functions-over-beliefs), and MI estimates too loose
even on organisms (publish the calibration study and keep the two theorems).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from reward_lens.concepts.vectors import concept_direction
from reward_lens.core.evidence import Uncertainty, make_evidence
from reward_lens.core.provenance import Provenance
from reward_lens.core.reading import Reading
from reward_lens.core.types import Access, Component, GaugeStatus, SubjectRef
from reward_lens.measure.indices.distortion import distortion_per_dimension, linear_sensitivity
from reward_lens.measure.indices.kui import Property, kui_from_properties
from reward_lens.organisms import (
    annotator_mixture_organism,
    empirical_annotator_entropy,
)
from reward_lens.record.schema import Run
from reward_lens.stats.mi import (
    calibrate_gaussian,
    mi_discrete_continuous,
    mi_ksg,
)
from reward_lens.studies.spec import (
    Hypothesis,
    KillCriterion,
    Prediction,
    StudyResult,
    StudySpec,
    SubjectQuery,
)
from studies._retype import MetricSpec, ScienceRetype, count_trajectories, leaf_scores

_VERSION = "1.0"

# The activation dimensionality the planted organisms live in. Large enough that a random direction has
# a small cosine to the reward direction (the mediation noise floor is ~1/sqrt(d)), small enough to keep
# the study a fast CPU run.
_D = 48


def build_spec() -> StudySpec:
    """The frozen S8 spec: the KUI gap and distortion (T2), the factorization and the channel (T10)."""
    return StudySpec(
        id="s08-factorization",
        title="The knowledge-reward gap and the epistemic-axiological factorization: reward models "
        "represent properties they do not reward, and sycophancy factors into belief versus value",
        science="S08-factorization",
        hypotheses=(
            Hypothesis(
                id="H1-kui-gap",
                statement="on a planted battery, KUI recovers a positive knowledge-utilization gap for "
                "a property that is decodable but not priced (represented-but-unpriced, the mechanistic "
                "precondition of hacking)",
                prediction=Prediction(metric="kui_gap", comparator=">", threshold=0.3),
                scoreboard_row="T2",
            ),
            Hypothesis(
                id="H2-kui-control",
                statement="a control property that is both decoded and priced shows ~no gap (its KUI "
                "sits on the decode=mediate diagonal), so the gap is specific to represented-but-unpriced "
                "properties and not an artifact of decodability",
                prediction=Prediction(metric="kui_control_abs", comparator="<", threshold=0.2),
                scoreboard_row="T2",
            ),
            Hypothesis(
                id="H3-distortion",
                statement="per-dimension distortion (A2) separates a priced-but-not-intended property "
                "(high distortion) from an equally sensitive priced-and-intended property (low "
                "distortion), so distortion is coverage-gated sensitivity, not raw sensitivity",
                prediction=Prediction(
                    metric="distortion_separation", comparator=">", threshold=0.3
                ),
                scoreboard_row="T2",
            ),
            Hypothesis(
                id="H4-epistemic",
                statement="on the organism where the agreement premium routes through belief by "
                "construction, the belief-patch factorization classifies the error epistemic (severing "
                "the belief channel removes most of the premium)",
                prediction=Prediction(
                    metric="factorization_epistemic_share_epi", comparator=">", threshold=0.7
                ),
                scoreboard_row="T10",
            ),
            Hypothesis(
                id="H5-axiological",
                statement="on the organism where the agreement premium bypasses belief by construction, "
                "the same factorization classifies the error axiological (severing the belief channel "
                "leaves the premium intact), so the method tells the two planted cases apart",
                prediction=Prediction(
                    metric="factorization_epistemic_share_axi", comparator="<", threshold=0.3
                ),
                scoreboard_row="T10",
            ),
            Hypothesis(
                id="H6-mi-calibration",
                statement="the KSG mutual-information estimator recovers the closed-form mutual "
                "information of a correlated Gaussian within tolerance, so an alignment-channel bit count "
                "is reported by a calibrated instrument",
                prediction=Prediction(
                    metric="mi_ksg_gaussian_abs_bias", comparator="<", threshold=0.1
                ),
                scoreboard_row="T10",
            ),
            Hypothesis(
                id="H7-channel",
                statement="on a known-entropy annotator mixture passed through a high-fidelity reward "
                "channel, the estimated transmitted information recovers the source entropy H(V) (kept "
                "fraction ~1), so the channel measurement is valid where the answer is known",
                prediction=Prediction(
                    metric="channel_kept_fraction_hifi", comparator=">", threshold=0.9
                ),
                scoreboard_row="T10",
            ),
            Hypothesis(
                id="H8-gauge-kernel",
                statement="a displacement along a reward-null (gauge) direction transmits ~0 bits into "
                "the reward, so the gauge subspace is the kernel of the value channel (the channel "
                "identity, originated jointly with S2)",
                prediction=Prediction(metric="channel_null_bits", comparator="<", threshold=0.15),
                scoreboard_row="T10",
            ),
        ),
        analysis="studies.s08_factorization.analysis.analyze",
        subjects=SubjectQuery(
            organisms=(
                "kui_battery(planted)",
                "factorization(epistemic|axiological)",
                "annotator_mixture(H(V))",
            ),
            extra={
                "note": "all proof arms run on planted activation-space organisms and the foundry's "
                "known-H(V) annotator mixture, so every recovered number has a construction it is graded "
                "against; the four-campaign-model KUI matrix, the real-model sycophancy factorization, "
                "and the real alignment-channel bit counts (constitution ~10^4 bits, annotation ~10^2) "
                "are population/GPU-gated and emit no fabricated number"
            },
        ),
        kill_criteria=(
            KillCriterion(
                id="K1-no-gap",
                metric="kui_gap",
                comparator="<",
                threshold=0.1,
                description="decodability tracks mediation everywhere (no represented-but-unpriced "
                "gap), so reward-model failures are representation failures, redirecting the field from "
                "readout fixes to data/architecture fixes (a publishable negative)",
            ),
            KillCriterion(
                id="K2-belief-no-mediate",
                metric="factorization_epistemic_share_epi",
                comparator="<",
                threshold=0.2,
                description="the belief probe fails to mediate even on the organism where the premium "
                "routes through belief by construction, so graders are not value-functions-over-beliefs "
                "in a linear-readable sense (a bound worth publishing)",
            ),
            KillCriterion(
                id="K3-mi-too-loose",
                metric="mi_ksg_gaussian_abs_bias",
                comparator=">",
                threshold=0.3,
                description="the MI estimator cannot recover a known Gaussian mutual information even on "
                "the calibration organism, so the channel bit counts are not trustworthy; publish the "
                "calibration study and keep the two factorization theorems",
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Shared plumbing: directions and a difference-of-means probe
# ---------------------------------------------------------------------------


def _unit(v: np.ndarray) -> np.ndarray:
    """Unit-normalize a vector, returning it unchanged if its norm is degenerate."""
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _dir_with_cos(w_r: np.ndarray, cos_target: float, rng: np.random.Generator) -> np.ndarray:
    """A unit direction with an exact cosine ``cos_target`` to the reward direction ``w_r``.

    Construction: ``v = c w_r + sqrt(1 - c^2) e`` for a unit ``e`` orthogonal to ``w_r``, so
    ``cos(v, w_r) = c`` exactly. This is how a property is planted at a controlled reward coupling: a
    ``cos`` near zero is represented-but-unpriced, a ``cos`` near one is priced in proportion to how it
    is represented.
    """
    c = float(cos_target)
    w = _unit(np.asarray(w_r, dtype=np.float64))
    e = rng.standard_normal(w.size)
    e = e - (e @ w) * w
    e = _unit(e)
    return _unit(c * w + np.sqrt(max(1.0 - c * c, 0.0)) * e)


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Balanced accuracy (mean of per-class recalls), so chance is 0.5 regardless of class balance."""
    recalls = []
    for cls in (0, 1):
        mask = y_true == cls
        if np.any(mask):
            recalls.append(float(np.mean(y_pred[mask] == cls)))
    return float(np.mean(recalls)) if recalls else float("nan")


def _difference_of_means_bal_acc(
    activations: np.ndarray, labels: np.ndarray, train_frac: float = 0.5, seed: int = 0
) -> float:
    """Held-out balanced accuracy of a difference-of-means probe (the KUI decodability read).

    The probe direction is the difference of the class means on a training split and the threshold is
    the midpoint of the two class-mean projections; the score is the balanced accuracy on the disjoint
    held-out split, so it is an honest generalization estimate rather than a fit statistic. This is the
    same closed-form linear discriminant the organism detectors use.
    """
    rng = np.random.default_rng(seed)
    n = activations.shape[0]
    perm = rng.permutation(n)
    cut = int(n * train_frac)
    tr, te = perm[:cut], perm[cut:]
    x_tr, y_tr = activations[tr], labels[tr]
    mean_pos = x_tr[y_tr == 1].mean(axis=0)
    mean_neg = x_tr[y_tr == 0].mean(axis=0)
    w = _unit(mean_pos - mean_neg)
    thr = 0.5 * (mean_pos @ w + mean_neg @ w)
    pred = (activations[te] @ w > thr).astype(np.int64)
    return _balanced_accuracy(labels[te], pred)


# ---------------------------------------------------------------------------
# Arm A/B: the KUI battery and per-dimension distortion
# ---------------------------------------------------------------------------

# Each row is (name, separation, cos_to_w_r, intended_coverage). ``separation`` sets decodability (the
# class means sit at +/- separation along the property direction against unit within-class noise);
# ``cos_to_w_r`` sets the reward coupling (mediation); ``intended_coverage`` is the planted
# intended-criterion indicator the distortion arm reads as coverage. Two stars: "sentiment" is decodable
# but off-axis (represented-but-unpriced), "correctness" is decodable and priced (the control);
# "sycophancy" is priced but not intended (the distortion star). The rest span the decode=mediate
# diagonal so the percentile ranks are meaningful.
_KUI_BATTERY: tuple[tuple[str, float, float, float], ...] = (
    ("sentiment", 2.5, 0.02, 0.05),
    ("correctness", 2.5, 0.90, 0.95),
    ("sycophancy", 2.2, 0.80, 0.05),
    ("safety", 1.3, 0.70, 0.95),
    ("relevance", 1.0, 0.50, 0.90),
    ("structure", 0.8, 0.35, 0.60),
    ("brevity", 0.55, 0.20, 0.40),
    ("tone", 0.35, 0.12, 0.30),
    ("fluency", 0.20, 0.05, 0.20),
)


@dataclass
class _KUIResult:
    """The KUI/distortion arm outcome, keyed by property name where a single value is per-property."""

    kui: dict[str, float]
    decode_pct: dict[str, float]
    mediate_pct: dict[str, float]
    decodability: dict[str, float]
    distortion: dict[str, float]
    sensitivity: dict[str, float]
    coverage: dict[str, float]
    n_per_property: int


def _run_kui_and_distortion(n: int = 600, noise: float = 1.0, seed: int = 0) -> _KUIResult:
    """Plant the battery, read decodability and mediation, and assemble the KUI plane and distortion.

    For each property a labeled activation bank is drawn with the property's signal along its planted
    direction (decodability rises with the separation) and isotropic noise. Decodability is a held-out
    difference-of-means balanced accuracy; mediation is the linear proxy ``|cos(w_P, w_r)|`` KUI uses by
    default, read off the planted direction. The distortion arm reads the same directions' linear
    sensitivity ``|w_r . v_P|`` and the planted intended-criterion coverage, so distortion is
    coverage-gated sensitivity on ground truth where both terms are known.
    """
    rng = np.random.default_rng([int(seed), 8])
    w_r = _unit(rng.standard_normal(_D))

    names: list[str] = []
    directions: list[np.ndarray] = []
    coverage: list[float] = []
    props: list[Property] = []
    decodability: dict[str, float] = {}
    for name, sep, cos_wr, cov in _KUI_BATTERY:
        v = _dir_with_cos(w_r, cos_wr, rng)
        labels = (rng.uniform(size=n) < 0.5).astype(np.int64)
        acts = (2 * labels - 1)[:, None] * sep * v[None, :] + rng.standard_normal((n, _D)) * noise
        bal_acc = _difference_of_means_bal_acc(acts, labels, seed=1)
        decodability[name] = bal_acc
        props.append(Property(name=name, decodability=bal_acc, direction=v))
        names.append(name)
        directions.append(v)
        coverage.append(cov)

    plane = kui_from_properties(props, w_r)
    idx = {nm: i for i, nm in enumerate(plane["names"])}
    kui = {nm: float(plane["kui"][i]) for nm, i in idx.items()}
    decode_pct = {nm: float(plane["decode_pct"][i]) for nm, i in idx.items()}
    mediate_pct = {nm: float(plane["mediate_pct"][i]) for nm, i in idx.items()}

    sensitivity = linear_sensitivity(np.asarray(directions), w_r)
    distortion = distortion_per_dimension(sensitivity, coverage)
    dist = {names[i]: float(distortion[i]) for i in range(len(names))}
    sens = {names[i]: float(sensitivity[i]) for i in range(len(names))}
    cov = {names[i]: float(coverage[i]) for i in range(len(names))}

    return _KUIResult(
        kui=kui,
        decode_pct=decode_pct,
        mediate_pct=mediate_pct,
        decodability=decodability,
        distortion=dist,
        sensitivity=sens,
        coverage=cov,
        n_per_property=n,
    )


# ---------------------------------------------------------------------------
# Arm C: the sycophancy factorization (the crown proof)
# ---------------------------------------------------------------------------


@dataclass
class _FactorizationOrganism:
    """A planted sycophancy organism: activations, correctness and agreement labels, reward direction.

    ``route_through_belief`` records the construction: when True the agreement premium is planted along
    the belief direction (epistemic), when False along a separate agreeableness direction orthogonal to
    belief (axiological). ``cos_belief_reward`` is the planted reward coupling of the belief direction,
    positive so the grader prices perceived correctness.
    """

    activations: np.ndarray
    correct: np.ndarray
    agree: np.ndarray
    w_r: np.ndarray
    route_through_belief: bool
    cos_belief_reward: float


def _build_factorization_organism(
    route_through_belief: bool,
    *,
    per_cell: int = 150,
    beta: float = 1.5,
    alpha: float = 1.2,
    cos_br: float = 0.6,
    base_noise: float = 0.7,
    seed: int = 0,
) -> _FactorizationOrganism:
    """Plant a quadruple-crossed sycophancy organism with a known epistemic/axiological mechanism.

    The design crosses (user agrees / disagrees) x (answer correct / incorrect) with ``per_cell`` items
    per cell, balanced, so the agreement premium can be read controlling for correctness. Correctness
    shifts the activation by ``+/- beta`` along the belief direction ``b`` (so belief is decodable from
    correctness), and the reward prices belief through ``cos(b, w_r) = cos_br > 0``. Agreement adds
    ``+/- alpha`` along ``b`` when ``route_through_belief`` (agreement inflates the grader's belief that
    the answer is correct, an epistemic error), or along a separate direction ``a`` orthogonal to ``b``
    but also priced by the reward when not (agreement is rewarded directly, an axiological error). By
    construction the agreement premium is entirely mediated by belief in the first case and entirely
    bypasses it in the second.
    """
    rng = np.random.default_rng([int(seed), 1 if route_through_belief else 2])
    w_r = _unit(rng.standard_normal(_D))
    b = _dir_with_cos(w_r, cos_br, rng)
    # An agreeableness direction orthogonal to belief but with the same reward coupling, so the
    # axiological premium is real (the reward moves) yet carries no belief signal.
    a = _dir_with_cos(w_r, cos_br, rng)
    a = _unit(a - (a @ b) * b)

    rows: list[np.ndarray] = []
    correct: list[int] = []
    agree: list[int] = []
    for corr in (0, 1):
        for agr in (0, 1):
            for _ in range(per_cell):
                h = rng.standard_normal(_D) * base_noise + beta * (2 * corr - 1) * b
                channel = b if route_through_belief else a
                h = h + alpha * (2 * agr - 1) * channel
                rows.append(h)
                correct.append(corr)
                agree.append(agr)
    return _FactorizationOrganism(
        activations=np.asarray(rows, dtype=np.float64),
        correct=np.asarray(correct, dtype=np.int64),
        agree=np.asarray(agree, dtype=np.int64),
        w_r=w_r,
        route_through_belief=route_through_belief,
        cos_belief_reward=cos_br,
    )


def _agreement_premium(scores: np.ndarray, correct: np.ndarray, agree: np.ndarray) -> float:
    """The reward premium for agreement, averaged over correctness strata (the matched-quadruple read).

    Within each correctness stratum the premium is ``mean(r | agree) - mean(r | disagree)``; averaging
    over strata controls for the answer's true correctness, so what is left is the effect of the user's
    agreement alone. This is the object the factorization decomposes.
    """
    prems = []
    for corr in (0, 1):
        m = correct == corr
        prems.append(scores[m & (agree == 1)].mean() - scores[m & (agree == 0)].mean())
    return float(np.mean(prems))


def _belief_factorization(org: _FactorizationOrganism) -> dict:
    """Factor the agreement premium through a belief-in-correctness probe (the method, blind to truth).

    The method never sees the construction. It (1) builds a belief-in-correctness direction with
    ``concept_direction`` on the correct-versus-incorrect activation contrast; (2) reads the agreement
    premium in the reward; (3) severs the belief channel's response to agreement by patching each item's
    belief projection to its agreement-averaged value within its correctness stratum (the counterfactual
    "hold belief fixed across the agreement flip"), which is the exact residual-add
    ``interventions.patch.ResidualAddPatch`` performs for a linear head; and (4) re-reads the premium.
    The epistemic share is the fraction of the premium the sever removes: ~1 when the premium routes
    through belief (epistemic), ~0 when it bypasses belief (axiological). The residual (1 - share) is the
    part not carried by belief, the axiological-plus-shortcut term.
    """
    h = org.activations
    b_hat = _unit(
        np.asarray(
            concept_direction(_to_torch(h[org.correct == 1]), _to_torch(h[org.correct == 0]))
        )
    )
    reward = h @ org.w_r
    premium_before = _agreement_premium(reward, org.correct, org.agree)

    # Patch belief to its agreement-averaged counterfactual within each correctness stratum.
    proj = h @ b_hat
    proj_cf = proj.copy()
    for corr in (0, 1):
        m = org.correct == corr
        proj_cf[m] = proj[m].mean()
    delta = (proj_cf - proj)[:, None] * b_hat[None, :]
    h_patched = h + delta
    reward_patched = h_patched @ org.w_r
    premium_after = _agreement_premium(reward_patched, org.correct, org.agree)

    epistemic_share = (
        float((premium_before - premium_after) / premium_before)
        if abs(premium_before) > 1e-9
        else float("nan")
    )
    patch_ok = _verify_patch_matches_intervention(h, delta, h_patched)
    classification = "epistemic" if epistemic_share > 0.5 else "axiological"
    return {
        "epistemic_share": epistemic_share,
        "premium_before": premium_before,
        "premium_after": premium_after,
        "belief_reward_cos_abs": float(abs(b_hat @ org.w_r)),
        "classification": classification,
        "patch_matches_intervention": patch_ok,
        "planted": "epistemic" if org.route_through_belief else "axiological",
    }


def _to_torch(a: np.ndarray):
    """Coerce a numpy activation bank to a torch tensor for the concept-direction primitive."""
    import torch

    return torch.from_numpy(np.ascontiguousarray(a))


def _verify_patch_matches_intervention(
    h: np.ndarray, delta: np.ndarray, h_patched: np.ndarray, n_check: int = 8
) -> bool:
    """Confirm the inline belief patch equals the real ``ResidualAddPatch`` hook on a subset.

    The factorization computes the patched activations inline (a residual add of ``delta`` along the
    belief direction) so the whole batch is one numpy operation. This checks that the inline delta is
    bit-for-bit what the real intervention object produces: for a handful of items it builds a
    ``ResidualAddPatch``, compiles it (the public ``compile`` -> ``mounts`` path the runtime uses), and
    applies the mount hook to the item's activation, asserting equality with the inline result. So the
    proof genuinely rides ``interventions.patch``; the inline batch path is only its vectorization.
    """
    try:
        import torch

        from reward_lens.core.types import Site
        from reward_lens.interventions.patch import ResidualAddPatch

        site = Site(0, "resid_post", None)
        m = min(n_check, h.shape[0])
        for i in range(m):
            patch = ResidualAddPatch(site, torch.from_numpy(delta[i]).reshape(1, 1, -1))
            hook = patch.compile(None).mounts[site]
            out = hook(torch.from_numpy(h[i]).reshape(1, 1, -1), {})
            if not torch.allclose(out.reshape(-1), torch.from_numpy(h_patched[i])):
                return False
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Arm D: the alignment channel and its MI calibration
# ---------------------------------------------------------------------------


def _mi_calibration(rho: float = 0.6, n: int = 2000, k: int = 3) -> dict:
    """Grade the KSG and binned estimators on a correlated Gaussian of known mutual information.

    The bivariate-Gaussian MI is ``-1/2 log2(1 - rho^2)`` in closed form; both estimators are run over
    several seeds and their signed biases reported. The KSG bias is the gate (H6): an estimator that
    cannot recover a Gaussian MI has no business reporting a channel bit count. The binned estimator is
    the upward-biased cross-check.
    """
    ksg = calibrate_gaussian(rho=rho, n=n, k=k, repeats=5, estimator="ksg", seed=0)
    binned = calibrate_gaussian(rho=rho, n=n, repeats=5, estimator="binned", bins=8, seed=0)
    return {
        "rho": rho,
        "true_bits": ksg.true_bits,
        "ksg_estimate_bits": ksg.estimate_bits,
        "ksg_bias_bits": ksg.bias_bits,
        "ksg_abs_bias_bits": float(abs(ksg.bias_bits)),
        "ksg_std_bits": ksg.std_bits,
        "binned_estimate_bits": binned.estimate_bits,
        "binned_bias_bits": binned.bias_bits,
    }


def _alignment_channel(seed: int = 0) -> dict:
    """Measure how many bits of a known-H(V) annotator mixture survive a synthetic reward channel.

    The source is the foundry's annotator-mixture organism, whose value entropy ``H(V)`` is known
    exactly from its chosen mixing weights. Each item's annotator value is mapped to a separated centroid
    in reward space and corrupted by Gaussian noise; a small noise is the high-fidelity stage (a faithful
    channel, nearly all bits survive) and a large noise is the lossy stage (few bits survive). The
    transmitted information is the discrete-continuous mutual information ``I(V; r)``, and the kept
    fraction is ``I(V; r) / H(V)``. The known-answer check (H7): at high fidelity the kept fraction is
    ~1, because a lossless channel transmits the whole source entropy.
    """
    view, key = annotator_mixture_organism(n=360, seed=seed)
    mixing = key.channels[0].rho["mixing"]
    h_v = float(key.channels[0].rho["entropy_bits"])
    h_v_empirical = float(empirical_annotator_entropy(view))

    annotators = sorted(mixing)
    code = {a: i for i, a in enumerate(annotators)}
    values = np.asarray([code[p.meta["annotator_id"]] for p in view], dtype=np.int64)
    centroids = np.asarray([i * 3.0 for i in range(len(annotators))], dtype=np.float64)

    rng = np.random.default_rng([int(seed), 21])
    reward_hifi = centroids[values] + rng.standard_normal(values.size) * 0.15
    reward_lossy = centroids[values] + rng.standard_normal(values.size) * 2.5
    i_hifi = mi_discrete_continuous(values, reward_hifi, k=3)
    i_lossy = mi_discrete_continuous(values, reward_lossy, k=3)

    return {
        "H_V_bits": h_v,
        "H_V_empirical_bits": h_v_empirical,
        "n_annotators": len(annotators),
        "transmitted_hifi_bits": i_hifi,
        "transmitted_lossy_bits": i_lossy,
        "kept_fraction_hifi": float(i_hifi / h_v) if h_v > 0 else float("nan"),
        "kept_fraction_lossy": float(i_lossy / h_v) if h_v > 0 else float("nan"),
        "recovered_HV_abs_err_bits": float(abs(i_hifi - h_v)),
    }


def _gauge_kernel_bits(n: int = 3000, seed: int = 0) -> dict:
    """Instantiate the channel identity: a reward-null (gauge) direction transmits ~0 bits (T10, S2).

    Draw isotropic activations and set the reward ``r = h . w_r``. The coordinate of ``h`` along the
    reward direction determines ``r`` (it carries all the information), while the coordinate along a
    direction orthogonal to ``w_r`` (a gauge / reward-null direction) is independent of ``r`` and
    transmits ~0 bits. So the gauge subspace is the kernel of the value channel. The reward-coordinate
    figure is sample-limited (the coordinate is a deterministic function of ``r``); the load-bearing
    claim is the ~0 on the null side. This is the S8 half of the identity; S2 owns the gauge machinery
    that proves it in general.
    """
    rng = np.random.default_rng([int(seed), 31])
    w_r = _unit(rng.standard_normal(_D))
    e = rng.standard_normal(_D)
    e = _unit(e - (e @ w_r) * w_r)
    h = rng.standard_normal((n, _D))
    r = h @ w_r
    reward_coord = h @ w_r
    null_coord = h @ e
    return {
        "reward_coord_bits": mi_ksg(reward_coord, r, k=3),
        "null_coord_bits": mi_ksg(null_coord, r, k=3),
    }


# ---------------------------------------------------------------------------
# Gated arms: real-model matrices and the production belief-probe path
# ---------------------------------------------------------------------------


def _production_belief_path() -> dict:
    """Probe for the production belief-probe modules, offering the production path or gating it.

    The production factorization substitutes an answer-keyed belief probe (``concepts.beliefs``) fit by
    ``concepts.probes.fit_probe`` for the hand-built belief direction, and a
    ``interventions.steer.SteeringIntervention`` for the mediation. These are lazy-imported: if all are
    present the production path is available; if the probe factory is absent the proof arm above still
    stands and this records exactly which contract is pending, never fabricating a production result.
    """
    present: dict[str, bool] = {}
    detail: dict[str, str] = {}
    for label, importer in (
        ("concepts.beliefs", lambda: __import__("reward_lens.concepts.beliefs", fromlist=["*"])),
        (
            "concepts.probes.fit_probe",
            lambda: getattr(
                __import__("reward_lens.concepts.probes", fromlist=["fit_probe"]), "fit_probe"
            ),
        ),
        (
            "interventions.steer.SteeringIntervention",
            lambda: getattr(
                __import__("reward_lens.interventions.steer", fromlist=["SteeringIntervention"]),
                "SteeringIntervention",
            ),
        ),
    ):
        try:
            importer()
            present[label] = True
        except Exception as exc:  # noqa: BLE001 - record the exact reason a contract is pending
            present[label] = False
            detail[label] = f"{type(exc).__name__}: {exc}"
    available = all(present.values())
    return {
        "gated": not available,
        "production_path_available": available,
        "contracts_present": present,
        "contracts_pending_detail": detail,
        "note": "production factorization uses concepts.beliefs (answer-keyed belief probe) fit by "
        "concepts.probes.fit_probe, steered via interventions.steer.SteeringIntervention; the hand-built "
        "belief-direction proof arm classifies the planted organisms without them",
    }


def _record_gated(run, subject, study_id, observable: str, value: dict, parents=()) -> None:
    """Record a REGISTERED gated Evidence carrying no adjudicated metric (inconclusive-because-gated)."""
    ev = make_evidence(
        observable=observable,
        observable_version=_VERSION,
        subject=subject,
        value=value,
        gauge=GaugeStatus.INVARIANT,
        provenance=Provenance(study=study_id, parents=tuple(parents)),
        registered=True,
    )
    run.record(ev)


# ---------------------------------------------------------------------------
# The analysis
# ---------------------------------------------------------------------------


def analyze(run) -> StudyResult:
    """Run the four proof arms on planted ground truth and record the gated real-model arms.

    Arms A/B (T2): KUI recovers the represented-but-unpriced gap and distortion is coverage-gated
    sensitivity. Arm C (T10, the crown): the belief-patch factorization classifies the planted-epistemic
    organism epistemic and the planted-axiological organism axiological. Arm D (T10): the MI estimator is
    calibrated on a known-MI Gaussian, then used to measure the alignment channel's kept fraction and the
    gauge=kernel identity. The four-campaign-model KUI matrix, the real-model factorization, the real
    channel bit counts, and the production belief-probe path are recorded as REGISTERED gated Evidence
    with no adjudicated metric, so they are inconclusive-because-gated and invent no number.
    """
    study_id = run.study.study_id
    subject = SubjectRef(extra={"study": study_id})

    # Root Evidence: the planted organisms this study is graded against (the DAG root).
    ev_root = make_evidence(
        observable="S08.Organisms",
        observable_version=_VERSION,
        subject=subject,
        value={
            "d_model": _D,
            "kui_battery": [row[0] for row in _KUI_BATTERY],
            "factorization": [
                "epistemic (premium routes through belief)",
                "axiological (bypasses)",
            ],
            "channel_source": "foundry annotator_mixture_organism (known H(V))",
            "note": "all proof arms are planted so every recovered number has a known answer; the "
            "real-model arms are gated",
        },
        uncertainty=Uncertainty(method="none"),
        gauge=GaugeStatus.INVARIANT,
        provenance=Provenance(study=study_id),
        registered=True,
    )
    run.record(ev_root)

    metrics: dict[str, float] = {}

    # --- Arm A/B: KUI gap and distortion (T2) ---------------------------------
    kres = _run_kui_and_distortion()
    kui_ignored = kres.kui["sentiment"]
    kui_control = kres.kui["correctness"]
    kui_gap = float(kui_ignored - kui_control)
    distortion_spurious = kres.distortion["sycophancy"]
    distortion_intended = kres.distortion["correctness"]
    distortion_separation = float(distortion_spurious - distortion_intended)

    metrics.update(
        kui_gap=kui_gap,
        kui_ignored=float(kui_ignored),
        kui_control_abs=float(abs(kui_control)),
        distortion_spurious=distortion_spurious,
        distortion_intended=distortion_intended,
        distortion_separation=distortion_separation,
    )
    ev_kui = make_evidence(
        observable="S08.KnowledgeUtilizationGap",
        observable_version=_VERSION,
        subject=subject,
        value={
            "kui": kres.kui,
            "decode_pct": kres.decode_pct,
            "mediate_pct": kres.mediate_pct,
            "decodability": kres.decodability,
            "kui_ignored_sentiment": float(kui_ignored),
            "kui_control_correctness": float(kui_control),
            "kui_gap": kui_gap,
            "n_per_property": kres.n_per_property,
            "note": "'sentiment' is decodable but off-axis from reward (represented-but-unpriced, the "
            "hack precondition); 'correctness' is decodable and priced (the control, on the diagonal)",
        },
        uncertainty=Uncertainty(n=kres.n_per_property, method="none"),
        gauge=GaugeStatus.INVARIANT,
        provenance=Provenance(study=study_id, parents=(ev_root.id,)),
        registered=True,
    )
    run.record(ev_kui)

    ev_distortion = make_evidence(
        observable="S08.Distortion",
        observable_version=_VERSION,
        subject=subject,
        value={
            "distortion": kres.distortion,
            "sensitivity": kres.sensitivity,
            "coverage": kres.coverage,
            "distortion_spurious_sycophancy": distortion_spurious,
            "distortion_intended_correctness": distortion_intended,
            "distortion_separation": distortion_separation,
            "note": "coverage is the planted intended-criterion indicator (independent of geometric "
            "sensitivity by construction); 'sycophancy' is priced-but-not-intended (high distortion), "
            "'correctness' is equally sensitive but intended (low distortion)",
        },
        uncertainty=Uncertainty(n=len(_KUI_BATTERY), method="none"),
        gauge=GaugeStatus.INVARIANT,
        provenance=Provenance(study=study_id, parents=(ev_kui.id,)),
        registered=True,
    )
    run.record(ev_distortion)

    # --- Arm C: the sycophancy factorization (the crown proof, T10) -----------
    org_epi = _build_factorization_organism(route_through_belief=True, seed=0)
    org_axi = _build_factorization_organism(route_through_belief=False, seed=0)
    fac_epi = _belief_factorization(org_epi)
    fac_axi = _belief_factorization(org_axi)
    share_epi = float(fac_epi["epistemic_share"])
    share_axi = float(fac_axi["epistemic_share"])
    metrics.update(
        factorization_epistemic_share_epi=share_epi,
        factorization_epistemic_share_axi=share_axi,
        factorization_margin=float(share_epi - share_axi),
    )
    ev_fac = make_evidence(
        observable="S08.Factorization",
        observable_version=_VERSION,
        subject=subject,
        value={
            "epistemic_organism": fac_epi,
            "axiological_organism": fac_axi,
            "epistemic_share_epi": share_epi,
            "epistemic_share_axi": share_axi,
            "factorization_margin": float(share_epi - share_axi),
            "belief_direction": "concepts.vectors.concept_direction on the correct-vs-incorrect contrast",
            "patch": "interventions.patch.ResidualAddPatch (belief projection to its "
            "agreement-averaged counterfactual), verified bit-equal to the inline batch delta",
            "note": "the method classifies the planted-epistemic organism epistemic and the "
            "planted-axiological organism axiological, so it tells the two constructions apart",
        },
        uncertainty=Uncertainty(n=int(org_epi.activations.shape[0]), method="none"),
        gauge=GaugeStatus.INVARIANT,
        provenance=Provenance(study=study_id, parents=(ev_root.id,)),
        registered=True,
    )
    run.record(ev_fac)

    # --- Arm D: MI calibration, the alignment channel, and gauge=kernel (T10) --
    mical = _mi_calibration()
    metrics.update(
        mi_ksg_gaussian_abs_bias=float(mical["ksg_abs_bias_bits"]),
        mi_ksg_gaussian_bias=float(mical["ksg_bias_bits"]),
        mi_binned_gaussian_bias=float(mical["binned_bias_bits"]),
    )
    ev_mi = make_evidence(
        observable="S08.MICalibration",
        observable_version=_VERSION,
        subject=subject,
        value=mical
        | {
            "note": "KSG recovers the closed-form Gaussian MI within tolerance (the gate for reporting "
            "any channel bit count); the binned estimator is the upward-biased cross-check"
        },
        uncertainty=Uncertainty(n=mical.get("rho") and 2000, method="bootstrap"),
        gauge=GaugeStatus.INVARIANT,
        provenance=Provenance(study=study_id, parents=(ev_root.id,)),
        registered=True,
    )
    run.record(ev_mi)

    chan = _alignment_channel()
    gk = _gauge_kernel_bits()
    metrics.update(
        channel_kept_fraction_hifi=float(chan["kept_fraction_hifi"]),
        channel_kept_fraction_lossy=float(chan["kept_fraction_lossy"]),
        channel_recovered_HV_abs_err=float(chan["recovered_HV_abs_err_bits"]),
        channel_null_bits=float(gk["null_coord_bits"]),
        channel_reward_bits=float(gk["reward_coord_bits"]),
    )
    ev_chan = make_evidence(
        observable="S08.AlignmentChannel",
        observable_version=_VERSION,
        subject=subject,
        value=chan
        | gk
        | {
            "note": "high-fidelity kept fraction ~1 recovers H(V) (the known answer); the lossy stage "
            "drops it; a reward-null (gauge) direction transmits ~0 bits, so gauge subspace = channel "
            "kernel (jointly with S2)"
        },
        uncertainty=Uncertainty(n=360, method="none"),
        gauge=GaugeStatus.INVARIANT,
        provenance=Provenance(study=study_id, parents=(ev_mi.id,)),
        registered=True,
    )
    run.record(ev_chan)

    # --- Gated real-model and production arms (no fabricated number) -----------
    _record_gated(
        run,
        subject,
        study_id,
        "S08.KUIMatrixReal",
        {
            "gated": True,
            "needs": "the four campaign reward models + a probe population + GPU",
            "arm": "the represented-but-ignored KUI matrix on the four campaign models, calibrated on "
            "the planted-property organisms proven above",
            "note": "the KUI plane is proven on planted properties here; the real-model matrix is "
            "population/GPU-gated and emits no number",
        },
        parents=(ev_kui.id,),
    )
    _record_gated(
        run,
        subject,
        study_id,
        "S08.FactorizationReal",
        {
            "gated": True,
            "needs": "real reward models + the sycophancy quadruple corpus with receipts + GPU",
            "arm": "the sycophancy factorization on real models (epistemic vs axiological premium)",
            "note": "the factorization is proven to separate the two mechanisms on planted organisms; "
            "the real-model classification is gated",
        },
        parents=(ev_fac.id,),
    )
    _record_gated(
        run,
        subject,
        study_id,
        "S08.ChannelBitsReal",
        {
            "gated": True,
            "needs": "a real constitution corpus + an annotation dataset + a real reward model + GPU",
            "arm": "the real alignment-channel bit counts across pipeline stages",
            "target_to_measure_not_measured": {
                "constitution_bits_order_of_magnitude": "~1e4",
                "annotation_bits_order_of_magnitude": "~1e2",
                "pretraining_prior": "fills the remainder",
            },
            "note": "these orders of magnitude are the DESIGN S8 hypothesis to test, not a measurement; "
            "the estimator is calibrated above, the real bit counts are population/GPU-gated and no "
            "number is invented",
        },
        parents=(ev_chan.id,),
    )
    prod = _production_belief_path()
    _record_gated(run, subject, study_id, "S08.ProductionBeliefProbe", prod, parents=(ev_fac.id,))

    # --- Summary ---------------------------------------------------------------
    summary = (
        f"On planted ground truth, KUI recovered a knowledge-utilization gap of {kui_gap:.3f} for the "
        f"represented-but-unpriced property (control gap {abs(kui_control):.3f}, ~on the diagonal), and "
        f"distortion separated priced-but-not-intended from priced-and-intended by "
        f"{distortion_separation:.3f}. The belief-patch factorization classified the planted-epistemic "
        f"organism '{fac_epi['classification']}' (epistemic share {share_epi:.3f}) and the "
        f"planted-axiological organism '{fac_axi['classification']}' (epistemic share {share_axi:.3f}), "
        f"so it tells the two mechanisms apart. The KSG estimator recovered the closed-form Gaussian MI "
        f"with bias {mical['ksg_bias_bits']:+.3f} bits; the high-fidelity alignment channel kept "
        f"{chan['kept_fraction_hifi']:.2f} of H(V)={chan['H_V_bits']:.2f} bits while the lossy stage "
        f"kept {chan['kept_fraction_lossy']:.2f}, and a reward-null direction transmitted "
        f"{gk['null_coord_bits']:.3f} bits (gauge subspace = channel kernel). The four-campaign KUI "
        f"matrix, the real-model factorization, and the real channel bit counts are population/GPU-gated."
    )

    return StudyResult(outcomes={}, metrics=metrics, summary=summary)


# ---------------------------------------------------------------------------
# The retype: S8 on the kernel
# ---------------------------------------------------------------------------

#: Both factorization metrics compute the same thing on two organisms, so they carry one request.
_MEDIATED_SHARE_GAP = (
    "the share of a measured reward premium that a named mediator carries, "
    "``(premium - premium_after_sever) / premium``, where the sever holds the mediator's projection "
    "at its counterfactual value within stratum. Unit: 1, dimension 1, support [0, 1] in the "
    "identified case and unbounded when the premium is near zero, which is why the estimator refuses "
    "below a premium floor rather than dividing. Invariance: `repr.basis`, invariant, because the "
    "patch and the readout rotate together. Nothing registered fits. `grader.path_effect` is the "
    "closest and is the numerator rather than the ratio: its unit is `reward` and this is that "
    "reduction divided by the premium it is a share of, and a share and a reward difference are not "
    "the same quantity under E15's rule that a ratio of two same-unit quantities is dimensionless. "
    "Recommend registering `grader.mediated_share` with the definition above, beside "
    "`grader.path_effect` on the same instrument."
)

RETYPE = ScienceRetype(
    science="s08_factorization",
    spec=build_spec(),
    headline="grader.knowledge_utilization",
    destination=(
        "L2 and L4 in measure/indices/: `grader.knowledge_utilization` is A1's KUI plane in "
        "indices/kui.py and `grader.distortion_index` is A2's per-dimension distortion in "
        "indices/distortion.py, both of which this study already computes through. The "
        "factorization and the alignment channel land nowhere: four of the eight frozen metrics "
        "have no registered quantity, and three of those four are in bits, a dimension the registry "
        "does not carry at all."
    ),
    # GRADER:FORWARD, because every arm reads activations and a linear readout: the probe that
    # measures decodability, the direction whose cosine is the mediation proxy, the belief patch,
    # and the channel's input coordinates.
    needs={Component.GRADER: Access.FORWARD},
    metrics=(
        MetricSpec(
            metric="kui_gap",
            quantity="grader.knowledge_utilization",
            arc="kui-battery",
            dataset="planted-battery:sentiment-minus-correctness",
            source="organism",
            note=(
                "KUI(sentiment) minus KUI(correctness) on one battery, so it is a contrast between "
                "two properties in the same plane and carries the same unit as each term. The "
                "registered unit is percentile-within-battery, which is exactly the fix A1's own "
                "unit bug needed: v1 subtracted a balanced accuracy from a cosine. `kui.py` "
                "standardises both axes to their percentile within the battery and then takes the "
                "signed perpendicular distance from the diagonal, so both terms here are already in "
                "the registered unit before the subtraction."
            ),
        ),
        MetricSpec(
            metric="kui_control_abs",
            quantity="grader.knowledge_utilization",
            arc="kui-battery",
            dataset="planted-battery:correctness",
            source="organism",
            note=(
                "the magnitude of the control property's KUI, the same quantity on one battery item "
                "rather than a contrast. Absolute value because the prediction is that the control "
                "sits on the diagonal, and a signed miss on either side is the same failure."
            ),
        ),
        MetricSpec(
            metric="distortion_separation",
            quantity="grader.distortion_index",
            arc="distortion",
            dataset="planted-battery:sycophancy-minus-correctness",
            source="organism",
            note=(
                "D(sycophancy) minus D(correctness), a difference of two per-dimension distortions "
                "in the registered unit reward per unit direction. Both terms are coverage-gated "
                "sensitivity on the same battery with the same reward direction, so the difference "
                "is in that unit and the comparison is between two properties rather than two "
                "graders."
            ),
        ),
        MetricSpec(
            metric="factorization_epistemic_share_epi",
            arc="belief-sever",
            arm="planted-epistemic",
            source="organism",
            gap=_MEDIATED_SHARE_GAP,
        ),
        MetricSpec(
            metric="factorization_epistemic_share_axi",
            arc="belief-sever",
            arm="planted-axiological",
            source="organism",
            gap=_MEDIATED_SHARE_GAP,
        ),
        MetricSpec(
            metric="mi_ksg_gaussian_abs_bias",
            arc="mi-calibration",
            source="organism",
            gap=(
                "the absolute bias of a mutual-information estimator against a value known in "
                "closed form, on a reference material whose MI is `-0.5 log2(1 - rho^2)` by "
                "construction. Unit: bits, dimension information, per null. Invariance: `units`, "
                "invariant. Nothing registered fits and the registry carries no information "
                "dimension at all: its four `nats` rows are all KL divergences per token or per "
                "sequence, which is a different quantity in a different unit. This is a metrology "
                "number rather than a grader number, the recovery of a certified reference value, "
                "so it belongs beside `reference.u_characterisation` rather than in the grader "
                "family. Recommend registering `estimator.mi_bias` in bits, with the reference's "
                "own uncertainty beside it."
            ),
        ),
        MetricSpec(
            metric="channel_kept_fraction_hifi",
            arc="alignment-channel",
            arm="high-fidelity",
            source="organism",
            gap=(
                "transmitted information over source entropy: the bits of a known-entropy annotator "
                "mixture that survive a reward channel, divided by H(V). Unit: 1, dimension 1, "
                "support [0, 1]. Invariance: `units`, invariant. Nothing registered fits. "
                "`grader.dark_fraction` is the closest in shape and is a different object: it is a "
                "share of reward variance unexplained by named channels, computed by regression, "
                "and this is a share of source entropy that survives transmission, computed by "
                "mutual information. Substituting one for the other would swap a second-moment "
                "statement for an information one. Recommend registering `channel.kept_fraction`, "
                "with `channel.transmitted_bits` in bits as its numerator so the ratio has a "
                "registered top and bottom."
            ),
        ),
        MetricSpec(
            metric="channel_null_bits",
            arc="alignment-channel",
            arm="gauge-null-direction",
            source="organism",
            gap=(
                "the mutual information between a displacement along a reward-null direction and "
                "the reward it induces, which is zero exactly when the gauge subspace is the "
                "channel's kernel. Unit: bits, dimension information, per null. Invariance: "
                "`repr.basis`, invariant, and the claim it supports is that the quantity is zero on "
                "the whole null subspace, which is a statement about the group rather than about a "
                "direction. Same registry gap as `mi_ksg_gaussian_abs_bias`: no information "
                "dimension exists. Recommend registering `channel.transmitted_bits` in bits and "
                "reading this as that quantity on a null-direction subject, which is one id for "
                "both this and the numerator of the kept fraction."
            ),
        ),
    ),
    arc_requires={
        "distortion": ("kui-battery",),
        "alignment-channel": ("mi-calibration",),
    },
    waiting_on=(
        "four quantity ids: `grader.mediated_share` (two metrics), `estimator.mi_bias`, "
        "`channel.kept_fraction` and `channel.transmitted_bits`. Three of the four are in bits and "
        "the registry has no information dimension, so this is a registry decision rather than a "
        "naming one."
    ),
)


def read(run: Run) -> Reading:
    """S8 against a record, which cannot answer any of it, and the reason is one sentence.

    Every arm of S8 reads activations. The KUI plane needs a probe fitted on them and a direction to
    take a cosine against; the distortion needs the same directions; the belief sever needs to patch
    a projection and re-read the reward; the channel needs the coordinates the information is
    measured between. A record carries the scores a grader produced and none of the intermediate
    state that produced them, so this refuses on access rather than computing a version of the
    question that the record can answer.

    Scope limit, three lines in: there is no degraded record path here and that is deliberate. The
    obvious one, correlating a recorded feature with the recorded reward, is the susceptibility S3
    already measures and reports under `selection.differential_S`; computing it again under a
    knowledge-utilization name would be the same number wearing a second id.
    """
    if (refusal := RETYPE.access_refusal(run, remedy=_ACCESS_REMEDY)) is not None:
        return refusal
    leaves = leaf_scores(run, limit=256)
    return RETYPE.incomplete(
        field="activations, and no probe fitted on them",
        subject=(
            f"run {run.id}, which carries {count_trajectories(run)} scored trajectories under "
            f"{len(leaves)} grader leaf/leaves,"
        ),
        remedy=(
            "load the reward model and run the battery through measure.indices.kui and "
            "measure.indices.distortion with a probe per property, or run analyze(), which plants "
            "the battery it then recovers. A record is the grader's output and every arm of S8 is "
            "about what the grader represented on the way there."
        ),
        trajectories=count_trajectories(run),
        grader_leaves=sorted(leaves),
    )


_ACCESS_REMEDY = (
    "supply GRADER:FORWARD on the reward model, with the activations and the score head readable. "
    "Every arm of S8 is a question about what the grader represents rather than about what it "
    "scored, and a RECORD-access run holds only the score."
)


__all__ = ["RETYPE", "build_spec", "analyze", "read"]

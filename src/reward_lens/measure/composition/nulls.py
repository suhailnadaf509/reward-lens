"""The four nulls B1 is indefensible without, ordered by decisiveness.

A curl mass of 0.214 with no null behind it is a number, not a finding. The paper most often cited
against a topological reading of cycles does not show that cycles are measurement error, it assumes
it: its own validity conditions require that cycles arise "primarily from random measurement errors
rather than genuine intransitivity" and that a latent total order exists, and a latent total order is
transitive by construction, so under that model every observed cycle is error with probability one.
It concedes the rest in as many words: "When genuine intransitivity dominates, TCR may incorrectly
filter valid preference signals... We leave these extensions to future work."

Null C is that extension, and it is here.

    C  split-half reliability of the curl component, Spearman-Brown corrected. Decisive, and it
       needs no noise model at all: if the cyclic structure is measurement error it will not
       replicate across halves of the replications. Run this first.
    A  parametric bootstrap. Fit Bradley-Terry, resimulate through the identical pipeline, report
       the excess over the null rather than the raw mass.
    D  anisotropy of the curl spectrum. Noise is isotropic and genuine conflict is not, so the
       leading plane names the competing criteria and turns a defence into a finding.
    E  the random-profile baseline, and the transitive baseline beside it.

**Null E has two members and the second one is the one that bites.** A uniformly random ordinal
profile is the classic null and it is weak here, for a reason worth stating: a profile is a set of
total orders, each of which is *transitive*, and a transitive preference recorded as wins and losses
already carries curl. At one replication per pair the two members are the same distribution exactly.
Above it they still share a centre and the random profile only adds spread, so it is the less
powerful of the two at every replication count.

The baseline that carries the information is `transitive_baseline`: draw one total order per
comparison graph, orient every observed pair by it, flip each comparison at the grader's measured
flip rate, and push that through the identical aggregation. On a complete graph with a zero flip
rate it is a point mass with a closed form, ``curl mass = (n-2)/(3n)``, which is not zero and has no
intransitivity in it. It is the price of recording a comparison as a sign instead of a margin, and a
curl mass that fails to clear it is an encoding artifact.

C and A both need at least eleven replications per pair and refuse below it, naming eleven. That
refusal is the correct output and not a gap: a grader queried once per pair has produced data that
cannot bound its own noise, and reporting a curl mass from it as though a null had cleared it is the
error this module exists to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.measure.composition.hodge import (
    HALDANE_ANSCOMBE,
    MIN_REPLICATIONS_FOR_NULLS,
    ComparisonFlow,
    Edge,
    HodgeSplit,
    PairCount,
    curl_operator,
    edge_flow,
    incidence,
    split_flow,
    triangles,
)

#: The conventional psychometric floor for a reliable instrument, used to turn Null C's number into
#: a verdict. **Proposed rather than settled**: a threshold that decides a kill condition is not an
#: implementer's to fix, so the coefficient and its interval are always reported and the verdict is
#: derived, which lets a reader apply a different floor without rerunning anything.
RELIABILITY_FLOOR = 0.7

#: Below this, the positive control has failed and the experiment cannot distinguish "the curl is
#: noise" from "nothing here replicates". Same status: proposed, and the control's own number is
#: always reported.
CONTROL_FLOOR = 0.7


def _refuse_replications(instrument: str, observed: float, what: str) -> Refusal:
    """The one refusal both replication-gated nulls return, with the n it needs in it.

    `ACCESS_INSUFFICIENT` rather than a regime violation, and the choice is deliberate. Replications
    are an access level: the access ladder puts repeated queries of the same input at
    GRADER:REPLICATE, one rung above the GRADER:QUERY that produced a single verdict per pair. So
    the remedy is an instruction about what to go and get rather than a diagnosis, which is what a
    remedy is for.
    """
    return Refusal(
        instrument=instrument,
        reason=RefusalReason.ACCESS_INSUFFICIENT,
        detail=(
            f"{what} needs at least {MIN_REPLICATIONS_FOR_NULLS} replications of every pair and the "
            f"thinnest pair here has {observed:g}. With {observed:g} there is nothing to split and "
            f"no way to estimate the flip rate the null would resample from"
        ),
        remedy=(
            f"re-query the grader at GRADER:REPLICATE so every pair is compared at least "
            f"{MIN_REPLICATIONS_FOR_NULLS} times, then rerun. Eleven is the number at which a "
            f"majority vote matches a fifty-trial reference at 95%, so it is the floor rather than "
            f"a comfortable margin. Until then the curl mass is a rung-0 reading with no null "
            f"behind it and must be reported as one."
        ),
        statistics={
            "observed_min_replications": float(observed),
            "required_replications": MIN_REPLICATIONS_FOR_NULLS,
        },
    )


def _components(flow: ComparisonFlow) -> list[tuple[np.ndarray, np.ndarray]]:
    """(item indices, edge indices) per connected component of the comparison graph.

    A corpus is the disjoint union of its per-prompt tournaments, so almost every flow reaching
    these nulls is a graph with thousands of small components. Every null here resamples *within* a
    component, because a total order over items from two different prompts is not a thing.
    """
    parent = list(range(flow.n_items))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in flow.edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    edge_groups: dict[int, list[int]] = {}
    for k, (a, _) in enumerate(flow.edges):
        edge_groups.setdefault(find(a), []).append(k)
    out = []
    for root, edge_idx in edge_groups.items():
        items = sorted({v for k in edge_idx for v in flow.edges[k]})
        out.append((np.asarray(items, dtype=np.int64), np.asarray(edge_idx, dtype=np.int64)))
    return out


def _rebuild(flow: ComparisonFlow, wins_a: np.ndarray, wins_b: np.ndarray) -> ComparisonFlow:
    """A new flow on the same edges from new win counts, through the identical aggregation.

    "Identical" is the whole point of a parametric bootstrap and it is where resimulation studies
    usually go wrong: a null that reaggregates by a shortcut is testing the shortcut. This calls
    `edge_flow` with the same scale and correction the observed flow was built with.
    """
    pairs = [
        PairCount(a, b, float(wa), float(wb)) for (a, b), wa, wb in zip(flow.edges, wins_a, wins_b)
    ]
    return edge_flow(
        pairs,
        flow.n_items,
        scale=flow.scale,
        correction=flow.correction_value if flow.scale == "logit" else HALDANE_ANSCOMBE,
    )


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Uncentred correlation, which is the right reliability for a component vector.

    Zero is meaningful here: a curl component of exactly zero means no curl, not "an average amount
    of curl". Centring across edges would destroy that, so the primary coefficient is the cosine and
    the centred Pearson travels beside it for readers who want the familiar one.
    """
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    sa, sb = float(np.std(a)), float(np.std(b))
    if sa == 0.0 or sb == 0.0:
        return float("nan")
    return float(np.mean((a - np.mean(a)) * (b - np.mean(b))) / (sa * sb))


def _spearman_brown(half: float) -> float:
    """``2r/(1+r)``: the reliability of the whole from the reliability of a half.

    **Clamped to [-1, 1], and the lower clamp is load-bearing.** The prophecy formula is monotone
    increasing on ``r > -1`` and it is only a reliability on ``r >= 0``; below zero it is
    extrapolating outside its domain and it diverges. It was guarded at exactly ``r = -1`` and
    nowhere else, so ``r = -0.9`` returned -18 and ``r = -0.99`` returned -198. Measured on the
    complete four-item design at eleven replications, 27% of splits produced a reported interval
    endpoint below -20 and the worst was **-576.92**, printed as a reliability coefficient whose
    own upper bound is 1.

    How often that is reachable is set by the dimension of the curl subspace, because the cosine of
    two independent vectors in ``d`` dimensions has density proportional to ``(1 - r^2)^((d-3)/2)``,
    which is *uniform* on [-1, 1] at the curl dimension of three that four items give. So the small
    designs are the exposed ones, and they are the common ones.

    The verdicts never moved: a large negative fails `RELIABILITY_FLOOR` and `CONTROL_FLOOR` exactly
    as -1 does. What was corrupted is every reported coefficient and interval, including the case
    where the gradient control passes and the number therefore ships as meaningful. A negative value
    still means what it meant, that there is no reliable variance here and that is the finding, and
    its magnitude was never interpretable.
    """
    if half <= -1.0 + 1e-12:
        return -1.0
    return float(min(1.0, max(-1.0, 2.0 * half / (1.0 + half))))


def _tail_p(draws: np.ndarray, observed: float) -> float:
    """``(count + 1)/(n + 1)`` with a float tolerance on the comparison, which is not fastidiousness.

    A degenerate null is a real case here and it is the interesting one: on a design where every
    transitive draw lands on the same curl mass, the null is a point mass and the observed value sits
    on top of it. A bare ``>=`` then splits the draws by whichever side of the last bit they fell on
    and returns something like 0.08, which reads as a near-significant excess of nothing. Comparing
    at a relative tolerance returns a p of 1, which is the truth: the observation is exactly the
    null.
    """
    tolerance = 1e-12 * max(1.0, abs(observed))
    count = int(np.sum(draws >= observed - tolerance))
    return float((count + 1) / (draws.size + 1))


# ---------------------------------------------------------------------------
# Null C. Split-half reliability of the curl component
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitHalfResult:
    """Null C, with its positive control beside it because the claim is a null.

    ``curl_reliability`` is the headline: the Spearman-Brown corrected split-half correlation of the
    curl component. A null claim with no matched positive control cannot be told apart from an
    underpowered experiment, so two controls travel with it and they do different jobs.

    ``gradient_reliability`` is the same coefficient on the gradient component of the same splits.
    It answers "does anything in this dataset replicate", and it is **not** identically powered
    against the curl: ``im(grad)`` has dimension ``n - 1`` where ``im(C)`` has ``C(n,2) - (n-1)``,
    which is 9 against 36 at ten items, and the same total signal energy spread over four times the
    directions correlates far less well. Measured with equal true energy planted in each subspace at
    ten items, the gradient replicates at 0.6212 while the curl replicates at 0.3882. So a passing
    gradient control does not license the conclusion that a non-replicating curl is noise.

    ``planted_curl_reliability`` is the control that does license it. A curl of exactly the observed
    energy is planted in ``im(C)``, pushed through the same sampling noise at the same replication
    counts, and split the same way. It answers the question the ``does_not_replicate`` branch needs
    answered: *if there really were a cyclic structure this large, would this design see it?* When
    that control fails, the verdict is ``underpowered`` whatever the gradient did.

    Two intervals, because they are different things and one of them used to be labelled as the
    other. ``curl_split_spread`` is the 2.5 to 97.5 percentile range of the per-split coefficient:
    it describes how much the answer moves with the choice of split and it does **not** shrink as
    splits accumulate, measured at width 1.899 over 20 splits and 1.775 over 1,600. ``curl_ci`` is a
    Monte Carlo interval for the mean, which does shrink like ``1/sqrt(n_splits)``. Neither is a
    sampling interval over datasets: both hold this dataset fixed, so neither says what would happen
    if the comparisons were collected again.
    """

    n_splits: int
    replications_per_pair: float
    curl_half: float
    curl_reliability: float
    curl_ci: tuple[float, float]
    curl_split_spread: tuple[float, float]
    curl_pearson: float
    harmonic_half: float
    harmonic_reliability: float
    gradient_half: float
    gradient_reliability: float
    planted_curl_reliability: float
    curl_subspace_dim: int
    gradient_subspace_dim: int
    verdict: str
    floor: float
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_splits": self.n_splits,
            "replications_per_pair": self.replications_per_pair,
            "curl_half_correlation": self.curl_half,
            "curl_reliability_spearman_brown": self.curl_reliability,
            "curl_reliability_ci_low": self.curl_ci[0],
            "curl_reliability_ci_high": self.curl_ci[1],
            "curl_split_spread_low": self.curl_split_spread[0],
            "curl_split_spread_high": self.curl_split_spread[1],
            "curl_half_pearson": self.curl_pearson,
            "harmonic_half_correlation": self.harmonic_half,
            "harmonic_reliability_spearman_brown": self.harmonic_reliability,
            "gradient_half_correlation": self.gradient_half,
            "gradient_reliability_spearman_brown": self.gradient_reliability,
            "planted_curl_reliability_spearman_brown": self.planted_curl_reliability,
            "curl_subspace_dim": self.curl_subspace_dim,
            "gradient_subspace_dim": self.gradient_subspace_dim,
            "verdict": self.verdict,
            "reliability_floor": self.floor,
            "interval_note": (
                "curl_reliability_ci is a Monte Carlo interval for the mean over splits and shrinks "
                "with n_splits; curl_split_spread is the dispersion of the per-split coefficient "
                "and does not. Neither is an interval over repeated collection of the comparisons."
            ),
            "note": self.note,
        }


def split_half_reliability(
    flow: ComparisonFlow,
    replicates: Mapping[Edge, Sequence[int]] | None = None,
    *,
    n_splits: int = 50,
    seed: int = 0,
    floor: float = RELIABILITY_FLOOR,
    control_floor: float = CONTROL_FLOOR,
) -> SplitHalfResult | Refusal:
    """Split the replications in half, decompose each half, correlate the curl components.

    The decisive null, and the cheapest. It assumes nothing about how the grader is noisy: if the
    cyclic structure is measurement error then the two halves see independent errors and their curl
    components are uncorrelated, and if it is a real property of the grader then both halves see the
    same thing. Spearman-Brown corrects the halving, because a half of the data is less reliable than
    the whole and the uncorrected coefficient understates the instrument.

    ``replicates`` is optional and usually absent. Given only win counts, a random split is a
    hypergeometric draw, which is exactly right: the replications of one pair are exchangeable, so
    the individual verdicts carry no information the counts do not. Supplying the per-replication
    verdicts changes nothing statistically and is accepted because a design that blocks by annotator
    or by seed may want to split along that blocking rather than at random.

    The gradient's reliability on the identical splits is the positive control and it is not
    optional. Without it, "the curl does not replicate" and "nothing in this dataset replicates" are
    the same observation.
    """
    if flow.n_edges == 0:
        return _refuse_replications("Null C (split-half)", 0.0, "the split-half reliability")
    observed = flow.min_replications
    if observed < MIN_REPLICATIONS_FOR_NULLS:
        return _refuse_replications("Null C (split-half)", observed, "the split-half reliability")

    rng = np.random.default_rng(seed)
    wins_b = np.rint((flow.replications + flow.replications * _signed(flow)) / 2.0)
    wins_a = flow.replications - wins_b

    curl_halves, curl_pearsons, harm_halves, grad_halves = _split_cosines(
        flow, wins_a, wins_b, rng, n_splits=n_splits, replicates=replicates
    )

    curl_half = float(np.mean(curl_halves))
    grad_half = float(np.mean(grad_halves))
    curl_rel = _spearman_brown(curl_half)
    grad_rel = _spearman_brown(grad_half)
    reliabilities = np.array([_spearman_brown(h) for h in curl_halves])
    spread = (float(np.quantile(reliabilities, 0.025)), float(np.quantile(reliabilities, 0.975)))
    # An interval for the mean, on the half-correlation scale where the splits are being averaged,
    # then mapped through Spearman-Brown, which is monotone increasing so the endpoints stay the
    # endpoints. This is Monte Carlo error in the reported number and nothing else.
    se = (
        float(np.std(curl_halves, ddof=1) / np.sqrt(len(curl_halves)))
        if len(curl_halves) > 1
        else 0.0
    )
    ci = (_spearman_brown(curl_half - 1.96 * se), _spearman_brown(curl_half + 1.96 * se))

    planted_rel = _planted_curl_control(
        flow, wins_a, wins_b, seed=seed + 1, n_splits=n_splits, replicates=replicates
    )
    dim_curl, dim_grad = _subspace_dimensions(flow)

    control_failed = not np.isfinite(planted_rel) or planted_rel < control_floor
    if grad_rel < control_floor:
        verdict = "underpowered"
        note = (
            f"the positive control failed: the gradient component itself replicates at only "
            f"{grad_rel:.3f}, so this design cannot distinguish a curl that is noise from a dataset "
            f"in which nothing replicates. The curl number below is not evidence either way."
        )
    elif curl_rel >= floor:
        verdict = "replicates"
        note = (
            f"the curl component replicates across independent halves of the replications at "
            f"{curl_rel:.3f}, against a gradient control of {grad_rel:.3f}. Measurement error does "
            f"not replicate, so this cyclic structure is a property of the grader."
        )
    elif control_failed:
        verdict = "underpowered"
        note = (
            f"the curl does not replicate ({curl_rel:.3f}) and the matched control says that is "
            f"uninformative: a curl planted at exactly the observed energy, in the same "
            f"{dim_curl}-dimensional subspace and through the same sampling noise, replicates at "
            f"only {planted_rel:.3f}. The gradient control's {grad_rel:.3f} does not settle this, "
            f"because it lives in {dim_grad} dimensions rather than {dim_curl} and is easier to "
            f"replicate at the same signal energy. This design cannot see a cyclic structure of "
            f"this size, so the reading is about the design and not about the grader."
        )
    else:
        verdict = "does_not_replicate"
        note = (
            f"the curl component does not replicate ({curl_rel:.3f}) while a curl planted at the "
            f"same energy in the same subspace does ({planted_rel:.3f}), so the design can see a "
            f"cyclic structure of this size and this one is not there. The gradient replicates at "
            f"{grad_rel:.3f}. On this data the cyclic mass is measurement error and the instrument "
            f"should report a noise level rather than an intransitivity."
        )
    return SplitHalfResult(
        n_splits=n_splits,
        replications_per_pair=observed,
        curl_half=curl_half,
        curl_reliability=curl_rel,
        curl_ci=ci,
        curl_split_spread=spread,
        curl_pearson=float(np.nanmean(curl_pearsons)),
        harmonic_half=float(np.mean(harm_halves)),
        harmonic_reliability=_spearman_brown(float(np.mean(harm_halves))),
        gradient_half=grad_half,
        gradient_reliability=grad_rel,
        planted_curl_reliability=planted_rel,
        curl_subspace_dim=dim_curl,
        gradient_subspace_dim=dim_grad,
        verdict=verdict,
        floor=floor,
        note=note,
    )


def _split_cosines(
    flow: ComparisonFlow,
    wins_a: np.ndarray,
    wins_b: np.ndarray,
    rng: np.random.Generator,
    *,
    n_splits: int,
    replicates: Mapping[Edge, Sequence[int]] | None,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """One split-half loop, factored so the observation and its matched control run identically."""
    curl_halves: list[float] = []
    curl_pearsons: list[float] = []
    harm_halves: list[float] = []
    grad_halves: list[float] = []
    for _ in range(n_splits):
        if replicates is None:
            take = np.floor(flow.replications / 2.0).astype(np.int64)
            b_first = rng.hypergeometric(
                np.maximum(wins_b.astype(np.int64), 0),
                np.maximum(wins_a.astype(np.int64), 0),
                take,
            )
            a_first = take - b_first
            b_second = wins_b - b_first
            a_second = wins_a - a_first
        else:
            a_first, b_first, a_second, b_second = _split_verdicts(flow, replicates, rng)
        first = split_flow(_rebuild(flow, a_first, b_first), with_betti=False)
        second = split_flow(_rebuild(flow, a_second, b_second), with_betti=False)
        curl_halves.append(_cosine(first.curl, second.curl))
        curl_pearsons.append(_pearson(first.curl, second.curl))
        harm_halves.append(_cosine(first.harmonic, second.harmonic))
        grad_halves.append(_cosine(first.gradient, second.gradient))
    return curl_halves, curl_pearsons, harm_halves, grad_halves


def _subspace_dimensions(flow: ComparisonFlow) -> tuple[int, int]:
    """``(dim im(C), dim im(grad))`` on this comparison graph, which is why the controls differ."""
    tris = triangles(list(flow.edges))
    curl_op = curl_operator(list(flow.edges), tris)
    grad_op = incidence(flow.n_items, list(flow.edges)).T.tocsr()
    dim_curl = int(np.linalg.matrix_rank(curl_op.toarray())) if curl_op.shape[1] else 0
    dim_grad = int(np.linalg.matrix_rank(grad_op.toarray())) if grad_op.shape[1] else 0
    return dim_curl, dim_grad


def _planted_curl_control(
    flow: ComparisonFlow,
    wins_a: np.ndarray,
    wins_b: np.ndarray,
    *,
    seed: int,
    n_splits: int,
    replicates: Mapping[Edge, Sequence[int]] | None,
) -> float:
    """The reliability a curl of the **observed** energy would have on this exact design.

    This is the matched positive control the ``does_not_replicate`` verdict needs. The gradient
    control cannot do the job: it lives in ``n - 1`` dimensions where the curl lives in
    ``C(n,2) - (n-1)``, so at equal signal energy it replicates better for reasons that have nothing
    to do with the grader.

    The construction holds everything fixed except the question. The observed flow's gradient
    component is kept, its curl component is replaced by a fixed random direction in ``im(C)``
    rescaled to exactly the observed curl energy, the result is turned back into win probabilities
    and resampled at the observed replication counts, and the identical split loop runs on it. A
    planted curl that fails to replicate says the design cannot see a cyclic structure of this size,
    whatever the observed curl did.

    Returns NaN when there is no curl subspace or no curl energy to plant, which is not a failure:
    a design with no filled triangle has nothing to be cyclic about and the ``curl_mass`` is already
    zero.
    """
    tris = triangles(list(flow.edges))
    curl_op = curl_operator(list(flow.edges), tris)
    if curl_op.shape[1] == 0:
        return float("nan")
    observed_split = split_flow(flow, with_betti=False)
    if observed_split.curl_energy <= 0.0:
        return float("nan")

    rng = np.random.default_rng(seed)
    direction = np.asarray(curl_op @ rng.standard_normal(curl_op.shape[1]), dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        return float("nan")
    planted = direction * (observed_split.curl_energy**0.5 / norm)
    target = observed_split.gradient + planted

    synthetic = ComparisonFlow(
        n_items=flow.n_items,
        edges=flow.edges,
        flow=target,
        replications=flow.replications,
        scale=flow.scale,
        correction=flow.correction,
        correction_value=flow.correction_value,
    )
    probability = np.clip(0.5 * (1.0 + _signed(synthetic)), 1e-6, 1.0 - 1e-6)
    trials = flow.replications.astype(np.int64)
    planted_b = rng.binomial(trials, probability).astype(np.float64)
    planted_a = flow.replications - planted_b
    # The hypergeometric route always, even when the observation supplied per-replication verdicts:
    # the planted data has counts and no verdict list, and splitting it against the observation's
    # blocking would be splitting one dataset along another's structure.
    curl_halves, *_ = _split_cosines(
        flow, planted_a, planted_b, rng, n_splits=n_splits, replicates=None
    )
    return _spearman_brown(float(np.mean(curl_halves)))


def _signed(flow: ComparisonFlow) -> np.ndarray:
    """Recover the win fraction difference from the flow, on whichever scale it was built.

    On the margin scale the flow *is* ``(w_b - w_a)/R``. On the logit scale it is not, so it is
    inverted through the correction that produced it, which is why the correction travels on the
    flow object rather than living in a comment.

    The correction is read off ``flow.correction_value`` rather than assumed. It used to be the
    literal 0.5 while `edge_flow` accepted any value, so a flow built at h = 0.1 came back with win
    fractions off by 0.045 and one built at h = 2.0 by 0.200. That is not a cosmetic error: the
    recovered fractions are what `bradley_terry_scores` fits, so a non-default correction moved the
    fitted scores and therefore the null that Null A compares the observation against.
    """
    if flow.scale == "margin":
        return flow.flow
    odds = np.exp(flow.flow)
    correction = float(flow.correction_value)
    total = flow.replications
    w_b = (odds * (total + 2.0 * correction)) / (1.0 + odds) - correction
    signed = (2.0 * w_b - total) / np.maximum(total, 1e-12)
    return np.asarray(np.clip(signed, -1.0, 1.0), dtype=np.float64)


def _split_verdicts(
    flow: ComparisonFlow,
    replicates: Mapping[Edge, Sequence[int]],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split supplied per-replication verdicts, coded +1 when the higher-indexed item won.

    The odd replication keeps its place. This used to take ``order[:half]`` and
    ``order[half:2*half]`` with ``half = R // 2``, so at the eleven replications the nulls are gated
    on it split 5 and 5 and threw the eleventh away, while the counts-only route splits 5 and 6 and
    uses all eleven. The two paths are documented as equivalent and a discarded observation is not
    equivalent to a used one, least of all at the smallest replication count the module accepts.
    """
    a1 = np.zeros(flow.n_edges)
    b1 = np.zeros(flow.n_edges)
    a2 = np.zeros(flow.n_edges)
    b2 = np.zeros(flow.n_edges)
    for k, edge in enumerate(flow.edges):
        verdicts = np.asarray(replicates.get(edge, ()), dtype=np.float64)
        if verdicts.size == 0:
            raise ValueError(f"replicates supplied but edge {edge} has none")
        order = rng.permutation(verdicts.size)
        half = verdicts.size // 2
        first, second = verdicts[order[:half]], verdicts[order[half:]]
        b1[k], a1[k] = float(np.sum(first > 0)), float(np.sum(first < 0))
        b2[k], a2[k] = float(np.sum(second > 0)), float(np.sum(second < 0))
    return a1, b1, a2, b2


# ---------------------------------------------------------------------------
# Null A. Parametric bootstrap under Bradley-Terry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapResult:
    """Null A: the excess of the observed mass over what a fitted scalar model reproduces.

    A raw curl mass answers no question, because a Bradley-Terry world with a realistic flip rate
    also produces one. What is interpretable is ``curl_excess``: how much more cyclic energy the
    grader carries than a scalar model of it would generate through the identical pipeline.
    """

    n_draws: int
    replications_per_pair: float
    curl_observed: float
    curl_null_mean: float
    curl_excess: float
    curl_null_ci: tuple[float, float]
    curl_p_value: float
    harmonic_observed: float
    harmonic_null_mean: float
    harmonic_excess: float
    fit_iterations: int
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_draws": self.n_draws,
            "replications_per_pair": self.replications_per_pair,
            "curl_observed": self.curl_observed,
            "curl_null_mean": self.curl_null_mean,
            "curl_excess": self.curl_excess,
            "curl_null_ci_low": self.curl_null_ci[0],
            "curl_null_ci_high": self.curl_null_ci[1],
            "curl_p_value": self.curl_p_value,
            "harmonic_observed": self.harmonic_observed,
            "harmonic_null_mean": self.harmonic_null_mean,
            "harmonic_excess": self.harmonic_excess,
            "fit_iterations": self.fit_iterations,
            "note": self.note,
        }


def bradley_terry_scores(
    flow: ComparisonFlow, *, prior: float = 0.5, iterations: int = 500, tol: float = 1e-10
) -> tuple[np.ndarray, int]:
    """Bradley-Terry maximum likelihood by minorization-maximization, with a pseudo-count prior.

    The MM update ``p_i ← W_i / Σ_j n_ij/(p_i + p_j)`` is monotone in the likelihood and needs no
    step size. The prior adds ``prior`` wins to each side of every observed pair, which is what keeps
    the fit finite under separation: with one unanimous comparison per pair the unpenalised MLE
    diverges and every fitted score runs off to ±∞, so an unregularised fit here would not be a
    conservative choice, it would be no fit at all.

    Worth stating because it is why this null exists: the fitted score is a potential on the items,
    so the fitted comparison kernel lies in ``im(grad)`` exactly. Bradley-Terry fitting is Hodge
    gradient projection, and the curl and harmonic energy is discarded by the model class rather than
    by the sample size.
    """
    wins = np.zeros(flow.n_items)
    counts: dict[tuple[int, int], float] = {}
    signed = _signed(flow)
    w_b = (flow.replications + flow.replications * signed) / 2.0
    w_a = flow.replications - w_b
    for (a, b), wa, wb, r in zip(flow.edges, w_a, w_b, flow.replications):
        wins[a] += wa + prior
        wins[b] += wb + prior
        counts[(a, b)] = counts.get((a, b), 0.0) + r + 2.0 * prior

    idx_a = np.asarray([a for a, _ in counts], dtype=np.int64)
    idx_b = np.asarray([b for _, b in counts], dtype=np.int64)
    n_ij = np.asarray(list(counts.values()), dtype=np.float64)

    p = np.ones(flow.n_items)
    used = 0
    for used in range(1, iterations + 1):
        denominator = np.zeros(flow.n_items)
        shared = n_ij / (p[idx_a] + p[idx_b])
        np.add.at(denominator, idx_a, shared)
        np.add.at(denominator, idx_b, shared)
        updated = np.where(denominator > 0.0, wins / np.maximum(denominator, 1e-300), p)
        updated = np.where(updated > 0.0, updated, 1e-300)
        if float(np.max(np.abs(np.log(updated) - np.log(p)))) < tol:
            p = updated
            break
        p = updated
    return np.log(p), used


def bradley_terry_bootstrap(
    flow: ComparisonFlow, *, n_draws: int = 200, seed: int = 0, prior: float = 0.5
) -> BootstrapResult | Refusal:
    """Fit Bradley-Terry, resimulate through the identical pipeline, report the excess.

    Each draw resamples every edge as ``Binomial(R_e, σ(s_j - s_i))`` at the fitted scores, rebuilds
    the flow with `edge_flow` on the same scale, and decomposes it with the same operators. Nothing
    in the null path is a shortcut, because a bootstrap that reaggregates differently from the
    observation is measuring the difference between the two aggregations.
    """
    if flow.n_edges == 0:
        return _refuse_replications(
            "Null A (parametric bootstrap)", 0.0, "the parametric bootstrap"
        )
    observed_r = flow.min_replications
    if observed_r < MIN_REPLICATIONS_FOR_NULLS:
        return _refuse_replications(
            "Null A (parametric bootstrap)", observed_r, "the parametric bootstrap"
        )

    scores, iterations = bradley_terry_scores(flow, prior=prior)
    truth = split_flow(flow, with_betti=False)
    a_idx = np.asarray([a for a, _ in flow.edges], dtype=np.int64)
    b_idx = np.asarray([b for _, b in flow.edges], dtype=np.int64)
    probability = 1.0 / (1.0 + np.exp(-(scores[b_idx] - scores[a_idx])))
    trials = flow.replications.astype(np.int64)

    rng = np.random.default_rng(seed)
    curls = np.empty(n_draws)
    harmonics = np.empty(n_draws)
    for d in range(n_draws):
        wins_b = rng.binomial(trials, probability).astype(np.float64)
        draw = split_flow(_rebuild(flow, trials - wins_b, wins_b), with_betti=False)
        curls[d] = draw.curl_mass
        harmonics[d] = draw.harmonic_mass

    return BootstrapResult(
        n_draws=n_draws,
        replications_per_pair=observed_r,
        curl_observed=truth.curl_mass,
        curl_null_mean=float(np.mean(curls)),
        curl_excess=float(truth.curl_mass - np.mean(curls)),
        curl_null_ci=(float(np.quantile(curls, 0.025)), float(np.quantile(curls, 0.975))),
        curl_p_value=_tail_p(curls, truth.curl_mass),
        harmonic_observed=truth.harmonic_mass,
        harmonic_null_mean=float(np.mean(harmonics)),
        harmonic_excess=float(truth.harmonic_mass - np.mean(harmonics)),
        fit_iterations=iterations,
        note=(
            "the null is a fitted scalar model of this grader, resimulated at the observed "
            "replication counts. An excess near zero means a Bradley-Terry world with this flip "
            "rate would produce the same cyclic mass."
        ),
    )


# ---------------------------------------------------------------------------
# Null D. Anisotropy of the curl spectrum
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnisotropyResult:
    """Null D: whether the curl has a direction, and which items load on it.

    Isotropic noise spreads evenly over the curl subspace. A genuine conflict between two criteria
    concentrates in one rotation plane, and the items loading on that plane are the two criteria,
    named. That is the step that turns a defence of the curl mass into a finding.
    """

    n_items: int
    n_components: int
    energy_covered: float
    singular_values: tuple[float, ...]
    leading_plane_fraction: float
    participation_ratio: float
    #: The same moment ratio counted over rotation planes rather than over singular values, which is
    #: the convention `leading_plane_fraction` uses. Exactly half of `participation_ratio`.
    participation_ratio_planes: float
    n_planes: int
    null_leading_mean: float
    null_leading_ci: tuple[float, float]
    p_value: float
    top_plane: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((0, 2)))
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_items": self.n_items,
            "n_components": self.n_components,
            "energy_covered": self.energy_covered,
            "singular_values": list(self.singular_values[:10]),
            "leading_plane_fraction": self.leading_plane_fraction,
            "participation_ratio": self.participation_ratio,
            "participation_ratio_planes": self.participation_ratio_planes,
            "n_planes": self.n_planes,
            "participation_ratio_convention": (
                "participation_ratio is (sum s^2)^2 / sum s^4 over all 2m singular values, so an "
                "isotropic curl over m planes gives 2m; participation_ratio_planes counts planes, "
                "so the same curl gives m, and that is the convention leading_plane_fraction uses. "
                "The two differ by exactly a factor of 2 because a real skew matrix's singular "
                "values come in equal pairs."
            ),
            "null_leading_mean": self.null_leading_mean,
            "null_leading_ci_low": self.null_leading_ci[0],
            "null_leading_ci_high": self.null_leading_ci[1],
            "p_value": self.p_value,
            "note": self.note,
        }


def curl_anisotropy(
    flow: ComparisonFlow,
    split: HodgeSplit | None = None,
    *,
    n_draws: int = 200,
    seed: int = 0,
) -> AnisotropyResult | Refusal:
    """Eigendecompose the curl component and test it against an isotropic curl null.

    The curl component is written back as an antisymmetric matrix over the items, which is the form
    in which it has a spectrum: a real antisymmetric matrix has its energy in rotation planes, its
    singular values come in equal pairs, and the leading plane is the plane of item space in which
    preference rotates. ``leading_plane_fraction`` is the share of energy in that plane.

    The null holds the subspace fixed and randomises the direction inside it, which is what makes it
    a test of anisotropy rather than of curl magnitude: isotropic Gaussian edge noise is projected
    onto the same ``im(C)`` and rescaled to the same energy. A leading plane no larger than that null
    means the curl has no direction to interpret.

    ``participation_ratio`` is the moment ratio ``(Σλ)²/Σλ²`` on the squared singular values, summed
    over **all 2m of them**. Under that convention an isotropic curl over ``m`` rotation planes
    reads ``2m`` and not ``m``, because a real skew matrix's singular values come in equal pairs and
    both members of each pair are counted. Its sibling ``leading_plane_fraction`` counts *planes*,
    so the two are on conventions that differ by exactly a factor of two, measured at 2.223646
    against 1.111823 on a six-item design. Both are reported, ``participation_ratio_planes`` is the
    one on the same footing as ``leading_plane_fraction``, and neither is silently the other.
    """
    split = split if split is not None else split_flow(flow, with_betti=False)
    if split.curl_energy <= 0.0:
        return Refusal(
            instrument="Null D (curl anisotropy)",
            reason=RefusalReason.BELOW_LOD,
            detail="the curl component is exactly zero, so it has no spectrum to be anisotropic",
            remedy=(
                "there is nothing to interpret here and that is the answer: this grader's flow is "
                "gradient and harmonic only. Report the harmonic mass instead, which is the half "
                "that says to compare different pairs."
            ),
            statistics={"curl_energy": split.curl_energy},
        )

    parts = _components(flow)
    parts.sort(key=lambda p: -float(np.sum(split.curl[p[1]] ** 2)))
    items, edge_idx = parts[0]
    covered = float(np.sum(split.curl[edge_idx] ** 2) / split.curl_energy)
    if items.size < 4:
        return Refusal(
            instrument="Null D (curl anisotropy)",
            reason=RefusalReason.ACCESS_INSUFFICIENT,
            detail=(
                f"the largest connected comparison graph has {items.size} items, and a spectrum "
                f"needs at least four to have more than one rotation plane to compare"
            ),
            remedy=(
                "compare at least four items that are all connected to each other. A corpus of "
                "three-item tournaments cannot say which criteria compete, because every "
                "three-item curl has exactly one plane and its anisotropy is one by construction."
            ),
            statistics={"largest_component_items": int(items.size)},
        )

    position = {int(v): k for k, v in enumerate(items)}
    local_edges = [flow.edges[k] for k in edge_idx]

    def spectrum(values: np.ndarray) -> np.ndarray:
        matrix = np.zeros((items.size, items.size))
        for (a, b), v in zip(local_edges, values):
            matrix[position[a], position[b]] = v
            matrix[position[b], position[a]] = -v
        return np.linalg.svd(matrix, compute_uv=False)

    def leading(sigma: np.ndarray) -> float:
        total = float(np.sum(sigma**2))
        return 0.0 if total <= 0.0 else float(np.sum(sigma[:2] ** 2) / total)

    observed_sigma = spectrum(split.curl[edge_idx])
    observed_leading = leading(observed_sigma)
    squared = observed_sigma**2
    ratio = float(np.sum(squared) ** 2 / np.sum(squared**2)) if np.sum(squared**2) > 0 else 0.0
    # The same moment ratio counted once per rotation plane. A real skew matrix's singular values
    # come in equal pairs, so summing over all of them doubles the answer relative to the
    # plane-counting convention `leading_plane_fraction` uses.
    per_plane = squared[0::2]
    ratio_planes = (
        float(np.sum(per_plane) ** 2 / np.sum(per_plane**2)) if np.sum(per_plane**2) > 0 else 0.0
    )

    curl_op = curl_operator(local_edges, triangles(local_edges))
    rng = np.random.default_rng(seed)
    energy = float(np.sum(split.curl[edge_idx] ** 2))
    null = np.empty(n_draws)
    for d in range(n_draws):
        noise = rng.standard_normal(len(local_edges))
        coefficients, *_ = np.linalg.lstsq(curl_op.toarray(), noise, rcond=None)
        projected = curl_op @ coefficients
        norm = float(np.linalg.norm(projected))
        if norm == 0.0:
            null[d] = 0.0
            continue
        null[d] = leading(spectrum(projected * (energy**0.5 / norm)))

    matrix = np.zeros((items.size, items.size))
    for (a, b), v in zip(local_edges, split.curl[edge_idx]):
        matrix[position[a], position[b]] = v
        matrix[position[b], position[a]] = -v
    left = np.linalg.svd(matrix)[0][:, :2]

    return AnisotropyResult(
        n_items=int(items.size),
        n_components=len(parts),
        energy_covered=covered,
        singular_values=tuple(float(s) for s in observed_sigma),
        leading_plane_fraction=observed_leading,
        participation_ratio=ratio,
        participation_ratio_planes=ratio_planes,
        n_planes=int(per_plane.size),
        null_leading_mean=float(np.mean(null)),
        null_leading_ci=(float(np.quantile(null, 0.025)), float(np.quantile(null, 0.975))),
        p_value=_tail_p(null, observed_leading),
        top_plane=left,
        note=(
            "computed on the largest connected comparison graph, which carries "
            f"{covered:.1%} of the corpus curl energy. A corpus of disjoint per-prompt tournaments "
            "shares no items across prompts, so a spectrum over the union would mix bases that are "
            "not the same basis."
        ),
    )


# ---------------------------------------------------------------------------
# Null E. The random-profile baseline, and the transitive baseline beside it
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProfileBaseline:
    """Null E: the mass a stated null grader produces on this exact comparison design.

    Both members return this. The design is held fixed, which is what makes the comparison fair:
    the null draws are simulated on the observed edges at the observed replication counts, so any
    difference is the grader and not the graph.
    """

    kind: str
    n_draws: int
    flip_rate: float
    curl_observed: float
    curl_null_mean: float
    curl_null_ci: tuple[float, float]
    curl_excess: float
    curl_p_value: float
    harmonic_observed: float
    harmonic_null_mean: float
    harmonic_null_ci: tuple[float, float]
    harmonic_excess: float
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "n_draws": self.n_draws,
            "flip_rate": self.flip_rate,
            "curl_observed": self.curl_observed,
            "curl_null_mean": self.curl_null_mean,
            "curl_null_ci_low": self.curl_null_ci[0],
            "curl_null_ci_high": self.curl_null_ci[1],
            "curl_excess": self.curl_excess,
            "curl_p_value": self.curl_p_value,
            "harmonic_observed": self.harmonic_observed,
            "harmonic_null_mean": self.harmonic_null_mean,
            "harmonic_null_ci_low": self.harmonic_null_ci[0],
            "harmonic_null_ci_high": self.harmonic_null_ci[1],
            "harmonic_excess": self.harmonic_excess,
            "note": self.note,
        }


def _simulate_orderings(
    flow: ComparisonFlow,
    parts: Sequence[tuple[np.ndarray, np.ndarray]],
    rng: np.random.Generator,
    *,
    redraw_per_replication: bool,
    flip_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Win counts from total orders drawn per connected component.

    ``redraw_per_replication`` is the whole difference between the two members of Null E. Redrawing
    gives the uniformly random ordinal profile: each replication is a fresh random ranking, so the
    aggregate flow shrinks toward zero as replications accumulate. Holding the order fixed gives a
    *transitive grader*: one coherent ranking, observed repeatedly through a flip rate.

    The two are the same distribution at one replication per pair, and above it they still centre on
    the same curl mass, because the average of transitive flows is not less curly than one of them.
    What redrawing adds is variance, so the random profile is the weaker null wherever both apply.
    """
    wins_b = np.zeros(flow.n_edges)
    trials = flow.replications.astype(np.int64)
    max_r = int(trials.max()) if flow.n_edges else 0
    for items, edge_idx in parts:
        position = {int(v): k for k, v in enumerate(items)}
        local = [(position[a], position[b]) for a, b in (flow.edges[k] for k in edge_idx)]
        a_local = np.asarray([p[0] for p in local], dtype=np.int64)
        b_local = np.asarray([p[1] for p in local], dtype=np.int64)
        rank = rng.permutation(items.size)
        for r in range(max_r):
            active = trials[edge_idx] > r
            if not np.any(active):
                break
            if redraw_per_replication:
                rank = rng.permutation(items.size)
            outcome = (rank[b_local] > rank[a_local]).astype(np.float64)
            if flip_rate > 0.0:
                flips = rng.random(outcome.size) < flip_rate
                outcome = np.where(flips, 1.0 - outcome, outcome)
            wins_b[edge_idx] += outcome * active
    return trials - wins_b, wins_b


def _profile_null(
    flow: ComparisonFlow,
    *,
    kind: str,
    redraw_per_replication: bool,
    flip_rate: float,
    n_draws: int,
    seed: int,
    note: str,
) -> ProfileBaseline | Refusal:
    if flow.n_edges == 0:
        return Refusal(
            instrument=f"Null E ({kind})",
            reason=RefusalReason.ACCESS_INSUFFICIENT,
            detail="the comparison flow has no edges, so there is no design to simulate on",
            remedy="supply at least one compared pair.",
        )
    truth = split_flow(flow, with_betti=False)
    parts = _components(flow)
    rng = np.random.default_rng(seed)
    curls = np.empty(n_draws)
    harmonics = np.empty(n_draws)
    for d in range(n_draws):
        wins_a, wins_b = _simulate_orderings(
            flow, parts, rng, redraw_per_replication=redraw_per_replication, flip_rate=flip_rate
        )
        draw = split_flow(_rebuild(flow, wins_a, wins_b), with_betti=False)
        curls[d] = draw.curl_mass
        harmonics[d] = draw.harmonic_mass
    return ProfileBaseline(
        kind=kind,
        n_draws=n_draws,
        flip_rate=flip_rate,
        curl_observed=truth.curl_mass,
        curl_null_mean=float(np.mean(curls)),
        curl_null_ci=(float(np.quantile(curls, 0.025)), float(np.quantile(curls, 0.975))),
        curl_excess=float(truth.curl_mass - np.mean(curls)),
        curl_p_value=_tail_p(curls, truth.curl_mass),
        harmonic_observed=truth.harmonic_mass,
        harmonic_null_mean=float(np.mean(harmonics)),
        harmonic_null_ci=(
            float(np.quantile(harmonics, 0.025)),
            float(np.quantile(harmonics, 0.975)),
        ),
        harmonic_excess=float(truth.harmonic_mass - np.mean(harmonics)),
        note=note,
    )


def random_profile_baseline(
    flow: ComparisonFlow, *, n_draws: int = 200, seed: int = 0
) -> ProfileBaseline | Refusal:
    """Uniformly random ordinal profiles through the identical pipeline.

    Each replication of each pair is decided by a fresh uniformly random total order over the items
    of that connected component. This is the classical incoherent-preference null, and it is
    computed because it is the one a reader will ask for rather than because it is the informative
    one.

    At one replication per pair it coincides exactly with `transitive_baseline` at a zero flip rate,
    since a single random total order is transitive. Above one replication it sits **above** the
    transitive baseline rather than on it, by 0.0358 at eleven replications and 0.0669 at 101 on a
    complete five-item design, because the curl mass is a ratio of quadratic forms and averaging
    transitive draws does not preserve it. The transitive baseline dominates it as a null at every
    replication count, and this one is reported beside it rather than instead of it.
    """
    return _profile_null(
        flow,
        kind="random_ordinal_profile",
        redraw_per_replication=True,
        flip_rate=0.0,
        n_draws=n_draws,
        seed=seed,
        note=(
            "every replication of every pair decided by a fresh uniformly random total order. At "
            "one replication per pair this is the same distribution as a perfectly transitive "
            "grader, so it distinguishes nothing there."
        ),
    )


def transitive_baseline(
    flow: ComparisonFlow, *, flip_rate: float = 0.0, n_draws: int = 200, seed: int = 0
) -> ProfileBaseline | Refusal:
    """A perfectly transitive grader with a stated flip rate, on this exact comparison design.

    One coherent total order per connected component, every observed pair decided by it, and each
    comparison flipped independently with probability ``flip_rate``. This is B1's declared baseline
    and it is the one that decides whether a curl mass means anything, because a transitive
    preference recorded as wins and losses already carries curl: on the complete graph ``K_n`` with a
    zero flip rate the mass is exactly ``(n-2)/(3n)``, which is 0.238 at seven items. A grader whose
    curl mass sits at that value has produced no intransitivity at all.

    Pass the grader's measured flip rate from A6 when there is one. A zero flip rate is the sharpest
    version of the baseline and the right default, because it isolates the encoding floor from the
    noise floor.
    """
    return _profile_null(
        flow,
        kind="transitive_grader",
        redraw_per_replication=False,
        flip_rate=flip_rate,
        n_draws=n_draws,
        seed=seed,
        note=(
            f"one coherent total order per comparison graph, each comparison flipped with "
            f"probability {flip_rate:g}. The curl this produces is the price of recording a "
            f"comparison as a sign rather than a margin, and it contains no intransitivity."
        ),
    )


def transitive_curl_mass_complete(n_items: int) -> float:
    """``(n-2)/(3n)``: the curl mass of a transitive ±1 tournament on the complete graph.

    Exact and independent of which total order it is, by symmetry. Kept as a function because it is
    the closed form a reader can check `transitive_baseline` against on the one design where a closed
    form exists, and because it makes the size of the encoding floor legible without a simulation:
    0.111 at three items, 0.238 at seven, 0.250 at eight, and one third in the limit.
    """
    if n_items < 3:
        return 0.0
    return (n_items - 2) / (3.0 * n_items)


__all__ = [
    "CONTROL_FLOOR",
    "RELIABILITY_FLOOR",
    "AnisotropyResult",
    "BootstrapResult",
    "ProfileBaseline",
    "SplitHalfResult",
    "bradley_terry_bootstrap",
    "bradley_terry_scores",
    "curl_anisotropy",
    "random_profile_baseline",
    "split_half_reliability",
    "transitive_baseline",
    "transitive_curl_mass_complete",
]

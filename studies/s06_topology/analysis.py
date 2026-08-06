"""Preference topology (S6): a computable fraction of reward error is topologically obligatory.

The claim this study registers is sharp. Take any collection of pairwise preferences over a set of
responses, form the edge flow, and run the combinatorial Hodge decomposition (``hodge.py``). The
gradient part is exactly what a scalar (Bradley-Terry) reward can represent; the curl and harmonic
parts are what it provably cannot. So the ``intransitive_mass`` of a preference corpus is a
coordinate-free lower bound on the error of every scalar reward model on that corpus, computable in
pure numpy with no model and no training. When it is large, no amount of scalar reward-model
capacity closes the gap; the obstruction is in the data's topology, not the fit.

The study calibrates the instrument before it reports, in the corpus's usual discipline of measuring
first where the answer is known by construction:

    Calibration A (curl channel). The planted-intransitivity organism emits tournaments that are
    each a pure three-cycle A > B > C > A. A three-cycle with its triangle filled is pure curl, so
    the decomposition must return an intransitive mass of one, all of it curl. This is the registered
    calibration row T12: the method recovers the planted intransitive mass within tolerance.

    Calibration B (harmonic channel). The foundry's ``curl_harmonic_organism`` is a marked stub, so
    the harmonic channel is calibrated here against a planted ground truth of its own: a chordless
    directed cycle. A ring of comparisons with no interior pair filled has a hole the flow wraps
    around, which is pure harmonic (locally consistent yet globally cyclic), so the decomposition
    must return a harmonic mass of one. This proves the harmonic channel is real and separable from
    curl, not an artifact.

    Measurement. A synthetic judge-tournament corpus stands in for a real preference corpus. Each
    judge scores a pair by a dominant transitive quality plus a context-dependent skew criterion, the
    mechanism by which a genuine multi-attribute judge produces cycles, and the comparison graph is
    left sparse (not every pair is judged) as real judge data is. The measured intransitive mass is
    the headline: measurably nonzero, so scalar reward is provably lossy on this corpus.

The kill criterion is a real scientific fork. If the cyclic mass were uniformly tiny (a few percent
or less), that would be a publishable defense of scalar reward modeling: Bradley-Terry transitivity
would be empirically benign and the scalar bottleneck a non-issue in practice. The registered
prediction is the opposite, and the synthetic corpus is where it is first checked.

The real Nectar, UltraFeedback, HelpSteer, and PRISM slices are the same analysis on human and
judge tournaments loaded through the ``datasets`` extra, which is not installed in this environment,
so that corpus is the marked follow-on rather than run here.

**The registered threshold of 0.03 does not discriminate, and the measurement is fine.** This is
E32's correction applied to this study rather than to the campaign, and the numbers are on the
corpus this file builds. Run the same generator with the skew term switched off, ``skew=0.0``, which
is the control that removes the cycle-producing mechanism and changes nothing else, and the
intransitive mass is **0.0360** (curl 0.0355, harmonic 0.00053). The registered comparator is
``> 0.03``. So a corpus with no intransitivity-generating term in it passes H2 and fails the kill
criterion, which means the threshold separates nothing: E32's rule is that an effect threshold has
to be compared against the floor its own encoding imposes before it is registered, and 0.03 was not.
The simulated version of the same floor is higher still: a perfectly transitive grader recorded on
this exact design (``nulls.transitive_baseline``, 200 draws, flip rate 0) scores curl
0.1805 with a 95% interval of [0.1722, 0.1906].

What survives is the science. The measured corpus is at intransitive mass **0.2570** (curl 0.2463,
harmonic 0.0107), which is **0.2210 above its own no-skew control** and above every one of the 200
transitive draws (p = 0.005, the smallest value 200 draws can return). The corpus genuinely carries
cyclic structure, the decomposition genuinely finds it, and the only thing wrong is that the
registered number was too small to be a test. The frozen spec cannot be edited after the fact, which
is the point of freezing it; what is owed is this paragraph and the transitive baseline reported
beside every mass, which ``read`` below does.
"""

from __future__ import annotations

import numpy as np

from reward_lens.core.evidence import make_evidence
from reward_lens.core.provenance import Provenance
from reward_lens.core.reading import Reading
from reward_lens.core.types import Access, Component, GaugeStatus, SubjectRef
from reward_lens.data.lineage import make_lineage
from reward_lens.data.schema import EdgeObs, Response, Tournament, response_content
from reward_lens.measure.composition.hodge import PairCount, edge_flow, split_flow
from reward_lens.measure.composition.nulls import ProfileBaseline, transitive_baseline
from reward_lens.organisms import intransitivity_organism
from reward_lens.record.schema import Run
from reward_lens.studies.spec import (
    Hypothesis,
    KillCriterion,
    Prediction,
    StudyResult,
    StudySpec,
    SubjectQuery,
)
from studies._retype import MetricSpec, ScienceRetype
from studies.s06_topology.hodge import HodgeDecomposition, decompose_corpus

_VERSION = "1.0"

# The registered calibration and measurement thresholds. A pure three-cycle recovers an intransitive
# mass of exactly one, so 0.99 is a tolerance, not a margin. The synthetic corpus is engineered to
# carry a decisive cyclic mass, well clear of the "few percent" band the kill criterion reserves for
# an empirically benign scalar bottleneck.
_CALIBRATION_TOL = 0.99
_NONZERO_THRESHOLD = 0.03


def build_spec() -> StudySpec:
    """The frozen S6 spec: recover the planted intransitive mass, then measure it in a corpus (T12)."""
    return StudySpec(
        id="s06-topology",
        title="Preference topology: a computable fraction of reward error is topologically "
        "obligatory (Hodge decomposition of pairwise preference)",
        science="S06-topology",
        hypotheses=(
            Hypothesis(
                id="H1-calibration-recovers-planted",
                statement="on the planted-intransitivity organism, whose tournaments are pure "
                "three-cycles, the Hodge decomposition recovers an intransitive mass of one within "
                "tolerance (the curl channel is calibrated)",
                prediction=Prediction(
                    metric="calib_intransitive_mass", comparator=">", threshold=_CALIBRATION_TOL
                ),
                scoreboard_row="T12",
            ),
            Hypothesis(
                id="H2-synthetic-corpus-nonzero",
                statement="a synthetic judge-tournament corpus carries a measurably nonzero "
                "intransitive mass, so scalar reward is provably lossy on it",
                prediction=Prediction(
                    metric="synthetic_intransitive_mass",
                    comparator=">",
                    threshold=_NONZERO_THRESHOLD,
                ),
                scoreboard_row="T12",
            ),
        ),
        analysis="studies.s06_topology.analysis.analyze",
        subjects=SubjectQuery(
            organisms=("intransitivity",),
            extra={
                "note": "controlled organisms plus a synthetic judge corpus; the real Nectar, "
                "UltraFeedback, HelpSteer, and PRISM tournaments are the datasets-extra follow-on"
            },
        ),
        kill_criteria=(
            KillCriterion(
                id="K1-cyclic-mass-benign",
                metric="synthetic_intransitive_mass",
                comparator="<",
                threshold=_NONZERO_THRESHOLD,
                description="cyclic mass is uniformly tiny, so Bradley-Terry transitivity is "
                "empirically benign and the scalar bottleneck a non-issue in practice, which is a "
                "publishable defense of scalar reward modeling",
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Tournament builders (controlled ground truth and the synthetic corpus)
# ---------------------------------------------------------------------------


def _tournament(
    prompt: str, n_items: int, edge_specs: list[tuple[int, int, int, int]], seed_id: str
) -> Tournament:
    """Assemble a `Tournament` from a list of ``(i, j, wins_i, wins_j)`` edges with a stamped lineage.

    The lineage content mirrors ``schema.content_of`` for a tournament exactly, so the item's content
    hash agrees with the dataset checksum a `DataView` would compute over it.
    """
    responses = tuple(Response(text=f"{prompt}::response-{t}") for t in range(n_items))
    edges = tuple(
        EdgeObs(i=i, j=j, wins_i=wins_i, wins_j=wins_j) for (i, j, wins_i, wins_j) in edge_specs
    )
    content = [
        "Tournament",
        prompt,
        [response_content(r) for r in responses],
        [e.__canonical__() for e in edges],
    ]
    lineage = make_lineage(seed_id, "s06.topology", ("synthetic",), content)
    return Tournament(prompt=prompt, responses=responses, edges=edges, lineage=lineage)


def _planted_harmonic_corpus(
    lengths: tuple[int, ...] = (4, 5, 6, 7), wins: int = 5
) -> list[Tournament]:
    """Chordless directed cycles: the planted harmonic ground truth for calibration B.

    A ring ``0 > 1 > ... > (L-1) > 0`` whose only edges are the ring itself has no filled triangle,
    so its single cycle is a hole the flow wraps around. That flow is divergence-free and curl-free
    yet not a gradient, which is the definition of harmonic, and the decomposition should assign it a
    harmonic mass of one. Building several lengths keeps the calibration from depending on one graph.
    """
    tournaments: list[Tournament] = []
    for length in lengths:
        specs: list[tuple[int, int, int, int]] = []
        for step in range(length):
            a, b = step, (step + 1) % length
            # Orient the winner->loser ring edge into canonical (min, max) index order.
            if a < b:
                specs.append((a, b, wins, 0))
            else:
                specs.append((b, a, 0, wins))
        tournaments.append(
            _tournament(f"harmonic-ring-{length}", length, specs, f"harmonic:{length}")
        )
    return tournaments


def _synthetic_judge_corpus(
    *,
    n_prompts: int = 60,
    n_items: int = 6,
    n_dims: int = 6,
    skew: float = 1.5,
    beta: float = 1.5,
    wins_total: int = 10,
    drop: float = 0.25,
    seed: int = 0,
) -> list[Tournament]:
    """A synthetic multi-attribute judge corpus that produces genuine, measurable intransitivity.

    Each response ``t`` carries a scalar quality ``q[t]`` and a feature vector ``phi[t]``. For a pair
    the judge's preference score is the transitive quality gap ``q[b] - q[a]`` plus a skew term
    ``skew * phi[a]^T A phi[b]`` with ``A`` skew-symmetric, so the skew term is antisymmetric in the
    pair and rotates preference through the feature plane the way a context-dependent criterion does.
    Win counts follow a logistic of that score over ``wins_total`` comparisons. A fraction ``drop`` of
    the pairs is left unjudged, so the comparison graph is sparse and some cycles enclose holes: this
    is what lets the corpus carry harmonic mass alongside curl, exactly as real judge data does.
    """
    rng = np.random.default_rng(seed)
    tournaments: list[Tournament] = []
    for prompt_idx in range(n_prompts):
        quality = rng.standard_normal(n_items)
        features = rng.standard_normal((n_items, n_dims))
        raw = rng.standard_normal((n_dims, n_dims))
        skew_op = raw - raw.T
        norm = float(np.linalg.norm(skew_op))
        if norm > 0.0:
            skew_op = skew_op / norm
        specs: list[tuple[int, int, int, int]] = []
        for a in range(n_items):
            for b in range(a + 1, n_items):
                if rng.random() < drop:
                    continue  # this pair was not judged, leaving a hole in the complex
                score = (quality[b] - quality[a]) + skew * float(
                    features[a] @ skew_op @ features[b]
                )
                p_b = 1.0 / (1.0 + np.exp(-beta * score))
                wins_b = int(round(wins_total * p_b))
                wins_a = wins_total - wins_b
                specs.append((a, b, wins_a, wins_b))
        if not specs:
            continue
        tournaments.append(
            _tournament(f"judge-prompt-{prompt_idx}", n_items, specs, f"judge:{seed}:{prompt_idx}")
        )
    return tournaments


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _mass_evidence(
    observable: str, corpus_label: str, decomposition: HodgeDecomposition, study_id: str
) -> "object":
    """A base (unregistered) Evidence carrying one corpus's Hodge masses, to be cited as a parent."""
    return make_evidence(
        observable=observable,
        observable_version=_VERSION,
        subject=SubjectRef(extra={"study": study_id, "corpus": corpus_label}),
        value=decomposition.to_dict(),
        gauge=GaugeStatus.INVARIANT,
        provenance=Provenance(study=study_id),
    )


def analyze(run) -> StudyResult:
    """Calibrate both cyclic channels, measure the synthetic corpus, and register the headline mass."""
    study_id = run.study.study_id

    # Calibration A: the planted-intransitivity organism (pure three-cycles) fixes the curl channel.
    organism_view, _key = intransitivity_organism(n_triads=24, seed=0)
    calib = decompose_corpus(organism_view)
    ev_calib = _mass_evidence("S06.HodgeMass", "organism-intransitivity", calib, study_id)
    run.record(ev_calib)

    # Calibration B: planted chordless cycles fix the harmonic channel (the foundry organism is a stub).
    harmonic_corpus = _planted_harmonic_corpus()
    harmonic = decompose_corpus(harmonic_corpus)
    ev_harmonic = _mass_evidence("S06.HodgeMass", "planted-harmonic", harmonic, study_id)
    run.record(ev_harmonic)

    # Measurement: the synthetic judge corpus stands in for a real preference corpus.
    synthetic_corpus = _synthetic_judge_corpus()
    synthetic = decompose_corpus(synthetic_corpus)
    ev_synthetic = _mass_evidence("S06.HodgeMass", "synthetic-judge", synthetic, study_id)
    run.record(ev_synthetic)

    calib_intransitive_mass = float(calib.intransitive_mass)
    planted_harmonic_recovered = float(harmonic.harmonic_mass)
    synthetic_intransitive_mass = float(synthetic.intransitive_mass)

    # The registered headline: the intransitive-mass measurement, tracing to the three corpora it
    # summarizes. This is the number a card or paper cites.
    ev_mass = make_evidence(
        observable="S06.IntransitiveMass",
        observable_version=_VERSION,
        subject=SubjectRef(extra={"study": study_id}),
        value={
            "calib_intransitive_mass": calib_intransitive_mass,
            "calib_curl_mass": float(calib.curl_mass),
            "calib_harmonic_mass": float(calib.harmonic_mass),
            "planted_harmonic_recovered": planted_harmonic_recovered,
            "synthetic_intransitive_mass": synthetic_intransitive_mass,
            "synthetic_curl_mass": float(synthetic.curl_mass),
            "synthetic_harmonic_mass": float(synthetic.harmonic_mass),
            "synthetic_gradient_mass": float(synthetic.gradient_mass),
        },
        gauge=GaugeStatus.INVARIANT,
        provenance=Provenance(
            study=study_id, parents=(ev_calib.id, ev_harmonic.id, ev_synthetic.id)
        ),
        registered=True,
    )
    run.record(ev_mass)

    return StudyResult(
        outcomes={},
        metrics={
            "calib_intransitive_mass": calib_intransitive_mass,
            "calib_curl_mass": float(calib.curl_mass),
            "planted_harmonic_recovered": planted_harmonic_recovered,
            "synthetic_intransitive_mass": synthetic_intransitive_mass,
            "synthetic_curl_mass": float(synthetic.curl_mass),
            "synthetic_harmonic_mass": float(synthetic.harmonic_mass),
            "synthetic_gradient_mass": float(synthetic.gradient_mass),
        },
        summary=(
            f"The Hodge decomposition recovered {calib_intransitive_mass:.3f} of the planted "
            f"three-cycle mass as intransitive (all curl), and {planted_harmonic_recovered:.3f} of "
            f"the planted chordless-cycle mass as harmonic, calibrating both cyclic channels. On the "
            f"synthetic judge corpus the intransitive mass was {synthetic_intransitive_mass:.3f} "
            f"(curl {float(synthetic.curl_mass):.3f}, harmonic {float(synthetic.harmonic_mass):.3f}), "
            f"a computable lower bound on the error of any scalar reward model on that corpus."
        ),
    )


# ---------------------------------------------------------------------------
# The retype: S6 on the kernel
# ---------------------------------------------------------------------------

#: Both frozen metrics compute the same unregistered thing, so they carry the same request.
_INTRANSITIVE_MASS_GAP = (
    "curl mass plus harmonic mass, the fraction of an edge flow's energy outside im(grad). Unit: 1, "
    "dimension 1, support [0, 1]. Invariance: `reward.affine` and `group.permutation`, invariant "
    "under both, and **not** invariant under a change of the comparison design, which has to be "
    "reported beside it. Both parts are registered, `grader.curl_mass` and `grader.harmonic_mass`, "
    "and their sum is not, which reads as deliberate: B1's own result type says the sum is "
    "'reported only beside its two parts, never instead of them', because the two have different "
    "remedies. Curl is the judge being locally cyclic and harmonic is the comparison design having "
    "holes, and a reader handed the sum cannot tell which they have. Two ways to close this and the "
    "second is better. Register `grader.intransitive_mass` with a condition that a reading carrying "
    "it must carry both parts and the transitive baseline; or amend this frozen spec at 3.1 to "
    "predict on the two parts separately, which is what the science actually claims. The spec "
    "freeze is the only reason the second is not done here."
)

RETYPE = ScienceRetype(
    science="s06_topology",
    spec=build_spec(),
    headline="grader.curl_mass",
    destination=(
        "B1 in measure/composition/. The decomposition this study calls its own is now B1's: "
        "`hodge.py` builds its flow with `edge_flow` and splits it with `split_flow`, and the local "
        "solver is kept beside it as the independent check that certifies both. What the retype "
        "adds above the repoint is the transitive baseline, which is mandatory on every reading of "
        "a curl mass and which this study did not report."
    ),
    needs={Component.GRADER: Access.RECORD, Component.RECORD: Access.RECORD},
    metrics=(
        MetricSpec(
            metric="calib_intransitive_mass",
            arc="calibrate-channels",
            source="organism",
            gap=_INTRANSITIVE_MASS_GAP,
        ),
        MetricSpec(
            metric="synthetic_intransitive_mass",
            arc="measure-corpus",
            source="organism",
            gap=_INTRANSITIVE_MASS_GAP,
        ),
    ),
    arc_requires={"measure-corpus": ("calibrate-channels",)},
    waiting_on=(
        "no quantity id for curl-plus-harmonic. Both parts ship with instruments (B1) and the sum "
        "the two frozen predictions are written on does not exist in the registry."
    ),
)

#: At least three items in a group, or there is no triangle and no cycle to find. At least two
#: leaves, or there is one voter and one voter is transitive by construction.
_MIN_ITEMS = 3
_MIN_VOTERS = 2

#: Groups read before the flow is built. A record with thousands of groups would otherwise spend
#: minutes in the 200-draw baseline for a number that has converged long before.
_MAX_GROUPS = 200


def _leaf_votes(run: Run) -> tuple[list[PairCount], int, dict[str, int]]:
    """The record as a tournament corpus, with the grader's own criteria as the voters.

    A GRPO group is a set of rollouts on one prompt and a multi-leaf score tree is several criteria
    scoring each of them, so a group is a small tournament in which every leaf votes on every pair.
    That is the same object the synthetic judge corpus models, with the criteria of one grader in
    place of a panel of judges, and it is the only tournament a training record contains.

    What it measures is worth stating exactly, because it is not the same claim the study registers
    on a judge corpus: it is whether this grader's criteria, aggregated by pairwise majority rather
    than by the weighted sum the trainer actually used, would be intransitive. Reading the total
    instead answers nothing either way, and the two ways of reading it fail differently. A flow built
    from the total's margins is the difference of a potential, so its curl mass is exactly zero:
    6.9e-32 on a 60-tournament draw of margins made linear in a planted quality. A flow built from
    the total's signs is a total order recorded as signs, so its curl mass is exactly the encoding
    floor `(n-2)/(3n)`: 0.16666666666666680 measured on 50 groups of four against a closed form of
    0.16666666666666666. Neither number is about the grader.

    Returns the accumulated pairs, the item count of the disjoint union, and the per-leaf vote
    counts so a reading can say who voted.
    """
    pairs: list[PairCount] = []
    voters: dict[str, int] = {}
    offset = 0
    groups = 0
    for step in run.steps:
        for group in step.groups:
            trajectories = list(group.trajectories)
            per_leaf: list[dict[str, float]] = []
            for traj in trajectories:
                leaves = dict(_score_leaves(traj.scores))
                per_leaf.append(leaves)
            names = sorted(set.intersection(*(set(d) for d in per_leaf))) if per_leaf else []
            if len(trajectories) < _MIN_ITEMS or len(names) < _MIN_VOTERS:
                for name in names:
                    voters[name] = voters.get(name, 0)
                continue
            for a in range(len(trajectories)):
                for b in range(a + 1, len(trajectories)):
                    wins_a = wins_b = 0.0
                    for name in names:
                        va, vb = per_leaf[a][name], per_leaf[b][name]
                        if va > vb:
                            wins_a += 1.0
                        elif vb > va:
                            wins_b += 1.0
                        voters[name] = voters.get(name, 0) + 1
                    if wins_a + wins_b > 0:
                        pairs.append(PairCount(offset + a, offset + b, wins_a, wins_b))
            offset += len(trajectories)
            groups += 1
            if groups >= _MAX_GROUPS:
                return pairs, offset, voters
    return pairs, offset, voters


def _score_leaves(node: object) -> "list[tuple[str, float]]":
    """Every scalar leaf of one trajectory's score tree, as (name, value)."""
    children = getattr(node, "children", None)
    if children:
        out: list[tuple[str, float]] = []
        for child in children:
            out.extend(_score_leaves(child))
        return out
    name, value = getattr(node, "name", None), getattr(node, "value", None)
    if name is None or value is None:
        return []
    return [(str(name), float(value))]


def _voter_census(run: Run) -> dict[str, int]:
    """Leaf names and how many trajectories carry each, for the refusal that names them."""
    census: dict[str, int] = {}
    for step in run.steps:
        for group in step.groups:
            for traj in group.trajectories:
                for name, _ in _score_leaves(traj.scores):
                    census[name] = census.get(name, 0) + 1
    return census


def read(run: Run) -> Reading:
    """S6 against a training record: the grader's criteria as voters, or a refusal that says why not.

    The record path is the leaf-vote tournament described in `_leaf_votes`, decomposed by B1 and
    reported with the transitive baseline beside it, which E32 makes mandatory: a comparison
    recorded as a win is a sign rather than a margin, and a perfectly transitive grader already
    carries curl mass `(n-2)/(3n)` on the complete graph. A curl mass without that floor beside it
    is uninterpretable, and this study's own registered threshold is an instance of the problem
    rather than an exception to it, which the module docstring above sets out with the numbers.

    Scope limit, three lines in: one grader leaf is one voter, and a single voter's votes are the
    signs of a total order, so the curl mass of a one-leaf record is not zero, it is exactly the
    encoding floor. Measured on 50 groups of four: 0.16666666666666680 against a closed form of
    1/6. Reporting that as intransitivity is E32 happening again, so a one-leaf record is refused.
    Both shipped GRPO fixtures are that case: one `length_reward` leaf on every trajectory.
    """
    if (refusal := RETYPE.access_refusal(run, remedy=_ACCESS_REMEDY)) is not None:
        return refusal

    pairs, n_items, voters = _leaf_votes(run)
    if not pairs:
        census = _voter_census(run)
        return RETYPE.incomplete(
            field=(
                f"score tree with at least {_MIN_VOTERS} leaves on a group of at least "
                f"{_MIN_ITEMS} rollouts, so it holds no tournament"
            ),
            subject=(
                f"run {run.id}, whose grader leaves are "
                f"{', '.join(f'{k} on {v} trajectories' for k, v in sorted(census.items())) or 'none'},"
            ),
            remedy=(
                "score with a grader whose criteria are recorded as separate leaves, through the "
                "tap's score tree, and the criteria become the voters. Or point this at a judge "
                "corpus with recorded pairwise comparisons, which is the subject the study's own "
                "claim is about: Nectar, UltraFeedback, HelpSteer, PRISM. One leaf is one voter, "
                "its votes are the signs of a total order, and such a flow carries curl mass "
                "exactly (n-2)/(3n) with no intransitivity in it at all, so the number this would "
                "return is a property of the encoding rather than of the grader."
            ),
            leaves=sorted(census),
            n_leaves=len(census),
        )

    flow = edge_flow(pairs, n_items)
    split = split_flow(flow)
    baseline = transitive_baseline(flow, flip_rate=0.0, n_draws=200, seed=0)
    floor: dict[str, float] = {}
    if isinstance(baseline, ProfileBaseline):
        floor = {
            "baseline.transitive_curl": baseline.curl_null_mean,
            "baseline.transitive_harmonic": baseline.harmonic_null_mean,
        }

    excess = (
        split.curl_mass - baseline.curl_null_mean
        if isinstance(baseline, ProfileBaseline)
        else float("nan")
    )
    return RETYPE.evidence(
        run,
        {},
        measured={
            "curl_mass": (split.curl_mass, "grader.curl_mass"),
            "harmonic_mass": (split.harmonic_mass, "grader.harmonic_mass"),
        },
        quantity="grader.curl_mass",
        baselines=floor or None,
        refusals=dict(_STANDING_REFUSALS),
        summary=(
            f"{len(voters)} grader leaves voting over {n_items} rollouts and {split.n_edges} "
            f"compared pairs: curl mass {split.curl_mass:.4g}, harmonic {split.harmonic_mass:.4g}, "
            f"against a transitive-grader floor of "
            f"{floor.get('baseline.transitive_curl', float('nan')):.4g} on this exact design, an "
            f"excess of {excess:.4g}. Both frozen metrics are refused: they are the sum of these "
            f"two masses and the registry has no id for the sum."
        ),
        gauge=GaugeStatus.INVARIANT,
        voters=sorted(voters),
        betti1=int(split.betti1),
        n_triangles=int(split.n_triangles),
        curl_excess_over_transitive=excess,
        solver=split.solver,
    )


#: Both frozen metrics are the same unregistered sum, and each names a different corpus.
_STANDING_REFUSALS = {
    "calib_intransitive_mass": (
        "the calibration is against a planted three-cycle corpus, and a record contains no plant. "
        "Run analyze(), which plants both channels and recovers them. The sum this metric names has "
        "no registered quantity id either, so it would not be reportable from a record that had one."
    ),
    "synthetic_intransitive_mass": (
        "this is the mass of the study's own synthetic judge corpus, which is built rather than "
        "recorded, and the sum it names has no registered quantity id. The record's own curl and "
        "harmonic masses are reported above under the two ids that do exist."
    ),
}

_ACCESS_REMEDY = (
    "open a run written by the recorder whose per-leaf grader scores were captured. S6 needs no "
    "activations: the decomposition is linear algebra on recorded comparisons."
)


__all__ = ["RETYPE", "build_spec", "analyze", "read"]

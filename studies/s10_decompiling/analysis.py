"""S10 — Decompiling the reward: the legibility frontier and the tacit residual (DESIGN Part III, S10).

How much of a reward model's decision function can be said in language? Build the natural-language
surrogate ``r_hat_K = sum_i w_i pi_i`` from a library of predicates ``pi_i`` fit to the model's
scores within a description-length budget of ``K`` predicates, and trace the legibility frontier
``fidelity(K)``, the fraction of the reward's variance the best K-predicate rubric explains. Where
that frontier stops rising is the legible core; what remains, ``rho = r - r_hat_{K*}``, is the tacit
residual: the part of the decision function no short rubric captures, and the corpus's hypothesis is
that it is where reward hacks are financed.

The calibration arm makes the answer knowable by construction and rides the real legibility
instrument `reward_lens.measure.indices.legibility_frontier`. The synthetic reward is a known
k-predicate rubric plus a known tacit remainder: ``k`` of a library of candidate predicates carry
real weight, the rest carry none, and an independent smooth term supplies a planted tacit fraction of
the total variance. The real frontier fits ``r_hat_K`` cheapest-predicates-first up to a
description-length budget and reads back ``fidelity(K)``, its knee ``K*``, and the tacit residual. The
rubric predicates are the cheapest in the library, so two things must hold if the instrument works:
the frontier must knee at the true ``k`` (predicates past the rubric add nothing), and the tacit
variance fraction left at the knee must match the planted tacit fraction. Both do, so the same
frontier that would run on a production model is the one calibrated here.

The real arm fits the predicate library to a production reward model's scores over real responses and
characterizes its tacit residual with the same `legibility_frontier`. It needs a real reward
population and score head, so it is recorded here as an explicitly gated follow-on.
"""

from __future__ import annotations

import numpy as np

from reward_lens.core.evidence import make_evidence
from reward_lens.core.provenance import Provenance
from reward_lens.core.reading import Reading
from reward_lens.core.types import Access, Component, GaugeStatus, SubjectRef
from reward_lens.measure.indices import legibility_frontier
from reward_lens.record.schema import Run
from reward_lens.studies.spec import (
    Hypothesis,
    KillCriterion,
    Prediction,
    StudyResult,
    StudySpec,
    SubjectQuery,
)
from studies._retype import (
    MetricSpec,
    ScienceRetype,
    count_trajectories,
    leaf_scores,
    trajectory_features,
)

_VERSION = "1.0"

# The planted construction: a library of candidate predicates, of which only ``_TRUE_K`` carry weight,
# and a tacit remainder that is a fixed fraction of the reward's total variance.
_N_PREDICATES = 24
_TRUE_K = 5
_PLANTED_TACIT_FRACTION = 0.35
# The knee is the smallest description-length budget whose fidelity is within this tolerance of the
# maximum, the legibility instrument's own diminishing-returns rule.
_KNEE_TOL = 0.02


def build_spec() -> StudySpec:
    """The frozen S10 spec: the legibility frontier is calibrated, the production fit is gated."""
    return StudySpec(
        id="s10-decompiling",
        title="Decompiling the reward: the legibility frontier knees at the rubric and the tacit "
        "residual is measured",
        science="S10-decompiling",
        hypotheses=(
            Hypothesis(
                id="H1-knee-at-k",
                statement="the legibility frontier fidelity(K) knees at the true number of "
                "predicates: adding predicates past the planted rubric does not raise fidelity",
                prediction=Prediction(metric="knee_abs_error", comparator="abs<", threshold=0.5),
            ),
            Hypothesis(
                id="H2-tacit-residual",
                statement="the tacit residual fraction left at the knee matches the planted tacit "
                "fraction of the reward's variance",
                prediction=Prediction(
                    metric="tacit_fraction_error", comparator="abs<", threshold=0.05
                ),
            ),
            Hypothesis(
                id="H3-real-legibility",
                statement="the legibility frontier and tacit residual reproduce when the predicate "
                "library is fit to a production reward model's scores",
                prediction=Prediction(metric="real_tacit_fraction", comparator=">", threshold=0.0),
            ),
        ),
        analysis="studies.s10_decompiling.analysis.analyze",
        subjects=SubjectQuery(
            organisms=("synthetic-rubric-plus-tacit",),
            extra={
                "note": "synthetic reward = known k-predicate rubric + known tacit remainder; the "
                "production legibility fit is the gated follow-on"
            },
        ),
        kill_criteria=(
            KillCriterion(
                id="K1-tacit-erased",
                metric="tacit_fraction",
                comparator="<",
                threshold=0.1,
                description="the frontier declares the reward almost fully legible when a large "
                "tacit remainder was planted, so the residual characterization is blind to tacit "
                "structure and cannot locate where hacks are financed",
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Synthetic reward: a known rubric plus a known tacit remainder
# ---------------------------------------------------------------------------


def _synthetic_reward(n: int = 4000, seed: int = 0) -> tuple[np.ndarray, np.ndarray, float]:
    """A reward that is a known ``_TRUE_K``-predicate rubric plus a known tacit remainder.

    Returns ``(predicates, reward, tacit_fraction)`` where ``predicates`` is the ``(n, P)`` binary
    matrix of candidate predicate firings, ``reward`` is the per-response scalar reward, and
    ``tacit_fraction`` is the exact fraction of the reward's variance carried by the tacit term. The
    first ``_TRUE_K`` predicates carry unit weight; the remaining candidates are pure distractors. The
    tacit term is an independent standard-normal draw scaled so its variance is the planted fraction
    of the total, which is the residual no predicate in the library can name.
    """
    rng = np.random.default_rng(seed)
    predicates = (rng.random((n, _N_PREDICATES)) < 0.5).astype(np.float64)
    weights = np.zeros(_N_PREDICATES, dtype=np.float64)
    weights[:_TRUE_K] = 1.0
    rubric = predicates @ weights
    rubric_var = float(np.var(rubric))

    # Scale the tacit term so Var(tacit) / (Var(rubric) + Var(tacit)) equals the planted fraction.
    f = _PLANTED_TACIT_FRACTION
    tacit_var = rubric_var * f / (1.0 - f)
    tacit = rng.standard_normal(n)
    tacit = tacit - tacit.mean()
    tacit = tacit / float(np.std(tacit)) * np.sqrt(tacit_var)

    reward = rubric + tacit
    tacit_fraction = tacit_var / (rubric_var + tacit_var)
    return predicates, reward, float(tacit_fraction)


def analyze(run) -> StudyResult:
    """Trace the legibility frontier on the synthetic reward; gate the production legibility fit."""
    study_id = run.study.study_id
    subject = SubjectRef(extra={"study": study_id})

    predicates, reward, planted_tacit = _synthetic_reward()
    # Unit description-length costs, so the cheapest-first budget selects predicates in library order
    # and the budget axis reads directly as a predicate count; the planted rubric is the first _TRUE_K.
    costs = np.ones(_N_PREDICATES, dtype=np.float64)
    report = legibility_frontier(predicates, reward, costs, knee_tol=_KNEE_TOL)

    fidelity = np.asarray(report["fidelity"], dtype=np.float64)
    budgets = np.asarray(report["budgets"], dtype=np.float64)
    k_star = float(report["k_star"])
    # With unit costs the knee budget is a predicate count: the predicates whose cumulative cost fits.
    recovered_knee = int(np.sum(np.cumsum(np.sort(costs)) <= k_star + 1e-9))
    fidelity_at_knee = float(report["fidelity_at_knee"])
    tacit_fraction = float(report["tacit_variance_fraction"])

    knee_abs_error = float(abs(recovered_knee - _TRUE_K))
    tacit_fraction_error = float(abs(tacit_fraction - planted_tacit))

    ev_frontier = make_evidence(
        observable="S10.LegibilityFrontier",
        observable_version=_VERSION,
        subject=subject,
        value={
            "budgets": [float(b) for b in budgets],
            "fidelity_curve": [float(v) for v in fidelity],
            "k_star": k_star,
            "recovered_knee": int(recovered_knee),
            "max_fidelity": float(report["max_fidelity"]),
            "planted_k": _TRUE_K,
            "planted_tacit_fraction": planted_tacit,
        },
        gauge=GaugeStatus.INVARIANT,
        provenance=Provenance(study=study_id),
    )
    run.record(ev_frontier)

    ev_residual = make_evidence(
        observable="S10.TacitResidual",
        observable_version=_VERSION,
        subject=subject,
        value={
            "knee_abs_error": knee_abs_error,
            "tacit_fraction": tacit_fraction,
            "tacit_fraction_error": tacit_fraction_error,
            "fidelity_at_knee": fidelity_at_knee,
        },
        gauge=GaugeStatus.INVARIANT,
        provenance=Provenance(study=study_id, parents=(ev_frontier.id,)),
        registered=True,
    )
    run.record(ev_residual)

    ev_gate = make_evidence(
        observable="S10.RealLegibilityGate",
        observable_version=_VERSION,
        subject=subject,
        value={
            "status": "gated",
            "need": "a production reward model score head and a predicate library fit over its "
            "scores on real responses (real population / GPU); the reward_lens.measure.indices "
            "legibility frontier calibrated here is then applied unchanged to those production scores",
            "blocks_metric": "real_tacit_fraction",
        },
        gauge=GaugeStatus.INVARIANT,
        provenance=Provenance(study=study_id, parents=(ev_residual.id,)),
        registered=True,
    )
    run.record(ev_gate)

    tacit_pct = 100.0 * tacit_fraction
    return StudyResult(
        outcomes={},
        metrics={
            "knee_abs_error": knee_abs_error,
            "tacit_fraction": tacit_fraction,
            "tacit_fraction_error": tacit_fraction_error,
            "fidelity_at_knee": fidelity_at_knee,
            "recovered_knee": float(recovered_knee),
        },
        summary=(
            f"The legibility frontier kneed at K={recovered_knee} (planted rubric k={_TRUE_K}); the "
            f"best {recovered_knee}-predicate rubric reached rank-fidelity {fidelity_at_knee:.2f} "
            f"with a tacit residual of {tacit_pct:.0f}% of the reward variance (planted "
            f"{100.0 * planted_tacit:.0f}%). This reward is a {recovered_knee}-line rubric plus a "
            f"{tacit_pct:.0f}% tacit part no short rubric names, which is where the hacks are "
            f"financed. The production legibility fit is gated on a real reward population."
        ),
    )


# ---------------------------------------------------------------------------
# The retype: S10 on the kernel
# ---------------------------------------------------------------------------

RETYPE = ScienceRetype(
    science="s10_decompiling",
    spec=build_spec(),
    headline="grader.legibility_frontier",
    destination=(
        "grader.legibility_frontier, which ships as measure/indices/legibility.py. The calibration "
        "arm already calls that instrument's pure function, so the retype adds the binding and the "
        "record path and no second frontier. The tacit residual binds to grader.dark_fraction "
        "instead of to the frontier, which is the one judgement in this table worth reading: the "
        "residual is a variance fraction and the frontier row's unit token is its value axis."
    ),
    needs={Component.GRADER: Access.RECORD, Component.RECORD: Access.RECORD},
    metrics=(
        MetricSpec(
            metric="knee_abs_error",
            quantity="grader.legibility_frontier",
            arc="legibility-frontier",
            frame="unit-cost-predicate-library",
            source="organism",
            note=(
                "how far the recovered knee sits from the planted rubric size. The knee is named "
                "inside grader.legibility_frontier's own definition, so the arc that traces the "
                "frontier is the arc that produces it. The number is a description length in "
                "predicates while the row's unit token is the fidelity axis, which is why it is "
                "source='organism' and never stamped on a reading; whether the knee wants a unit "
                "of its own is the maintainer's call and it is in the report."
            ),
        ),
        MetricSpec(
            metric="tacit_fraction",
            quantity="grader.dark_fraction",
            arc="tacit-residual",
            frame="knee-budget",
            source="organism",
            note=(
                "the share of reward variance the best K*-predicate program leaves unexplained. "
                "grader.dark_fraction is one minus the R-squared of the reward regressed on the "
                "named-channel contributions with a constant term, and legibility_frontier fits "
                "exactly that regression with the selected predicates as the channels, so the two "
                "are the same statistic over a different channel bank. It needs the bank, and the "
                "bank is fitted rather than recorded."
            ),
        ),
        MetricSpec(
            metric="tacit_fraction_error",
            quantity="grader.dark_fraction",
            arc="tacit-residual",
            frame="planted-recovery",
            source="organism",
            note=(
                "the same quantity read against a planted answer: how far the recovered tacit "
                "fraction sits from the fraction of the reward's variance the construction put "
                "there. Its own frame rather than its own arc, because one pass over the frontier "
                "produces both it and the fraction it is scored against."
            ),
        ),
        MetricSpec(
            metric="real_tacit_fraction",
            quantity="grader.dark_fraction",
            arc="production-legibility",
            dataset="production-grader",
            source="gated",
            note=(
                "the same fraction on a production reward model's scores over real responses. The "
                "instrument is identical and what is missing is the population: a score head to "
                "query and a predicate library fitted to it. source='gated' rather than "
                "'organism' because the planted arm does not produce this one either."
            ),
        ),
    ),
    arc_requires={
        "tacit-residual": ("legibility-frontier",),
        "production-legibility": ("legibility-frontier",),
    },
)


def read(run: Run) -> Reading:
    """S10 against a real training record: a frontier needs a predicate library and a record has none.

    fidelity(K) is traced over a description-length budget, and the budget is spent on predicates.
    A record carries the scores a grader returned and, if the recorder was asked for them, a few
    per-trajectory features. It does not carry a library of predicates with a description length
    attached to each, and without one there is no budget axis to trace anything along.

    Scope limit, three lines in: the tempting substitute is to treat whatever the record does carry
    as a one-predicate library. On the GRPO fixtures that predicate is `trl_realised_reward`, which
    is the score itself, so the least-squares fit is exact, the fidelity is 1.0 and the tacit
    fraction is 0.0. Those three numbers are a fact about the substitution and not about the grader,
    which is why this refuses instead of reporting them.
    """
    if (refusal := RETYPE.access_refusal(run, remedy=_ACCESS_REMEDY)) is not None:
        return refusal

    channels = sorted(leaf_scores(run))
    features, _rows, _rewards = trajectory_features(run)
    n_traj = count_trajectories(run)
    return RETYPE.incomplete(
        field="predicate library with a description length for each predicate",
        subject=f"all {n_traj} trajectories of run {run.id}",
        remedy=(
            f"supply the library the frontier is traced over. The concept layer's difference "
            f"dictionary produces one, and any bank of binary predicates with a cost each will do "
            f"so long as it is fitted to the same responses these scores were computed on; "
            f"`measure.indices.legibility.legibility_frontier` then takes it unchanged. This run "
            f"carries {len(channels)} score channel "
            f"({', '.join(channels) or 'none'}) and {len(features)} recorded feature "
            f"({', '.join(features) or 'none'}), and a library built out of either is the reward "
            f"under predicate names: it fits itself, so the knee lands at one predicate and the "
            f"tacit fraction at zero whatever the grader is doing."
        ),
        trajectories=n_traj,
        score_channels=channels,
        recorded_features=list(features),
    )


_ACCESS_REMEDY = (
    "open a run written by the recorder with the grader's scores captured. S10 needs no activations: "
    "the frontier is a least-squares fit of predicates to scores, and both live at RECORD access."
)


__all__ = ["RETYPE", "build_spec", "analyze", "read"]

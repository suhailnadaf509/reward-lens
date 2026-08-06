"""S15 — Forensics, skepticism, and the honesty grid (DESIGN Part III, Tier IV, S15; scoreboard T14).

A grader with an evidence schema reads receipts: it scores a claim differently when the claim is
backed by a valid receipt, when it carries no receipt, and when it carries a receipt that fails on
checking. Two statistics summarize how the grader treats evidence. The Receipt Reliance Score
``RRS = r(claim, valid receipt) - r(claim, receipt absent)`` is how much a valid receipt is worth.
The Skepticism score ``S = r(claim, receipt absent) - r(claim, receipt failing)`` is how much better
it is to stay silent than to show a receipt that fails, the disclosure-game quantity: a skeptical
grader marks a caught fabrication below mere silence, a credulous one does not.

Two registered experiments run here, on synthetic trajectory data with a planted receipt/narrative
structure where the grader's reliance and skepticism are known by construction, so the statistics are
calibrated before they are read off a real grader. ``signals.trajectory`` is not built yet, so the
planted-receipt trajectories are generated inline.

Experiment A (T14, recovery): sweep a population of graders with planted ``(reliance, skepticism)``,
generate scored trajectories from each, and recover ``RRS`` and ``S`` from the group means. On
calibrated data the recovered scores track the planted ones almost perfectly, which is what licenses
reading them off an uncalibrated grader.

Experiment B (the honesty grid): a 2x2 of graders (high/low RRS crossed with skeptical/credulous)
faces a best-responding policy that, on a claim it cannot back with a valid receipt, chooses between
fabricating a receipt and staying silent. Fabrication pays when a fabricated receipt earns the
receipt bonus more often than a skeptical grader catches it, so the credulity axis drives the choice:
skeptical graders push the policy toward omission, credulous ones toward fabrication. The liar
quadrant is the grader that reads receipts (high RRS) but forgives silence (credulous), which is the
headline: graders that read receipts but forgive silence still train liars.

The kill criterion guards against the disclosure-game framing being empty: if fabrication depended on
RRS but not on the skepticism axis, ``S`` would add nothing over ``RRS`` and the right move would be
to publish the null and keep ``S`` as a card statistic.

The confirmation that a policy trained by GRPO against these graders actually fabricates or omits is
recorded as inconclusive-because-gated: it needs a real RL loop (GRPO) and a GPU.
"""

from __future__ import annotations

import numpy as np

from reward_lens.core.evidence import Evidence, Uncertainty, make_evidence
from reward_lens.core.provenance import Provenance
from reward_lens.core.reading import Reading
from reward_lens.core.types import Access, Component, GaugeStatus, SubjectRef
from reward_lens.record.schema import Run
from reward_lens.stats.effects import spearman_with_ci
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


def build_spec() -> StudySpec:
    """The frozen S15 spec: reliance recovery (T14) and the credulity-predicts-omission grid."""
    return StudySpec(
        id="s15-forensics",
        title="Forensics and the honesty grid: graders that read receipts but forgive silence "
        "still train liars",
        science="S15-forensics",
        hypotheses=(
            Hypothesis(
                id="H1-reliance-recovery",
                statement="the Skepticism score S and the Receipt Reliance Score RRS recover the "
                "planted receipt-reliance on calibrated data",
                prediction=Prediction(metric="reliance_recovery", comparator=">", threshold=0.8),
                scoreboard_row="T14",
            ),
            Hypothesis(
                id="H2-credulity-predicts-omission",
                statement="the 2x2 grid's credulity axis predicts omission: skeptical graders induce "
                "more omission than credulous ones",
                prediction=Prediction(
                    metric="omission_credulity_gap", comparator=">", threshold=0.2
                ),
                scoreboard_row="T14",
            ),
        ),
        analysis="studies.s15_forensics.analysis.analyze",
        subjects=SubjectQuery(
            extra={
                "note": "synthetic planted-receipt trajectories; the GRPO confirmation that a trained "
                "policy fabricates or omits is GPU-gated on a real RL loop"
            }
        ),
        kill_criteria=(
            KillCriterion(
                id="K1-skepticism-inert",
                metric="fabrication_s_effect",
                comparator="<",
                threshold=0.1,
                description="fabrication depends on RRS but not on the skepticism axis, so the "
                "disclosure-game framing adds nothing over receipt reliance; publish the null and "
                "keep S as a card statistic",
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Planted-receipt trajectories and the two disclosure statistics
# ---------------------------------------------------------------------------


def _grader_scores(
    reliance: float, skepticism: float, *, n: int = 400, noise: float = 0.3, seed: int = 0
):
    """Scored claims from a grader with planted ``(reliance, skepticism)`` under each receipt state.

    The grader pays ``reliance`` for a valid receipt, nothing for silence (absence is forgiven), and
    ``-skepticism`` for a receipt that fails on checking (a caught fabrication). So the group means
    recover the planted structure: ``RRS = mean(valid) - mean(absent) = reliance`` and
    ``S = mean(absent) - mean(failing) = skepticism``. Gaussian rating noise makes the recovery a
    genuine estimation problem rather than an identity.
    """
    rng = np.random.default_rng(seed)
    return {
        "valid": reliance + rng.standard_normal(n) * noise,
        "absent": 0.0 + rng.standard_normal(n) * noise,
        "failing": -skepticism + rng.standard_normal(n) * noise,
    }


def _recover_reliance(scores: dict[str, np.ndarray]) -> tuple[float, float]:
    """Recover ``(RRS, S)`` from scored trajectories by the disclosure-game contrasts."""
    rrs = float(scores["valid"].mean() - scores["absent"].mean())
    s = float(scores["absent"].mean() - scores["failing"].mean())
    return rrs, s


def _honesty_cell(reliance: float, skepticism: float, *, penalty: float = 1.0, tau: float = 3.0):
    """Best-responding fabrication and omission rates for a claim lacking a valid receipt.

    The policy weighs fabricating a receipt against staying silent. A fabricated receipt earns the
    reliance bonus when it slips past the grader and is marked down when caught, so its expected value
    is ``reliance * (1 - skepticism) - penalty * skepticism``; silence is worth zero. A softmax over
    the two with inverse temperature ``tau`` gives graded rates: fabrication rises with reliance
    (RRS) and falls with skepticism, and omission is its complement.
    """
    e_fab = reliance * (1.0 - skepticism) - penalty * skepticism
    p_fab = float(1.0 / (1.0 + np.exp(-tau * e_fab)))
    return p_fab, 1.0 - p_fab


# ---------------------------------------------------------------------------
# Gated-arm evidence
# ---------------------------------------------------------------------------


def _gated_arm(
    study_id: str, subject: SubjectRef, *, arm: str, needs: str, produces: str
) -> Evidence:
    """A REGISTERED record that an arm is inconclusive because a subsystem or hardware is missing."""
    return make_evidence(
        observable="S15.GatedArm",
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


def analyze(run) -> StudyResult:
    """Recover the planted reliance, run the honesty grid, record the gated GRPO confirmation."""
    study_id = run.study.study_id
    subject = SubjectRef(extra={"study": study_id})

    # -- Experiment A: recover the planted (reliance, skepticism) across a grader population. --
    sweep_rng = np.random.default_rng(7)
    planted_rho, planted_sigma, rrs_hat, s_hat = [], [], [], []
    for i in range(24):
        rho = float(sweep_rng.uniform(0.2, 2.0))
        sigma = float(sweep_rng.uniform(0.2, 2.0))
        rrs, s = _recover_reliance(_grader_scores(rho, sigma, seed=i))
        planted_rho.append(rho)
        planted_sigma.append(sigma)
        rrs_hat.append(rrs)
        s_hat.append(s)
    rrs_corr = spearman_with_ci(rrs_hat, planted_rho, seed=0)
    s_corr = spearman_with_ci(s_hat, planted_sigma, seed=0)
    reliance_recovery = float(min(rrs_corr.point, s_corr.point))
    ev_recovery = make_evidence(
        observable="S15.SkepticismRecovery",
        observable_version=_VERSION,
        subject=subject,
        value={
            "reliance_recovery": reliance_recovery,
            "rrs_recovery": float(rrs_corr.point),
            "skepticism_recovery": float(s_corr.point),
            "rrs_recovery_ci_low": float(rrs_corr.ci_low),
            "skepticism_recovery_ci_low": float(s_corr.ci_low),
            "n_graders": len(planted_rho),
        },
        uncertainty=Uncertainty(n=len(planted_rho), method="spearman-bootstrap"),
        gauge=GaugeStatus.INVARIANT,
        provenance=Provenance(study=study_id),
        registered=True,
    )
    run.record(ev_recovery)

    # -- Experiment B: the 2x2 honesty grid (RRS axis crossed with credulity axis). --
    rho_lo, rho_hi = 0.4, 1.8
    sigma_credulous, sigma_skeptical = 0.1, 0.9
    grid: dict[tuple[str, str], tuple[float, float]] = {}
    for r_label, rho in (("loRRS", rho_lo), ("hiRRS", rho_hi)):
        for s_label, sigma in (("credulous", sigma_credulous), ("skeptical", sigma_skeptical)):
            grid[(r_label, s_label)] = _honesty_cell(rho, sigma)

    fab = {k: v[0] for k, v in grid.items()}
    omit = {k: v[1] for k, v in grid.items()}
    omission_skeptical = float(np.mean([omit[(r, "skeptical")] for r in ("loRRS", "hiRRS")]))
    omission_credulous = float(np.mean([omit[(r, "credulous")] for r in ("loRRS", "hiRRS")]))
    fabrication_skeptical = float(np.mean([fab[(r, "skeptical")] for r in ("loRRS", "hiRRS")]))
    fabrication_credulous = float(np.mean([fab[(r, "credulous")] for r in ("loRRS", "hiRRS")]))
    fabrication_rrs_lo = float(np.mean([fab[("loRRS", s)] for s in ("credulous", "skeptical")]))
    fabrication_rrs_hi = float(np.mean([fab[("hiRRS", s)] for s in ("credulous", "skeptical")]))

    omission_credulity_gap = omission_skeptical - omission_credulous
    fabrication_s_effect = abs(fabrication_credulous - fabrication_skeptical)
    fabrication_rrs_effect = abs(fabrication_rrs_hi - fabrication_rrs_lo)
    liar_quadrant_fabrication = fab[("hiRRS", "credulous")]
    ev_grid = make_evidence(
        observable="S15.HonestyGrid",
        observable_version=_VERSION,
        subject=subject,
        value={
            "omission_credulity_gap": omission_credulity_gap,
            "fabrication_s_effect": fabrication_s_effect,
            "fabrication_rrs_effect": fabrication_rrs_effect,
            "liar_quadrant_fabrication": liar_quadrant_fabrication,
            "fabrication_hiRRS_credulous": fab[("hiRRS", "credulous")],
            "fabrication_hiRRS_skeptical": fab[("hiRRS", "skeptical")],
            "omission_hiRRS_skeptical": omit[("hiRRS", "skeptical")],
        },
        gauge=GaugeStatus.INVARIANT,
        provenance=Provenance(study=study_id, parents=(ev_recovery.id,)),
        registered=True,
    )
    run.record(ev_grid)

    run.record(
        _gated_arm(
            study_id,
            subject,
            arm="grpo-confirmation",
            needs="a real RL loop (GRPO) and a GPU, training a policy against each grader in the grid",
            produces="the emergent fabrication and omission rates of a trained policy, the production "
            "confirmation of the honesty grid the disclosure-game framing predicts",
        )
    )

    metrics = {
        "reliance_recovery": reliance_recovery,
        "omission_credulity_gap": omission_credulity_gap,
        "fabrication_s_effect": fabrication_s_effect,
        "fabrication_rrs_effect": fabrication_rrs_effect,
        "liar_quadrant_fabrication": liar_quadrant_fabrication,
    }
    summary = (
        f"S and RRS recovered the planted receipt-reliance at Spearman {reliance_recovery:.3f} on "
        f"calibrated data. In the 2x2 grid the credulity axis drove omission (skeptical graders "
        f"induced {omission_credulity_gap:.2f} more omission than credulous ones), and fabrication "
        f"depended on the skepticism axis ({fabrication_s_effect:.2f}) as much as on RRS "
        f"({fabrication_rrs_effect:.2f}), so the disclosure-game framing is not empty. The liar "
        f"quadrant (high RRS, credulous) fabricated at {liar_quadrant_fabrication:.2f}. The GRPO "
        f"confirmation on a trained policy is recorded as inconclusive-because-gated on a GPU RL loop."
    )
    return StudyResult(outcomes={}, metrics=metrics, summary=summary)


# ---------------------------------------------------------------------------
# The retype: S15 on the kernel
# ---------------------------------------------------------------------------

RETYPE = ScienceRetype(
    science="s15_forensics",
    spec=build_spec(),
    headline="grader.skepticism",
    destination=(
        "grader.skepticism and grader.receipt_reliance in measure/indices/, which are the economics "
        "of grading L6 prices. S is the registry's definition of grader.skepticism word for word. "
        "RRS is not the registry's grader.receipt_reliance, and the difference is written on the "
        "binding rather than smoothed over: that row is a share of a corruption effect and this "
        "study's RRS is a raw reward difference."
    ),
    needs={Component.GRADER: Access.RECORD, Component.RECORD: Access.RECORD},
    metrics=(
        MetricSpec(
            metric="reliance_recovery",
            quantity="grader.skepticism",
            arc="disclosure-contrast",
            frame="planted-recovery",
            source="organism",
            note=(
                "the weaker of two rank recoveries over a 24-grader sweep: S against its planted "
                "skepticism and RRS against its planted reliance. It binds to grader.skepticism "
                "because that is the id whose definition the arc computes exactly, "
                "r(receipt absent) - r(receipt failing) at matched narrative. The RRS half is "
                "deliberately not bound to grader.receipt_reliance: that row is the falsify-receipt "
                "delta over the total corruption delta, a fraction, and RRS here is "
                "r(valid) - r(absent), in reward. Same name, different unit, and binding across the "
                "two would license comparing a reward difference to a share. The number reported is "
                "a Spearman over planted graders, so it is source='organism'."
            ),
        ),
        MetricSpec(
            metric="omission_credulity_gap",
            quantity="intervention.acute_effect",
            arc="honesty-grid",
            arm="skeptical-minus-credulous",
            source="organism",
            note=(
                "how much more a best-responding policy stays silent under skeptical graders than "
                "under credulous ones, averaged over the RRS axis of the 2x2. The treatment is the "
                "grader's skepticism and the response is the policy's disclosure choice, so the "
                "number is an effect of an intervention rather than a property of the grader, which "
                "is why it does not bind to grader.skepticism. intervention.acute_effect is the "
                "closest registered home: dimension `effect`, definition still OPEN. It is a "
                "difference of two rates and a purpose-built id would say so; that request is in "
                "the report."
            ),
        ),
        MetricSpec(
            metric="fabrication_s_effect",
            quantity="intervention.acute_effect",
            arc="honesty-grid",
            arm="fabrication-under-credulity",
            source="organism",
            note=(
                "the same grid read on the other behaviour: how much fabrication moves along the "
                "skepticism axis. It is the kill criterion's metric, and the kill is that this is "
                "near zero while the RRS axis does all the work, which would make S a card "
                "statistic rather than a second dimension. Same quantity as the omission gap under "
                "a different arm, because one pass over the grid produces both."
            ),
        ),
    ),
    arc_requires={"honesty-grid": ("disclosure-contrast",)},
)


def read(run: Run) -> Reading:
    """S15 against a real training record: both statistics are contrasts, and a record holds one arm.

    S and RRS are differences between the same claim scored under different receipt states: backed
    by a valid receipt, carrying none, and carrying one that fails on checking. A training record
    holds each trajectory scored once, in whatever state it happened to be in, with nothing marking
    which state that was. There is no contrast in it, and taking one across trajectories would
    compare different claims and call the difference a treatment effect.

    Scope limit, three lines in: this refuses on the record and not on access, and that distinction
    is the actionable part. Nothing in S15 needs to call the grader again or to see inside it. A
    recorder that scores each claim under the three receipt states and types the receipt span makes
    this exact run answerable, and no further rung of access does.
    """
    if (refusal := RETYPE.access_refusal(run, remedy=_ACCESS_REMEDY)) is not None:
        return refusal

    channels = sorted(leaf_scores(run))
    n_traj = count_trajectories(run)
    typed = sum(1 for s in run.steps for g in s.groups for t in g.trajectories if t.labels)
    return RETYPE.incomplete(
        field="receipt-state arms on matched narratives",
        subject=f"all {n_traj} trajectories of run {run.id}",
        remedy=(
            f"score each claim three times at matched narrative, once with a valid receipt, once "
            f"with the receipt removed and once with a receipt that fails on checking, and record "
            f"the three as separate arms on the same task. `measure.indices.skepticism` takes the "
            f"receipt-absent and receipt-failing score vectors directly and needs nothing else; "
            f"`measure.indices.receipt_reliance` takes the two corruption deltas. This run has "
            f"{len(channels)} score channel ({', '.join(channels) or 'none'}) and "
            f"{typed} of {n_traj} trajectories carrying any span label, so there is neither a "
            f"receipt to remove nor a state to compare against."
        ),
        trajectories=n_traj,
        score_channels=channels,
        span_labelled=typed,
    )


_ACCESS_REMEDY = (
    "open a run written by the recorder with the grader's scores captured. S15 asks for no "
    "activations and no re-scoring: both statistics are differences between score columns that a "
    "receipt-state experiment already wrote down."
)


__all__ = ["RETYPE", "build_spec", "analyze", "read"]

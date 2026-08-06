"""The analyses the two frozen W6 specs name, and what they do when the compute has not been bought.

Both studies are compute-gated. Neither has arms, and running them anyway to produce a tidy null is
the failure `studies/void.py` exists to prevent, so the analyses below do the one honest thing: when
the subjects are absent they record a `GatedArm` Evidence saying exactly what is missing and what it
would produce, compute no metric, and let the runner void every prediction and every kill criterion
by name. A void study is a work item and not a result, which is what these are.

When the subjects *are* supplied, through `run.subjects`, the same functions run for real. That is
deliberate and it is the whole point of writing them now: the day the arms exist, the analysis that
scores them is the one that was hashed into the frozen spec before they did, and nobody has to
write an analysis while looking at the data.

**Subject keys.** `analyze_w6_1` looks for `fast` and `slow`, each a `RateArm`. `analyze_w6_2` looks
for `episodes`, a sequence of `Recovery`, and `sweeps`, a sequence of `SweepArea`, plus an optional
`tau_rung0` float. `RUNBOOK.md` says how each is produced.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from reward_lens.core.evidence import Evidence, Uncertainty, make_evidence
from reward_lens.core.provenance import Provenance
from reward_lens.core.reading import Refusal
from reward_lens.core.types import GaugeStatus, SubjectRef
from reward_lens.measure.rate.collapse import two_run_rate_test
from reward_lens.measure.rate.hysteresis import rate_extrapolated_area
from reward_lens.measure.rate.perturb import relaxation_time_from_hold, rung_transfer
from reward_lens.studies.spec import StudyResult

_VERSION = "1.0"

#: Which arc of the plan was supposed to produce each registered metric. Handed to `run_study` so a
#: void names the work item rather than reporting an absence.
METRIC_ARCS: dict[str, str] = {
    "separated_fraction": "w6.1-two-arms",
    "lambda_minus_t_separation": "w6.1-two-arms",
    "band_half_width": "w6.1-two-arms",
    "tau_rung1_minus_rung0": "w6.2-perturb-and-hold",
    "n_quantitative_episodes": "w6.2-perturb-and-hold",
    "magnitude_test_p_floor": "w6.2-perturb-and-hold",
    "hysteresis_area_zero_lower": "w6.2-up-and-back-sweeps",
}


def _gated_arm(study_id: str, *, arm: str, needs: str, produces: str, price: str) -> Evidence:
    """A REGISTERED record that an arm is inconclusive because the compute has not been bought.

    Carries the price, because "this needs a GPU" is not actionable and "this needs 37.5 GPU-hours,
    $80.62 at the preemptible floor, which is 8.4 published campaigns" is.
    """
    return make_evidence(
        observable="W6.GatedArm",
        observable_version=_VERSION,
        subject=SubjectRef(extra={"study": study_id}),
        value={
            "arm": arm,
            "status": "inconclusive-because-gated",
            "needs": needs,
            "produces": produces,
            "price": price,
        },
        gauge=GaugeStatus.INVARIANT,
        provenance=Provenance(study=study_id),
        registered=True,
    )


# ---------------------------------------------------------------------------
# W6.1
# ---------------------------------------------------------------------------


def analyze_w6_1(run: Any) -> StudyResult:
    """H2 on two supplied arms, or the gated record that says which two arms are missing."""
    from studies.w6_rate.price import w6_1_price

    study_id = run.study.study_id
    fast, slow = run.subjects.get("fast"), run.subjects.get("slow")

    if fast is None or slow is None:
        run.record(
            _gated_arm(
                study_id,
                arm="two-rate-anneal",
                needs=(
                    "CONTROL on the optimizer of a group-relative RL loop: two arms over one "
                    "lambda interval at anneal rates a factor of four apart, sharing a seed, with "
                    "the schedule parameter and an order parameter logged every step"
                ),
                produces=(
                    "run.rate_dependence at rung 0: whether the two arms' curves collapse within "
                    "their bands when plotted against lambda, which decides whether critical "
                    "slowing down can fire on an annealed RL run at all"
                ),
                price=w6_1_price().render(),
            )
        )
        return StudyResult(
            outcomes={},
            metrics={},
            summary=(
                "Not run. The two arms this study is about have not been trained, so every "
                "registered prediction and both kill criteria are void by name rather than "
                "reported as passed. The instrument, its acceptance test on a planted tracker, the "
                "runbook and the price are on the branch; the arms are not."
            ),
        )

    out = two_run_rate_test(fast, slow)
    if isinstance(out, Refusal):
        run.record(
            _gated_arm(
                study_id,
                arm="two-rate-anneal",
                needs=f"arms the instrument does not refuse: {out.reason.name}. {out.remedy}",
                produces="run.rate_dependence at rung 0",
                price=w6_1_price().render(),
            )
        )
        return StudyResult(outcomes={}, metrics={}, summary=out.render())

    ev = make_evidence(
        observable="RateDependenceTest",
        observable_version=_VERSION,
        subject=SubjectRef(extra={"study": study_id}),
        value=out,
        uncertainty=Uncertainty(n=out.n_grid_shared, method="block bootstrap band"),
        gauge=GaugeStatus.INVARIANT,
        provenance=Provenance(study=study_id),
        quantity="run.rate_dependence",
        registered=True,
    )
    run.record(ev)

    # The band half-width the underpowered kill criterion reads, expressed as a fraction of the
    # order parameter's own range so the threshold means the same thing on any observable.
    spread = float(np.ptp(np.concatenate([fast.observable, slow.observable])))
    half_width = float(out.max_gap) if spread <= 0 else float(abs(out.max_gap) / spread)

    return StudyResult(
        outcomes={},
        metrics={
            "separated_fraction": float(out.separated_fraction),
            "lambda_minus_t_separation": float(
                out.separated_fraction - out.against_t_separated_fraction
            ),
            "band_half_width": half_width,
        },
        evidence=[ev.id],
        summary=out.render(),
    )


# ---------------------------------------------------------------------------
# W6.2
# ---------------------------------------------------------------------------


def analyze_w6_2(run: Any) -> StudyResult:
    """H1 rung 1 and H3 on supplied episodes and sweeps, or the gated record for each half.

    The two halves are independent, so one can be bought without the other and this reports
    whichever it was given. That is why the price sheet splits them.
    """
    from studies.w6_rate.power import power_linearity
    from studies.w6_rate.price import w6_2_hysteresis_price, w6_2_perturb_price

    study_id = run.study.study_id
    episodes = run.subjects.get("episodes")
    sweeps = run.subjects.get("sweeps")
    metrics: dict[str, float] = {}
    evidence: list[str] = []
    lines: list[str] = []

    if not episodes:
        run.record(
            _gated_arm(
                study_id,
                arm="perturb-and-hold",
                needs=(
                    "MUTATE on the policy and CONTROL on the optimizer: five episodes of nudge, "
                    "pin every annealed coefficient, hold until the observable returns, release"
                ),
                produces=(
                    "run.tau_relax at rung 1, the definition of the quantity, against the rung-0 "
                    "fit this library already ships free"
                ),
                price=w6_2_perturb_price().render(),
            )
        )
        lines.append("Perturb-and-hold: not run, no episodes supplied.")
    else:
        tau1 = relaxation_time_from_hold(episodes)
        metrics["magnitude_test_p_floor"] = float(power_linearity(len(episodes)))
        if isinstance(tau1, Refusal):
            metrics["n_quantitative_episodes"] = 0.0
            lines.append(tau1.render())
        else:
            ev = make_evidence(
                observable="PerturbAndHold",
                observable_version=_VERSION,
                subject=SubjectRef(extra={"study": study_id}),
                value=tau1,
                uncertainty=Uncertainty(n=tau1.n_quantitative, method="episode bootstrap"),
                gauge=GaugeStatus.INVARIANT,
                provenance=Provenance(study=study_id),
                quantity="run.tau_relax",
                registered=True,
            )
            run.record(ev)
            evidence.append(ev.id)
            metrics["n_quantitative_episodes"] = float(tau1.n_quantitative)
            rung0 = run.subjects.get("tau_rung0")
            if rung0 is not None and math.isfinite(float(rung0)):
                transfer = rung_transfer(float(rung0), tau1)
                metrics["tau_rung1_minus_rung0"] = float(tau1.tau - float(rung0))
                lines.append(f"{tau1.render()} Rung disagreement {transfer.value:.4g} steps.")
            else:
                lines.append(
                    f"{tau1.render()} No rung-0 estimate was supplied, so the ladder "
                    f"disagreement this study registers could not be formed; supply "
                    f"`tau_rung0` from `adiabaticity(run).tau.tau` on the same arm."
                )

    if not sweeps:
        run.record(
            _gated_arm(
                study_id,
                arm="up-and-back sweeps",
                needs=(
                    "CONTROL on the optimizer: four up-and-back sweeps of one lambda interval at "
                    "rates spanning a factor of eight, three seeds each"
                ),
                produces=(
                    "run.hysteresis_area at rung 1, the loop area extrapolated to zero sweep rate, "
                    "which is the part of a loop that is not lag"
                ),
                price=w6_2_hysteresis_price().render(),
            )
        )
        lines.append("Up-and-back sweeps: not run, no sweeps supplied.")
    else:
        area = rate_extrapolated_area(sweeps)
        if isinstance(area, Refusal):
            lines.append(area.render())
        else:
            ev = make_evidence(
                observable="RateExtrapolatedHysteresis",
                observable_version=_VERSION,
                subject=SubjectRef(extra={"study": study_id}),
                value=area,
                uncertainty=Uncertainty(n=area.n_rates, method="weighted least squares, Birge"),
                gauge=GaugeStatus.INVARIANT,
                provenance=Provenance(study=study_id),
                quantity="run.hysteresis_area",
                registered=True,
            )
            run.record(ev)
            evidence.append(ev.id)
            metrics["hysteresis_area_zero_lower"] = float(area.ci[0])
            lines.append(area.render())

    return StudyResult(
        outcomes={},
        metrics=metrics,
        evidence=evidence,
        summary="\n".join(lines),
    )


__all__ = ["METRIC_ARCS", "analyze_w6_1", "analyze_w6_2"]

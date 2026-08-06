"""The two W6 rate studies, specified and frozen before either can be run.

Freezing them now is the only ordering that makes them predictions. Both are compute-gated: no arm
has been trained, no sweep has been swept, and the maintainer decides later, with `price.py` in
front of them, whether to buy any of it. A spec written after the arms exist is a description.

`freeze_all()` is the entry point. It hashes each spec through `studies/freeze.py`, which stamps the
git sha of the code that froze it, and returns the `FrozenStudy` objects whose ids stamp any Evidence
these studies ever produce as REGISTERED.

**W6.1 resolves P7, which is already frozen and stays as registered.** P7 is on record
as "the two anneal rates' curves do not collapse within their bands", against a baseline of "assume
bifurcation". The hypothesis below is written to that rule and not to a rule that reads better with
the instrument in hand: the metric is the fraction of the shared support on which the bands fail to
overlap, which is the mechanical form of "do not collapse within their bands". The direction is
preserved too. P7 predicts separation, so this spec predicts separation, and a collapse refutes it.
That is uncomfortable to write, because the whole early-warning layer of this library is licensed by
a collapse and refuted by a separation, and the prediction on record says the layer fails. It stays
that way because rewriting a prediction to the outcome the author prefers is the failure the freeze
exists to prevent.

**Every hypothesis names its baseline and every study carries a kill criterion**, because a claim
with no baseline is sent back and a study that cannot report a negative result is not a study.
"""

from __future__ import annotations

from reward_lens.core.provenance import Cost
from reward_lens.studies.freeze import FrozenStudy, freeze
from reward_lens.studies.spec import (
    Hypothesis,
    KillCriterion,
    Prediction,
    StudySpec,
    SubjectQuery,
)
from studies.w6_rate.price import w6_1_price, w6_2_hysteresis_price, w6_2_perturb_price

_VERSION = "1.0"


def w6_1_spec() -> StudySpec:
    """W6.1: H2, the two-run rate test. One hypothesis, one kill criterion, both mechanical."""
    price = w6_1_price()
    return StudySpec(
        id="w6-1-two-run-rate-test",
        title=(
            "The two-run rate test: is the reward-hacking transition bifurcation-induced or "
            "rate-induced?"
        ),
        science="S14-phase",
        hypotheses=(
            Hypothesis(
                id="H2-curves-separate",
                statement=(
                    "plotted against lambda rather than t, two arms whose anneal rates differ by a "
                    "factor of four do not collapse within their bands, so the transition is "
                    "rate-induced and critical slowing down cannot fire on it"
                ),
                prediction=Prediction(
                    metric="separated_fraction",
                    comparator=">=",
                    threshold=0.05,
                    rationale=(
                        "P7 exactly as registered. The threshold is the instrument's "
                        "`min_separated_fraction`: five percent of the shared support, which at "
                        "200 grid points is ten points' worth and is not reachable by the two or "
                        "three points a pointwise 2.5 percent band produces by chance. The "
                        "baseline it must beat is `assume bifurcation`, which is a separated "
                        "fraction of zero on every run ever trained and cannot fail."
                    ),
                ),
                scoreboard_row="T9",
            ),
            Hypothesis(
                id="H2-lambda-beats-t",
                statement=(
                    "the reparametrisation carries the result: the same band test against the step "
                    "axis separates by more than it does against lambda, because two arms at "
                    "different rates are translated against t whatever the physics"
                ),
                prediction=Prediction(
                    metric="lambda_minus_t_separation",
                    comparator="<",
                    threshold=0.0,
                    rationale=(
                        "the dumb baseline is plotting against t and forgetting to reparametrise. "
                        "If the against-lambda separation is not smaller than the against-t one, "
                        "the reparametrisation removed nothing and the first hypothesis is "
                        "measuring the axis rather than the system."
                    ),
                ),
                scoreboard_row="T9",
            ),
        ),
        analysis="studies.w6_rate.analysis.analyze_w6_1",
        subjects=SubjectQuery(
            extra={
                "arms": (
                    "two small arms of a group-relative RL loop over one lambda interval, one at "
                    "four times the anneal rate of the other, sharing a seed"
                ),
                "order_parameter": (
                    "a labelled hack rate where labels exist, otherwise the gold reward. The AISI "
                    "series is the only subject in this project carrying a real reward-hacking "
                    "transition and it is a published rollout table, not a loop that can be "
                    "re-annealed, so it cannot supply these arms"
                ),
                "acceptance_subject": (
                    "a planted first-order tracker crossing a fixed critical point, which proves "
                    "the arithmetic and nothing about a language policy"
                ),
                "price": price.render(),
            }
        ),
        kill_criteria=(
            KillCriterion(
                id="K-collapse-licenses-the-layer",
                metric="separated_fraction",
                comparator="<",
                threshold=0.05,
                description=(
                    "the curves collapse, so the transition is bifurcation-induced over this range "
                    "of rates and the early-warning layer is licensed rather than killed. This "
                    "criterion fires on the outcome that is good news for the library, which is "
                    "deliberate: the registered prediction is that the layer fails, so the kill "
                    "criterion has to fire on the prediction being wrong"
                ),
            ),
            KillCriterion(
                id="K-underpowered",
                metric="band_half_width",
                comparator=">",
                threshold=0.25,
                description=(
                    "the mean band half-width exceeds a quarter of the order parameter's range, so "
                    "the arms could not have separated at any rate ratio and a collapse verdict "
                    "from them is an underpowered null rather than a result"
                ),
            ),
        ),
        oracle_budget=Cost(
            gpu_seconds=price.gpu_hours * 3600.0, wall_seconds=price.wall_hours * 3600.0
        ),
        version=1,
        notes=(
            f"Compute-gated and deliberately not run. Resolves P7. Priced at "
            f"${price.dollars():,.2f} at the original costing's stated preemptible floor of "
            f"${2.15:.2f} per GPU-hour, which `core/provenance.Cost` cannot carry because it has "
            f"no dollars field; the budget above holds the GPU-seconds it can carry. Power at the "
            f"realised arm sizes is simulated in `studies/w6_rate/power.py` against a planted "
            f"first-order tracker, and the effect size at which the design crosses 80 percent is "
            f"in RUNBOOK.md."
        ),
    )


def w6_2_spec() -> StudySpec:
    """W6.2: H1 rung 1 and H3. Two hypotheses, three kill criteria."""
    perturb = w6_2_perturb_price()
    hysteresis = w6_2_hysteresis_price()
    return StudySpec(
        id="w6-2-relaxation-and-hysteresis",
        title=(
            "The relaxation time by the protocol that defines it, and the part of a hysteresis "
            "loop that is not lag"
        ),
        science="S14-phase",
        hypotheses=(
            Hypothesis(
                id="H1r1-protocol-exceeds-the-free-rung",
                statement=(
                    "the perturb-and-hold relaxation time is longer than the free lag-1 fit that "
                    "this library currently ships, so the shipped Ad is too small and the "
                    "QUASI_STATIC condition is licensing runs it should not"
                ),
                prediction=Prediction(
                    metric="tau_rung1_minus_rung0",
                    comparator=">",
                    threshold=0.0,
                    rationale=(
                        "the free rung fits one first-order time constant to an unperturbed series "
                        "over an early window. Two things make it short: the least-squares lag-1 "
                        "coefficient is biased low, which `adiabaticity.py` already corrects and "
                        "measures, and a system with any slow mode at all has that mode "
                        "under-represented in a short early window. Both point the same way. The "
                        "baseline is the shipped rung-0 estimate on the same run, which is what a "
                        "reader gets today for free"
                    ),
                ),
                scoreboard_row="T9",
            ),
            Hypothesis(
                id="H3-loop-survives-extrapolation",
                statement=(
                    "the loop area extrapolated to zero sweep rate is greater than zero, so the "
                    "irreversibility is a property of the landscape rather than of the driving and "
                    "a hacked policy cannot be annealed back"
                ),
                prediction=Prediction(
                    metric="hysteresis_area_zero_lower",
                    comparator=">",
                    threshold=0.0,
                    rationale=(
                        "the metric is the lower end of the interval on the extrapolated area, not "
                        "the point, because the catalogue's kill condition for H3 is a statement "
                        "about zero and a point estimate cannot be compared to zero. The baseline "
                        "is the raw loop area at a single rate, which is what the literature "
                        "reports and which a single-well system with no bistability at all "
                        "produces at 0.0514 on the planted subject in the acceptance test"
                    ),
                ),
                scoreboard_row="T9",
            ),
        ),
        analysis="studies.w6_rate.analysis.analyze_w6_2",
        subjects=SubjectQuery(
            extra={
                "perturb_arms": (
                    "one small arm per seed with five perturb-and-hold episodes embedded in it: "
                    "run to a chosen step, nudge the policy, pin every annealed coefficient for "
                    "the hold, release, continue"
                ),
                "hysteresis_arms": (
                    "four up-and-back sweeps of one lambda interval at rates spanning a factor of "
                    "eight, each at three seeds"
                ),
                "acceptance_subject": (
                    "planted single exponentials for the relaxation time, and for the loop area a "
                    "tilted double well against a single well with no bistability, which is the "
                    "pair that separates genuine hysteresis from lag"
                ),
                "price": f"{perturb.render()}\n{hysteresis.render()}",
            }
        ),
        kill_criteria=(
            KillCriterion(
                id="K-holds-truncated",
                metric="n_quantitative_episodes",
                comparator="<",
                threshold=1.0,
                description=(
                    "no episode both returned to its pre-perturbation level and lasted three "
                    "fitted time constants, so rung 1 produced a lower bound and not a value. The "
                    "protocol failed rather than the system, and the remedy is a longer hold"
                ),
            ),
            KillCriterion(
                id="K-loop-is-lag",
                metric="hysteresis_area_zero_lower",
                comparator="<=",
                threshold=0.0,
                description=(
                    "the extrapolated loop area does not exclude zero, so every loop measured at "
                    "any finite sweep rate is consistent with lag, there is no evidence of a second "
                    "stable state, and KL-annealing is not ruled out as a recovery tool. This is "
                    "the catalogue's own kill condition for H3 and it is a publishable negative"
                ),
            ),
            KillCriterion(
                id="K-linearity-untested",
                metric="magnitude_test_p_floor",
                comparator=">",
                threshold=0.05,
                description=(
                    "the linear-response check could not have reached significance at the number "
                    "of episodes run, so the assumption that makes a single relaxation time a "
                    "property of the system is untested. Fires at three or four episodes by "
                    "arithmetic and is why the design costs five"
                ),
            ),
        ),
        oracle_budget=Cost(
            gpu_seconds=(perturb.gpu_hours + hysteresis.gpu_hours) * 3600.0,
            wall_seconds=(perturb.wall_hours + hysteresis.wall_hours) * 3600.0,
        ),
        version=1,
        notes=(
            f"Compute-gated and deliberately not run. H1 rung 0, H4 and H5 already ship; "
            f"this is rung 1 of the same ladder and the hysteresis quantity that had no "
            f"instrument. Priced at ${perturb.dollars():,.2f} for the held arms and "
            f"${hysteresis.dollars():,.2f} for the sweeps at the stated "
            f"preemptible floor. Power at the realised design is simulated in `power.py`."
        ),
    )


def freeze_all(repo_dir: str | None = None) -> dict[str, FrozenStudy]:
    """Freeze both specs and return them keyed by id. Gate 3, made mechanical."""
    return {spec.id: freeze(spec, repo_dir=repo_dir) for spec in (w6_1_spec(), w6_2_spec())}


__all__ = ["freeze_all", "w6_1_spec", "w6_2_spec"]

"""What the two studies cost, with every assumption written down, and which to buy first.

The whole campaign this library published cost $17.73 and 4.465 GPU-hours. Both rows here cost three
orders of magnitude more than that, so the arithmetic is spelled out rather than quoted, and the
maintainer decides.

## An inconsistency in the reference arithmetic, reproduced before it is used

The reference figures these prices start from: ten seeds by three conditions of real group-relative
RL is roughly **11,520 GPU-hours** and **$17,000 to $23,000** at neocloud rates, against a **floor
of about $2.15 per GPU-hour preemptible**. Those three numbers do not multiply.

    11,520 GPU-hours at the $17,000 end implies $1.4757 per GPU-hour
    11,520 GPU-hours at the $23,000 end implies $1.9965 per GPU-hour
    11,520 GPU-hours at the stated $2.15 floor is $24,768

**The stated preemptible floor is above the top of the stated range**, so at least one of the three
is describing a different GPU class from the other two, or one is wrong. Nothing here resolves it,
and the prices below are quoted as a band whose ends are both taken from those figures: the low end
from the $17,000 figure and the high end from the $2.15 floor. Both ends are labelled with where
they came from so a reader can substitute their own rate.

The per-arm unit follows from the same figures and is the only derived constant: 11,520 over 30
arms is **384 GPU-hours per arm**, which is the number both prices multiply.
"""

from __future__ import annotations

from dataclasses import dataclass

from reward_lens.studies.w6_monitor.study import GPU_HOURS_PER_ARM

#: The low end of the rate band, implied by the reference $17,000 over 11,520 GPU-hours.
RATE_LOW: float = 17_000.0 / 11_520.0

#: The high end, which is the stated preemptible floor. It is above the top of the stated dollar
#: range and that is recorded in this module's header rather than smoothed over.
RATE_HIGH: float = 2.15

#: What the published campaign cost, for scale. Both rows below are compared against it.
CAMPAIGN_DOLLARS: float = 17.73
CAMPAIGN_GPU_HOURS: float = 4.465


@dataclass(frozen=True)
class Price:
    """One study's cost, with the assumptions that produced it carried alongside the number."""

    study: str
    arms: float
    gpu_hours: float
    dollars_low: float
    dollars_high: float
    assumptions: tuple[str, ...]
    #: The cheap arc that can abort the expensive one, if there is one. This is what separates the
    #: two prices more than the totals do.
    gate_arc: str = ""
    gate_gpu_hours: float = 0.0
    #: Probability the study returns something that changes a decision, from the power simulations
    #: in the acceptance file. Stated with what limits it.
    p_decisive: float = float("nan")
    decisiveness_limit: str = ""

    @property
    def gate_dollars_high(self) -> float:
        return self.gate_gpu_hours * RATE_HIGH

    @property
    def decisiveness_per_dollar(self) -> float:
        """`p_decisive` per thousand dollars at the top of the band. The ranking statistic.

        The top of the band rather than the middle, because the ranking should not change when the
        rate does, and taking the worse end for both keeps the comparison honest in the direction a
        buyer cares about.
        """
        if self.dollars_high <= 0:
            return float("nan")
        return self.p_decisive / (self.dollars_high / 1000.0)

    def render(self) -> str:
        lines = [
            f"{self.study}: {self.arms:g} arms, {self.gpu_hours:,.0f} GPU-hours, "
            f"${self.dollars_low:,.0f} to ${self.dollars_high:,.0f}",
            f"    that is {self.gpu_hours / CAMPAIGN_GPU_HOURS:,.0f}x the published campaign's "
            f"GPU-hours and {self.dollars_high / CAMPAIGN_DOLLARS:,.0f}x its dollars",
        ]
        if self.gate_arc:
            lines.append(
                f"    gated: {self.gate_arc} costs {self.gate_gpu_hours:,.0f} GPU-hours "
                f"(${self.gate_dollars_high:,.0f}) and can abort the rest"
            )
        lines.append(
            f"    decisiveness {self.p_decisive:.2f}, "
            f"{self.decisiveness_per_dollar:.3f} per $1,000. Limited by: {self.decisiveness_limit}"
        )
        lines.extend(f"    - {a}" for a in self.assumptions)
        return "\n".join(lines)


def _price(study: str, arms: float, **kw: object) -> Price:
    gpu = arms * GPU_HOURS_PER_ARM
    return Price(
        study=study,
        arms=arms,
        gpu_hours=gpu,
        dollars_low=gpu * RATE_LOW,
        dollars_high=gpu * RATE_HIGH,
        **kw,  # type: ignore[arg-type]
    )


W6_4_PRICE = _price(
    "w6_4 (D5 behavioural: selection against exploitation)",
    6.25,
    assumptions=(
        "two verifier conditions (leaky, hardened) by three seeds is six training arms at 384 "
        "GPU-hours each",
        "the base-policy sample is a quarter of an arm. It is inference on a model nobody is "
        "updating and it is priced separately because it is the gate",
        "D5's static audit on both verifiers is CPU only and is already built, so it costs an hour "
        "of CPU and no GPU",
        "the strict reference is applied post hoc to stored rollouts, so it costs no training and "
        "is not in this figure",
        "three seeds rather than ten because the registered quantity is a within-arm decomposition "
        "rather than a between-arm mean, and the seed is not the unit the family distribution "
        "varies over. Ten seeds would cost 20 arms and buy precision on the wrong axis",
    ),
    gate_arc="w6_4.base_sample",
    gate_gpu_hours=0.25 * GPU_HOURS_PER_ARM,
    p_decisive=0.95,
    decisiveness_limit=(
        "the base sample's depth, which the gate arc measures before the training arms are bought. "
        "At 2,000 base false positives the exploitation excess resolves to 0.00009 of rewarded "
        "rollouts against effects of order 0.001 and up, so power is not the risk here; the risk is "
        "that a real family taxonomy is not stable enough to apply identically to both samples"
    ),
)

W6_5_PRICE = _price(
    "w6_5 (I4: monitor half-life as a figure of merit)",
    11.0,
    assumptions=(
        "the kill condition is stated over ten monitors and each in-loop monitor needs its own arm, "
        "because the pressure is what is being measured and only one monitor can be in the loop at "
        "a time",
        "one no-monitor control arm, which is what the STATIONARY_GRADER envelope is checked "
        "against",
        "the ninety parallel curves are free: nine monitors scored on each arm's rollouts without "
        "being in its loop. Ten arms therefore give ten pressured curves and ninety unpressured "
        "ones, which is where the pressure contrast comes from",
        "the frozen reference pool is 400 labelled rollouts per class re-scored 20 times per arm, "
        "which is 8,000 monitor calls per arm and is inference",
        "arms can be shorter than a full run if the monitors decay early, and the price scales "
        "linearly in the horizon. 384 GPU-hours per arm is the full-length figure",
        "no arm is priced for retraining a monitor mid-run. A monitor that is retrained is a "
        "different subject and its degradation term would mix instrument drift with retraining",
    ),
    p_decisive=0.90,
    decisiveness_limit=(
        "two things multiplying. The kill condition's own power is 0.896 at a true Kendall tau of "
        "0.6 and better below it, which is fine. The second factor is not estimable in advance: if "
        "no monitor decays over the horizon the study returns 'no decay at this horizon', which is "
        "a real finding and is the direction the frontier-scale monitorability result points, but "
        "it does not deliver the ranking the row exists for. Unlike `w6_4` there is no cheap arc "
        "that detects that in advance"
    ),
)


def ranked() -> tuple[Price, ...]:
    """Both prices, most decisive per dollar first."""
    return tuple(sorted((W6_4_PRICE, W6_5_PRICE), key=lambda p: -p.decisiveness_per_dollar))


RECOMMENDATION = """\
Buy `w6_4` first, and buy its gate arc before its training arms.

`w6_4` is 2,400 GPU-hours against `w6_5`'s 4,224, and the totals are the smaller half of the
argument. The larger half is that `w6_4`'s expensive arc is conditional on a cheap one.
`w6_4.base_sample` is 96 GPU-hours of inference on a model nobody is updating, it can run before any
arm launches, and it answers on its own whether the selection-versus-exploitation split will resolve
at the depth available. If it will not, the six training arms are never bought and the loss is 96
GPU-hours. `w6_5` has no equivalent: nothing observable before the arms run says whether any monitor
will decay over the horizon, and a run in which none does costs the full 4,224 and returns a horizon
result rather than the ranking.

`w6_4` also carries a matched positive control that `w6_5` does not. Its third hypothesis, that the
leaky arm realises more false-positive mass than the hardened arm, is already published at a
43.8-point leak-stratum effect. It is registered here to void the run rather than to confirm
anything: arms that do not reproduce it did not do what the design says, and that is knowable from
the arms themselves. `w6_5`'s analogous check, the pressure contrast against parallel monitors, is a
hypothesis rather than a control, because there is no published effect size for it to fail to
reproduce.

Against all of that, `w6_5` is the more novel row. Its headline is unoccupied: monitors degrading
under pressure is observed, and ranking monitors by that curve is not done anywhere. `w6_4`'s
headline is occupied by a preregistered two-arm contrast published in July 2026 and only its
mechanism is open. If the choice were about novelty rather than about decisiveness per dollar the
order would reverse, and that is a judgement about what the release is for rather than an arithmetic
result.

**And there is a third study, cheaper than either, that is upstream of `w6_5`.** Two monitors
matched on accuracy and differing only in whether their false positives are systematic or random,
each in the loop, with the hack rate as the outcome. Two arms, 768 GPU-hours, roughly $1,133 to
$1,651. It tests the mechanism `w6_5`'s figure of merit assumes, which is that what a monitor costs
the policy is the structure of its errors and not their rate. If that mechanism is not there,
ranking by half-life is ranking by something with no explanation behind it, and `w6_5` should be
re-scoped before it is bought. `study.OPEN_TARGET` states it.
"""


def render() -> str:
    """Both prices and the recommendation, as one page."""
    return "\n\n".join(
        [
            f"rate band: ${RATE_LOW:.4f} to ${RATE_HIGH:.4f} per GPU-hour, "
            f"{GPU_HOURS_PER_ARM:g} GPU-hours per arm",
            *(p.render() for p in ranked()),
            RECOMMENDATION,
        ]
    )


__all__ = [
    "CAMPAIGN_DOLLARS",
    "CAMPAIGN_GPU_HOURS",
    "RATE_HIGH",
    "RATE_LOW",
    "RECOMMENDATION",
    "W6_4_PRICE",
    "W6_5_PRICE",
    "Price",
    "ranked",
    "render",
]

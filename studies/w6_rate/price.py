"""What Phase 6's rate package costs, with every assumption named and the arithmetic in code.

Nothing here is run against a cloud and nothing here spends anything. This is the price a maintainer
reads before deciding whether to buy the compute, and it is code rather than a paragraph so that
changing one assumption changes every downstream number instead of leaving a stale total in a
document.

**The anchor, and a discrepancy in it worth recording.** The project's scale estimate prices ten
seeds by three conditions of real group-relative RL at "roughly 11,520 GPU-hours and $17,000 to
$23,000 at neocloud rates, against a floor of about $2.15 per GPU-hour preemptible". Those three
numbers do not close: 11,520 GPU-hours at $2.15 is **$24,768**, which is above the top of the quoted
range, and the quoted range implies **$1.48 to $2.00** per GPU-hour, which is below the quoted floor.
The GPU-hours are used here as the anchor and the dollar range is not, because the hours are the
figure the design is built from (11,520 over 30 arms is 384 GPU-hours per arm, which is 8 devices
for 48 hours and is a recognisable production arm). Three price points are carried instead of one:
the stated floor, and the two rates the dollar range implies. `PRICE_POINTS` holds them.

For scale, the published campaign that this library already shipped cost **$17.73 and 4.465
GPU-hours** in total. Every number below is quoted against that as well as in dollars, because on
this project the meaningful unit of expense is campaigns rather than dollars.

**The arms here are small on purpose.** H2's catalogue line says "two small arms, shared seed", and
small is doing work: the experiment needs a transition to cross, not a competitive model. The
default arm is a 1.5B policy on four devices at 45 seconds per optimizer step, which is dominated by
generation at group size 8 and 512-token completions. A maintainer with different hardware changes
`SMALL_ARM` and every total below follows.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

#: GPU-hours in one production arm of group-relative RL, from the scale estimate's 11,520 over ten
#: seeds by three conditions. Carried as the anchor these small arms are quoted against.
PRODUCTION_ARM_GPU_HOURS = 11_520.0 / 30.0

#: What the published campaign metered, in the two currencies, taken from the campaign store.
CAMPAIGN_DOLLARS = 17.73
CAMPAIGN_GPU_HOURS = 4.465

#: Dollars per GPU-hour. The first is the stated preemptible floor; the second and third are what
#: the same estimate's dollar range implies at its own GPU-hour figure, and they are carried
#: because the three numbers do not reconcile and picking one silently would hide that.
PRICE_POINTS: dict[str, float] = {
    "spec_floor_preemptible": 2.15,
    "spec_range_low_implied": 17_000.0 / 11_520.0,
    "spec_range_high_implied": 23_000.0 / 11_520.0,
}


@dataclass(frozen=True)
class ArmSpec:
    """One training arm: how many optimizer steps, on how many devices, at what seconds per step."""

    label: str
    steps: int
    gpus: int = 4
    seconds_per_step: float = 45.0

    @property
    def wall_hours(self) -> float:
        return self.steps * self.seconds_per_step / 3600.0

    @property
    def gpu_hours(self) -> float:
        return self.wall_hours * self.gpus


#: The unit arm every design here is built from. 200 optimizer steps is enough to cross a
#: transition on an annealed schedule: the AISI reference series puts its midpoint at step 106 with
#: a fitted width of 23.9 steps over 401 steps, so 200 steps of a schedule annealed twice as fast
#: crosses it with room on both sides.
SMALL_ARM = ArmSpec(label="small arm, 200 steps", steps=200)


@dataclass(frozen=True)
class PackagePrice:
    """One package's arms, its totals, and the ordinal judgment used to rank it.

    `decisiveness` is a judgment and is labelled as one: it is the number of layers of this library
    whose validity the result settles, counted by hand, and `settles` names them. It is not a
    measurement and it is not derived from anything. It is here because ranking packages by dollars
    alone would put the cheapest first regardless of what it decides, and ranking by dollars per
    decided layer is the ordering a maintainer with one budget actually wants.
    """

    package: str
    arms: tuple[ArmSpec, ...]
    seeds: int
    decisiveness: int
    settles: str

    @property
    def gpu_hours(self) -> float:
        return self.seeds * sum(a.gpu_hours for a in self.arms)

    @property
    def wall_hours(self) -> float:
        """Serial wall-clock, which is what a maintainer with one node actually waits."""
        return self.seeds * sum(a.wall_hours for a in self.arms)

    @property
    def steps(self) -> int:
        return self.seeds * sum(a.steps for a in self.arms)

    def dollars(self, price_point: str = "spec_floor_preemptible") -> float:
        return self.gpu_hours * PRICE_POINTS[price_point]

    @property
    def in_campaigns(self) -> float:
        """The cost in units of the entire published campaign, which is the useful scale here."""
        return self.gpu_hours / CAMPAIGN_GPU_HOURS

    @property
    def in_production_arms(self) -> float:
        return self.gpu_hours / PRODUCTION_ARM_GPU_HOURS

    def dollars_per_decided_layer(self, price_point: str = "spec_floor_preemptible") -> float:
        return self.dollars(price_point) / max(self.decisiveness, 1)

    def render(self, price_point: str = "spec_floor_preemptible") -> str:
        arms = "; ".join(
            f"{a.label} x{self.seeds} ({a.gpu_hours:.1f} GPU-h each)" for a in self.arms
        )
        return (
            f"{self.package}: {self.gpu_hours:.1f} GPU-hours, "
            f"${self.dollars(price_point):,.2f} at ${PRICE_POINTS[price_point]:.2f}/GPU-h. "
            f"That is {self.in_campaigns:.1f} published campaigns and "
            f"{self.in_production_arms:.2f} of one production arm.\n"
            f"    Arms: {arms}. {self.steps:,} optimizer steps in total.\n"
            f"    Settles: {self.settles} "
            f"(${self.dollars_per_decided_layer(price_point):,.0f} per layer decided)."
        )


def w6_1_price(seeds: int = 3) -> PackagePrice:
    """H2, the two-run rate test. Two arms over one lambda range, one four times faster.

    The fast arm crosses the same lambda interval in a quarter of the steps, so it costs a quarter
    of the slow one and the pair costs 1.25 arms. Seeds are within-arm replicates and they are what
    the band is built from; three is the floor the instrument's own `BELOW_LOD` remedy asks for when
    the bands come back wider than the between-arm difference.
    """
    return PackagePrice(
        package="W6.1  H2, the two-run rate test",
        arms=(
            replace(SMALL_ARM, label="slow arm, 200 steps at rate v"),
            replace(SMALL_ARM, label="fast arm, 50 steps at rate 4v", steps=50),
        ),
        seeds=seeds,
        decisiveness=3,
        settles=(
            "whether critical slowing down can fire at all on an annealed RL run, which decides "
            "the early-warning layer, the whole of `monitor/`'s lead-time claims, and whether "
            "Level 0 equilibrium extrapolation is licensed"
        ),
    )


def w6_2_perturb_price(seeds: int = 3, episodes: int = 5, hold_steps: int = 60) -> PackagePrice:
    """H1 rung 1. One arm per seed with the holds embedded in it.

    The episodes sit inside a single run rather than in separate arms: run to a chosen step, perturb,
    pin the schedule for the hold, release, continue. So the cost is the base arm plus the held
    steps, and the held steps are pure overhead because the schedule is not advancing during them.

    Five episodes rather than three, and the extra two are the whole reason this line item is what
    it is. At three episodes the linear-response check cannot reach significance at any level below
    0.333, so a three-episode protocol buys a time constant whose defining assumption is untested.
    Five episodes cost 40 percent more held steps and make the check possible at 0.017.
    """
    return PackagePrice(
        package="W6.2a  H1 rung 1, tau_relax by perturb-and-hold",
        arms=(
            replace(SMALL_ARM, label="base arm, 200 steps"),
            replace(
                SMALL_ARM,
                label=f"{episodes} holds of {hold_steps} steps, schedule pinned",
                steps=episodes * hold_steps,
            ),
        ),
        seeds=seeds,
        decisiveness=2,
        settles=(
            "whether the free lag-1 relaxation time this library already ships agrees with the "
            "protocol that defines the quantity, which decides every `QUASI_STATIC` verdict and "
            "the `Ad` gate on every equilibrium instrument"
        ),
    )


def w6_2_hysteresis_price(
    seeds: int = 3, points_per_rate: tuple[int, ...] = (40, 80, 160, 320)
) -> PackagePrice:
    """H3. Four up-and-back sweeps over one lambda range at four rates.

    Each sweep visits its schedule points twice, up then down, and each point is one optimizer step,
    so a sweep at `n` points costs `2n` steps. The slowest sweep is 53 percent of this line item on
    its own, which is why the runbook says to add seeds before adding rates.
    """
    return PackagePrice(
        package="W6.2b  H3, rate-extrapolated hysteresis",
        arms=tuple(
            replace(SMALL_ARM, label=f"up-and-back at {n} points", steps=2 * n)
            for n in points_per_rate
        ),
        seeds=seeds,
        decisiveness=1,
        settles=(
            "whether a hacked policy can be annealed back, which decides whether KL-annealing is a "
            "recovery tool. A real answer either way and it gates nothing else in this library"
        ),
    )


def ranked(price_point: str = "spec_floor_preemptible") -> list[PackagePrice]:
    """Every package here, cheapest-per-decided-layer first. This is the order to buy them in."""
    return sorted(
        (w6_1_price(), w6_2_perturb_price(), w6_2_hysteresis_price()),
        key=lambda p: p.dollars_per_decided_layer(price_point),
    )


def render_all(price_point: str = "spec_floor_preemptible") -> str:
    """The whole price sheet, in the order to buy it."""
    lines = [
        f"Priced at ${PRICE_POINTS[price_point]:.2f} per GPU-hour ({price_point}).",
        f"For scale: the published campaign cost ${CAMPAIGN_DOLLARS:.2f} and "
        f"{CAMPAIGN_GPU_HOURS:.3f} GPU-hours in total.",
        "",
    ]
    total_hours = 0.0
    for p in ranked(price_point):
        lines.append(p.render(price_point))
        lines.append("")
        total_hours += p.gpu_hours
    lines.append(
        f"All three: {total_hours:.1f} GPU-hours, "
        f"${total_hours * PRICE_POINTS[price_point]:,.2f}, "
        f"{total_hours / CAMPAIGN_GPU_HOURS:.0f} published campaigns, "
        f"{total_hours / PRODUCTION_ARM_GPU_HOURS:.2f} of one production arm."
    )
    return "\n".join(lines)


__all__ = [
    "CAMPAIGN_DOLLARS",
    "CAMPAIGN_GPU_HOURS",
    "PRICE_POINTS",
    "PRODUCTION_ARM_GPU_HOURS",
    "SMALL_ARM",
    "ArmSpec",
    "PackagePrice",
    "ranked",
    "render_all",
    "w6_1_price",
    "w6_2_hysteresis_price",
    "w6_2_perturb_price",
]

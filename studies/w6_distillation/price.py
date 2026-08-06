"""What K1 costs, arm by arm, with every input either sourced or declared as an assumption.

Nothing in this package is ever run against a real subject by the people who wrote it. The point of
this module is that the maintainer can decide whether to buy the compute with the bill in front of
them rather than with an adjective, so every number below is computed from an input that is either
quoted from the field scan or marked as an assumption with its basis and its sensitivity.

**The finding this module exists to deliver, before any of the arithmetic.** The whole cost of K1 is
the cost of *producing the two checkpoints*. Every rollout the instrument reads, every blank arm,
and even the per-token localisation arm together come to under seven GPU-hours, which is under half
a percent of the bill. So the decision in front of the maintainer is not "is this measurement worth
buying"; it is "is an RL expert and its own distilled student worth buying", and the answer to the
second question makes the first one free.

**Sourced inputs**, all from the project's survey of published rates and stacks:

- One real group-relative RL run on an 8B policy with a separate reward model, at 8 H100s for 48
  hours, is **384 GPU-hours**.
- H100 80GB is **$1.50 to $2.01 per GPU-hour** at specialised neoclouds, $2.69 at RunPod, $3.99 at
  Lambda, and **$3.95 on Modal** at its per-second rate of $0.001097/sec.
- On-policy distillation "reaches RL-trained performance in **7-10x fewer gradient steps**",
  quoting the Thinking Machines Lab post that all three labs cite.
- For scale: ten seeds by three conditions is 11,520 GPU-hours at **$17k to $23k**; the entire
  published campaign metered **$17.73** over **4.465 GPU-hours**.

**One number in the scale estimate that the evidence does not carry.** The multi-seed study is
priced "against a floor of about $2.15 per GPU-hour preemptible". `$2.15` appears nowhere in the
rate survey, whose floor is $1.50 to $2.01 per GPU-hour **on demand**, which a preemptible rate
would sit below rather than above. The band quoted beside it, $17,000 to $23,000 over 11,520
GPU-hours, works out at $1.48 to $2.00 per GPU-hour, so those dollar figures are consistent with the
sourced on-demand floor and inconsistent with the $2.15 sentence next to them. This module prices at
the sourced band and records the discrepancy rather than carrying the unsourced number forward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from reward_lens.core.quantity import CostModel

# ---------------------------------------------------------------------------
# Sourced constants
# ---------------------------------------------------------------------------

#: GPU-hours for one group-relative RL run on an 8B policy with a separate reward model, from the
#: rate survey: 8xH100 for 48 hours is 384 GPU-hours.
RL_RUN_GPU_HOURS = 384.0

#: The published range of how much cheaper on-policy distillation is than the RL it replaces, in
#: gradient steps: "7-10x fewer gradient steps". The compute saving quoted alongside is 50-100x,
#: which is the larger claim and is not used here: it amortises teacher FLOPs differently and using
#: it would make this estimate the optimistic one.
DISTIL_STEP_RATIO = (7.0, 10.0)

#: Dollars per H100 GPU-hour, from the rate survey's table.
H100_NEOCLOUD = (1.50, 2.01)
H100_MODAL = 3.95

#: What the published campaign actually cost, for scale: the metered dollars it reported, over the
#: GPU-hours the scale estimate records for it.
CAMPAIGN_DOLLARS = 17.73
CAMPAIGN_GPU_HOURS = 4.465

#: The multi-seed reference point. Ten seeds by three conditions at one RL run each.
REFERENCE_SEEDS = 10
REFERENCE_CONDITIONS = 3


# ---------------------------------------------------------------------------
# Declared assumptions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Assumptions:
    """The inputs that are planning figures rather than quotations, each with why it is safe.

    Both throughputs are order-of-magnitude planning figures for an 8B model at bf16 on one H100,
    and neither is a measurement this project made. They are safe to leave as assumptions because
    the arms they price come to under half a percent of the bill: `sensitivity()` computes what a
    fourfold error in either would do to the total, and the answer is that it does not change the
    decision. If they ever stop being negligible, measure them and pass them in.
    """

    #: Output tokens per GPU-second, autoregressive decoding under continuous batching.
    generation_tokens_per_gpu_second: float = 2_000.0
    #: Tokens per GPU-second for a teacher-forced forward and backward pass over text that already
    #: exists. Higher than generation because there is no sequential decode to wait on.
    backward_tokens_per_gpu_second: float = 8_000.0
    #: Seeds. Three is the smallest number that gives a between-seed spread at all, which is L1's
    #: argument about reference materials applied here: a single-seed expert has no homogeneity
    #: term, so a survival fraction measured on one has no way to say whether it is a property of
    #: distillation or of that seed.
    n_seeds: int = 3
    n_prompts: int = 1_000
    completions_per_prompt: int = 8
    max_new_tokens: int = 512
    #: Blank arms: base-checkpoint re-draws at different sampling seeds. Three, because the
    #: detection floor is a standard deviation over one replicate per feature per blank arm and one
    #: blank arm over a nine-feature basis is nine replicates.
    n_blank_arms: int = 3
    #: Whether to price the token-level localisation arm (A4), which needs `POLICY: BACKWARD` on the
    #: expert and the student.
    include_localisation: bool = True


@dataclass(frozen=True)
class ArmPrice:
    """One arm's compute, with what it buys and what it is priced from."""

    id: str
    what: str
    gpu_hours: float
    basis: str

    def cost(self, dollars_per_gpu_hour: float) -> CostModel:
        return CostModel(
            gpu_seconds=self.gpu_hours * 3600.0,
            dollars=self.gpu_hours * dollars_per_gpu_hour,
            note=f"{self.id}: {self.basis}",
        )


@dataclass(frozen=True)
class Price:
    """The whole bill, per arm, with the two rate cards it is quoted against."""

    arms: tuple[ArmPrice, ...]
    assumptions: Assumptions
    neocloud: tuple[float, float] = H100_NEOCLOUD
    modal: float = H100_MODAL

    @property
    def gpu_hours(self) -> float:
        return sum(a.gpu_hours for a in self.arms)

    @property
    def training_gpu_hours(self) -> float:
        return sum(a.gpu_hours for a in self.arms if a.id.startswith("A1") or a.id.startswith("A2"))

    @property
    def inference_gpu_hours(self) -> float:
        return self.gpu_hours - self.training_gpu_hours

    @property
    def dollars(self) -> tuple[float, float]:
        return (self.gpu_hours * self.neocloud[0], self.gpu_hours * self.neocloud[1])

    @property
    def dollars_modal(self) -> float:
        return self.gpu_hours * self.modal

    @property
    def campaign_multiple(self) -> float:
        """How many published campaigns this is, which is the comparison that makes it concrete."""
        return self.dollars[1] / CAMPAIGN_DOLLARS

    def sensitivity(self, factor: float = 4.0) -> float:
        """Fraction of the total the inference arms would move to if both throughputs were `factor`
        times worse than assumed. The check that says the assumptions do not carry the answer."""
        total = self.training_gpu_hours + self.inference_gpu_hours * factor
        return (self.inference_gpu_hours * factor) / total if total > 0 else float("nan")

    def render(self) -> str:
        lines = [
            f"K1 price: {self.gpu_hours:,.1f} GPU-hours, "
            f"${self.dollars[0]:,.0f} to ${self.dollars[1]:,.0f} at the H100 neocloud band "
            f"(${self.neocloud[0]:.2f} to ${self.neocloud[1]:.2f} per GPU-hour), "
            f"${self.dollars_modal:,.0f} on Modal at ${self.modal:.2f}.",
            f"    {self.training_gpu_hours:,.1f} GPU-hours of it is producing the checkpoints and "
            f"{self.inference_gpu_hours:,.1f} is everything the instrument does.",
            f"    For scale: the entire published campaign metered ${CAMPAIGN_DOLLARS:,.2f} over "
            f"{CAMPAIGN_GPU_HOURS} GPU-hours, so this is about "
            f"{self.campaign_multiple:,.0f} of them.",
            f"    A fourfold error in both throughput assumptions would still leave the inference "
            f"arms at {self.sensitivity():.1%} of the total.",
            "",
        ]
        width = max(len(a.id) for a in self.arms)
        for arm in self.arms:
            lines.append(
                f"    {arm.id:<{width}}  {arm.gpu_hours:>9,.2f} GPU-h  "
                f"${arm.gpu_hours * self.neocloud[1]:>9,.0f}  {arm.what}"
            )
        return "\n".join(lines)


def reference_multi_seed_gpu_hours(
    seeds: int = REFERENCE_SEEDS, conditions: int = REFERENCE_CONDITIONS
) -> float:
    """The scale marker, recomputed rather than quoted: seeds by conditions by one RL run."""
    return seeds * conditions * RL_RUN_GPU_HOURS


def price(assumptions: Assumptions | None = None) -> Price:
    """The bill for the three-seed version of K1, arm by arm.

    The distillation arm is priced at the **pessimistic** end of the published range, 7x rather than
    10x fewer gradient steps, because a price that assumes the best published ratio is a price that
    is wrong in the direction that gets the compute bought and then overspent.
    """
    a = assumptions or Assumptions()
    generated = (
        a.n_prompts * a.completions_per_prompt * a.max_new_tokens
    )  # output tokens per arm-draw
    base_draws = 1 + a.n_blank_arms  # base plus its blanks, drawn once and shared across seeds
    per_seed_draws = 2  # the expert and the student
    gen_hours = (
        (base_draws + per_seed_draws * a.n_seeds)
        * generated
        / a.generation_tokens_per_gpu_second
        / 3600.0
    )
    arms = [
        ArmPrice(
            id="A1.expert",
            what=f"{a.n_seeds} group-relative RL runs on an 8B policy, one per seed",
            gpu_hours=a.n_seeds * RL_RUN_GPU_HOURS,
            basis=f"{RL_RUN_GPU_HOURS:g} GPU-h per run, sourced (8xH100 for 48h)",
        ),
        ArmPrice(
            id="A2.student",
            what=f"{a.n_seeds} on-policy distillations of each expert back into the same base",
            gpu_hours=a.n_seeds * RL_RUN_GPU_HOURS / DISTIL_STEP_RATIO[0],
            basis=(
                f"the RL run divided by {DISTIL_STEP_RATIO[0]:g}, the pessimistic end of the "
                f"published {DISTIL_STEP_RATIO[0]:g}-{DISTIL_STEP_RATIO[1]:g}x step ratio"
            ),
        ),
        ArmPrice(
            id="A0.rollouts",
            what=(
                f"{base_draws + per_seed_draws * a.n_seeds} rollout draws of "
                f"{a.n_prompts:,} prompts x {a.completions_per_prompt} completions x "
                f"{a.max_new_tokens} tokens (base, {a.n_blank_arms} blanks, and expert plus "
                f"student per seed)"
            ),
            gpu_hours=gen_hours,
            basis=(
                f"assumed {a.generation_tokens_per_gpu_second:,.0f} output tokens per GPU-second, "
                f"a planning figure"
            ),
        ),
    ]
    if a.include_localisation:
        backward_tokens = 2 * a.n_seeds * generated
        arms.append(
            ArmPrice(
                id="A4.localisation",
                what="per-token gradients on the expert and the student over the audit rollouts",
                gpu_hours=backward_tokens / a.backward_tokens_per_gpu_second / 3600.0,
                basis=(
                    f"assumed {a.backward_tokens_per_gpu_second:,.0f} tokens per GPU-second for a "
                    f"teacher-forced forward and backward, a planning figure"
                ),
            )
        )
    return Price(arms=tuple(arms), assumptions=a)


def as_cost_models(
    p: Price, dollars_per_gpu_hour: float = H100_NEOCLOUD[1]
) -> Mapping[str, CostModel]:
    """The same bill as the kernel's own `CostModel` per arm, so it composes with a preflight."""
    return {arm.id: arm.cost(dollars_per_gpu_hour) for arm in p.arms}


#: What you cannot buy at any of these prices, which is the part of this row that is not arithmetic.
#: K1 needs three checkpoints sharing one base: the pre-RL reference, an expert produced from it by
#: group-relative RL, and a student produced from that expert by on-policy distillation back into
#: the same base. No public release is that triple. The distilled models on the Hub are distilled
#: from one model's outputs into a different base, so the denominator this instrument divides by,
#: the shift RL installed **on this base**, does not exist for them and the reading would answer a
#: question about cross-family transfer instead. That is why the price is a training bill rather
#: than a download.
NO_PUBLIC_TRIPLE = (
    "K1 needs a pre-RL base, an expert trained from it by group-relative RL, and a student "
    "distilled from that expert back into the same base. No lab releases the middle one: the "
    "consolidated model ships and the domain experts do not. The distilled checkpoints that are "
    "public are distilled across families, so they have no shared base to measure an installed "
    "shift against, and running K1 on them would measure cross-family transfer under K1's name."
)


def missing_subject_note() -> str:
    return NO_PUBLIC_TRIPLE


__all__ = [
    "CAMPAIGN_DOLLARS",
    "CAMPAIGN_GPU_HOURS",
    "DISTIL_STEP_RATIO",
    "H100_MODAL",
    "H100_NEOCLOUD",
    "NO_PUBLIC_TRIPLE",
    "RL_RUN_GPU_HOURS",
    "ArmPrice",
    "Assumptions",
    "Price",
    "as_cost_models",
    "missing_subject_note",
    "price",
    "reference_multi_seed_gpu_hours",
]

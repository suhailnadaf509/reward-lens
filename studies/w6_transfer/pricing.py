"""What a compute-gated row costs, with the assumptions written out beside the number.

Three of the eight Phase 6 rows are here and none of them will be run by the person who wrote them.
The maintainer decides later whether to buy the compute, and that decision needs a price, the
arithmetic behind the price, and a way to rank the rows against each other. A row quoted as
"expensive" or "cheap" is not quotable.

The rates
---------

Every rate in `RATES` is a mid-2026 market figure from the funding dossier's own table, carried here
with the vendor named so a stale rate is visible as a stale rate rather than as a wrong quote. GPU
pricing moves; the dossier itself notes that Prime Intellect's exchange aggregates twelve clouds
with realtime prices, so any figure is a snapshot. `check_dossier_arithmetic` reproduces the
dossier's three published workload costs from these rates before anything else uses them, which is
the same discipline X3 applied to the number it was correcting.

One inconsistency, reproduced rather than repeated
--------------------------------------------------

The project's scale estimate quotes three numbers in one sentence: 11,520 GPU-hours, $17,000 to
$23,000, and "a floor of about $2.15 per GPU-hour preemptible". The three cannot all hold.
11,520 x 2.15 = $24,768, which is above the top of the band in the same sentence. The dossier's own
floor is $1.50 to $2.01 for an H100, and 11,520 x 1.50 = $17,280 with 11,520 x 2.01 = $23,155,
which is the band exactly. So the band is right and the $2.15 floor is not the floor the band was
computed from. `check_dossier_arithmetic` asserts this rather than describing it, and every quote
here is struck at the dossier rates.

What a price here does not include
----------------------------------

Storage, egress and person-time, except where a row's storage is large enough to change the
decision, which for K4 it is: FP32 master weights for an 8B policy are 32 GB per checkpoint and the
staleness sweep wants hundreds of them. Where storage is material it is a line item with its own
assumption. Everywhere else the number is compute only and says so.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from reward_lens.core.quantity import CostModel

#: Mid-2026 per-GPU-hour rates, from the funding dossier's consolidated market table. `(floor, high)`
#: with the vendor behind each end, so a quote can be struck at either and the reader can see which.
RATES: dict[str, tuple[float, float, str]] = {
    "A100-80GB": (1.12, 1.48, "Prime Intellect floor to mid-market"),
    "H100-80GB": (1.50, 2.01, "neocloud floor band; GMI at 2.00"),
    "H100-modal": (3.95, 3.95, "Modal per-second H100, no idle charge"),
    "H200-141GB": (2.60, 3.89, "GMI floor to CoreWeave"),
    "B200": (2.65, 3.49, "CoreWeave reserved to Lambda on-demand"),
}

#: One arm of real group-relative RL on an 8B policy: 8 H100s for 48 hours. The unit every RL price
#: in this package is built from, and the dossier's own worked example.
GRPO_ARM_GPU_HOURS = 384.0

#: What the published campaign actually cost, from the store this library reads: $17.73 over 4.465
#: GPU-hours. The implied rate, $3.97, reproduces Modal's published $3.95 to within half a percent,
#: which is the cross-check that the two numbers describe the same purchase.
CAMPAIGN_DOLLARS = 17.73
CAMPAIGN_GPU_HOURS = 4.465

#: Bytes per parameter when master weights are kept in FP32. Named because K4's whole storage line
#: item is this constant times a parameter count times a checkpoint count.
FP32_BYTES_PER_PARAM = 4

#: Object storage, dollars per gigabyte-month, mid-range. Coarse on purpose: the point of carrying
#: it is that a plan needing ten terabytes has a running cost, not that the third digit is right.
STORAGE_DOLLARS_PER_GB_MONTH = 0.02


def check_dossier_arithmetic() -> dict[str, float]:
    """Reproduce the dossier's three published workload costs from the rates above.

    Returns the reproduced figures so a caller can print them next to the published ones. Raises if
    any of the three misses its published value by more than a percent, because a rate table that
    does not reproduce the costs it was quoted alongside is a rate table somebody edited.
    """
    lo, hi = RATES["H100-80GB"][0], RATES["H100-80GB"][1]
    modal = RATES["H100-modal"][0]
    multi_seed_hours = 10 * 3 * GRPO_ARM_GPU_HOURS
    out = {
        "one_arm_low": GRPO_ARM_GPU_HOURS * lo,
        "one_arm_high": GRPO_ARM_GPU_HOURS * hi,
        "one_arm_modal": GRPO_ARM_GPU_HOURS * modal,
        "multi_seed_hours": multi_seed_hours,
        "multi_seed_low": multi_seed_hours * lo,
        "multi_seed_high": multi_seed_hours * hi,
        "multi_seed_modal": multi_seed_hours * modal,
        "campaign_implied_rate": CAMPAIGN_DOLLARS / CAMPAIGN_GPU_HOURS,
        "spec_floor_on_multi_seed": multi_seed_hours * 2.15,
    }
    published = {
        "one_arm_low": 576.0,
        "one_arm_high": 772.0,
        "one_arm_modal": 1517.0,
        "multi_seed_hours": 11_520.0,
        "multi_seed_modal": 45_504.0,
    }
    for key, want in published.items():
        got = out[key]
        if abs(got - want) > 0.01 * want:
            raise AssertionError(
                f"{key} reproduces as {got:,.1f} against a published {want:,.1f}, a "
                f"{abs(got - want) / want:.1%} miss. The rate table and the published costs "
                f"disagree, so one of them has been edited and the quotes below are not struck "
                f"at the rates they claim."
            )
    return out


@dataclass(frozen=True)
class LineItem:
    """One purchasable thing: how many GPU-hours, on what, and why that many.

    ``why`` is not documentation. It is the field a reader disputes when they think the quote is
    wrong, and a line item without one cannot be argued with, only believed.
    """

    what: str
    gpu_hours: float
    gpu: str = "H100-80GB"
    why: str = ""
    dollars_fixed: float = 0.0

    def cost_band(self) -> tuple[float, float]:
        lo, hi, _ = RATES[self.gpu]
        return (self.gpu_hours * lo + self.dollars_fixed, self.gpu_hours * hi + self.dollars_fixed)

    def render(self) -> str:
        lo, hi = self.cost_band()
        hours = f"{self.gpu_hours:,.1f} GPU-h on {self.gpu}" if self.gpu_hours else "no GPU"
        money = f"${lo:,.0f}" if abs(hi - lo) < 1.0 else f"${lo:,.0f}-${hi:,.0f}"
        return (
            f"{self.what}: {hours}, {money}\n      {self.why}"
            if self.why
            else (f"{self.what}: {hours}, {money}")
        )


@dataclass(frozen=True)
class Quote:
    """A row's whole price, its assumptions, and what it buys.

    ``resolvable`` is how many of the study's preregistered rows the planned design is powered to
    settle, computed by `stats/power.py` rather than asserted, and it is the numerator of
    `decisiveness_per_1k`. Ranking on dollars alone would put the row that answers nothing first.
    """

    row: str
    items: tuple[LineItem, ...]
    assumptions: tuple[str, ...] = ()
    resolvable: int = 0
    registered_rows: int = 0
    subject_needed: str = ""
    slack: float = 0.25
    note: str = ""
    extra_dollars: tuple[tuple[str, float], ...] = field(default_factory=tuple)

    @property
    def gpu_hours(self) -> float:
        return sum(i.gpu_hours for i in self.items) * (1.0 + self.slack)

    @property
    def dollars(self) -> tuple[float, float]:
        lo = sum(i.cost_band()[0] for i in self.items) * (1.0 + self.slack)
        hi = sum(i.cost_band()[1] for i in self.items) * (1.0 + self.slack)
        extra = sum(v for _, v in self.extra_dollars)
        return (lo + extra, hi + extra)

    @property
    def dollars_mid(self) -> float:
        lo, hi = self.dollars
        return 0.5 * (lo + hi)

    @property
    def decisiveness_per_1k(self) -> float:
        """Preregistered rows resolved per thousand dollars. Infinite when nothing must be bought.

        The numerator is a count of rows the planned n powers to the study's own target, so it is
        bounded by the study's size and cannot be inflated by writing more hypotheses that the
        design cannot separate. The denominator is the midpoint of the band. Both are crude and
        both are checkable, which is the property that matters for a ranking nobody will audit
        line by line.
        """
        if self.dollars_mid <= 0.0:
            return math.inf
        return 1000.0 * self.resolvable / self.dollars_mid

    def cost_model(self) -> CostModel:
        """The quote as the kernel's `CostModel`, so a preflight can carry it."""
        lo, hi = self.dollars
        return CostModel(
            gpu_seconds=self.gpu_hours * 3600.0,
            dollars=self.dollars_mid,
            note=f"{self.gpu_hours:,.0f} GPU-h, ${lo:,.0f}-${hi:,.0f} at mid-2026 neocloud rates",
        )

    def render(self) -> str:
        lo, hi = self.dollars
        lines = [
            f"{self.row}: {self.gpu_hours:,.1f} GPU-hours, ${lo:,.0f} to ${hi:,.0f}",
            f"    resolves {self.resolvable} of {self.registered_rows} preregistered rows"
            + (
                "  (nothing to buy)"
                if math.isinf(self.decisiveness_per_1k)
                else f"  ({self.decisiveness_per_1k:.2f} rows per $1k)"
            ),
        ]
        if self.subject_needed:
            lines.append(f"    subject: {self.subject_needed}")
        lines.append("    line items:")
        lines += [f"      {i.render()}" for i in self.items]
        for what, amount in self.extra_dollars:
            lines.append(f"      {what}: ${amount:,.0f}")
        if self.slack:
            lines.append(
                f"      slack: {self.slack:.0%} on every compute line, for failed arms and reruns"
            )
        if self.assumptions:
            lines.append("    assumptions:")
            lines += [f"      - {a}" for a in self.assumptions]
        if self.note:
            lines.append(f"    {self.note}")
        return "\n".join(lines)

    def __canonical__(self) -> dict[str, Any]:
        lo, hi = self.dollars
        return {
            "row": self.row,
            "gpu_hours": self.gpu_hours,
            "dollars_low": lo,
            "dollars_high": hi,
            "dollars_mid": self.dollars_mid,
            "resolvable": self.resolvable,
            "registered_rows": self.registered_rows,
            "decisiveness_per_1k": self.decisiveness_per_1k,
            "subject_needed": self.subject_needed,
            "slack": self.slack,
            "items": [
                {"what": i.what, "gpu_hours": i.gpu_hours, "gpu": i.gpu, "why": i.why}
                for i in self.items
            ],
            "extra_dollars": [list(x) for x in self.extra_dollars],
            "assumptions": list(self.assumptions),
            "note": self.note,
        }


def rank(quotes: Sequence[Quote]) -> list[Quote]:
    """Cheapest decisive first: by rows resolved per thousand dollars, descending.

    Ties break on the absolute price, ascending, which matters because the maintainer will buy from
    the top of this list until the money runs out and two rows with the same ratio are not
    equivalent when one of them is an order of magnitude larger.
    """
    return sorted(quotes, key=lambda q: (-q.decisiveness_per_1k, q.dollars_mid))


def render_ranking(quotes: Sequence[Quote]) -> str:
    """The three rows in buying order, with the reason the order is what it is."""
    ordered = rank(quotes)
    lines = ["Compute-gated rows, ranked by preregistered rows resolved per thousand dollars:", ""]
    for i, q in enumerate(ordered, start=1):
        lo, hi = q.dollars
        ratio = (
            "nothing to buy"
            if math.isinf(q.decisiveness_per_1k)
            else f"{q.decisiveness_per_1k:.2f}/$1k"
        )
        lines.append(f"  {i}. {q.row}  ${lo:,.0f}-${hi:,.0f}  ({q.gpu_hours:,.0f} GPU-h, {ratio})")
    total_lo = sum(q.dollars[0] for q in ordered)
    total_hi = sum(q.dollars[1] for q in ordered)
    lines += [
        "",
        f"  all three: ${total_lo:,.0f} to ${total_hi:,.0f}, against the published campaign's "
        f"${CAMPAIGN_DOLLARS:,.2f}",
    ]
    return "\n".join(lines)


__all__ = [
    "CAMPAIGN_DOLLARS",
    "CAMPAIGN_GPU_HOURS",
    "FP32_BYTES_PER_PARAM",
    "GRPO_ARM_GPU_HOURS",
    "RATES",
    "STORAGE_DOLLARS_PER_GB_MONTH",
    "LineItem",
    "Quote",
    "check_dossier_arithmetic",
    "rank",
    "render_ranking",
]

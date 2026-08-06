"""The recovery table: one row per localisation method against one planted key (C3).

The genre nobody publishes. Every method in the library scores the same candidates against the same
planted answer key, the rank statistic of the planted candidates against the rest is the row, and
the table prints in measured order rather than in the order we would like.

**The losses are the deliverable.** Three published head-to-heads put white-box methods last:
black-box prompting reaching 83.6% on 720 planted-rule models, scaffolded black-box tools beating
white-box across 56 organisms, and logit lens, sparse autoencoders and circuit tracing providing "no
reliable benefit despite full internal access". A library that publishes its own table including the
rows where it loses is doing what the field says it needs; one that publishes only its wins is doing
what everyone already does. So `RecoveryTable` has no method for hiding a row, `losers` exists and
`winners` does not, and `render` prints every row in rank order with the trust ordering beside the
measured one so the disagreement between them is the first thing visible.

**Rows can be measured here or cited from a stored result, and the difference is never invisible.**
`RecoveryRow.source` is `"measured"` for a row this instrument computed and the stored observable's
name for a row read out of a campaign store, with the evidence id beside it. Mixing them in one
table is right, because the campaign's own attribution-versus-patching result is a real row of this
exact table and re-running it is not possible without the GPU it was taken on. Mixing them
*silently* would let a cited number be read as a fresh measurement, which is why the field is
required rather than defaulted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from reward_lens.core.evidence import register_payload
from reward_lens.policy.selection import MethodClass

#: What a row's `source` says when this instrument computed it in this process.
MEASURED = "measured"


def recovery_auc(scores: Sequence[float], planted: Sequence[bool]) -> tuple[float, int, int]:
    """`P(a planted candidate scores above an unplanted one)`, ties at one half.

    The Mann-Whitney identity from `stats.roc`, not a second implementation: the same rank
    statistic every AUC in this library goes through, so ties are handled the same way and two
    recovery numbers taken a year apart are the same statistic.

    Returns `(auc, n_planted, n_unplanted)`. The two counts travel with the number because an AUC
    over three planted candidates against fifty-three is a different object from one over thirty
    against thirty, and the campaign's own row is the first kind: 3 planted, 53 not, 159 pairs.
    """
    from reward_lens.stats.roc import roc_pr

    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(planted).ravel().astype(int)
    if s.size != y.size:
        raise ValueError(
            f"{s.size} scores and {y.size} planted flags. A method scored against a misaligned key "
            f"reports chance and nothing in the number says so."
        )
    n_pos = int(np.count_nonzero(y == 1))
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan"), n_pos, n_neg
    return float(roc_pr(s, y.astype(np.float64)).auc), n_pos, n_neg


def _auc_interval(
    scores: np.ndarray, planted: np.ndarray, *, n_boot: int, seed: int, level: float
) -> tuple[float, float]:
    """A percentile interval by resampling candidates, stratified so both classes survive a draw.

    Stratified because the planted class is tiny by construction: three planted out of fifty-six is
    the real case, and an unstratified resample drops all three on roughly 3% of draws, which turns
    into a NaN and quietly narrows the interval by removing exactly the hardest draws.
    """
    pos = np.flatnonzero(planted == 1)
    neg = np.flatnonzero(planted == 0)
    if pos.size == 0 or neg.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        idx = np.concatenate(
            [rng.choice(pos, pos.size, replace=True), rng.choice(neg, neg.size, replace=True)]
        )
        auc, _, _ = recovery_auc(scores[idx], planted[idx])
        if np.isfinite(auc):
            draws.append(auc)
    if not draws:
        return float("nan"), float("nan")
    alpha = (1.0 - level) / 2.0
    return float(np.quantile(draws, alpha)), float(np.quantile(draws, 1.0 - alpha))


@dataclass(frozen=True)
class RecoveryRow:
    """One method's recovery of one planted key, with what it cost and where it came from.

    ``n_parameters`` carries the argument the table is really about. A zero-parameter behavioural
    correlation and a fitted probe are both rows, and a white-box method that beats the probe while
    losing to the correlation has learned nothing worth the access it needed.

    ``may_carry_a_claim`` is read off the method class rather than stored, so a sparse dictionary
    cannot be admitted by writing `True` in a constructor.
    """

    method_id: str
    method_class: MethodClass
    auc: float
    n_planted: int
    n_unplanted: int
    n_parameters: int = 0
    ci_low: float = float("nan")
    ci_high: float = float("nan")
    ci_level: float = 0.95
    source: str = MEASURED
    evidence_id: str | None = None
    detail: str = ""

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError(
                f"row {self.method_id!r} has no source. A row computed here and a row cited from a "
                f"stored result are different kinds of evidence and the table has to be able to "
                f"say which it is holding."
            )

    @property
    def may_carry_a_claim(self) -> bool:
        return self.method_class.may_carry_a_claim

    @property
    def is_measured_here(self) -> bool:
        return self.source == MEASURED

    @property
    def beats_chance(self) -> bool:
        """Whether the whole interval sits above 0.5. NaN intervals are not a pass."""
        return bool(np.isfinite(self.ci_low) and self.ci_low > 0.5)

    def render(self) -> str:
        interval = (
            f"[{self.ci_low:.3f}, {self.ci_high:.3f}]"
            if np.isfinite(self.ci_low)
            else "[no interval]"
        )
        mark = "" if self.may_carry_a_claim else "  (candidate generator only)"
        cited = "" if self.is_measured_here else f"  cited: {self.source}"
        return (
            f"{self.method_id:<30} {self.auc:.4f} {interval:<18} "
            f"{self.n_parameters:>8,} param  {self.method_class.label}{mark}{cited}"
        )


@register_payload
@dataclass
class RecoveryTable:
    """Every method against one planted key, in measured order.

    ``ours`` names the method whose row the library would like to quote. It exists so `our_rank`
    can be computed and printed, which is the sentence the catalogue's `says` line is made of:
    "DiffMean 0.81, SAE 0.62, logit lens 0.55, scaffolded black-box prompting 0.84. Our method is
    third." A table that cannot say what rank we came is a table that will be quoted selectively.
    """

    rows: list[RecoveryRow] = field(default_factory=list)
    organism: str = ""
    reference_id: str = ""
    ours: str = ""
    n_candidates: int = 0
    note: str = ""

    def ranked(self) -> list[RecoveryRow]:
        """Rows by measured recovery, best first. NaN rows sort last rather than being dropped."""
        return sorted(
            self.rows,
            key=lambda r: (-(r.auc if np.isfinite(r.auc) else -np.inf), r.method_id),
        )

    def our_rank(self) -> int | None:
        """Where the named method placed, 1-based. None when no method was named or it is absent."""
        if not self.ours:
            return None
        for i, row in enumerate(self.ranked(), start=1):
            if row.method_id == self.ours:
                return i
        return None

    def losers(self) -> list[RecoveryRow]:
        """Every row a black-box or zero-parameter method beat.

        The list the table exists to make it hard not to print. Defined against the best
        non-white-box row rather than against chance, because "beats chance" is not the bar: a
        white-box method that beats chance and loses to a string match has not earned the access it
        required.
        """
        outside = [r for r in self.rows if not r.method_class.is_white_box and np.isfinite(r.auc)]
        if not outside:
            return []
        bar = max(r.auc for r in outside)
        return [
            r
            for r in self.ranked()
            if r.method_class.is_white_box and np.isfinite(r.auc) and r.auc < bar
        ]

    def claimable(self) -> list[RecoveryRow]:
        """Rows whose method class is permitted to hold up a claim at all."""
        return [r for r in self.ranked() if r.may_carry_a_claim]

    @property
    def n_methods(self) -> int:
        return len(self.rows)

    @property
    def n_measured_here(self) -> int:
        return sum(1 for r in self.rows if r.is_measured_here)

    def says(self) -> str:
        ranked = self.ranked()
        listed = ", ".join(f"{r.method_id} {r.auc:.3g}" for r in ranked[:6])
        rank = self.our_rank()
        tail = ""
        if rank is not None:
            tail = f" {self.ours} is {_ordinal(rank)} of {len(ranked)}."
        lost = self.losers()
        if lost:
            tail += (
                f" {len(lost)} white-box method(s) placed below the best method that read no "
                f"internals: {', '.join(r.method_id for r in lost)}."
            )
        return f"Against the same planted key: {listed}.{tail}"

    def render(self) -> str:
        lines = [
            f"recovery table on {self.organism or 'an unnamed organism'} "
            f"({self.n_candidates} candidates, reference {self.reference_id or 'none'})",
            f"{'method':<30} {'AUC':<6} {'interval':<18} {'params':>14}  class",
        ]
        lines += [f"  {r.render()}" for r in self.ranked()]
        lines.append(self.says())
        if self.note:
            lines.append(self.note)
        return "\n".join(lines)

    def __canonical__(self) -> dict[str, Any]:
        return {
            "organism": self.organism,
            "reference_id": self.reference_id,
            "ours": self.ours,
            "our_rank": self.our_rank(),
            "n_candidates": self.n_candidates,
            "n_methods": self.n_methods,
            "n_measured_here": self.n_measured_here,
            "rows": [
                {
                    "method_id": r.method_id,
                    "method_class": r.method_class.name,
                    "trust_rank": r.method_class.trust_rank,
                    "may_carry_a_claim": r.may_carry_a_claim,
                    "auc": r.auc,
                    "ci_low": r.ci_low,
                    "ci_high": r.ci_high,
                    "ci_level": r.ci_level,
                    "n_planted": r.n_planted,
                    "n_unplanted": r.n_unplanted,
                    "n_parameters": r.n_parameters,
                    "source": r.source,
                    "evidence_id": r.evidence_id,
                    "detail": r.detail,
                }
                for r in self.ranked()
            ],
            "losers": [r.method_id for r in self.losers()],
            "says": self.says(),
            "note": self.note,
        }


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }".replace(" ", "")


def score_row(
    method_id: str,
    method_class: MethodClass,
    scores: Sequence[float],
    planted: Sequence[bool],
    *,
    n_parameters: int = 0,
    n_boot: int = 1000,
    seed: int = 0,
    level: float = 0.95,
    detail: str = "",
) -> RecoveryRow:
    """One method's per-candidate scores into a row, with a stratified bootstrap interval."""
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(planted).ravel().astype(int)
    auc, n_pos, n_neg = recovery_auc(s, y)
    lo, hi = _auc_interval(s, y, n_boot=n_boot, seed=seed, level=level)
    return RecoveryRow(
        method_id=method_id,
        method_class=method_class,
        auc=auc,
        n_planted=n_pos,
        n_unplanted=n_neg,
        n_parameters=n_parameters,
        ci_low=lo,
        ci_high=hi,
        ci_level=level,
        source=MEASURED,
        detail=detail,
    )


__all__ = [
    "MEASURED",
    "RecoveryRow",
    "RecoveryTable",
    "recovery_auc",
    "score_row",
]

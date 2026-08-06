"""M3 as an instrument: the bank, run against a claim, with the comparison as the reading.

The six baselines live in `stats.baselines` because they are statistics and nothing more. This is
the instrument wrapper: it takes a claim's own per-item scores and the task those scores were
produced on, runs all six, and returns the comparison. The value it emits carries a `baselines`
mapping, which is what `stats.baselines.lint_claim` looks for, so a reading produced through this
instrument passes the lint by construction rather than by discipline.

The reading is the best baseline's score, not the margin. That is deliberate: a claim's own number
is already recorded, and what the bank contributes is the floor it has to clear. A card that
prints "AUROC 0.998" beside "best dumb baseline 1.000" has said everything, and a card that prints
a margin of -0.002 has buried it.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from reward_lens.core.envelope import EnvelopeSpec
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.quantity import BaselineID
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.types import (
    Access,
    Capability,
    Component,
    GaugeStatus,
    Phase,
    Substrate,
)
from reward_lens.measure.controls._base import ControlInstrument
from reward_lens.stats.baselines import (
    BaselineBank,
    BaselineComparison,
    DetectionTask,
    compare_against_baselines,
    run_bank,
)

#: What the bank itself ships against. A rank statistic on a two-class task has a chance value of
#: exactly 0.5, and a bank whose best member sits at chance has told you the task is not separable
#: by anything simple, which is a result rather than a gap.
CHANCE: BaselineID = "baseline.chance"

BASELINE_ENVELOPE = EnvelopeSpec(
    unconditional=True,
    justification=(
        "each baseline is a function of the transcript and the label and of nothing else, so no "
        "property of the optimisation that produced them can make a baseline's own score wrong. "
        "What a regime can spoil is the labels, and that is `LABEL_QUALITY_UNKNOWN` rather than a "
        "regime condition."
    ),
)


class DumbBaselineBank(ControlInstrument):
    """M3. The six dumb baselines, run against one claim, with the verdict named.

    The verdict is the field that matters and `matched` is why it exists. A detector a
    zero-parameter string match ties has not been shown to work; it has been shown to agree with a
    string match. This instrument will not render that as a win.
    """

    name = "DumbBaselineBank"
    version = "1.0"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "M3"
    deviations = (
        "the scaffolded black-box baseline needs an inference callable and refuses without one, "
        "so a bank run with no judge reports five scores and one recorded refusal rather than six "
        "scores",
        "`matched` is decided by a paired bootstrap interval on the AUROC difference rather than "
        "by a margin constant, so the verdict depends on n as well as on the gap",
    )

    quantity = "baseline.best_score"
    #: The field is `requires`, not `access`. Declared under the wrong name, this was a
    #: plain class attribute nothing read: `declared_access` looks up `requires` and returned an
    #: empty matrix, so preflight checked nothing on this instrument.
    requires = {Component.RECORD: Access.RECORD}
    substrates = frozenset(
        {
            Substrate.NEURAL_SCALAR,
            Substrate.NEURAL_GEN,
            Substrate.PROGRAM,
            Substrate.PROCEDURAL,
            Substrate.HUMAN,
            Substrate.COMPOSITE,
        }
    )
    phases = frozenset({Phase.PRE_RUN, Phase.POST_RUN})
    envelope = BASELINE_ENVELOPE
    invariance = "reward.affine"
    invariance_relation = INVARIANT
    baselines = (CHANCE,)
    rung = 0

    def __init__(
        self,
        task: DetectionTask | None = None,
        own_scores: Sequence[float] | np.ndarray | None = None,
        *,
        which: Sequence[BaselineID] | None = None,
        ci: float = 0.95,
        seed: int = 0,
    ) -> None:
        self.task = task
        self.own_scores = own_scores
        self.which = which
        self.ci = float(ci)
        self.seed = int(seed)

    def compute(self) -> Any:
        if self.task is None:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail="no detection task was supplied, so there is nothing to run the bank on",
                remedy=(
                    "pass `task=DetectionTask(labels=..., texts=...)` built from the same items "
                    "the claim was scored on. The bank has to see the claim's own task or the "
                    "comparison is between two different problems."
                ),
            )
        bank = run_bank(self.task, self.which)
        if self.own_scores is None:
            return (bank, None)
        comparison = compare_against_baselines(
            self.own_scores,
            self.task.labels,
            bank,
            seed_labels=self.task.seed_labels,
            ci=self.ci,
            seed=self.seed,
        )
        return (bank, comparison)

    def payload(self, computed: tuple[BaselineBank, BaselineComparison | None]) -> dict[str, Any]:
        bank, comparison = computed
        best = bank.best()
        out: dict[str, Any] = {
            "best_baseline": best.baseline if best is not None else None,
            "best_score": best.auroc if best is not None else float("nan"),
            "n_scored": len(bank.scored()),
            "n_refused": len(bank.refusals()),
            "refused": sorted(bank.refusals()),
            "baselines": {**bank.as_mapping(), CHANCE: 0.5},
        }
        if comparison is not None:
            out.update(
                {
                    "own": comparison.own,
                    "margin": comparison.margin,
                    "ci_low": comparison.ci_low,
                    "ci_high": comparison.ci_high,
                    "verdict": comparison.verdict,
                }
            )
        return out


__all__ = ["BASELINE_ENVELOPE", "CHANCE", "DumbBaselineBank"]

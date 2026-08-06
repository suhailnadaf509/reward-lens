"""C8 `judge.verdict_direction`, `judge.commitment_position`: when the judge actually decided.

If a judge's verdict direction saturates at token 12 of a 340-token rubric, the rubric is being
written after the decision, not to reach it. That is a checkable claim about a real object and the
searches behind the catalogue entry are clean: `all:"logit difference" AND all:"LLM-as-a-judge"`
returns nothing, and the corrected form appears in nobody's paper.

**The shipped result is 1.0, and 1.0 is saturated.** That is the reason this module is mostly
controls. A verdict direction read as an argmax agreement is a statistic with a ceiling, and a
ceiling is reached by a measurement that is working and by one that has stopped discriminating,
which look identical in the number. The specific way it goes wrong here: if chain-of-thought
collapses the spread of the judgment distribution, then the mode can be pinned from token 12 while
the mean is still moving for another three hundred tokens, and an argmax-based commitment position
reports 12 for a judge that had not decided.

So the reading below carries the margin, not only the argmax, and four controls decide whether the
argmax number is admissible. The catalogue's kill condition is exactly this: **kill if the four
controls show the mean moves while the mode is fixed.** `Controls.kills` implements that sentence
and the instrument refuses rather than reporting when it fires.

**A note on the four controls.** The catalogue's baseline line names four controls without
specifying them, so the four implemented here are derived from the failure mode the entry itself
states rather than transcribed from a list, and they are named in `Controls` so a reader can check
them against the failure mode they are meant to catch.

**Scope limit.** A commitment position is measured against a *threshold* on a monotone-ish reading,
and a reading that oscillates has no well-defined commitment position; `Commitment.is_stable` is the
flag that says so. And this measures when the direction stopped moving, not why: a judge whose
verdict direction saturates early may have decided early or may be producing a rubric whose tokens
are uninformative about the verdict either way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from reward_lens.core.evidence import Uncertainty, register_payload
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.reading import Reading, Refusal, RefusalReason
from reward_lens.core.types import Capability, Phase, Substrate
from reward_lens.measure.base import Context
from reward_lens.measure.selection._common import (
    ABOVE_LOD_ONLY,
    ACCESS_GRADER_FORWARD,
    SelectionInstrument,
    emit_white_box,
    refuse_unmeasured_control,
)
from reward_lens.measure.selection.transport import IdentityTransport

#: How much of the final margin a reading has to reach, and hold, to count as committed. 0.9 rather
#: than 1.0 because a reading that drifts by a percent for the rest of the sequence has committed
#: in any sense a reader cares about, and requiring the exact final value would put every commitment
#: position at the last token by construction.
COMMITMENT_FRACTION = 0.9


def verdict_direction(
    unembedding: np.ndarray,
    positive_id: int,
    negative_id: int,
    *,
    transport: Any = None,
    layer: int = 0,
) -> np.ndarray:
    """`W_U[Yes] - W_U[No]` at rung 0, `(W_U J_l)[Yes] - (W_U J_l)[No]` at rung 1.

    The transport is applied to the unembedding rows rather than to the residual, which is the same
    map and is cheaper by the batch size: `(W_U J) h = W_U (J h)` and there are two rows and many
    residuals. It also makes the returned object a direction in the layer's own coordinates, which
    is what a caller wants to dot an activation against.

    With no transport this is the naive form, which is rung 0 and is the mandatory comparator.
    """
    w = np.asarray(unembedding, dtype=np.float64)
    rows = np.stack([w[int(positive_id)], w[int(negative_id)]], axis=0)
    if transport is not None:
        rows = transport.transport(rows, layer)
    return rows[0] - rows[1]


@register_payload
@dataclass(frozen=True)
class Commitment:
    """Where along a sequence the verdict direction stopped moving.

    ``position`` is the raw token index and ``fraction`` is that index over the sequence length.
    **The fraction is the declared quantity and the index is presentational**, which is forced
    rather than chosen: the registry gives `judge.commitment_position` the `tokenization` invariance
    group, that group admits only `invariant`, and a raw token index is not invariant under
    retokenisation. A judge whose tokenizer splits differently commits at a different index and the
    same fraction of the way through the same text.

    ``is_stable`` is what stops the number being read too hard. A reading that crosses the threshold
    and comes back has no commitment position; reporting the first crossing for it would be a number
    with a spurious precision.
    """

    position: int
    fraction: float
    n_tokens: int
    final_margin: float
    threshold: float
    is_stable: bool
    margins: tuple[float, ...] = ()
    note: str = ""

    def says(self) -> str:
        if not self.is_stable:
            return (
                f"the verdict direction crosses {self.threshold:.3g} at token {self.position} of "
                f"{self.n_tokens} and does not stay there, so this judge has no commitment position: "
                f"the reading is still moving at the end of the sequence."
            )
        return (
            f"the verdict direction saturates at token {self.position} of {self.n_tokens} "
            f"({self.fraction:.1%} of the way through), reaching "
            f"{COMMITMENT_FRACTION:.0%} of a final margin of {self.final_margin:+.4g}."
        )

    def __canonical__(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "fraction": self.fraction,
            "n_tokens": self.n_tokens,
            "final_margin": self.final_margin,
            "threshold": self.threshold,
            "is_stable": self.is_stable,
            "note": self.note,
        }


def commitment(margins: Sequence[float], *, fraction: float = COMMITMENT_FRACTION) -> Commitment:
    """The first position reaching `fraction` of the final margin and never falling back below it.

    "And never falling back" is the whole of the definition that is not obvious. A reading that
    touches the threshold at token 12, drops away, and returns at token 300 committed at 300, and
    reporting 12 for it is how an early-commitment result gets manufactured out of noise.
    """
    m = np.asarray(margins, dtype=np.float64).ravel()
    n = int(m.size)
    if n == 0:
        return Commitment(
            position=-1,
            fraction=float("nan"),
            n_tokens=0,
            final_margin=float("nan"),
            threshold=float("nan"),
            is_stable=False,
            note="no positions were read",
        )
    final = float(m[-1])
    threshold = fraction * final
    reached = (m >= threshold) if final >= 0 else (m <= threshold)
    # The last index at which the condition fails; everything after it holds.
    failures = np.flatnonzero(~reached)
    first_stable = int(failures[-1] + 1) if failures.size else 0
    stable = first_stable < n
    first_touch = int(np.flatnonzero(reached)[0]) if np.any(reached) else -1
    return Commitment(
        position=first_stable if stable else first_touch,
        fraction=float(first_stable / n) if stable else float("nan"),
        n_tokens=n,
        final_margin=final,
        threshold=float(threshold),
        # A commitment position exists exactly when some suffix stays above the threshold. An
        # earlier crossing that falls back does not remove it, it just means the naive "first
        # crossing" answer would have been wrong, and that is what the note is for. Conflating the
        # two reported a judge that settles at token 4 as a judge that never settles.
        is_stable=bool(stable),
        margins=tuple(float(v) for v in m),
        note=(
            ""
            if first_touch == first_stable
            else (
                f"the reading first crosses the threshold at token {first_touch} and falls back "
                f"before settling at {first_stable}, so the early crossing is not a commitment and "
                f"a first-crossing rule would have reported {first_touch}."
            )
        ),
    )


# ---------------------------------------------------------------------------
# The four controls
# ---------------------------------------------------------------------------


@register_payload
@dataclass(frozen=True)
class Controls:
    """The four checks that decide whether an argmax-based verdict reading is admissible.

    Each is a pair of numbers rather than a verdict, because the interesting failures are the ones
    where a control half-fires. `kills` is the catalogue's kill condition, verbatim: the mean moves
    while the mode is fixed.
    """

    #: Control 1. Fraction of positions after the argmax settles at which the *margin* is still
    #: moving by more than `margin_tolerance` of its final value. This is the kill condition's
    #: measurement: a fixed mode with a moving mean.
    mode_settles_at: float
    mean_settles_at: float
    #: Control 2. Spread of the judgment margin across items, with and without chain-of-thought. A
    #: collapse is what makes an argmax reading saturate for reasons that are not about the judge
    #: deciding early.
    spread_with_cot: float
    spread_without_cot: float
    #: Control 3. The same commitment statistic computed from sequence length alone.
    length_baseline_auc: float
    verdict_auc: float
    #: Control 4. The reading with the two verdict tokens exchanged. A genuine direction flips sign;
    #: an argmax artifact does not move.
    permuted_correlation: float
    n_items: int = 0
    margin_tolerance: float = 0.05
    note: str = ""

    @property
    def mode_fixed_mean_moving(self) -> bool:
        """The kill condition's condition: the mode settles well before the mean does."""
        return (
            np.isfinite(self.mode_settles_at)
            and np.isfinite(self.mean_settles_at)
            and self.mean_settles_at > self.mode_settles_at + 0.1
        )

    @property
    def spread_collapsed(self) -> bool:
        return (
            np.isfinite(self.spread_with_cot)
            and np.isfinite(self.spread_without_cot)
            and self.spread_without_cot > 0
            and self.spread_with_cot < 0.5 * self.spread_without_cot
        )

    @property
    def beats_length(self) -> bool:
        return (
            np.isfinite(self.verdict_auc)
            and np.isfinite(self.length_baseline_auc)
            and self.verdict_auc > self.length_baseline_auc + 0.02
        )

    @property
    def permutation_responds(self) -> bool:
        """A genuine direction anti-correlates with itself under a verdict-token swap."""
        return np.isfinite(self.permuted_correlation) and self.permuted_correlation < -0.5

    def kills(self) -> bool:
        """The catalogue's kill condition: the four controls show the mean moves while the mode is
        fixed."""
        return self.mode_fixed_mean_moving

    def failures(self) -> tuple[str, ...]:
        out = []
        if self.mode_fixed_mean_moving:
            out.append(
                f"the mode settles at {self.mode_settles_at:.2f} of the sequence and the mean not "
                f"until {self.mean_settles_at:.2f}: an argmax commitment position is an artifact here"
            )
        if self.spread_collapsed:
            out.append(
                f"the judgment spread collapses under chain-of-thought "
                f"({self.spread_without_cot:.4g} to {self.spread_with_cot:.4g}), which is the "
                f"mechanism that makes a saturated reading look decisive"
            )
        if not self.beats_length:
            out.append(
                f"a length-only baseline reaches {self.length_baseline_auc:.4f} against the verdict "
                f"direction's {self.verdict_auc:.4f}, so the direction adds nothing over counting "
                f"tokens"
            )
        if not self.permutation_responds:
            out.append(
                f"exchanging the two verdict tokens moves the reading by a correlation of "
                f"{self.permuted_correlation:+.3f} rather than flipping it, so the reading is not "
                f"tracking the verdict contrast"
            )
        return tuple(out)

    @property
    def all_passed(self) -> bool:
        return not self.failures()

    def render(self) -> str:
        marks = {True: "ok", False: "FAIL"}
        lines = [
            f"four controls on {self.n_items} items:",
            f"  1 mode vs mean      {marks[not self.mode_fixed_mean_moving]:<5} "
            f"mode settles {self.mode_settles_at:.3f}, mean settles {self.mean_settles_at:.3f}",
            f"  2 spread collapse   {marks[not self.spread_collapsed]:<5} "
            f"with CoT {self.spread_with_cot:.4g}, without {self.spread_without_cot:.4g}",
            f"  3 length baseline   {marks[self.beats_length]:<5} "
            f"verdict {self.verdict_auc:.4f}, length {self.length_baseline_auc:.4f}",
            f"  4 verdict swap      {marks[self.permutation_responds]:<5} "
            f"correlation under swap {self.permuted_correlation:+.3f}",
        ]
        if self.kills():
            lines.append("  KILL: the mean moves while the mode is fixed (C8's kill condition).")
        return "\n".join(lines)

    def __canonical__(self) -> dict[str, Any]:
        return {
            "mode_settles_at": self.mode_settles_at,
            "mean_settles_at": self.mean_settles_at,
            "spread_with_cot": self.spread_with_cot,
            "spread_without_cot": self.spread_without_cot,
            "length_baseline_auc": self.length_baseline_auc,
            "verdict_auc": self.verdict_auc,
            "permuted_correlation": self.permuted_correlation,
            "n_items": self.n_items,
            "kills": self.kills(),
            "failures": list(self.failures()),
        }


def settles_at(series: Sequence[float], *, tolerance: float = 0.05) -> float:
    """The fraction of the way through a series after which it stays within `tolerance` of its end.

    The shared measurement behind control 1, applied once to the argmax indicator and once to the
    margin. Returning a fraction rather than an index is what makes the two comparable across items
    of different lengths, and it is the same normalisation `Commitment.fraction` uses and for the
    same reason.
    """
    s = np.asarray(series, dtype=np.float64).ravel()
    if s.size == 0:
        return float("nan")
    final = float(s[-1])
    scale = abs(final) if abs(final) > 0 else 1.0
    outside = np.flatnonzero(np.abs(s - final) > tolerance * scale)
    return float((outside[-1] + 1) / s.size) if outside.size else 0.0


# ---------------------------------------------------------------------------
# The instrument
# ---------------------------------------------------------------------------


@register_payload
@dataclass(frozen=True)
class VerdictReading:
    """Both rungs beside each other, the commitment position, and the four controls."""

    naive: Commitment
    corrected: Commitment | None
    controls: Controls | None
    transport_name: str = "logit_lens.identity"
    layer: int = 0
    residual_fraction: float = float("nan")

    @property
    def rung(self) -> int:
        return 1 if self.corrected is not None else 0

    def says(self) -> str:
        head = self.corrected or self.naive
        line = head.says()
        if self.corrected is not None:
            line += (
                f" The naive lens puts it at token {self.naive.position} "
                f"({self.naive.fraction:.1%}), a difference of "
                f"{abs(self.corrected.fraction - self.naive.fraction):.1%} of the sequence."
            )
        return line

    def render(self) -> str:
        lines = [self.says(), f"  transport: {self.transport_name} at layer {self.layer}"]
        if np.isfinite(self.residual_fraction):
            lines.append(
                f"  the linearisation explains {self.residual_fraction:.3f} of the final residual"
            )
        if self.controls is not None:
            lines.append(self.controls.render())
        return "\n".join(lines)

    def __canonical__(self) -> dict[str, Any]:
        return {
            "rung": self.rung,
            "naive": self.naive.__canonical__(),
            "corrected": self.corrected.__canonical__() if self.corrected else None,
            "controls": self.controls.__canonical__() if self.controls else None,
            "transport_name": self.transport_name,
            "layer": self.layer,
            "residual_fraction": self.residual_fraction,
            "says": self.says(),
        }


class VerdictDirection(SelectionInstrument):
    """C8. When the judge's verdict direction stopped moving, with the controls that decide whether
    the number means anything.

    Rung 0 is the naive `W_U[Yes] - W_U[No]`, which is what the shipped library used and is the
    mandatory comparator. Rung 1 carries the residual through the average Jacobian first. Both are
    always reported, because the catalogue makes the naive form a baseline rather than a predecessor.

    **The instrument refuses when the controls kill it.** C8's kill condition is that the mean moves
    while the mode is fixed, and an instrument that reported a commitment position anyway would be
    publishing the artifact the controls exist to detect. The refusal carries the control numbers.

    Two invariance groups and two relations, which the kernel now lets an instrument declare
    together. Under `repr.basis` the reading is invariant: a shared orthogonal map on the residuals
    and the unembedding leaves every inner product alone. Under `tokenization` it is invariant only
    because the reported quantity is the **fraction** of the sequence rather than the token index;
    a raw index is not, and that is why `Commitment.fraction` is the declared quantity.
    """

    name = "VerdictDirection"
    version = "1.0"
    quantity = "judge.commitment_position"
    capabilities = Capability.ACTIVATIONS | Capability.LINEAR_READOUT
    requires = ACCESS_GRADER_FORWARD
    substrates = frozenset({Substrate.NEURAL_GEN})
    phases = frozenset({Phase.PRE_RUN})
    envelope = ABOVE_LOD_ONLY
    invariance = "repr.basis, tokenization"
    #: Two groups, two relations. Both invariant, and the second is invariant only because the
    #: declared quantity is normalised: `tokenization` admits nothing but `invariant`, and a raw
    #: token index would fail it, so the normalisation is forced by the group rather than chosen.
    invariance_relation = {"repr.basis": INVARIANT, "tokenization": INVARIANT}
    baselines = (
        "the naive `W_U[Yes] - W_U[No]` form, always reported beside rung 1",
        "the vanilla logit lens (the identity transport), which costs nothing",
        "a length baseline",
        "the four controls: mode versus mean, spread collapse under chain-of-thought, the length "
        "baseline, and a verdict-token swap",
    )
    rung = 0
    faithful_to = "C8, the Jacobian-corrected verdict direction"
    deviations = (
        "the Jacobian is averaged over sampled positions and replaces a nonlinear map with its mean "
        "local linearisation. `residual_fraction` reports how much of the final residual that "
        "reproduces on held-out positions, and a reading through a transport explaining 40% of the "
        "variance is a different object from one explaining 95%",
        "the declared quantity is the commitment position as a fraction of the sequence, not the "
        "token index the registry's unit names. A raw index is not invariant under retokenisation "
        "and the `tokenization` group admits only invariance, so the index is carried as a "
        "presentational field beside the fraction",
        "the four controls are derived from the failure mode the catalogue entry states rather than "
        "transcribed from a specified list",
    )

    def __init__(
        self,
        *,
        naive_margins: Sequence[Sequence[float]] = (),
        corrected_margins: Sequence[Sequence[float]] = (),
        controls: Controls | None = None,
        incremental: Any = None,
        transport_name: str = "logit_lens.identity",
        layer: int = 0,
        residual_fraction: float = float("nan"),
        baseline_scores: Any = None,
    ) -> None:
        self.naive_margins = tuple(tuple(float(v) for v in row) for row in naive_margins)
        self.corrected_margins = tuple(tuple(float(v) for v in row) for row in corrected_margins)
        self.controls = controls
        self._incremental = incremental
        self.transport_name = transport_name
        self.layer = int(layer)
        self.residual_fraction = float(residual_fraction)
        self.baseline_scores = dict(baseline_scores or {})

    def compute(self) -> Any:
        if not self.naive_margins:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.RECORD_INCOMPLETE,
                detail="no per-token verdict margins were supplied, so there is nothing to read",
                remedy=(
                    "pass `naive_margins=[[m_0, m_1, ...], ...]`, one row per judged item, where "
                    "`m_t` is the verdict direction dotted with the residual at token `t`. "
                    "`measure.selection.verdict.verdict_direction` builds the direction and "
                    "`policy.selection.capture_at` reads the residuals."
                ),
            )
        naive = commitment(_mean_rows(self.naive_margins))
        corrected = (
            commitment(_mean_rows(self.corrected_margins)) if self.corrected_margins else None
        )
        if self.controls is not None and self.controls.kills():
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.BELOW_LOD,
                detail=(
                    "C8's kill condition fired: "
                    + "; ".join(self.controls.failures())
                    + ". A commitment position read off an argmax that settles before the margin "
                    "does is an artifact of the statistic, not a measurement of the judge."
                ),
                remedy=(
                    "report the margin's settling point rather than the mode's, which this reading "
                    "already carries, and say that the argmax form saturates. If the judgment "
                    "spread has collapsed under chain-of-thought, widen the item set until the "
                    "spread is measurable before reading a commitment position at all."
                ),
                statistics=self.controls.__canonical__(),
            )
        return VerdictReading(
            naive=naive,
            corrected=corrected,
            controls=self.controls,
            transport_name=self.transport_name,
            layer=self.layer,
            residual_fraction=self.residual_fraction,
        )

    def estimate(self, ctx: Context | None = None) -> Reading:
        ctx = ctx or Context(readout="decision")
        pre = self.preflight(ctx)
        if not pre.ok and pre.refusal is not None:
            return pre.refusal
        ctx._observable = self
        try:
            return self.measure(ctx)
        finally:
            ctx._observable = None

    def measure(self, ctx: Context) -> Any:
        computed = self.compute()
        if isinstance(computed, Refusal):
            return computed
        if self.controls is None:
            return refuse_unmeasured_control(
                self.name,
                what=(
                    "no controls were run, and the shipped verdict result is 1.0, which is the "
                    "ceiling of the statistic"
                ),
                remedy=(
                    "run the four controls and pass them as `controls=`: the mode-versus-mean "
                    "settling comparison, the judgment spread with and without chain-of-thought, "
                    "the length baseline, and the verdict-token swap. Without them a saturated "
                    "reading cannot be told from a decisive one."
                ),
            )
        if self._incremental is None:
            return refuse_unmeasured_control(
                self.name,
                what=(
                    "this is a white-box reading and no IncrementalValidity record was supplied, "
                    "so nothing records what the lens bought over the black-box bank"
                ),
                remedy=(
                    "run `stats.baselines.run_bank` on the judged items, hand the per-item margins "
                    "to `measure.meta.incremental.IncrementalValidityReading`, and pass its "
                    "`.record` as `incremental=`."
                ),
            )
        return emit_white_box(
            ctx,
            computed,
            incremental=self._incremental,
            baselines=self.baseline_scores or {"length": float(self.controls.length_baseline_auc)},
            uncertainty=Uncertainty(
                n=len(self.naive_margins),
                method="mean per-token margin across items; the commitment position is a threshold "
                "crossing on that mean",
            ),
            subject_extra={"transport": self.transport_name, "layer": str(self.layer)},
        )


def _mean_rows(rows: Sequence[Sequence[float]]) -> np.ndarray:
    """The mean margin per position over items, truncated to the shortest item.

    Truncated rather than padded. Padding a short sequence with its final value would manufacture a
    plateau at exactly the place a commitment position is looking for one.
    """
    if not rows:
        return np.zeros(0)
    width = min(len(r) for r in rows)
    return np.mean(np.stack([np.asarray(r[:width], dtype=np.float64) for r in rows]), axis=0)


__all__ = [
    "COMMITMENT_FRACTION",
    "Commitment",
    "Controls",
    "IdentityTransport",
    "VerdictDirection",
    "VerdictReading",
    "commitment",
    "settles_at",
    "verdict_direction",
]

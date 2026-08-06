"""W6.6, K2 rung 1: re-measure the transfer coefficient with the target itself as the calibrant.

The coefficient this row reports had never been measured until X3 measured it, and what X3 found is
not a number, it is a dependence. Calibrating an instrument bank on planted organisms and scoring it
on AISI's naturally-arising labelled hacks gives `t32 = 0.4732 [0.4543, 0.4933]` when the organism
was built by appending a hack to a working solution, and `0.0204` when it was built by substituting
the hack for the solution, which is what the policy in that run actually did. The spread across the
two designs is 0.4528, larger than either number is from zero, and both are legitimate renderings of
the same three techniques the environment declares. **A transfer coefficient quoted without its
organism design is not yet a measurement**, and that sentence, not the point estimate, is what K2
publishes. Every number in this paragraph is X3's, and each is bound to a row in the evidence store
X3 published with its run.

Standard addition is the fix the metrology already has for this
--------------------------------------------------------------

The design dependence is a matrix effect wearing different clothes. A planting design is a choice
about how the analyte presents *in the calibrant*, and it matters only because the calibrant is a
different system from the sample. Analytical chemistry stopped arguing about calibrant preparation
by removing the separate calibrant: spike the sample itself with known increments, fit the line
inside the sample's own matrix, and extrapolate to what was already there. There is no design to
choose because there is no clean organism to design.

Concretely: the target is the policy whose rollouts you hold. The increments are plants of known
strength installed **into that policy**, implementable as a LoRA at a known rank and scaling, or at
a cheaper rung as a rendered fragment injected into the target's own responses. The sensitivity that
comes out is the target's, so the coefficient computed against it is free of the multiplicative bias
that the clean-organism route carries and that no interval computed inside one arm can see.

What this predicts, and what it would mean if it fails
------------------------------------------------------

The prediction registered below is that standard addition collapses the design spread, not merely
that it lowers the coefficient. That is the sharper claim and it is the one the diagnosis actually
makes. If the spread survives, the gap is not a matrix effect, the diagnosis was wrong, and the
right conclusion is that organism-based calibration fails for a reason nobody has named yet. Both
outcomes are publishable and the second is more interesting.

What real subject this needs, and what it costs
-----------------------------------------------

It needs a **target policy you can fine-tune** together with a **labelled corpus of that same
policy's own behaviour**. The AISI series supplies the second half: 25,664 rollouts over 401 steps
from an Olmo 3.1 32B run against a coding grader, each carrying the publisher's `reward_hacked`
flag. It does not supply the first: the checkpoints are not published, so the LoRA increments have
to go into a policy of comparable class that you then sample from, and the corpus half becomes a
matched-class stand-in rather than the same model. That substitution is the largest weakness of the
plan and it is stated here rather than discovered later; a lab holding both halves for one model
runs this properly and cheaply.

Priced at 156 GPU-hours and $234 to $313 at mid-2026 neocloud rates, dominated by the two dozen LoRA
fits rather than by sampling. `quote()` carries the line items and the assumptions, and the numbers
in this sentence are what it returns. Nothing in this module runs any of it: it composes the shipped
instruments, freezes the study, and prints a runbook.

    python -m studies.w6_transfer.k2_standard_addition --runbook
    python -m studies.w6_transfer.k2_standard_addition --price
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from reward_lens.core.envelope import EnvelopeSpec
from reward_lens.core.evidence import Uncertainty, register_payload
from reward_lens.core.invariance import Relation
from reward_lens.core.reading import Reading, Refusal, RefusalReason
from reward_lens.core.reference import MatrixDescription
from reward_lens.core.types import Access, Capability, Component, GaugeStatus, Phase, Substrate
from reward_lens.measure.base import BaseObservable, Context
from reward_lens.organisms.chain import LadderRung, build_ladder
from reward_lens.organisms.standard_addition import (
    Addition,
    linearity_check,
    matrix_factor,
    standard_addition,
)
from reward_lens.organisms.transport import (
    SelectionDiagram,
    TransportVerdict,
    planted_to_real_diagram,
    transportable,
    untransportable_refusal,
)
from reward_lens.studies.freeze import FrozenStudy, freeze
from reward_lens.studies.spec import Hypothesis, KillCriterion, Prediction, StudySpec, SubjectQuery
from studies.w6_transfer.pricing import LineItem, Quote

# ---------------------------------------------------------------------------
# What X3 measured, carried as the comparator this row is registered against
# ---------------------------------------------------------------------------

#: X3's headline row. Every constant below is read off it rather than retyped from prose, and the
#: evidence id travels with them so a reader can check the source rather than the transcription.
X3_HEADLINE_EVIDENCE = "ev:e4cb1afba98e88f37e1c65764d8af425"

#: `t32` under external calibration, maximised over the two planting designs. The comparator.
T32_EXTERNAL_MAX = 0.47316351431543036

#: The same coefficient under the `substitute` design, which is what the policy actually did.
T32_EXTERNAL_SUBSTITUTE = 0.02041082148155071

#: The spread between the two designs. This, not either endpoint, is what the row exists to remove.
T32_DESIGN_SPREAD = 0.45275269283387964

#: The realised effective sample size of the AISI corpus: 25,664 rollouts over 1,096 problem
#: clusters, worth 919.5 independent items after the duplication in the response column.
#: `x3.corpus_census`, ev:43de2eb2b5b3fe0bb997a14d9ee626cd.
REALISED_N_CLUSTERS = 1096
REALISED_ESS = 919.5196568977841

#: The externally-calibrated arm's accuracy and the per-item correlation between arms, both
#: measured by X3's own power row (`x3.power`, ev:12da3b6d1ef7f55b0a1d109431bc06fb). They are the
#: planning marginals here, which is what a pilot is for.
PILOT_ACCURACY = 0.5998854863689911
PILOT_RHO = 0.36800812851864934

#: How far the coefficient's design spread must fall for the matrix-effect diagnosis to be
#: confirmed. 0.05 is the tolerance CAL-TRANSFER registered on this quantity and it is the only
#: pre-existing threshold on it, so it is the one worth testing rather than a fresh round number.
SPREAD_TOLERANCE = 0.05


# ---------------------------------------------------------------------------
# The reading
# ---------------------------------------------------------------------------


@register_payload
@dataclass(frozen=True)
class StandardAdditionTransferReading:
    """`t32` measured twice on one corpus, once per calibration route, and what separates them.

    ``design_spread_*`` are the fields the registered prediction is about. A coefficient that falls
    while its design spread survives has not been fixed, it has been moved, and reporting only the
    point estimate would hide that.
    """

    t32_external: float
    t32_standard_addition: float
    design_spread_external: float
    design_spread_standard_addition: float
    matrix_factor: float
    u_matrix_factor: float
    native_level: float
    u_native: float
    n_items: int
    n_instruments: int
    transport: str
    licence: tuple[str, ...] = ()
    per_instrument: Mapping[str, float] = None  # type: ignore[assignment]
    linearity_ok: bool = True
    linearity_note: str = ""
    note: str = ""

    @property
    def improvement(self) -> float:
        """`t32_standard_addition - t32_external`. Negative is the predicted direction."""
        return float(self.t32_standard_addition - self.t32_external)

    @property
    def spread_collapsed(self) -> bool:
        """Whether the design dependence is gone, which is the claim rather than the level."""
        return self.design_spread_standard_addition < SPREAD_TOLERANCE

    def render(self) -> str:
        lines = [
            f"t32 external {self.t32_external:.4f} (design spread "
            f"{self.design_spread_external:.4f})",
            f"t32 standard addition {self.t32_standard_addition:.4f} (design spread "
            f"{self.design_spread_standard_addition:.4f})",
            f"    moves it by {self.improvement:+.4f} on {self.n_items:,} items, "
            f"{self.n_instruments} instruments",
            f"    matrix factor {self.matrix_factor:.4g} +/- {self.u_matrix_factor:.4g}; "
            f"native level {self.native_level:.4g} +/- {self.u_native:.4g}",
            f"    transport: {self.transport}"
            + (f" on {{{', '.join(self.licence)}}}" if self.licence else ""),
        ]
        if not self.linearity_ok:
            lines.append(f"    LINEARITY: {self.linearity_note}")
        if self.note:
            lines.append(f"    {self.note}")
        return "\n".join(lines)

    def __canonical__(self) -> dict[str, Any]:
        return {
            "t32_external": self.t32_external,
            "t32_standard_addition": self.t32_standard_addition,
            "design_spread_external": self.design_spread_external,
            "design_spread_standard_addition": self.design_spread_standard_addition,
            "improvement": self.improvement,
            "matrix_factor": self.matrix_factor,
            "u_matrix_factor": self.u_matrix_factor,
            "native_level": self.native_level,
            "u_native": self.u_native,
            "n_items": self.n_items,
            "n_instruments": self.n_instruments,
            "transport": self.transport,
            "licence": list(self.licence),
            "per_instrument": dict(self.per_instrument or {}),
            "linearity_ok": self.linearity_ok,
            "linearity_note": self.linearity_note,
            "note": self.note,
        }


def transfer_coefficient(
    calibrated: Mapping[str, float], refit: Mapping[str, float]
) -> tuple[float, dict[str, float]]:
    """`max_i |AUC_i(calibrated elsewhere) - AUC_i(refit here)|`, and the per-instrument gaps.

    The maximum rather than the mean, matching what the campaign's CAL-TRANSFER card reported and
    what X3 reproduced, so the two numbers are comparable. The mean is the friendlier statistic and
    it is the wrong one for this question: the user of a calibration wants to know how badly the
    worst instrument in the bank is misled, not how the bank does on average.
    """
    gaps = {
        name: abs(float(calibrated[name]) - float(refit[name]))
        for name in sorted(set(calibrated) & set(refit))
    }
    return (max(gaps.values()) if gaps else float("nan")), gaps


#: K2's envelope, following the catalogue's own `envelope_requires: []`. The justification the
#: catalogue records is that the row's line names a `RefusalReason` rather than a `RegimeCondition`,
#: which is true and is not a reason for an instrument to be unconditional. The reason it actually
#: is: this reading describes a calibration that was performed, and both of its ways of being
#: quietly wrong are checked where they happen rather than in a regime of the training run. A
#: non-linear response over the extrapolation range is caught by `linearity_check` and reported on
#: the reading; a reference material with no certificate is caught by L1 and caps the trust. No
#: property of the run being measured can make an extrapolation through measured points wrong.
TRANSFER_ENVELOPE = EnvelopeSpec(
    unconditional=True,
    justification=(
        "the reading describes a calibration performed inside the target's own matrix, so no "
        "regime of the run under study can invalidate it. Its two quiet failure modes are checked "
        "at their source: curvature over the extrapolation range is reported by `linearity_check` "
        "on the reading, and an uncertified reference caps the trust through L1."
    ),
)


class StandardAdditionTransfer(BaseObservable):
    """K2 at rung 1: the transfer coefficient re-measured with the target as its own calibrant.

    **This instrument does not run the experiment.** It consumes the two calibration sweeps and the
    three scored arms and reports the coefficient, so the arithmetic is checkable on a planted
    subject before any GPU is bought. Producing the arms is the runbook's job.

    Three refusals it can return, all of them reached before a number is formed. Instruments present
    in one arm and not another refuse, because a maximum over a different instrument set is a
    different quantity. A standard-addition fit that does not identify the sensitivity propagates
    its own refusal. And a selection diagram that licenses no transport refuses with the rung-1
    number carried as `partial`, because the number is real and reading it as the target's
    coefficient is what the diagram says is not allowed.
    """

    name = "StandardAdditionTransfer"
    version = "1.0"
    quantity = "calibration.transfer_t32"
    capabilities = Capability.SCORES
    gauge_status = GaugeStatus.INVARIANT
    #: `MUTATE` on `GOLD` because you have to be able to plant, and `MUTATE` on `POLICY` because
    #: standard addition doses the target rather than a clean organism, which is the whole of the
    #: difference between this rung and rung 0.
    requires = {
        Component.GOLD: Access.MUTATE,
        Component.POLICY: Access.MUTATE,
        Component.RECORD: Access.RECORD,
    }
    substrates = frozenset({Substrate.NEURAL_GEN, Substrate.NEURAL_SCALAR, Substrate.COMPOSITE})
    phases = frozenset({Phase.POST_RUN, Phase.DEPLOYED})
    envelope = TRANSFER_ENVELOPE
    #: Two groups, both true and both checkable. A `ΔAUC` between two rank statistics is unchanged
    #: by an orthogonal change of representation basis and unchanged by an affine rescaling of the
    #: reward, because neither reorders any pair of items. The mapping form is what lets both be
    #: declared and therefore both be generated as tests.
    invariance = "repr.basis, reward.affine"
    invariance_relation = {
        "repr.basis": Relation("invariant"),
        "reward.affine": Relation("invariant"),
    }
    baselines = (
        "the organism-only number, which is what everyone reports and is this row's rung 0",
        "stats.baselines.ALL_SIX scored on both arms, so a coefficient cannot be reported for an "
        "instrument that a zero-parameter string match already beats",
    )
    rung = 1
    faithful_to = "standard addition, the analytical-chemistry method for calibrating in a matrix"
    deviations = (
        "the increments are plants of a declared technique at a nominal strength, and a nominal "
        "strength is not a certified one. The addition axis therefore carries L1's characterisation "
        "uncertainty, which is why the reading composes a reference certificate rather than "
        "treating the added dose as exact.",
        "the extrapolation assumes the instrument's response is linear in the added dose between "
        "the smallest addition and the intercept, and no measured point lies in that interval. "
        "`linearity_check` tests the part of the range that was measured and cannot test the part "
        "that carries the answer.",
        "the matrix factor propagates the two slopes' errors in quadrature, which assumes the two "
        "calibrations were fitted on disjoint measurements. Sharing items between them correlates "
        "the slopes and understates the factor's uncertainty.",
    )

    def __init__(
        self,
        *,
        target_additions: Sequence[Addition] = (),
        clean_additions: Sequence[Addition] = (),
        arm_external: Mapping[str, float] | None = None,
        arm_standard_addition: Mapping[str, float] | None = None,
        arm_refit: Mapping[str, float] | None = None,
        design_spread_external: float = T32_DESIGN_SPREAD,
        design_spread_standard_addition: float = float("nan"),
        n_items: int = REALISED_N_CLUSTERS,
        diagram: SelectionDiagram | None = None,
        note: str = "",
    ) -> None:
        self.target_additions = tuple(target_additions)
        self.clean_additions = tuple(clean_additions)
        self.arm_external = dict(arm_external or {})
        self.arm_standard_addition = dict(arm_standard_addition or {})
        self.arm_refit = dict(arm_refit or {})
        self.design_spread_external = float(design_spread_external)
        self.design_spread_standard_addition = float(design_spread_standard_addition)
        self.n_items = int(n_items)
        self.diagram = diagram if diagram is not None else planted_to_real_diagram()
        self.note = note

    def _arms_agree(self) -> Refusal | None:
        names = [set(self.arm_external), set(self.arm_standard_addition), set(self.arm_refit)]
        if not all(names):
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.RECORD_INCOMPLETE,
                detail=(
                    f"the three arms carry {[len(s) for s in names]} instruments. All three are "
                    f"needed: the externally-calibrated arm, the standard-addition arm, and the "
                    f"arm refit on the real corpus that both are measured against."
                ),
                remedy=(
                    "score the same instrument bank three ways on the same items: calibrated on "
                    "the clean organism, calibrated by standard addition into the target, and "
                    "refit directly on the labelled corpus. `stats.baselines.run_bank` produces "
                    "all three from one `DetectionTask` per calibration."
                ),
                statistics={
                    "n_external": len(names[0]),
                    "n_sa": len(names[1]),
                    "n_refit": len(names[2]),
                },
            )
        shared = names[0] & names[1] & names[2]
        missing = (names[0] | names[1] | names[2]) - shared
        if missing:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.RECORD_INCOMPLETE,
                detail=(
                    f"{len(missing)} instrument(s) appear in some arms and not others: "
                    f"{', '.join(sorted(missing))}. A maximum over a different instrument set in "
                    f"each arm is not one quantity measured twice."
                ),
                remedy=(
                    "restrict every arm to the instruments all three could score, and report the "
                    "dropped ones by name with the reason each could not be scored. An instrument "
                    "that refuses on one arm is a result about that arm."
                ),
                statistics={"n_shared": len(shared), "n_missing": len(missing)},
            )
        return None

    def _build(self) -> "tuple[StandardAdditionTransferReading, TransportVerdict] | Refusal":
        """The reading and the transport verdict, before the gate is applied to them.

        Separate from `compute` because the gate's refusal wants a recorded `Evidence` as its
        `partial` and there is no `Context` here to make one. `compute` applies the gate without a
        partial and `measure` applies it with one.
        """
        bad = self._arms_agree()
        if bad is not None:
            return bad

        target_fit = standard_addition(self.target_additions, dose_unit="rho")
        if isinstance(target_fit, Refusal):
            return target_fit
        clean_fit = standard_addition(self.clean_additions, dose_unit="rho")
        if isinstance(clean_fit, Refusal):
            return clean_fit
        factor = matrix_factor(target_fit, clean_fit)
        if isinstance(factor, Refusal):
            return factor

        t32_sa, per_instrument = transfer_coefficient(self.arm_standard_addition, self.arm_refit)
        t32_ext, _ = transfer_coefficient(self.arm_external, self.arm_refit)
        linear_ok, linear_note = linearity_check(target_fit)

        verdict = transportable(self.diagram, outcome="score", treatment="hack")
        if isinstance(verdict, Refusal):
            return verdict

        reading = StandardAdditionTransferReading(
            t32_external=float(t32_ext),
            t32_standard_addition=float(t32_sa),
            design_spread_external=self.design_spread_external,
            design_spread_standard_addition=self.design_spread_standard_addition,
            matrix_factor=factor.factor,
            u_matrix_factor=factor.u_factor,
            native_level=target_fit.native_level,
            u_native=target_fit.u_native,
            n_items=self.n_items,
            n_instruments=len(per_instrument),
            transport=verdict.verdict,
            licence=verdict.licence,
            per_instrument=per_instrument,
            linearity_ok=linear_ok,
            linearity_note=linear_note,
            note=self.note,
        )
        return reading, verdict

    def compute(self) -> StandardAdditionTransferReading | Refusal:
        """The whole reading, with no `Context`, so the arithmetic is testable on its own.

        A blocked transport comes back as a refusal whose `statistics` carry the coefficients that
        were computed. They are real numbers and reading them as the target's is what the diagram
        says is not licensed, so they travel where a reader has to look for them rather than where a
        reader would mistake them for the reading.
        """
        got = self._build()
        if isinstance(got, Refusal):
            return got
        reading, verdict = got
        if not verdict.may_cross:
            return self._transport_refusal(verdict, reading)
        return reading

    def _transport_refusal(
        self,
        verdict: TransportVerdict,
        reading: StandardAdditionTransferReading,
        partial: Any = None,
    ) -> Refusal:
        """The rung-2 gate: a number that was computed and may not be read as the target's.

        ``partial`` is an `Evidence` or nothing, never the payload. `Refusal.partial` is typed
        `Evidence | None` and `Refusal.render` reads `.value` off it, so handing it the bare reading
        makes the refusal raise the moment anybody prints it. There is no `Context` inside
        `compute`, so the numbers travel in `statistics` there and the recorded partial is attached
        on the `measure` path, which is the same split `measure/labels/reference.py` uses for L1's
        uncertified-reference bound.
        """
        base = untransportable_refusal(self.name, verdict, quantity=self.quantity)
        return Refusal(
            instrument=base.instrument,
            reason=base.reason,
            detail=base.detail,
            remedy=base.remedy,
            partial=partial,
            statistics={
                **base.statistics,
                "t32_standard_addition": reading.t32_standard_addition,
                "t32_external": reading.t32_external,
                "design_spread_standard_addition": reading.design_spread_standard_addition,
                "matrix_factor": reading.matrix_factor,
            },
        )

    def ladder(self) -> Any:
        """Both rungs as `organisms.chain.TransferLadder`, so M11 can publish the disagreement.

        A transport refusal stops the ladder rather than being routed around. The two rungs are real
        numbers either way, but composing them into a transfer term and a chain is the act the
        diagram says is not licensed, and returning the refusal is what keeps the gate a gate.
        """
        got = self.compute()
        if isinstance(got, Refusal):
            return got
        reading = got
        return build_ladder(
            [
                LadderRung(
                    rung=0,
                    value=reading.t32_external,
                    n=reading.n_items,
                    quantity=self.quantity,
                    estimator="external calibration on a clean planted organism",
                    access="ORGANISM:MUTATE + a labelled real corpus",
                    cost="CPU only; this is what X3 ran",
                ),
                LadderRung(
                    rung=1,
                    value=reading.t32_standard_addition,
                    n=reading.n_items,
                    quantity=self.quantity,
                    estimator="standard addition into the target",
                    access="POLICY:MUTATE + a labelled real corpus",
                    cost="one LoRA plant per addition level per seed, plus generation",
                ),
            ],
            instrument=self.name,
            note="rung 2 is the selection diagram, which licenses rather than estimates",
        )

    def measure(self, ctx: Context) -> Any:
        got = self._build()
        if isinstance(got, Refusal):
            return got
        reading, verdict = got
        evidence = ctx.emit(
            reading,
            uncertainty=Uncertainty(
                n=reading.n_items,
                n_effective=REALISED_ESS,
                method=(
                    "extrapolation uncertainty on the addition line, inverse-prediction form at "
                    "the x-intercept; intervals on the coefficient itself come from the cluster "
                    "bootstrap over problem ids, not from this fit"
                ),
            ),
            subject_extra={
                "route": "standard addition into the target",
                "transport": reading.transport,
                "matrix": MatrixDescription(
                    system="the target policy itself", scale="dosed by LoRA at known strength"
                ).render(),
            },
        )
        # A blocked transport still records what was measured, as the refusal's `partial`. The
        # number exists; what the diagram withholds is the licence to read it as the target's.
        if not verdict.may_cross:
            return self._transport_refusal(verdict, reading, partial=evidence)
        return evidence

    def estimate(self, ctx: Context | None = None) -> Reading:
        return super().estimate(ctx or Context(readout="score"))


# ---------------------------------------------------------------------------
# The registered study
# ---------------------------------------------------------------------------

#: What is already known at freeze time, said plainly, because a preregistration that overstates its
#: own blindness is worse than one that claims none. X3 has been run. Its coefficient, its design
#: spread and its corpus census are all published and are the comparators below. What has not been
#: computed is anything under standard addition: no plant has been made into a target, no addition
#: line has been fitted, and no arm of this comparison exists. The predictions are about those.
DISCLOSURE = (
    "informed about the comparator, blind about the outcome. X3 ran before this spec was frozen "
    "and its numbers are the registered comparators: t32 = 0.4732 under `append`, 0.0204 under "
    "`substitute`, a design spread of 0.4528, each bound to a row in the evidence store X3 "
    "published with its run. No standard-addition arm has been built, fitted or scored. "
    "This row is new and it does not replace P6. P6 predicts that standard addition drives t32 "
    "below 0.419 and it is not well posed as registered, for two reasons: "
    "0.419 is a simulation-to-real-model coefficient rather than the organism-only one the row "
    "names, and t32 depends on the organism design across a spread wider than the threshold. P6 "
    "stays as written, because rewriting a prediction after seeing the answer is the failure the "
    "freeze exists to prevent. This spec is registered beside it and names its designs."
)

STUDY = StudySpec(
    id="k2-standard-addition-transfer",
    title="Does dosing the target remove the transfer coefficient's dependence on organism design?",
    science="S12-metrology",
    hypotheses=(
        Hypothesis(
            id="H-spread-collapses",
            statement=(
                "Calibrating by standard addition into the target collapses the spread of `t32` "
                "across planting designs, from the 0.4528 X3 measured under external calibration "
                "to below the 0.05 tolerance CAL-TRANSFER registered."
            ),
            prediction=Prediction(
                metric="design_spread_standard_addition",
                comparator="<",
                threshold=SPREAD_TOLERANCE,
                effect=T32_DESIGN_SPREAD,
                ci_excludes=SPREAD_TOLERANCE,
                rationale=(
                    "the design is a property of the clean calibrant and standard addition has no "
                    "clean calibrant, so if the design dependence is a matrix effect it has "
                    "nowhere left to enter. This is the sharp form of the claim: a coefficient "
                    "that falls while its design spread survives has been moved, not fixed."
                ),
            ),
            scoreboard_row="K2",
        ),
        Hypothesis(
            id="H-below-external",
            statement=(
                "`t32` under standard addition is below the external-calibration maximum over "
                "designs, 0.4732, with its interval excluding it."
            ),
            prediction=Prediction(
                metric="t32_standard_addition",
                comparator="<",
                threshold=T32_EXTERNAL_MAX,
                effect=T32_EXTERNAL_MAX - T32_EXTERNAL_SUBSTITUTE,
                ci_excludes=T32_EXTERNAL_MAX,
                rationale=(
                    "the comparator is the worse of the two designs a competent person might have "
                    "chosen, which is the conservative reading and the one a user of an organism "
                    "calibration is exposed to. Beating the better design, 0.0204, is not "
                    "predicted: at that level the coefficient is already inside its own noise."
                ),
            ),
            scoreboard_row="K2",
        ),
    ),
    analysis="studies.w6_transfer.k2_standard_addition.StandardAdditionTransfer.compute",
    subjects=SubjectQuery(
        datasets=(
            "ai-safety-institute/reward-hacking-olmo3.1-32b-kl0.0-seed2-rollouts",
            "k2.standard-addition.target-dosed",
        ),
        organisms=("declared-techniques-uniform", "target-dosed-lora"),
        extra={
            "addition_levels": [0.0, 0.25, 0.5, 0.75],
            "seeds": [0, 1, 2],
            "designs": ["append", "substitute"],
            "instruments": "stats.baselines.ALL_SIX",
            "n_items": REALISED_N_CLUSTERS,
            "ess": REALISED_ESS,
        },
    ),
    kill_criteria=(
        KillCriterion(
            id="K-fix-works",
            metric="t32_standard_addition",
            comparator="<",
            threshold=SPREAD_TOLERANCE,
            description=(
                "standard addition drives the coefficient to within the registered tolerance of "
                "zero. The matrix-effect diagnosis was right, the fix works, and K2 is answered: "
                "organism-based calibration is usable provided the organism is the target. This is "
                "the catalogue's own kill and it is the good outcome."
            ),
        ),
        KillCriterion(
            id="K-diagnosis-wrong",
            metric="design_spread_standard_addition",
            comparator=">=",
            threshold=T32_DESIGN_SPREAD - SPREAD_TOLERANCE,
            description=(
                "the design spread survives dosing the target. Then the gap is not a matrix "
                "effect, the diagnosis this whole rung was built on is wrong, and the finding is "
                "that organism-based calibration fails for a reason nobody has named. That is the "
                "more interesting outcome and it is why this criterion is registered rather than "
                "left as a footnote."
            ),
        ),
    ),
    version=1,
    notes=DISCLOSURE,
)


def power_plan(replicates: int = 4000, seed: int = 0) -> Any:
    """M10's plan for the primary comparison, at the realised n rather than at a hoped-for one.

    The design is the per-item paired comparison of the two calibration routes on the AISI corpus.
    Both marginals are measured rather than assumed: 0.5999 is what the externally-calibrated bank
    achieved and 0.8997 is what the same bank achieved refit on the corpus, from X3's own power row
    (`x3.power`, ev:12da3b6d1ef7f55b0a1d109431bc06fb), and 0.368 is the measured per-item
    correlation between the two arms. The n is the effective sample size after the corpus's 63%
    duplicate fraction, not the nominal 25,664, because planning on rows the corpus does not contain
    is how an underpowered study gets registered as an adequate one.
    """
    from reward_lens.stats.power import PairedBinaryDesign, plan

    design = PairedBinaryDesign(
        n=int(REALISED_ESS),
        accuracy_a=PILOT_ACCURACY,
        accuracy_b=0.8996703780534716,
        rho=PILOT_RHO,
    )
    return plan(design, replicates=replicates, seed=seed, ess=REALISED_ESS)


def freeze_study(repo_dir: str | None = None) -> FrozenStudy:
    """Hash the spec and stamp the commit. The StudyID is what makes a later reading REGISTERED."""
    return freeze(STUDY, repo_dir=repo_dir)


def resolvable_rows(replicates: int = 400, seed: int = 0) -> int:
    """How many preregistered rows the realised n settles, from the power plan rather than by hand.

    All four rows here are decided from the same paired comparison, so they stand or fall together
    and the count is the whole spec when the design resolves and zero when it does not. Written as
    a computation rather than a constant because the constant would survive a change to the design
    that invalidated it.
    """
    got = power_plan(replicates=replicates, seed=seed)
    if not got.resolution.resolved:
        return 0
    return len(STUDY.hypotheses) + len(STUDY.kill_criteria)


# ---------------------------------------------------------------------------
# The price
# ---------------------------------------------------------------------------


def quote(resolvable: int | None = None) -> Quote:
    """What K2 rung 1 costs, at the matched model class rather than at a convenient one.

    The plants go into a 32B policy because the corpus that carries the labels came from a 32B
    policy, and calibrating in a 7B matrix to correct a 32B matrix effect would be the same mistake
    one size down. That choice roughly quadruples the price against an 8B stand-in and it is the
    only version of this row that answers the question it asks.
    """
    items = (
        LineItem(
            what="LoRA plants, target and clean organism",
            gpu_hours=24 * 4.0,
            why=(
                "4 addition levels by 3 seeds, twice: once into the target for the standard-"
                "addition arm and once into a clean organism for the rung-0 comparator. Each is "
                "rank-16 LoRA over about 4,000 pairs, one hour on 4 H100s for a 32B policy at "
                "bf16 with gradient checkpointing. One hour is a round over-estimate; the fits are "
                "minutes of compute and the hour absorbs load and spin-up."
            ),
        ),
        LineItem(
            what="calibration-sweep generation",
            gpu_hours=30 * 0.44,
            why=(
                "2,000 completions per arm at about 600 tokens, 30 arms (24 dosed plus 6 "
                "undosed), through vLLM at roughly 3,000 tokens per second aggregate on 4 H100s. "
                "2,000 is what a stable mean response per addition level needs; the coefficient "
                "itself is scored on the corpus that already exists."
            ),
        ),
        LineItem(
            what="unspiked anchor at corpus scale",
            gpu_hours=3 * 1.94,
            why=(
                "the unspiked response has to be measured with the same sampling parameters and "
                "at comparable scale to the corpus, or the intercept is compared against a "
                "differently-sampled quantity. 8,768 completions per seed, three seeds."
            ),
        ),
        LineItem(
            what="plant verification sweep",
            gpu_hours=24 * 0.4,
            why=(
                "every plant is checked on held-out probes to confirm it installed the behaviour "
                "at the intended strength. Without this the addition axis is nominal, and a "
                "nominal dose is exactly what L1 says a reference material may not have."
            ),
        ),
    )
    return Quote(
        row="W6.6 / K2 rung 1, standard addition into the target",
        items=items,
        assumptions=(
            "the target is a 32B-class open-weights policy that can be LoRA fine-tuned. AISI "
            "publishes the rollouts and their labels but not the Olmo 3.1 32B checkpoints, so the "
            "plants go into a matched-class stand-in and the corpus half is that stand-in's "
            "nearest available proxy. This is the plan's largest weakness and a lab holding both "
            "halves for one model avoids it entirely.",
            "H100 at the mid-2026 neocloud floor band of $1.50 to $2.01 per GPU-hour. On Modal at "
            "$3.95 the whole row roughly doubles and is still under a thousand dollars.",
            "the instrument bank is the six black-box baselines. A white-box probe needs "
            "activations on both arms and roughly triples the generation line; X3 refused that "
            "arm for the same reason and the refusal is on its page.",
            "compute only. Storage is a few hundred gigabytes of completions and is not itemised.",
        ),
        resolvable=resolvable if resolvable is not None else 0,
        registered_rows=len(STUDY.hypotheses) + len(STUDY.kill_criteria),
        subject_needed=(
            "a 32B-class policy you can fine-tune, plus a labelled corpus of that policy's own "
            "behaviour. The AISI series supplies the labelled half for a model whose weights are "
            "not published."
        ),
        note=(
            "cheap because it is a dozen LoRA fits and some sampling, not a training run. The "
            "expensive version of this question is the one nobody should run: re-doing the whole "
            "calibration study in every candidate matrix."
        ),
    )


# ---------------------------------------------------------------------------
# The runbook
# ---------------------------------------------------------------------------


def runbook() -> str:
    """What the maintainer types, in order, and what a failed arm looks like."""
    q = quote()
    lo, hi = q.dollars
    return f"""W6.6 / K2 rung 1 -- standard addition into the target

Price: {q.gpu_hours:,.0f} GPU-hours, ${lo:,.0f} to ${hi:,.0f}. Nothing below has been run.

What to fetch
  1. The AISI rollout table, 189 MB of parquet:
       ai-safety-institute/reward-hacking-olmo3.1-32b-kl0.0-seed2-rollouts
     X3 already knows how to fetch and cache it; point REWARD_LENS_AISI_ROLLOUTS at the file.
  2. A 32B-class open-weights policy you can fine-tune. Olmo 3.1 32B is the matched class and its
     RL checkpoints are not published, so this is a stand-in and the substitution goes on the page.
  3. Nothing else. The instrument bank is stats/baselines and runs on CPU.

The arms, and what each is for
  A  undosed target, 3 seeds       the unspiked response. The only measurement that can check the
                                   extrapolation, which is why it is not optional.
  B  target + plant at rho in {{0.25, 0.5, 0.75}}, 3 seeds each
                                   the addition line, fitted inside the target's own matrix.
  C  clean organism at the same doses and seeds
                                   the rung-0 comparator, which is what everyone reports.
  D  the labelled corpus, unmodified
                                   where every calibrated instrument is scored.

Run, in this order
  1. Freeze first. `studies.w6_transfer.k2_standard_addition.freeze_study()` on a clean tree; the
     StudyID it returns is what makes every later reading REGISTERED. A dirty tree records a sha
     nobody can check out, so commit before freezing rather than after.
  2. Plants, arm C before arm B. The clean organism is the cheaper failure: if the plants do not
     install the behaviour there, they will not install it in the target either and you have found
     that out for a quarter of the money.
  3. Verify every plant on held-out probes before generating from it. A plant whose realised
     strength does not track its nominal one is a plant that must be dropped from the line, and
     dropping it after the line is fitted is a decision made with the answer in view.
  4. Generate arms A, B and C. Same sampling parameters throughout, recorded.
  5. Score all three arms with `stats.baselines.run_bank`, plus the refit arm on D.
  6. `StandardAdditionTransfer(...).compute()`, then `.ladder()` for both rungs, then
     `measure.meta.rungs.compare_rungs` on `as_rung_readings` for the M11 transfer term.

What a failed arm looks like
  * `organisms.standard_addition` refuses BELOW_LOD: the response did not move with the spike. The
    plants did not take, or the instrument does not respond to this analyte in this matrix. Check
    the verification sweep from step 3 before touching the doses.
  * It refuses on a negative slope: the instrument is anti-correlated with the planted behaviour in
    the target. That is a finding about the instrument. Do not fix it by flipping the sign after
    seeing it; `stats.baselines.base.oriented_score` fixes orientation on the calibration arm,
    which is where the decision belongs.
  * `linearity_check` returns False: the addition line curves over the measured range, so the
    extrapolation through the unmeasured range is biased and the direction depends on the curvature.
    Report the native level as a bound, not as an estimate.
  * The transport verdict is `not_transportable`: the coefficient computed is real and may not be
    read as the target's. The refusal carries the number as `partial`. The remedy is to measure one
    of the blocking variables in the target, not to collect more organism data.
  * Every plant verifies, the line fits, and the coefficient does not move. That is
    `K-diagnosis-wrong` and it is a result. Publish it; do not add doses until it does move.

What to publish either way
  Both rungs, both designs, the matrix factor with its uncertainty, and the transport verdict. The
  sentence this row exists for is that a transfer coefficient quoted without its organism design is
  not yet a measurement, and it is true whichever way the numbers land.
"""


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runbook", action="store_true", help="print the runbook and exit")
    parser.add_argument("--price", action="store_true", help="print the quote and exit")
    parser.add_argument("--power", action="store_true", help="print the power plan and exit")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.price:
        print(quote(resolvable=resolvable_rows()).render())
    elif args.power:
        print(power_plan().render())
    else:
        print(runbook())
    return 0


__all__ = [
    "DISCLOSURE",
    "PILOT_ACCURACY",
    "PILOT_RHO",
    "REALISED_ESS",
    "REALISED_N_CLUSTERS",
    "SPREAD_TOLERANCE",
    "STUDY",
    "T32_DESIGN_SPREAD",
    "T32_EXTERNAL_MAX",
    "T32_EXTERNAL_SUBSTITUTE",
    "X3_HEADLINE_EVIDENCE",
    "StandardAdditionTransfer",
    "StandardAdditionTransferReading",
    "TRANSFER_ENVELOPE",
    "freeze_study",
    "main",
    "power_plan",
    "quote",
    "resolvable_rows",
    "runbook",
    "transfer_coefficient",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

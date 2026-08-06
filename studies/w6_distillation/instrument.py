"""K1, the distillation gap. Kill risk number one for the whole in-run layer of this library.

**What this instrument measures.** Take the behavioural features the in-run layer already measures
off a record, evaluate them on a pre-RL reference, on the post-RL expert, and on the model that
actually ships after the on-policy distillation step, and report what fraction of the shift RL
installed is still present in the shipped artifact. Per feature, and with the reward-hacking-relevant
features called out separately, because the claim worth testing is not that the student is a bit
different: it is that the capabilities and the hacking propensity might survive at **different**
rates.

**What this instrument cannot do.** It reads rollouts, so it sees behaviour and not mechanism: a
distillation step that preserves every measured behaviour by a different circuit reads as full
survival here, and only the token-level arm (A4 in the runbook, `POLICY: BACKWARD` on both
artifacts) could tell those apart. It is a statement about the mean shift per prompt, so a step that
preserves the mean and destroys the variance also reads as full survival. And the fraction is
conditional on the feature basis: a basis that omits the axis distillation moved reports high
survival, which is why `feature_names` is on every reading.

**Why this is kill risk number one, and it is the reason the package exists.** Multi-teacher
on-policy distillation is the default final step at seven or more labs and not one publishes a
behavioural audit of it. Kimi K3 (arXiv:2607.24653 §4.1) consolidates nine RL experts into the
shipped model with MOPD; DeepSeek-V4 (arXiv:2606.19348) reports that "the mixed RL stage was
entirely replaced by On-Policy Distillation" with more than ten teachers; GLM-5 (arXiv:2602.15763)
ships On-Policy Cross-Stage Distillation at group size 1. Two independent searches for a behavioural
comparison between an RL expert and its distilled student returned zero. **The shipped model is
never the RL policy.** If group-relative geometry does not survive that step, everything this
library measures during a run is measured on a model nobody deploys, and the whole in-run layer is
for the lab's internal use rather than for the artifact anyone ships.

**The one existing audit is the design brief.** arXiv:2607.07050 found a distillation shift
"invisible from aggregate losses alone", not explained by token exposure or by full-sequence
divergence, localised to "behavior leverage imbalance: local token-level signals at mode-entry and
structural positions", and correctable to cut tool over-calling from 13.7% to 9.0% at fixed decision
accuracy. Its closing line is the specification for this instrument: "multi-teacher OPD should
monitor *where* teacher signals act, not only *how large* they are in aggregate." They used two
teachers. K3 uses nine.

**The subject this claim needs, and what that subject costs.** Three checkpoints sharing one base:
the pre-RL reference, one expert produced by group-relative RL against a real grader, and one
student produced by on-policy distillation from that expert back into the same base. No public
release is that triple. The distilled models on the Hub are distilled from a *different* model's
outputs into a *different* base, so the denominator this instrument divides by, the shift RL
installed **on this base**, does not exist for them, and measuring them would answer a question
about cross-family transfer instead. `studies.w6_distillation.price` computes the bill from the
rate survey's reference unit of 384 GPU-hours for one 8B group-relative RL run: the three-seed
version of this study is **1,323.1 GPU-hours and $1,985 to $2,659** at the sourced H100 neocloud
band, against $17.73 and 4.465 GPU-hours for the entire published campaign. Of that, 1,316.6
GPU-hours is producing the checkpoints and 6.5 is everything this instrument does, which is the
number that decides the question: what costs money is the pair of models, not the measurement.
`studies.w6_distillation.runbook` prints the commands and `RUNBOOK.md` is the rendered copy.
"""

from __future__ import annotations

from typing import Any, Sequence

from reward_lens.core.envelope import EnvelopeSpec, RegimeCondition
from reward_lens.core.evidence import Uncertainty
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.types import (
    Access,
    AccessMatrix,
    Capability,
    Component,
    GaugeStatus,
    Phase,
    Substrate,
)
from reward_lens.measure.base import BaseObservable, Context
from reward_lens.measure.ledger.features import TrajectoryFeaturiser
from reward_lens.measure.rate.regime import MEASURED_BY
from reward_lens.stats.baselines import ALL_SIX
from studies.w6_distillation.reading import DistillationSurvival, distillation_survival
from studies.w6_distillation.survival import Arm

#: `ABOVE_LOD` is the one of the thirteen registered conditions that this estimator depends on and
#: that a record can measure: every number K1 reports is a ratio whose denominator is the shift RL
#: installed, and a denominator below the substrate's disagreement with itself makes the ratio noise
#: over noise. The blank arm is what measures it, and the reading carries the verdict per feature.
DISTILLATION_ENVELOPE = EnvelopeSpec(
    requires=frozenset({RegimeCondition.ABOVE_LOD}),
    measured_by={RegimeCondition.ABOVE_LOD: MEASURED_BY[RegimeCondition.ABOVE_LOD]},
    on_violation="refuse",
)


class DistillationGap(BaseObservable):
    """K1. What fraction of the RL-installed behaviour is present in the model that ships.

    The reading is `artifact.distillation_delta`, in percentage points: how much of the installed
    shift did **not** survive. Zero is perfect survival, 100 is nothing surviving, and a negative
    value means the distillation step amplified what RL installed, which is a real outcome and is
    the reason nothing here is clamped into [0, 1].

    Scope limit, three lines in as the house style asks. This is a black-box behavioural comparison
    over records: it needs no activations and no gradients, and it therefore says nothing about
    whether a surviving behaviour survived by the same mechanism. It is also a statement about
    means: a step that halves the variance of a behaviour while leaving its mean where it was reads
    here as full survival.

    Two preconditions are checked in `measure` and returned as refusals rather than declared in the
    envelope, and the gap is named here rather than hidden. The three arms must answer the **same
    prompts** and must be sampled under the **same decoding policy**, and the thirteen registered
    `RegimeCondition` members carry no term for a matched sampling design. That is the same shape as
    the crossed-design qualifier before `DESIGN_CROSSED` was added, and a fourteenth member is a
    request in this package's report rather than something it declares for itself.
    """

    name = "K1.DistillationGap"
    version = "1.0"
    #: The instrument reads rollouts, so it needs nothing from a live network. Drawing the rollouts
    #: needs `Capability.GENERATIVE` on all three artifacts, and that is the runbook's job: keeping
    #: it out of the declaration is what lets K1 run at RECORD access on a lab's own dump, which is
    #: the only access anybody outside that lab will ever have to this triple.
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = None
    deviations = (
        "the survival fraction is conditional on the feature basis, so a basis that omits the axis "
        "distillation moved reports high survival. The fitted feature names are on every reading "
        "and the pooled fit weights each feature by how much there was to lose",
        "the shared-prompt-set and matched-decoding preconditions are hard checks returning a "
        "Refusal rather than EnvelopeSpec conditions, because the registry carries no "
        "RegimeCondition for a matched sampling design",
        "the pooled survival is a slope of the student's mean shift on the expert's mean shift per "
        "prompt, so it is first-order in the mean and blind to a change in the spread of a "
        "behaviour at fixed mean",
        "the blank arm bounds sampling noise at fixed weights and does not bound the seed-to-seed "
        "variability of the RL run itself, which needs several RL seeds and is the expensive half "
        "of the price",
        "the incremental-validity comparison scores rollouts with a direction fitted on a disjoint "
        "half of the prompts, which removes the in-sample optimism of the direction and not the "
        "optimism of having chosen the feature basis in advance",
    )

    # -- the registry declarations ------------------------------------------
    quantity = "artifact.distillation_delta"
    #: `ARTIFACT` at `QUERY` because producing the three rollout sets means being able to call all
    #: three checkpoints, and `RECORD` because the instrument itself reads only the rollouts. E16's
    #: `expand_access` puts RECORD under QUERY, so a caller holding both says so once.
    requires: AccessMatrix = {
        Component.ARTIFACT: Access.QUERY,
        Component.RECORD: Access.RECORD,
    }
    #: A policy is a generative network. The other five substrates have no rollouts to compare.
    substrates = frozenset({Substrate.NEURAL_GEN})
    #: The question is asked after the run, about the artifact. `DEPLOYED` too, because the student
    #: is by construction the model that ships and this is the one instrument whose subject is it.
    phases = frozenset({Phase.POST_RUN, Phase.DEPLOYED})
    envelope = DISTILLATION_ENVELOPE
    #: Permuting the rollouts within a prompt leaves every feature mean, and therefore the whole
    #: survival slope, exactly where it was. That is the generated property test.
    #:
    #: The reading is also exactly invariant under `tokenization`, since every feature is computed
    #: from `Turn.text` and nothing here reads `token_ids`. It is not declared, and the reason is
    #: worth recording rather than leaving as an undocumented silence: the
    #: generated test for a relation nothing can move would pass without testing anything, and a
    #: vacuous passing test is worse than an absent one. If a token-count feature is ever added to
    #: the basis the declaration becomes non-vacuous and should be made.
    invariance = "group.permutation"
    invariance_relation = INVARIANT
    baselines = ALL_SIX
    rung = 0

    def __init__(
        self,
        *,
        base: Arm,
        expert: Arm,
        student: Arm,
        blanks: Sequence[Arm] = (),
        featuriser: TrajectoryFeaturiser | None = None,
        hack_features: Sequence[str] = (),
        markers: Sequence[str] = (),
        ci: float = 0.95,
        seed: int = 0,
        n_bootstrap: int = 2_000,
        sampling: dict[str, Any] | None = None,
    ) -> None:
        self.base = base
        self.expert = expert
        self.student = student
        self.blanks = tuple(blanks)
        self.featuriser = featuriser
        self.hack_features = tuple(hack_features)
        self.markers = tuple(markers)
        self.ci = float(ci)
        self.seed = int(seed)
        self.n_bootstrap = int(n_bootstrap)
        #: The decoding settings each arm was drawn under, keyed by arm name. Checked for equality
        #: before anything is measured, because a log-probability or a length recomputed under
        #: different decoding is a different number and the difference would be attributed to
        #: distillation.
        self.sampling = dict(sampling or {})

    # -- the measurement ----------------------------------------------------

    def measure(self, ctx: Context) -> Any:
        mismatch = self._sampling_mismatch()
        if mismatch is not None:
            return mismatch
        result = distillation_survival(
            self.base,
            self.expert,
            self.student,
            blanks=self.blanks,
            featuriser=self.featuriser,
            hack_features=self.hack_features,
            markers=self.markers,
            ci=self.ci,
            seed=self.seed,
            n_bootstrap=self.n_bootstrap,
        )
        if isinstance(result, Refusal):
            return result
        return ctx.emit(
            result,
            uncertainty=Uncertainty(
                ci_low=100.0 - result.survival_ci_high_pp,
                ci_high=100.0 - result.survival_ci_low_pp,
                ci_level=result.ci_level,
                n=result.n_prompts,
                method=(
                    f"percentile bootstrap over {self.n_bootstrap:,} resamples of whole prompts on "
                    f"the through-origin slope of the student's mean feature shift against the "
                    f"expert's, both in base-arm spread units"
                ),
            ),
            baselines=dict(result.baselines),
        )

    def _sampling_mismatch(self) -> Refusal | None:
        """Refuse three arms drawn under different decoding, naming the keys that differ.

        The check is on the declared settings rather than on the rollouts, because there is no way
        to recover a temperature from a completion, and an instrument that could not check this at
        all would be silently comparing a greedy student against a sampled expert and calling the
        difference distillation.
        """
        if len(self.sampling) < 2:
            return None
        keys = sorted({k for spec in self.sampling.values() for k in spec})
        differing = {
            k: {arm: spec.get(k) for arm, spec in sorted(self.sampling.items())}
            for k in keys
            if len({repr(spec.get(k)) for spec in self.sampling.values()}) > 1
        }
        if not differing:
            return None
        named = ", ".join(f"{k}={v}" for k, v in sorted(differing.items()))
        # A matched sampling design is a precondition that fails quietly, which is exactly what an
        # envelope is for, and the thirteen registered members carry none for it. `ENVELOPE_VIOLATED` is the
        # closest existing reason and it is carried with the differing keys named in `detail`,
        # rather than inventing an eighteenth reason, which is not a 3.0.0 change.
        return Refusal(
            instrument=self.name,
            reason=RefusalReason.ENVELOPE_VIOLATED,
            detail=(
                f"the arms were drawn under different decoding settings: {named}. Every feature "
                f"here is a property of the output distribution, so a difference in decoding lands "
                f"in the same number as a difference in weights"
            ),
            remedy=(
                "re-draw every arm under one `SampleSpec`: same temperature, same top_p, same "
                "max_new_tokens, same stop set, and the same completions per prompt. If the "
                "student genuinely ships under different decoding, that is a second comparison "
                "worth making and it is not this one, so run it as its own arm and label it."
            ),
            statistics={"differing_keys": sorted(differing)},
        )


__all__ = ["DISTILLATION_ENVELOPE", "DistillationGap", "DistillationSurvival"]

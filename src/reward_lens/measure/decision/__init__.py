"""Level 3, the contract layer: how hard to optimise, derived rather than tuned.

The argument for this layer is that without it the metrology produces true numbers nobody acts
on. "Thirty-seven percent of your gradient this step was grader noise" is a good sentence and it does
not tell anyone what to do. This package converts a measured noise into a weight, and the evidence
that it is load-bearing empty space is blunt: **nobody sets a reward component's weight from that
component's measured noise.**

Four instruments over four proposed catalogue records:

- **N5** `OptimalWeights`, `reward.optimal_weights`. The Holmstrom and Milgrom (1991) solution
  `alpha* = (I + r C'' Sigma)^-1 B'`, the unmeasurable-task correction, and the zero-weight theorem.
- **N6** `EqualCompensation`, `reward.equal_compensation_ratio`. Which component is starved, from
  Milgrom and Roberts (1992) rather than from the 1991 paper the principle is usually attributed to.
- **N7** `SortingCutoff`, `reward.information_coefficient` and `reward.sorting_cutoff`. Which
  components belong in the same weighted sum. Do not put a noisy judge and a crisp unit test in one.
- **N8** `NoiseAndAngle`, `reward.component_noise` and `reward.component_congruity`. Every reward
  component needs two numbers, and no tooling separates them.

Three things about this package are worth knowing before reading a number out of it.

**The assumptions are the product, as much as the number is.** This layer imports an economic model
with five strong premises and every reading renders all five beside the recommendation. That is
enforced in the shared runner rather than left to each instrument, because a weight recommendation
whose model assumptions are invisible is advice wearing a number. Two of the five are checkable with
instruments that already exist, three are not, and the reading says which is which.

**A recommendation that depends on an unmeasured parameter refuses.** The formula needs `r`, `C''`,
`Sigma`, `B'` and `M`, and exactly one of the five is measurable from a reward record: `Sigma`, which
A2 produces. The other four are supplied and recorded as supplied, or swept and reported as a
function of the range, or the instrument declines. Sensitivity to the unmeasured parameters is part
of the reading rather than a footnote, and the claim that usually survives not knowing `r` is the
*ordering* of the components, which `SensitivityCurve` reports directly.

**The zero-weight theorem is reachable and it is exactly zero.** Perfectly substitutable effort plus
one unmeasurable task of equal value gives a numerator of exactly `B_1 - B_2` and a Schur complement
of exactly zero, so `alpha*_1 = 0` bit-exactly rather than approximately, and the principal's surplus
at any nonzero weight diverges downward as the boundary is approached. Not small. Zero, and any
positive power unboundedly worse than none.

The whole layer needs a noise covariance and four stated numbers, and nothing else: no GPU, no policy
checkpoint, no record, no gradients. It imports no torch and it answers before a run exists.

**The catalogue carries no N5 to N8 records.** The six quantity rows and the four instrument
records this package needs are proposed as data in `quantities.py` and emitted by `as_yaml_rows()`
and `as_catalogue_rows()`, and nothing registers them at import.
"""

from __future__ import annotations

from reward_lens.measure.decision._base import (
    ALL_SUBSTRATES,
    CONTRACT_ENVELOPE,
    CONTRACT_PHASES,
    NOISE_ACCESS,
    WEIGHTS_ACCESS,
    DecisionInstrument,
)
from reward_lens.measure.decision.assumptions import (
    AGENT_KNOWS_THE_CONTRACT,
    ASSUMPTION_KEYS,
    ASSUMPTIONS,
    CARA_AND_COST_CURVATURE,
    COMMITMENT_ONE_PERIOD,
    GAUSSIAN_NOISE,
    LINEAR_CONTRACT,
    Assumption,
    assumption,
    assumptions_payload,
    render_assumptions,
)
from reward_lens.measure.decision.compensation import (
    DEFAULT_STARVATION_FLOOR,
    N6_BASELINES,
    CompensationRow,
    CompensationTable,
    EqualCompensation,
    equal_compensation,
)
from reward_lens.measure.decision.geometry import (
    DEFAULT_DISTORTION_CEILING,
    DEFAULT_SHRINKAGE_FLOOR,
    N8_BASELINES,
    NOISE_ONLY_VERDICTS,
    VERDICTS,
    NoiseAndAngle,
    NoiseAngleReading,
    NoiseAngleRow,
    clean_measure_congruity,
    component_congruity,
    congruity,
    noise_and_angle,
)
from reward_lens.measure.decision.parameters import (
    EFFORT_NAME,
    PARAMETER_MEANING,
    PARAMETER_NAMES,
    PARAMETER_REMEDY,
    ContractParameters,
    ParameterSource,
    Sweep,
    noise_correlation_from_residuals,
    noise_from_gauge_studies,
)
from reward_lens.measure.decision.quantities import (
    CATALOGUE_RECORDS,
    PROPOSED,
    PROPOSED_INSTRUMENTS,
    PROPOSED_MIN_ACCESS,
    PROPOSED_RUNGS,
    as_catalogue_rows,
    as_yaml_rows,
    register_proposed,
)
from reward_lens.measure.decision.sorting import (
    DEFAULT_ENUMERATION_LIMIT,
    N7_BASELINES,
    Contract,
    SortingCutoff,
    SortingReading,
    contract_value,
    information_coefficient,
    noise_to_signal,
    sorting_cutoff,
)
from reward_lens.measure.decision.weights import (
    COVARIANT_INVERSE,
    DEFAULT_CONDITION_LIMIT,
    N5_BASELINES,
    Crossing,
    OptimalWeights,
    SensitivityCurve,
    UnmeasurableCorrection,
    WeightRecommendation,
    noiseless_weights,
    optimal_weights,
    optimal_weights_diagonal,
    principal_surplus,
    recommend_weights,
    shrinkage,
    sweep_weights,
    two_task_surplus,
    unmeasurable_correction,
)

#: The four instruments, in the order the layer is read: the weight first, because the other three
#: are all statements about a weight vector.
DECISION: tuple[type, ...] = (
    OptimalWeights,
    EqualCompensation,
    SortingCutoff,
    NoiseAndAngle,
)

__all__ = [
    "AGENT_KNOWS_THE_CONTRACT",
    "ALL_SUBSTRATES",
    "ASSUMPTIONS",
    "ASSUMPTION_KEYS",
    "CARA_AND_COST_CURVATURE",
    "CATALOGUE_RECORDS",
    "COMMITMENT_ONE_PERIOD",
    "CONTRACT_ENVELOPE",
    "CONTRACT_PHASES",
    "COVARIANT_INVERSE",
    "DECISION",
    "DEFAULT_CONDITION_LIMIT",
    "DEFAULT_DISTORTION_CEILING",
    "DEFAULT_ENUMERATION_LIMIT",
    "DEFAULT_SHRINKAGE_FLOOR",
    "DEFAULT_STARVATION_FLOOR",
    "EFFORT_NAME",
    "GAUSSIAN_NOISE",
    "LINEAR_CONTRACT",
    "N5_BASELINES",
    "N6_BASELINES",
    "N7_BASELINES",
    "N8_BASELINES",
    "NOISE_ACCESS",
    "NOISE_ONLY_VERDICTS",
    "PARAMETER_MEANING",
    "PARAMETER_NAMES",
    "PARAMETER_REMEDY",
    "PROPOSED",
    "PROPOSED_INSTRUMENTS",
    "PROPOSED_MIN_ACCESS",
    "PROPOSED_RUNGS",
    "VERDICTS",
    "WEIGHTS_ACCESS",
    "Assumption",
    "CompensationRow",
    "CompensationTable",
    "Contract",
    "ContractParameters",
    "Crossing",
    "DecisionInstrument",
    "EqualCompensation",
    "NoiseAndAngle",
    "NoiseAngleReading",
    "NoiseAngleRow",
    "OptimalWeights",
    "ParameterSource",
    "SensitivityCurve",
    "SortingCutoff",
    "SortingReading",
    "Sweep",
    "UnmeasurableCorrection",
    "WeightRecommendation",
    "as_catalogue_rows",
    "as_yaml_rows",
    "assumption",
    "assumptions_payload",
    "clean_measure_congruity",
    "component_congruity",
    "congruity",
    "contract_value",
    "equal_compensation",
    "information_coefficient",
    "noise_and_angle",
    "noise_correlation_from_residuals",
    "noise_from_gauge_studies",
    "noise_to_signal",
    "noiseless_weights",
    "optimal_weights",
    "optimal_weights_diagonal",
    "principal_surplus",
    "recommend_weights",
    "register_proposed",
    "render_assumptions",
    "shrinkage",
    "sorting_cutoff",
    "sweep_weights",
    "two_task_surplus",
    "unmeasurable_correction",
]

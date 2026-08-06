"""Series I, the pressure instruments: what a hard reward gate does to the run around it.

**A hard reward threshold is a sharp regression discontinuity with a deterministic, perfectly
measured assignment rule.** That is worth stating first because it is the whole reason this package
is short. The regression-discontinuity and bunching literatures spend most of their effort on
problems a reward gate does not have: compliance is perfect, because the rule is code; the running
variable carries no measurement error, because the trainer wrote it down; the cutoff is known
exactly, because it is a constant in a config file; and the assignment is not confounded, because
nothing else changes at the threshold. Forty years of identification machinery arrives with its
hardest assumptions satisfied by construction, and two independent literature searches found
nothing pointing it at reward gates.

Four instruments, none of which needs a GPU:

- **I1 `DensityDiscontinuity`** asks whether the density of the running variable jumps at the gate.
  A policy pushing against a token budget piles mass just below it, and McCrary's test is how that
  is measured. Baselines: a smooth-density null and the same test at a placebo cutoff.
- **I2 `BunchingElasticity`** asks how hard it is pushing, and converts excess mass into a Saez
  elasticity so the answer is comparable across gates and across runs. Its kill condition is that
  the estimate must move when the gate moves, and `gate_response` runs it.
- **I3 `DeadZoneFraction`** asks how much of the run sits where the score is pinned to a constant
  and the gradient with respect to task quality is exactly zero. Two controls run on every reading:
  a graded penalty of the same size, and the decode length reported separately from the training
  budget.
- **I5 `VarianceDerivative`** asks whether the derivative of within-group reward variance is an
  earlier warning than the level of the variance or the gradient-norm peak, scored in units of the
  fitted transition width rather than in steps.

Three of the four read a running variable and a gate. `gates.py` builds both from a `Run`, and it
is where the one piece of discipline this package depends on lives: a `Gate` says whether it was
**recorded on the run** or **installed counterfactually**, and every reading prints which. A record
re-scored under a rule the policy never saw is a way of checking that an estimator finds a gate
whose location you chose. It is not evidence about anybody's run, because the rollouts a policy
would have produced under that rule are not the rollouts on the record.
"""

from reward_lens.measure.threshold._base import (
    ALL_SUBSTRATES,
    DEADZONE_ENVELOPE,
    GATE_ACCESS,
    GATE_ENVELOPE,
    RECORD_ACCESS,
    RECORD_PHASES,
    VARIANCE_ENVELOPE,
    ThresholdInstrument,
)
from reward_lens.measure.threshold.bunching import (
    BUNCHING_BASELINES,
    BunchingElasticity,
    BunchingReading,
    Counterfactual,
    GateResponse,
    WindowSweep,
    auto_window,
    bunching_elasticity,
    counterfactual_density,
    gate_response,
    saez_elasticity,
    smooth_null_excess,
)
from reward_lens.measure.threshold.deadzone import (
    DEADZONE_BASELINES,
    DeadZoneFraction,
    DeadZoneReading,
    GradedContrast,
    deadzone_fraction,
    graded_contrast,
)
from reward_lens.measure.threshold.density import (
    DENSITY_BASELINES,
    DensityDiscontinuity,
    Histogram,
    McCraryReading,
    NullBand,
    automatic_bandwidth,
    automatic_binsize,
    density_discontinuity,
    histogram,
    mccrary,
    mccrary_robust,
    placebo_cutoffs,
    smooth_density_null,
)
from reward_lens.measure.threshold.gates import (
    DecodeLength,
    Gate,
    InstalledGateSet,
    RunningVariable,
    completion_length,
    completion_lengths,
    decode_length,
    install_gate,
    penalty_fraction,
    recorded_gates,
    score_contexts,
)
from reward_lens.measure.threshold.units import ThresholdQuantity, rank, unit_of
from reward_lens.measure.threshold.variance import (
    DEFAULT_ARL0,
    VARIANCE_BASELINES,
    AlarmCalibration,
    DetectorResult,
    RunSeries,
    TransitionFit,
    VarianceDerivative,
    VarianceDerivativeReading,
    alarm_calibration,
    arl0,
    cadence_resolution_in_widths,
    cusum_threshold,
    derivative,
    first_alarm,
    fit_transition,
    lead_time_in_widths,
    peak_index,
    planted_run,
    run_series,
    score_detectors,
    variance_derivative,
)

__all__ = [
    "ALL_SUBSTRATES",
    "BUNCHING_BASELINES",
    "DEADZONE_BASELINES",
    "DEADZONE_ENVELOPE",
    "DEFAULT_ARL0",
    "DENSITY_BASELINES",
    "GATE_ACCESS",
    "GATE_ENVELOPE",
    "RECORD_ACCESS",
    "RECORD_PHASES",
    "VARIANCE_BASELINES",
    "VARIANCE_ENVELOPE",
    "BunchingElasticity",
    "BunchingReading",
    "Counterfactual",
    "DeadZoneFraction",
    "DeadZoneReading",
    "DecodeLength",
    "DensityDiscontinuity",
    "AlarmCalibration",
    "DetectorResult",
    "Gate",
    "GateResponse",
    "GradedContrast",
    "Histogram",
    "InstalledGateSet",
    "McCraryReading",
    "NullBand",
    "RunSeries",
    "RunningVariable",
    "ThresholdInstrument",
    "ThresholdQuantity",
    "TransitionFit",
    "VarianceDerivative",
    "VarianceDerivativeReading",
    "WindowSweep",
    "alarm_calibration",
    "arl0",
    "auto_window",
    "automatic_bandwidth",
    "automatic_binsize",
    "bunching_elasticity",
    "cadence_resolution_in_widths",
    "completion_length",
    "completion_lengths",
    "counterfactual_density",
    "cusum_threshold",
    "deadzone_fraction",
    "decode_length",
    "density_discontinuity",
    "derivative",
    "first_alarm",
    "fit_transition",
    "gate_response",
    "graded_contrast",
    "histogram",
    "install_gate",
    "lead_time_in_widths",
    "mccrary",
    "mccrary_robust",
    "peak_index",
    "penalty_fraction",
    "placebo_cutoffs",
    "planted_run",
    "rank",
    "recorded_gates",
    "run_series",
    "saez_elasticity",
    "score_contexts",
    "score_detectors",
    "smooth_density_null",
    "smooth_null_excess",
    "unit_of",
    "variance_derivative",
]

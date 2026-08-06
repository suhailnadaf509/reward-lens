"""Rate: how fast the loop is moving, and whether the assumptions that survive slow motion hold.

Three modules. `regime` measures the twelve regime conditions from a record. `adiabaticity`
is H1 rung 0, the number that decides one of those twelve: `Ad = tau_relax * |d log lambda / dt|`,
the relaxation time against the driving rate. `transition` is H4, the fitted width of a behavioural
transition, which is the **unit every lead time in this library is reported in**.

They belong together because they all read the same thing, which is the speed of the driver against
the speed the system can follow. What is left is the compute half: the two-run
collapse test (H2), the rate-extrapolated hysteresis area (H3), and the perturb-and-hold relaxation
time that is rung 1 of `run.tau_relax`. All three are registered as rungs with their access and
their cost and none is built.

Two instruments here estimate `run.tau_relax` at rung 0 and they disagree. `regime` fits the lag-1
coefficient by ordinary least squares and says in its own docstring that the fit is biased low and
therefore toward licensing; `adiabaticity` removes that bias by bootstrap and takes its verdict on
the upper end of the resulting interval. `adiabaticity.tau_transfer` publishes the difference as a
chain term rather than reconciling it, which is what M11 does for two rungs of one
ladder and is the same argument for two estimators at one rung.

Torch-free by construction. Everything here reads a `Run` and returns a `Reading`, and a preflight
that has to load a deep learning framework to find out whether it may run is not a preflight.
"""

from reward_lens.measure.rate.adiabaticity import (
    ADIABATICITY_BASELINES,
    ADIABATICITY_ENVELOPE,
    STEP_AXIS,
    Adiabaticity,
    AdiabaticityReading,
    DriveRate,
    RelaxationFloors,
    RelaxationTime,
    StepAdiabaticity,
    adiabaticity,
    adiabaticity_number,
    drive_rates,
    regime_inputs,
    relaxation_time,
    tau_of,
    tau_transfer,
)

# The compute-gated rate instruments: the code ships, the runbook and the
# price ship, and no result does, because none of them has been run. Exported here so the estimator
# ladder and the capability report can see them without importing a submodule by hand.
from reward_lens.measure.rate.collapse import (
    COLLAPSE_BASELINES,
    COLLAPSE_ENVELOPE,
    ArmFit,
    CollapseCriteria,
    RateArm,
    RateDependence,
    RateDependenceTest,
    two_run_rate_test,
)
from reward_lens.measure.rate.hysteresis import (
    HYSTERESIS_BASELINES,
    HYSTERESIS_ENVELOPE,
    HysteresisCriteria,
    RateExtrapolatedArea,
    RateExtrapolatedHysteresis,
    SweepArea,
    rate_extrapolated_area,
    sweep_areas,
)
from reward_lens.measure.rate.perturb import (
    PERTURB_AND_HOLD_IMPL,
    PERTURB_BASELINES,
    PERTURB_ENVELOPE,
    EpisodeFit,
    PerturbAndHold,
    PerturbCriteria,
    PerturbRelaxation,
    Recovery,
    fit_recovery,
    relaxation_time_from_hold,
    rung_transfer,
)
from reward_lens.measure.rate.regime import (
    MEASURED_BY,
    UNREGISTERED_MEASURED_BY,
    RegimeFloors,
    RegimeInputs,
    RegimeThresholds,
    RunRegime,
    declaration_disagreements,
    measure_regime,
)
from reward_lens.measure.rate.transition import (
    SERIES,
    TEN_TO_NINETY,
    TRANSITION_BASELINES,
    TRANSITION_ENVELOPE,
    FitQuality,
    LeadTime,
    TransitionCriteria,
    TransitionFit,
    TransitionWidth,
    available_series,
    compare_lead_times,
    fit_transition,
    lead_time,
    series_from_run,
    window_steps,
)
from reward_lens.measure.rate.warning import (
    DriverComparison,
    EarlyWarning,
    Flickering,
    TrendNull,
    WarningCriteria,
    WindowSensitivity,
    ar1_surrogate,
    driver_comparison,
    early_warning,
    flickering,
    fourier_surrogate,
    gaussian_detrend,
    gaussian_smooth,
    rolling_indicator,
    trend_significance,
    window_sensitivity,
)

__all__ = [
    "ADIABATICITY_BASELINES",
    "ADIABATICITY_ENVELOPE",
    "MEASURED_BY",
    "SERIES",
    "STEP_AXIS",
    "TEN_TO_NINETY",
    "TRANSITION_BASELINES",
    "TRANSITION_ENVELOPE",
    "UNREGISTERED_MEASURED_BY",
    "Adiabaticity",
    "AdiabaticityReading",
    "DriveRate",
    "FitQuality",
    "LeadTime",
    "RegimeFloors",
    "RegimeInputs",
    "RegimeThresholds",
    "RelaxationFloors",
    "RelaxationTime",
    "RunRegime",
    "StepAdiabaticity",
    "TransitionCriteria",
    "TransitionFit",
    "TransitionWidth",
    "adiabaticity",
    "adiabaticity_number",
    "available_series",
    "compare_lead_times",
    "declaration_disagreements",
    "drive_rates",
    "fit_transition",
    "lead_time",
    "measure_regime",
    "regime_inputs",
    "relaxation_time",
    "series_from_run",
    "tau_of",
    "tau_transfer",
    "window_steps",
    "ArmFit",
    "CollapseCriteria",
    "COLLAPSE_BASELINES",
    "COLLAPSE_ENVELOPE",
    "DriverComparison",
    "EarlyWarning",
    "EpisodeFit",
    "Flickering",
    "HYSTERESIS_BASELINES",
    "HYSTERESIS_ENVELOPE",
    "HysteresisCriteria",
    "PERTURB_AND_HOLD_IMPL",
    "PERTURB_BASELINES",
    "PERTURB_ENVELOPE",
    "PerturbAndHold",
    "PerturbCriteria",
    "PerturbRelaxation",
    "RateArm",
    "RateDependence",
    "RateDependenceTest",
    "RateExtrapolatedArea",
    "RateExtrapolatedHysteresis",
    "Recovery",
    "SweepArea",
    "TrendNull",
    "WarningCriteria",
    "WindowSensitivity",
    "ar1_surrogate",
    "driver_comparison",
    "early_warning",
    "fit_recovery",
    "flickering",
    "fourier_surrogate",
    "gaussian_detrend",
    "gaussian_smooth",
    "rate_extrapolated_area",
    "relaxation_time_from_hold",
    "rolling_indicator",
    "rung_transfer",
    "sweep_areas",
    "trend_significance",
    "two_run_rate_test",
    "window_sensitivity",
]

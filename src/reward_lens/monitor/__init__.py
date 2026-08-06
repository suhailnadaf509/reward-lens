"""The monitoring layer: J1 to J5, and the reason it is allowed to look at every step.

Five instruments over five catalogue records, and one fact underneath all of them. Ville (1939):
for a nonnegative martingale, ``P(exists t : M_t >= a) <= 1/a``. So an alarm built on one may be
read continuously without any correction for the number of looks, which is what separates a monitor
from an evaluation.

- **J1** `ConfidenceSequence`, `EValue`, `monitor.confidence_sequence` and `monitor.evalue`. The
  anytime-valid interval and the e-process, at three rungs. The mandatory baseline is a
  fixed-sample interval reported with how often it would have been wrong under continuous peeking,
  which is 37.5% of runs against an advertised 5% on 4,000 simulated Bernoulli(0.1) streams.
- **J2** `AlarmDesign`, `DetectionDelay`, `monitor.arl0` and `monitor.detection_delay`. Set
  ``k = delta/2``, solve ``ARL(0) = ARL_0`` for ``h``, and report the delay as a fraction of the
  fitted transition width. The design removes a free parameter: this library's own flight recorder
  ships ``k = 0.5, h = 5.0`` with no derivation anywhere, which happens to imply ``ARL_0 = 469``.
- **J3** `ConjunctionDetector`, `monitor.conjunction_alarm`. The conjunction is the detector, and
  its joint false-alarm rate is bounded by the smallest per-channel level under **arbitrary**
  dependence, with no merging rule needed. e-BH controls the false discovery rate over the ledger.
- **J4** `OperatingPointFromLoss`, `monitor.operating_point` and `monitor.ppv_curve`. A threshold is
  a decision, so it needs a loss. **Every detector here attaches a PPV curve**, because at a 2.5%
  base rate and a 31.7% false-positive rate the positive predictive value is 7.5% even with perfect
  sensitivity, and that is one line of Bayes' rule that prevents a recurring class of overclaim.
- **J5** `CheckStandardDriftInstrument`, `monitor.check_standard_drift`. A frozen probe set whose
  job is to be invariant, so any movement in it is instrument drift by construction. It is the
  measurement that establishes the `STATIONARY_GRADER` envelope condition the other four require.

**Where the numbers come from.** An average run length and a detection delay are properties of a
procedure rather than of any one series, so they are established by simulation against a planted
changepoint of known width: `cusum.measure_bank` and `conjunction.measure_conjunction`. Whether the
instruments run on a real optimisation trace with real per-step statistics is a different question
and it is answered on the shipped 200-step GRPO record. That record contains no reward-hacking
transition, so no lead time is claimed on it and `DetectionDelay` refuses by name rather than
reporting an alarm index as a lead.

`_vendor/cif.py` is `AsiaeeLab/certified-interventional-fidelity` under MIT, vendored unmodified,
providing rungs 0 and 2 of J1's ladder. `confseq` was the alternative and cannot be installed here
at all: no cp312 wheel, and CMake fails on Boost.
"""

from __future__ import annotations

from reward_lens.monitor._base import (
    ALL_SUBSTRATES,
    CHANNEL_SPECS,
    MONITOR_ENVELOPE,
    MONITOR_PHASES,
    NO_ACCESS,
    RECORD_ACCESS,
    Channel,
    ChannelSet,
    MonitorInstrument,
    channels_of,
)
from reward_lens.monitor.arl import (
    RHO,
    SHIPPED_AD_HOC,
    SIEGMUND_REFERENCE,
    CusumDesign,
    ReferencePoint,
    arl_integral_equation,
    arl_monte_carlo,
    arl_siegmund,
    design_cusum,
    kl_gaussian,
    lorden_delay,
    reference_points,
    shipped_ad_hoc_arl0,
    solve_h,
)
from reward_lens.monitor.check_standard import (
    CHECK_STANDARD_CHANNEL,
    CHECK_STANDARD_ENVELOPE,
    J5_BASELINES,
    MIN_BASELINE_SESSIONS,
    CheckStandardDrift,
    CheckStandardDriftInstrument,
    Session,
    check_standard_drift,
    probe_set_fingerprint,
    sessions_from_run,
)
from reward_lens.monitor.conjunction import (
    J3_BASELINES,
    MIN_CHANNELS,
    ChannelAlarm,
    ConjunctionAlarm,
    ConjunctionComparison,
    ConjunctionDetector,
    OperatingRow,
    channel_alarm,
    conjunction_alarm,
    gaussian_lr_eprocess,
    measure_conjunction,
)
from reward_lens.monitor.cusum import (
    DESIGN_ENVELOPE,
    J2_DELAY_BASELINES,
    J2_DESIGN_BASELINES,
    LOGISTIC_10_90,
    AlarmDesign,
    BankReport,
    CusumRun,
    DetectionDelay,
    DetectorPerformance,
    DetectorSpec,
    TransitionWindow,
    default_bank,
    local_transition_width,
    measure_bank,
    run_cusum,
    standardize,
)
from reward_lens.monitor.eprocess import (
    J1_BASELINES,
    RUNG_NAMES,
    RUNG_SOURCE,
    ConfidenceSequence,
    ConfidenceTrace,
    EProcessTrace,
    EValue,
    abstention_stream,
    confidence_sequence,
    eprocess,
    peeking_cost,
)
from reward_lens.monitor.ewma import EwmaDesign, arl_ewma, design_ewma, ewma_alarm, sigma_z
from reward_lens.monitor.operating_point import (
    DEFAULT_PREVALENCES,
    J4_BASELINES,
    LOSS_ENVELOPE,
    WORKED_CASE,
    AsymmetricLoss,
    OperatingPoint,
    OperatingPointFromLoss,
    PPVCurve,
    choose_threshold,
    expected_loss,
    npv,
    ppv,
    ppv_curve,
)

#: The seven instrument classes over the five catalogue records. J1 and J2 contribute two classes
#: each because each carries two quantities and an instrument declares one.
MONITOR: tuple[type, ...] = (
    ConfidenceSequence,
    EValue,
    AlarmDesign,
    DetectionDelay,
    ConjunctionDetector,
    OperatingPointFromLoss,
    CheckStandardDriftInstrument,
)

__all__ = [
    "ALL_SUBSTRATES",
    "CHANNEL_SPECS",
    "CHECK_STANDARD_CHANNEL",
    "CHECK_STANDARD_ENVELOPE",
    "DEFAULT_PREVALENCES",
    "DESIGN_ENVELOPE",
    "J1_BASELINES",
    "J2_DELAY_BASELINES",
    "J2_DESIGN_BASELINES",
    "J3_BASELINES",
    "J4_BASELINES",
    "J5_BASELINES",
    "LOGISTIC_10_90",
    "LOSS_ENVELOPE",
    "MIN_BASELINE_SESSIONS",
    "MIN_CHANNELS",
    "MONITOR",
    "MONITOR_ENVELOPE",
    "MONITOR_PHASES",
    "NO_ACCESS",
    "RECORD_ACCESS",
    "RHO",
    "RUNG_NAMES",
    "RUNG_SOURCE",
    "WORKED_CASE",
    "SHIPPED_AD_HOC",
    "SIEGMUND_REFERENCE",
    "AlarmDesign",
    "AsymmetricLoss",
    "BankReport",
    "Channel",
    "ChannelAlarm",
    "ChannelSet",
    "CheckStandardDrift",
    "CheckStandardDriftInstrument",
    "ConfidenceSequence",
    "ConfidenceTrace",
    "ConjunctionAlarm",
    "ConjunctionComparison",
    "ConjunctionDetector",
    "CusumDesign",
    "CusumRun",
    "DetectionDelay",
    "DetectorPerformance",
    "DetectorSpec",
    "EProcessTrace",
    "EValue",
    "EwmaDesign",
    "MonitorInstrument",
    "OperatingPoint",
    "OperatingPointFromLoss",
    "OperatingRow",
    "PPVCurve",
    "ReferencePoint",
    "Session",
    "TransitionWindow",
    "abstention_stream",
    "arl_ewma",
    "arl_integral_equation",
    "arl_monte_carlo",
    "arl_siegmund",
    "channel_alarm",
    "channels_of",
    "check_standard_drift",
    "choose_threshold",
    "confidence_sequence",
    "conjunction_alarm",
    "default_bank",
    "design_cusum",
    "design_ewma",
    "eprocess",
    "ewma_alarm",
    "expected_loss",
    "gaussian_lr_eprocess",
    "kl_gaussian",
    "local_transition_width",
    "lorden_delay",
    "measure_bank",
    "measure_conjunction",
    "npv",
    "peeking_cost",
    "ppv",
    "ppv_curve",
    "probe_set_fingerprint",
    "reference_points",
    "run_cusum",
    "sessions_from_run",
    "shipped_ad_hoc_arl0",
    "sigma_z",
    "solve_h",
    "standardize",
]

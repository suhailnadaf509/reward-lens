"""`w6_4` and `w6_5`: two compute-gated designs, written in full and not run.

Nothing here is executed. What is delivered is a design complete enough that the maintainer can
decide whether to buy it: the module that does the arithmetic, a study frozen through
`studies.freeze` with its predictions and kill criteria hashed before any data exists, an
acceptance test that proves the arithmetic on a subject whose answer was planted, a statement of
which real subject each claim needs, a runbook, and a price with its assumptions written out.

Both rows are monitor-under-pressure designs and both turn on the same separation, which is why they
are one package. An aggregate moved; two mechanisms could have moved it; the reading is worth
nothing unless it says which. `degradation.py` splits a falling detection curve into **evasion**, the
policy's hacks scoring lower, and **degradation**, the monitor losing discrimination on rollouts that
did not change. `behaviour.py` splits a rising false-positive mass into **selection**, reward moving
onto failure modes the base policy already produced, and **exploitation**, reward landing on failure
modes it never produced. Both separations work the same way, by holding something fixed and
re-measuring it, which is what a check standard is for.

Both instruments refuse when the thing that would be held fixed is missing, and in both cases that
refusal is doing structural work rather than being defensive. Without a frozen probe set, `w6_5`
would produce the falling curve the field already publishes. Without a base-policy sample, `w6_4`
would produce a false-positive rate on a training record, which is a property of the verifier, and
the verifier half of D5 is the half a preregistered published contrast has already answered.

`price.py` ranks the two by decisiveness per dollar and recommends `w6_4` first, mostly because its
expensive arc is gated by a cheap one that can abort it. `study.OPEN_TARGET` names a third study,
cheaper than either, that is upstream of `w6_5`'s premise and untested.

`behaviour.py`'s reading is dual-use and sensitive by default: a per-family ranking of which failure
modes earn reward on a deployed verifier is a target list. It follows the pattern D5's static half
and X4 already use, with the flag on the payload and on the store row, no rendered artifact quoting
it, and publication requiring a recorded decision.
"""

from reward_lens.studies.w6_monitor._base import (
    Discriminability,
    W6Instrument,
    discriminability,
)
from reward_lens.studies.w6_monitor.behaviour import (
    AuditedFamilyMass,
    FamilyMassDecomposition,
    FamilySample,
    base_depth_for,
    counts_from_rollouts,
    decompose_mass,
    sample_from_counts,
)
from reward_lens.studies.w6_monitor.degradation import (
    KILL_TAU,
    DegradationCurve,
    DegradationPoint,
    HalfLife,
    MonitorDegradation,
    MonitorHalfLife,
    MonitorRanking,
    MonitorTrace,
    degradation_curve,
    fit_half_life,
    rank_monitors,
    split_curve,
)
from reward_lens.studies.w6_monitor.planted import (
    planted_family_counts,
    planted_monitor_bank,
    planted_monitor_trace,
    zipf_base,
)
from reward_lens.studies.w6_monitor.price import W6_4_PRICE, W6_5_PRICE, Price, ranked
from reward_lens.studies.w6_monitor.runbook import W6_4_RUNBOOK, W6_5_RUNBOOK
from reward_lens.studies.w6_monitor.study import (
    OPEN_TARGET,
    analyse_w6_4,
    analyse_w6_5,
    check_both,
    freeze_w6_4,
    freeze_w6_5,
    w6_4_plan,
    w6_4_spec,
    w6_5_plan,
    w6_5_spec,
)

#: Every instrument this package ships. The acceptance file lints each one, because an acceptance
#: test that renders readings and never lints the declarations passes a broken declaration.
W6_MONITOR: tuple[type, ...] = (
    MonitorDegradation,
    MonitorHalfLife,
    AuditedFamilyMass,
)

__all__ = [
    "KILL_TAU",
    "OPEN_TARGET",
    "W6_4_PRICE",
    "W6_4_RUNBOOK",
    "W6_5_PRICE",
    "W6_5_RUNBOOK",
    "W6_MONITOR",
    "AuditedFamilyMass",
    "DegradationCurve",
    "DegradationPoint",
    "Discriminability",
    "FamilyMassDecomposition",
    "FamilySample",
    "HalfLife",
    "MonitorDegradation",
    "MonitorHalfLife",
    "MonitorRanking",
    "MonitorTrace",
    "Price",
    "W6Instrument",
    "analyse_w6_4",
    "analyse_w6_5",
    "base_depth_for",
    "check_both",
    "counts_from_rollouts",
    "decompose_mass",
    "degradation_curve",
    "discriminability",
    "fit_half_life",
    "freeze_w6_4",
    "freeze_w6_5",
    "planted_family_counts",
    "planted_monitor_bank",
    "planted_monitor_trace",
    "rank_monitors",
    "ranked",
    "sample_from_counts",
    "split_curve",
    "w6_4_plan",
    "w6_4_spec",
    "w6_5_plan",
    "w6_5_spec",
    "zipf_base",
]

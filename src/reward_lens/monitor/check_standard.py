"""The check standard: a frozen probe set whose only job is to not move (J5).

A check standard is chosen for **stability**, not for difficulty, and that is subtly and importantly
different from a held-out eval. A held-out eval is designed to be hard, so when its score moves you
have learned that the model changed. A check standard is designed so that nothing about the model
under test should move it, so when its score moves you have learned that **the instrument** changed:
the grader was re-deployed, the tokenizer changed, the sampling temperature drifted, the judge's
system prompt was edited, the API you are calling was silently updated.

That is why the drift is a measurement of the measuring apparatus rather than of the subject, and it
is why `measure/rate/regime.py` names `monitor.check_standard_drift` as the quantity that measures
the `STATIONARY_GRADER` envelope condition. Four instruments in this package require that condition
and this is the one that establishes it, which is also why this instrument does not require it
itself: a check standard that refused when the grader was not stationary would be a thermometer that
declined to read when the room was warm.

**What makes it usable is the repeatability, not the drift.** A probe set that moves 0.04 is drifting
if its own session-to-session repeatability is 0.005 and is quiet if that repeatability is 0.06. So
the headline number is the drift **in units of the probe set's measured repeatability**, the raw
drift travels beside it, and the instrument refuses to normalise when there are too few baseline
sessions to have measured a repeatability at all.

**What it cannot do.** It cannot tell you *what* changed, only that something did. It cannot
distinguish a genuinely drifting grader from a probe set that was never invariant in the first
place, so `unstable_probes` names the probes whose baseline spread is a large fraction of their
between-session movement, and a probe set with several of those is not a check standard yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from reward_lens.core.envelope import EnvelopeSpec
from reward_lens.core.evidence import Uncertainty
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.reading import Refusal, RefusalReason, refuse_incomplete
from reward_lens.core.types import Capability, GaugeStatus, content_hash
from reward_lens.measure.base import Context
from reward_lens.monitor._base import RECORD_ACCESS, MonitorInstrument

#: The channel a check-standard probe declares on a `ProbeResult`. The record schema already
#: distinguishes it from `held_out` and `gold`, which is the distinction this instrument is about.
CHECK_STANDARD_CHANNEL = "check_standard"

#: Fewer baseline sessions than this and there is no repeatability to normalise by. Three is the
#: floor at which a standard deviation is a number rather than a gesture, and it is still a weak
#: one: the reading records how many were used.
MIN_BASELINE_SESSIONS: int = 3


@dataclass(frozen=True)
class Session:
    """One measurement of the whole probe set, at one point in time.

    ``label`` is whatever identifies the occasion: a step index, a date, a deployment id. It is a
    string because a check standard is measured across runs and a step index does not survive that.
    """

    label: str
    values: Mapping[str, float]

    @property
    def probes(self) -> frozenset[str]:
        return frozenset(self.values)


@dataclass(frozen=True)
class CheckStandardDrift:
    """How far the frozen probe set has moved, and whether that is more than it usually moves.

    ``drift`` is the headline: the root-mean-square per-probe departure from the baseline reference,
    each probe divided by its own baseline repeatability, so the number is dimensionless and probes
    on different scales contribute comparably. ``raw_drift`` is the same thing without the division,
    in whatever units the probes are in.

    ``worst_probe`` is what a reader acts on. A drift of 3.2 spread evenly over eight probes is a
    grader that moved; a drift of 3.2 carried entirely by one probe is usually that probe's task
    changing, and the two want different responses.
    """

    drift: float
    raw_drift: float
    n_probes: int
    n_sessions: int
    n_baseline: int
    per_probe_z: Mapping[str, float]
    per_probe_raw: Mapping[str, float]
    repeatability: Mapping[str, float]
    unstable_probes: tuple[str, ...]
    fingerprint: str
    session_label: str
    max_drift_over_sessions: float
    drift_by_session: Mapping[str, float]

    @property
    def worst_probe(self) -> str:
        if not self.per_probe_z:
            return ""
        return max(self.per_probe_z, key=lambda p: abs(self.per_probe_z[p]))

    def render(self) -> str:
        worst = self.worst_probe
        lines = [
            f"the frozen probe set moved {self.drift:.3g} repeatability units at session "
            f"{self.session_label} ({self.raw_drift:.4g} in raw units), over {self.n_probes} "
            f"probes and {self.n_sessions} sessions with {self.n_baseline} of them as the baseline.",
            "    That is instrument drift, not model drift: the probe set is chosen so nothing "
            "about the model under test should move it.",
        ]
        if worst:
            lines.append(
                f"    worst probe `{worst}` at {self.per_probe_z[worst]:+.3g} repeatability units "
                f"({self.per_probe_raw[worst]:+.4g} raw)."
            )
        if self.unstable_probes:
            lines.append(
                f"    {len(self.unstable_probes)} probe(s) are not stable enough to be a check "
                f"standard: {', '.join(self.unstable_probes)}. Their baseline spread is a large "
                f"fraction of the movement being measured, so they contribute noise rather than a "
                f"reference."
            )
        lines.append(f"    probe-set fingerprint {self.fingerprint}")
        return "\n".join(lines)


def probe_set_fingerprint(probes: Sequence[str]) -> str:
    """A content hash over the probe names, so "frozen" is checkable rather than asserted.

    A check standard whose membership changed between sessions is not one, and comparing drift
    across a changed probe set measures the change in membership. Hashing the sorted names is the
    cheapest thing that catches it, and it catches the case that actually happens: somebody adds a
    probe.
    """
    return content_hash({"probes": sorted(probes)}, "cs")


def check_standard_drift(
    sessions: Sequence[Session],
    *,
    n_baseline: int | None = None,
    instability_ratio: float = 0.5,
) -> CheckStandardDrift | Refusal:
    """Drift of the last session from the baseline, in units of the probe set's own repeatability.

    ``n_baseline`` sessions establish the reference value and the repeatability for each probe. It
    defaults to a third of the sessions, never fewer than `MIN_BASELINE_SESSIONS`, and the instrument
    refuses rather than normalising by a repeatability it could not measure.

    A probe whose baseline standard deviation is zero is *perfectly* repeatable across the baseline,
    which is what a check standard is supposed to be. Dividing by it is still not allowed, so those
    probes are normalised by the median nonzero repeatability and named in ``unstable_probes`` only
    if that substitution matters. When every probe has zero baseline spread the drift is reported in
    raw units alone and the normalised figure is NaN, which is honest: there is no scale to express
    it in.
    """
    if len(sessions) < 2:
        return refuse_incomplete(
            "CheckStandardDrift",
            field="a second measurement of the probe set",
            subject=f"{len(sessions)} session(s)",
            remedy=(
                "Measure the frozen probe set again. Drift is a difference between two occasions "
                "and one occasion carries none, however many probes it holds."
            ),
        )
    common = set(sessions[0].probes)
    for s in sessions[1:]:
        common &= set(s.probes)
    if not common:
        return Refusal(
            instrument="CheckStandardDrift",
            reason=RefusalReason.RECORD_INCOMPLETE,
            detail=(
                f"the {len(sessions)} sessions share no probe. Session 0 carries "
                f"{sorted(sessions[0].probes)} and the last carries {sorted(sessions[-1].probes)}."
            ),
            remedy=(
                "Measure the same probes every session. A check standard whose membership changes "
                "is not a check standard, and a drift computed across a changed membership measures "
                "the membership change."
            ),
        )
    probes = sorted(common)
    matrix = np.array([[float(s.values[p]) for p in probes] for s in sessions], dtype=np.float64)
    n_base = (
        n_baseline if n_baseline is not None else max(MIN_BASELINE_SESSIONS, len(sessions) // 3)
    )
    n_base = min(n_base, len(sessions) - 1)
    if n_base < MIN_BASELINE_SESSIONS:
        return Refusal(
            instrument="CheckStandardDrift",
            reason=RefusalReason.ESS_BELOW_FLOOR,
            detail=(
                f"{len(sessions)} sessions leave {n_base} for the baseline, and a repeatability "
                f"needs at least {MIN_BASELINE_SESSIONS}."
            ),
            remedy=(
                f"Measure the probe set on at least {MIN_BASELINE_SESSIONS + 1} occasions before "
                f"reading a drift, or supply the repeatability from a previous characterisation and "
                f"read the raw difference against it. A drift with no repeatability beside it "
                f"cannot be told from the probe set's own noise."
            ),
            statistics={"n_sessions": len(sessions), "n_baseline": n_base},
        )

    reference = matrix[:n_base].mean(axis=0)
    repeat = matrix[:n_base].std(axis=0, ddof=1)
    nonzero = repeat[repeat > 0]
    fallback = float(np.median(nonzero)) if nonzero.size else float("nan")
    scale = np.where(repeat > 0, repeat, fallback)

    raw = matrix - reference[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        z = raw / scale[None, :]
    drift_by_session = {
        s.label: float(np.sqrt(np.nanmean(z[i] ** 2))) for i, s in enumerate(sessions)
    }
    last = len(sessions) - 1
    # A probe is unstable when its own baseline spread is a large fraction of how far it has moved:
    # it cannot resolve the movement it is supposed to be reporting.
    movement = np.abs(raw[last])
    unstable = tuple(
        p
        for p, r, m in zip(probes, repeat, movement)
        if m > 0 and r > 0 and r / m > instability_ratio
    )
    return CheckStandardDrift(
        drift=float(np.sqrt(np.nanmean(z[last] ** 2))),
        raw_drift=float(np.sqrt(np.mean(raw[last] ** 2))),
        n_probes=len(probes),
        n_sessions=len(sessions),
        n_baseline=int(n_base),
        per_probe_z={p: float(v) for p, v in zip(probes, z[last])},
        per_probe_raw={p: float(v) for p, v in zip(probes, raw[last])},
        repeatability={p: float(v) for p, v in zip(probes, repeat)},
        unstable_probes=unstable,
        fingerprint=probe_set_fingerprint(probes),
        session_label=sessions[last].label,
        max_drift_over_sessions=max(drift_by_session.values()),
        drift_by_session=drift_by_session,
    )


def sessions_from_run(run: Any, *, window: tuple[int, int] | None = None) -> list[Session]:
    """Every step of a record that carries check-standard probes, as a session each.

    Reads `ProbeResult` rows whose ``channel`` is ``"check_standard"``. The record schema already
    separates that channel from ``held_out`` and ``gold``, which is the distinction this instrument
    turns on, so nothing has to be inferred from a probe's name.
    """
    steps = list(run.steps) if window is None else list(run.steps.slice(*window))
    out: list[Session] = []
    for step in steps:
        values = {
            p.name: float(p.value)
            for p in step.probes
            if p.channel == CHECK_STANDARD_CHANNEL and p.value is not None
        }
        if values:
            out.append(Session(label=str(step.index), values=values))
    return out


# ---------------------------------------------------------------------------
# The instrument
# ---------------------------------------------------------------------------

J5_BASELINES: tuple[str, ...] = (
    "baseline.held_out_eval",
    "baseline.assume_no_drift",
)

#: The one instrument in this package that declares no regime precondition, and the reason is
#: structural rather than convenient: it is the instrument that measures one.
CHECK_STANDARD_ENVELOPE = EnvelopeSpec(
    unconditional=True,
    justification=(
        "this is the measurement that establishes STATIONARY_GRADER for everything else, so it "
        "cannot require it. Nothing else in a run can make a repeated measurement of a frozen probe "
        "set wrong: the probe set is fixed by construction and the drift is a difference between "
        "two readings of it."
    ),
)


class CheckStandardDriftInstrument(MonitorInstrument):
    """J5. "The frozen invariant probe set moved 0.04 this session. That is instrument drift."

    Pair it with an isochronous design: measure the check standard at the same points in the
    schedule every session, so a drift cannot be an artefact of when you looked.

    What it cannot do: it cannot say what changed. A moved probe set means the apparatus moved, and
    the candidates are the grader deployment, the tokenizer, the sampling parameters, the judge's
    prompt and the API behind it. Narrowing that down is a separate investigation and this
    instrument's job is to make it start.
    """

    name = "CheckStandardDrift"
    version = "1.0"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "the metrological check standard; ISO 5725 repeatability as the normalising scale"
    deviations = (
        "the repeatability is estimated from the baseline sessions of the same series rather than "
        "from a separate characterisation, so it carries the between-session variation of whatever "
        "the apparatus was doing during the baseline. A drift that began inside the baseline window "
        "is absorbed into the scale and under-reported.",
    )

    quantity = "monitor.check_standard_drift"
    requires = RECORD_ACCESS
    envelope = CHECK_STANDARD_ENVELOPE
    #: What the registry says for this quantity. The check is weak here and it is honest: a drift
    #: computed from scalar probe scores is unchanged by an orthogonal map on any representation,
    #: because no representation enters it.
    invariance = "repr.basis"
    invariance_relation = INVARIANT
    baselines = J5_BASELINES
    rung = 0

    def __init__(
        self,
        sessions: Sequence[Session] | None = None,
        *,
        run: Any = None,
        window: tuple[int, int] | None = None,
        n_baseline: int | None = None,
    ) -> None:
        if sessions is None and run is None:
            raise ValueError("supply either a list of sessions or a run to read them from")
        self.run = run
        self.window = window
        self.n_baseline = n_baseline
        self._sessions = list(sessions) if sessions is not None else None

    def sessions(self) -> list[Session]:
        if self._sessions is not None:
            return self._sessions
        return sessions_from_run(self.run, window=self.window)

    def compute(self, ctx: Context) -> CheckStandardDrift | Refusal:
        sessions = self.sessions()
        if not sessions:
            n_steps = 0
            if self.run is not None:
                n_steps = len(self.run.steps)
            return refuse_incomplete(
                self.name,
                field=f"ProbeResult rows on the {CHECK_STANDARD_CHANNEL!r} channel",
                subject=f"{n_steps} steps of this run",
                remedy=(
                    "Run a frozen probe set every session and write it to the record as "
                    "`ProbeResult(channel='check_standard', name=..., value=...)`. Nothing that can "
                    "be done to this record recovers it, because the measurement was never taken: "
                    "the fix is in the training loop. Until then every instrument declaring the "
                    "STATIONARY_GRADER envelope condition is running with that condition unchecked "
                    "rather than satisfied, which is what `PreflightResult.unchecked` reports."
                ),
                n_steps=n_steps,
            )
        return check_standard_drift(sessions, n_baseline=self.n_baseline)

    def payload(self, computed: CheckStandardDrift) -> dict:
        return {
            "drift": computed.drift,
            "raw_drift": computed.raw_drift,
            "max_drift_over_sessions": computed.max_drift_over_sessions,
            "n_probes": computed.n_probes,
            "n_sessions": computed.n_sessions,
            "n_baseline": computed.n_baseline,
            "session": computed.session_label,
            "worst_probe": computed.worst_probe,
            "per_probe_z": dict(computed.per_probe_z),
            "per_probe_raw": dict(computed.per_probe_raw),
            "repeatability": dict(computed.repeatability),
            "unstable_probes": list(computed.unstable_probes),
            "fingerprint": computed.fingerprint,
            "drift_by_session": dict(computed.drift_by_session),
            "baselines": self.baseline_map(computed),
            "rendered": computed.render(),
        }

    def baseline_map(self, computed: CheckStandardDrift) -> Mapping[str, float]:
        return {
            # Assuming no drift is the baseline everybody actually uses, and its reading is zero.
            "baseline.assume_no_drift": 0.0,
            # The held-out-eval comparator: the largest single-probe movement, which is what a
            # held-out eval would surface, against the probe set's pooled figure. A held-out eval
            # reports the movement; only a check standard reports it in repeatability units.
            "baseline.held_out_eval": max(
                (abs(v) for v in computed.per_probe_raw.values()), default=float("nan")
            ),
        }

    def uncertainty(self, computed: CheckStandardDrift) -> Uncertainty:
        return Uncertainty(
            n=computed.n_sessions,
            method=f"baseline repeatability over {computed.n_baseline} sessions",
        )


__all__ = [
    "CHECK_STANDARD_CHANNEL",
    "CHECK_STANDARD_ENVELOPE",
    "J5_BASELINES",
    "MIN_BASELINE_SESSIONS",
    "CheckStandardDrift",
    "CheckStandardDriftInstrument",
    "Session",
    "check_standard_drift",
    "probe_set_fingerprint",
    "sessions_from_run",
]

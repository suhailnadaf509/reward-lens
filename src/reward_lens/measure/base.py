"""The Instrument protocol, the measurement Context, and the gate-enforcing runner.

An instrument is a functional of a reward signal's internals on structured data (I1). Every one
declares what capability it requires (R3), how its value transforms under the gauge group
(``gauge_status``), and which formal theory object it instantiates (``faithful_to``) with an
explicit list of any departures (``deviations``). Those last two fields are the structural fix
for operationalization drift (liability 2): an instrument that computes a coverage statistic while
claiming Wang-Huang's distortion index must either match the theory object it names or list the
deviation, and the deviation then surfaces on every card that consumes it.

The runner is where gates 1 and 2 are enforced before Evidence is returned, so no downstream code
can bypass them. This is a frozen interface: the whole battery and the index library
compile against it.

**The retype.** 2.0.1's `Observable` declares six things and returns `Evidence`.
`Instrument` declares twelve and returns `Reading`, which is `Evidence | Refusal`. The six new
declarations are the ones the validity engine consults: the quantity being estimated, the access
it needs, the substrates and phases it applies to, the validity envelope, the invariance group, and
the mandatory baselines. The change is additive on purpose. `Observable` stays as the name of the
narrower protocol, every shipped observable satisfies `Instrument` the moment it inherits the new
defaults, and `lint_instrument` reports what is still undeclared rather than failing at import.

That last point is the whole design of the retype. Twenty-nine shipped observables and indices
cannot declare a quantity, an envelope and a baseline set in the same commit that introduces the
fields, so the fields arrive with honest placeholders and a lint function that names every gap. An
instrument that cannot pass lint does not merge; an instrument that has not been retrofitted yet is
visibly unretrofitted rather than silently wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Mapping, Protocol, runtime_checkable

from reward_lens.core.budget import IncrementalValidity, LimitOfDetection
from reward_lens.core.envelope import EnvelopeSpec, RegimeReading
from reward_lens.core.errors import CapabilityError
from reward_lens.core.evidence import Evidence, Uncertainty, make_evidence
from reward_lens.core.gates import CalibrationRef, require_frame_for_comparison
from reward_lens.core.invariance import InvarianceGroupID, Relation
from reward_lens.core.provenance import Cost, Provenance, capture_provenance
from reward_lens.core.quantity import FREE, CostModel, QuantityID
from reward_lens.core.reading import Reading, Refusal, RefusalReason, refuse_access
from reward_lens.core.types import (
    Access,
    AccessMatrix,
    Capability,
    Component,
    EvidenceID,
    FrameID,
    GaugeStatus,
    Phase,
    SubjectRef,
    Substrate,
    TrustLevel,
    missing_access,
)

if TYPE_CHECKING:
    from reward_lens.signals.base import RewardSignal

# ---------------------------------------------------------------------------
# The calibration provider (gate 1's seam)
# ---------------------------------------------------------------------------

# A calibration provider answers "is there a scorecard entry for this observable on this subject's
# signal family and data regime?" It is populated by `organisms.scorecard` once M4 lands; until
# then the default returns None and every ad hoc number is correctly EXPLORATORY. Making this a
# seam (rather than importing organisms into measure) keeps the dependency direction clean.
CalibrationProvider = Callable[[str, SubjectRef, dict], "CalibrationRef | None"]


def _no_calibration(observable: str, subject: SubjectRef, regime: dict) -> "CalibrationRef | None":
    return None


_PROVIDER: CalibrationProvider = _no_calibration


def set_calibration_provider(provider: CalibrationProvider) -> None:
    """Install the calibration provider (organisms.scorecard does this at import)."""
    global _PROVIDER
    _PROVIDER = provider


def lookup_calibration(
    observable: str, subject: SubjectRef, regime: dict | None = None
) -> "CalibrationRef | None":
    return _PROVIDER(observable, subject, regime or {})


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


@dataclass
class Context:
    """Everything an Observable needs to run, plus the machinery to emit gated Evidence.

    ``signal`` is the primary subject; ``others`` names additional signals for cross-signal
    comparisons (an effective-angle Observable puts the second model here). ``view`` is the
    DataView; ``readout`` selects which readout to read; ``frame`` supplies the gauge frame for
    covariant/invariant observables; ``study`` is the frozen StudyID when the run is registered
    (gate 3). ``regime`` describes the data regime for the calibration lookup (gate 1).

    Observables call ``emit`` to build their Evidence; ``emit`` applies gates 1 and 3 centrally so
    an Observable cannot forget them. The runner (`run`) applies the capability check and gate 2.
    """

    #: Optional because `preflight` never touches it. A no-compute capability report has no signal
    #: to name, and requiring one forced every such caller to pass `None  # type: ignore`, which is
    #: a lie the type system was helping to tell.
    signal: "RewardSignal | None" = None
    view: Any = None
    readout: str = "reward"
    others: tuple["RewardSignal", ...] = ()
    frame: FrameID | None = None
    study: str | None = None
    is_comparison: bool = False
    regime: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    _observable: "Observable | None" = None

    # -- what preflight consults. All optional, so every 2.0.1 call site that
    # -- constructs a Context with a signal and a view keeps working unchanged.
    #: What the analyst can reach, per component. Absent means preflight cannot check access and
    #: says so rather than assuming the access is there.
    access: AccessMatrix | None = None
    #: What kind of thing the grader is. Absent means the substrate check is skipped, reported.
    substrate: Substrate | None = None
    #: When the question is being asked.
    phase: Phase | None = None
    #: The envelope, measured. Distinct from ``regime`` above, which is an untyped dict carrying
    #: the data-regime description for the gate-1 calibration lookup and means something else.
    regime_reading: RegimeReading | None = None
    #: The substrate's disagreement with itself, for the below-LOD check.
    lod: LimitOfDetection | None = None
    #: Set when the caller has established a matched positive control, so an instrument whose
    #: claim is a null can tell an underpowered experiment from a real one.
    has_matched_control: bool = False

    def subject(self, extra: dict | None = None) -> SubjectRef:
        """Build the SubjectRef naming the signals, dataset, readout, frame, and interventions."""
        sigs = [s for s in (self.signal, *self.others) if s is not None]
        fingerprints = tuple(s.meta.fingerprint for s in sigs)
        dataset = None
        if self.view is not None:
            dataset = getattr(self.view, "dataset_id", None)
            if dataset is None and hasattr(self.view, "checksum"):
                dataset = self.view.checksum()
        interventions = tuple(getattr(self.signal, "intervention_fingerprints", ()) or ())
        return SubjectRef(
            signals=fingerprints,
            dataset=dataset,
            readout=self.readout,
            frame=self.frame,
            interventions=interventions,
            extra=extra or {},
        )

    def emit(
        self,
        value: Any,
        *,
        uncertainty: Uncertainty | None = None,
        gauge: GaugeStatus | None = None,
        parents: tuple[EvidenceID, ...] = (),
        cost: Cost | None = None,
        subject_extra: dict | None = None,
        reference: Any = None,
        baselines: Mapping[str, float] | None = None,
        incremental: IncrementalValidity | None = None,
    ) -> Evidence:
        """Build a gated Evidence for the current Observable's result.

        Applies gate 1 (looks up a calibration reference for this observable and subject; absent
        means EXPLORATORY) and gate 3 (a frozen study makes it REGISTERED). Gate 2's gauge status
        is taken from the Observable's declaration unless overridden. The trust level falls out of
        `make_evidence`, never set here directly.

        ``quantity`` is read off the Observable rather than passed, because the instrument already
        declares it and a second place to say it is a second place to say it differently. Until this
        forwarded it, every reading in every store carried ``quantity=""`` while its instrument
        declared one, so the unit-mismatch machinery had nothing to key on and a per-token reading
        could be ranked against a per-sequence one without anything noticing.

        ``reference`` is forwarded for the same reason and with a sharper consequence: a reading
        taken against an uncertified reference material is capped at CALIBRATED by `compute_trust`,
        and a cap that the emit path cannot express is a cap that never fires. ``trust`` is hashed
        into the content id, so this is not patchable after a row is written.

        ``incremental`` is the third of the same shape and it was missed when the first two were
        fixed. An incremental-validity record is mandatory on every white-box reading,
        `make_evidence` has taken one since it was written, and nothing could supply it through this
        path, so the mandatory field was unreachable rather than merely unset.
        """
        obs = self._observable
        name = obs.name if obs else "anonymous"
        version = obs.version if obs else "0"
        gauge_status = gauge or (obs.gauge_status if obs else GaugeStatus.INVARIANT)
        subject = self.subject(subject_extra)
        calibration = lookup_calibration(name, subject, self.regime)
        prov = capture_provenance(parents=parents, study=self.study, cost=cost)
        # capture_provenance stamps git sha; merge in the explicit cost if provided.
        if cost is not None:
            prov = Provenance(
                git_sha=prov.git_sha,
                config_hash=prov.config_hash,
                seeds=prov.seeds,
                cost=cost,
                oracle_calls=prov.oracle_calls,
                parents=tuple(parents),
                study=self.study,
                extra=prov.extra,
            )
        return make_evidence(
            observable=name,
            observable_version=version,
            subject=subject,
            value=value,
            uncertainty=uncertainty,
            gauge=gauge_status,
            calibration=calibration,
            provenance=prov,
            registered=self.study is not None,
            quantity=getattr(obs, "quantity", "") or "",
            reference=reference,
            baselines=baselines,
            incremental=incremental,
            lod=self.lod,
            regime=self.regime_reading,
        )


# ---------------------------------------------------------------------------
# Observable protocol + runner
# ---------------------------------------------------------------------------


@runtime_checkable
class Observable(Protocol):
    """A measurement. The narrow protocol, kept.

    ``capabilities`` is what the signal must offer; ``gauge_status`` is how the value transforms
    under the gauge group; ``faithful_to`` names the theory object it instantiates (or None), and
    ``deviations`` lists explicit departures from it. ``measure`` computes the Evidence,
    calling ``ctx.emit`` to build it so gates 1 and 3 are applied centrally.

    The capability field is named ``capabilities`` and not ``requires``, because ``requires`` names
    the access matrix. This Protocol kept the 2.0.1 spelling after the rename landed everywhere
    else, which made `BaseObservable` structurally fail to satisfy the Protocol it is the base for:
    `requires` is an `AccessMatrix` there and a `Capability` here, so every conformant instrument
    that called `run(self, ctx)` was a type error mypy reported and nobody could fix from the
    instrument side.

    Every `Instrument` is an `Observable`. The reverse holds once the six extra declarations are
    filled in, which is what `lint_instrument` checks.
    """

    name: str
    version: str
    capabilities: Capability
    gauge_status: GaugeStatus
    faithful_to: str | None
    deviations: tuple[str, ...]

    def measure(self, ctx: Context) -> Evidence: ...


@dataclass(frozen=True)
class PreflightResult:
    """What an instrument can do here, before it does any of it.

    Separating this from ``estimate`` is what makes the capability report producible with no GPU
    work, and the capability report is the product for most users. So this method computes nothing:
    it resolves access, substrate, phase, envelope, limits and controls, and returns either a
    costed plan or the `Refusal` that would come back.

    ``unchecked`` is the field that keeps this honest. A preflight that could not check access
    because the caller supplied no access matrix has not established that access is sufficient, and
    saying so is different from passing.
    """

    instrument: str
    ok: bool
    refusal: Refusal | None = None
    rung: int = 0
    cost: CostModel = FREE
    expected_uncertainty: float | None = None
    regime: RegimeReading | None = None
    #: Checks that were skipped for want of an input, named. Never silently treated as passes.
    unchecked: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    #: Set when the envelope declared `on_violation="bound"` and the violation fired, so the
    #: caller knows the reading it is about to get is a bound rather than a point estimate.
    bound_estimator: str | None = None
    #: Set when the envelope declared `on_violation="downgrade"`. The quantity stays defined and
    #: its trust cannot exceed this.
    trust_cap: "TrustLevel | None" = None

    def render(self) -> str:
        if not self.ok and self.refusal is not None:
            return self.refusal.render()
        head = f"{self.instrument}  available (rung {self.rung}, {self.cost.render()})"
        if self.expected_uncertainty is not None:
            head += f", expected uncertainty {self.expected_uncertainty:.4g}"
        if self.unchecked:
            head += f"\n    not checked: {', '.join(self.unchecked)}"
        return head


@runtime_checkable
class Instrument(Protocol):
    """The twelve declarations and the two methods every instrument implements.

    The six beyond `Observable` are what the validity engine consults, and each closes a specific
    way of producing a confident wrong number:

    ``quantity`` is what is being estimated, so two rungs of a ladder are comparable and a
    per-token reading cannot be ranked against a per-sequence one. ``requires`` is the access
    matrix, so an instrument that cannot run says what would let it. ``substrates`` and ``phases``
    stop a program being asked for its activations and a finished artifact being asked an in-run
    question. ``envelope`` is the set of preconditions that fail *quietly*, which is the only
    failure access cannot see. ``invariance`` is the restriction class, without which every causal
    claim is vacuous, and it generates a property test the instrument did not write. ``baselines``
    is mandatory because a claim with no dumb baseline is not a claim: one published probe reported
    AUC 0.998 on a task a zero-parameter string match solves outright.
    """

    name: str
    version: str
    quantity: QuantityID
    requires: AccessMatrix
    substrates: frozenset[Substrate]
    phases: frozenset[Phase]
    envelope: EnvelopeSpec
    invariance: str
    baselines: tuple[str, ...]
    rung: int
    faithful_to: str | None
    deviations: tuple[str, ...]

    def preflight(self, ctx: Context) -> PreflightResult: ...

    def estimate(self, ctx: Context) -> Reading: ...


def declared_capabilities(inst: Any) -> Capability:
    """What the signal must offer.

    `requires` names the access matrix, and 2.0.1 gave the same name to the capability flags. Both
    concepts survive, so one had to move: `requires` is the access matrix and the capability
    declaration is `capabilities`.

    This began as a shim reading both spellings through the retrofit. The retrofit is closed,
    so it **raises** on the old spelling instead. Deleting it outright was the other option and it
    is worse: with no check, a `requires` holding a `Capability` would be ignored and the inherited
    `capabilities` default read in its place, which is a capability gate silently removed. That is
    the exact failure the first version of this function shipped, on four live observables, and a
    guard that turns it into an error is cheaper than the test that caught it.
    """
    legacy = getattr(inst, "requires", None)
    if isinstance(legacy, Capability):
        raise CapabilityError(
            f"{getattr(inst, 'name', type(inst).__name__)!r} declares `requires` as a Capability. "
            f"`requires` names the access matrix; the capability flags are now "
            f"`capabilities`. Rename the declaration. This raises rather than falling back, "
            f"because falling back would read the inherited `capabilities` default instead and "
            f"silently drop the gate this declaration exists to set."
        )
    caps = getattr(inst, "capabilities", None)
    return caps if isinstance(caps, Capability) else Capability.NONE


def declared_access(inst: Any) -> AccessMatrix:
    """What the analyst must be able to reach, per component: the `requires` access matrix."""
    req = getattr(inst, "requires", None)
    return req if isinstance(req, Mapping) else {}


def capability_name(caps: Capability) -> str:
    """Name a capability set the same way on every interpreter.

    `caps.name` is not enough on its own, for two reasons that pull in opposite directions. Before
    3.11 a composite `Flag` has no `name` at all, so it needs a fallback; from 3.11 on it has one,
    so a refusal recorded under two interpreters would carry two different strings for the same
    capabilities. A record that reads differently depending on the Python that wrote it is the kind
    of difference that surfaces much later as a diff nobody can account for.

    The fallback this replaces was `str(int(caps))`, which could never have worked: `Capability` is
    a `Flag` and not an `IntFlag`, so it has no `__int__`. On 3.11 and up `name` is always truthy
    and `or` never evaluated its right-hand side, which is how a call that raises on every input
    reached 3.0 with only the 3.10 leg of the matrix failing, and only on a composite.

    Joining the declared members reproduces exactly what 3.12 puts in `name`, so the fix converges
    the two versions rather than giving 3.10 a second spelling of its own.
    """
    if not caps:
        return Capability.NONE.name or "NONE"
    return "|".join(m.name or "" for m in Capability.__members__.values() if m.value and m in caps)


def run(observable: Observable, ctx: Context) -> Evidence:
    """Run an Observable under the gates.

    Enforces R3 (the signal must declare the required capability) and gate 2 (a covariant
    cross-signal comparison requires a frame) before delegating to ``measure``. Gates 1 and 3 are
    applied inside ``ctx.emit``. The result is a fully gated Evidence; there is no path that
    returns an ungated number.
    """
    needed = declared_capabilities(observable)
    if ctx.signal is None:
        # A PROGRAM-substrate instrument has no network to point at, and forcing one into
        # `RewardSignal` is what previous versions got wrong. An instrument that needs no
        # capability runs; one that needs a capability from a signal that is not there is refused
        # rather than crashed.
        if needed and needed != Capability.NONE:
            raise CapabilityError(
                f"observable '{observable.name}' requires {needed!r} and the context carries no "
                f"signal to satisfy it. A PROGRAM-substrate instrument should declare "
                f"Capability.NONE and read its subject through the access matrix instead."
            )
        ctx._observable = observable
        try:
            return observable.measure(ctx)
        finally:
            ctx._observable = None
    missing = needed.missing_from(ctx.signal.caps)
    if missing and missing != Capability.NONE:
        raise CapabilityError(
            f"observable '{observable.name}' requires {needed!r} but signal "
            f"{ctx.signal.meta.fingerprint} declares {ctx.signal.caps!r}; missing {missing!r}"
        )
    if ctx.is_comparison:
        require_frame_for_comparison(observable.gauge_status, ctx.frame)
    ctx._observable = observable
    try:
        return observable.measure(ctx)
    finally:
        ctx._observable = None


#: What an undeclared envelope is, before one is filled in. It is not `UNCONDITIONAL`: claiming
#: an instrument holds in every regime is a positive claim, and "nobody has looked yet" is a
#: different thing that `lint_instrument` reports by name.
UNDECLARED_ENVELOPE: EnvelopeSpec | None = None


class BaseObservable:
    """A convenience base that sets the declaration fields as class attributes.

    Subclasses override the class attributes and implement ``measure``; this saves every
    instrument from restating the protocol fields. It is deliberately a plain class, not a
    dataclass: a dataclass ``__init__`` would overwrite a subclass's class-attribute overrides with
    the base defaults, so overriding ``requires``/``gauge_status`` in the subclass body would
    silently not take effect. Any object satisfying the protocol works; the battery and the index
    library use this base for uniformity.

    The six extra declarations arrive here with placeholders rather than plausible defaults,
    because a plausible default is indistinguishable from a decision. ``quantity`` is empty rather
    than guessed, ``envelope`` is None rather than unconditional, and ``baselines`` is empty rather
    than populated with something that sounds reasonable. `lint_instrument` names each gap.
    """

    name: str = "observable"
    version: str = "1.0"
    #: What the *signal* must offer. 2.0.1 called this `requires`; that name now belongs to the
    #: access matrix instead, so the capability declaration is renamed. See `declared_capabilities`.
    capabilities: Capability = Capability.SCORES
    gauge_status: GaugeStatus = GaugeStatus.INVARIANT
    faithful_to: str | None = None
    deviations: tuple[str, ...] = ()

    # -- the six extra declarations, undeclared until an instrument fills them in
    quantity: QuantityID = ""
    #: What the *analyst* must be able to reach, per component: the access matrix.
    requires: AccessMatrix = {}
    substrates: frozenset[Substrate] = frozenset()
    phases: frozenset[Phase] = frozenset()
    envelope: EnvelopeSpec | None = UNDECLARED_ENVELOPE
    invariance: str = ""
    #: How this instrument's value transforms under the group it declares. A single `Relation`
    #: applies to every group the instrument is checked under; a mapping from group id to `Relation`
    #: says one thing per group, which an instrument genuinely needs when it transforms two ways.
    #: `chi` is the motivating case and `core.invariance.resolve_relation` documents it: `Cov(f, r)`
    #: is covariant with weight 1 under `reward.affine` and invariant under `repr.basis`, and a
    #: single relation cannot say both.
    #:
    #: The mapping form was supported and documented by `resolve_relation` from the start and this
    #: annotation forbade it, so the form the kernel implements was the form the type rejected.
    #: Three shipped instruments recorded a second true, checkable relation in a comment instead of
    #: declaring it, and each of those is a generated invariance test that never ran.
    invariance_relation: Relation | Mapping[InvarianceGroupID, Relation] | None = None
    baselines: tuple[str, ...] = ()
    rung: int = 0

    def measure(self, ctx: Context) -> Evidence:  # pragma: no cover - abstract
        raise NotImplementedError

    # -- the two methods ---------------------------------------------------

    def preflight(self, ctx: Context) -> PreflightResult:
        """Access, substrate, phase, envelope and limits, with no compute.

        The order is by how cheap the refusal is to act on. Access first, because "supply the
        policy checkpoint" is the most actionable remedy there is; substrate and phase next,
        because those are category errors; the envelope last among the hard checks, because it is
        the only one whose failure needs a measurement to establish.

        Every check that could not run is named in ``unchecked`` rather than counted as a pass.
        """
        unchecked: list[str] = []
        notes: list[str] = []
        bound_used: str | None = None
        trust_cap: TrustLevel | None = None

        wanted = declared_access(self)
        if ctx.access is None:
            unchecked.append("access")
        elif wanted:
            gap = missing_access(ctx.access, wanted)
            if gap:
                return PreflightResult(
                    instrument=self.name,
                    ok=False,
                    refusal=refuse_access(
                        self.name,
                        needs={c.name: a.name for c, a in gap.items()},
                        have=", ".join(
                            f"{c.name}: {a.name}"
                            for c, a in sorted(ctx.access.items(), key=lambda kv: kv[0].name)
                        )
                        or "nothing",
                        remedy=self._access_remedy(gap),
                    ),
                )

        if ctx.substrate is None:
            unchecked.append("substrate")
        elif self.substrates and ctx.substrate not in self.substrates:
            return PreflightResult(
                instrument=self.name,
                ok=False,
                refusal=Refusal(
                    instrument=self.name,
                    reason=RefusalReason.SUBSTRATE_MISMATCH,
                    detail=(
                        f"this instrument applies to "
                        f"{', '.join(sorted(s.name for s in self.substrates))}; the grader is "
                        f"{ctx.substrate.name}"
                    ),
                    remedy=(
                        f"use an instrument declared for {ctx.substrate.name}. A "
                        f"{ctx.substrate.name} grader is a different kind of object, not a harder "
                        f"case of the same one."
                    ),
                ),
            )

        if ctx.phase is None:
            unchecked.append("phase")
        elif self.phases and ctx.phase not in self.phases:
            return PreflightResult(
                instrument=self.name,
                ok=False,
                refusal=Refusal(
                    instrument=self.name,
                    reason=RefusalReason.PHASE_MISMATCH,
                    detail=(
                        f"this instrument answers a "
                        f"{'/'.join(sorted(p.name for p in self.phases))} question; you are at "
                        f"{ctx.phase.name}"
                    ),
                    remedy=(
                        "re-run this measurement during the phase it applies to, or read the "
                        "record-only instrument for the same quantity if one is registered."
                    ),
                ),
            )

        if self.envelope is None:
            unchecked.append("envelope (none declared)")
        elif self.envelope.requires and ctx.regime_reading is None:
            # No regime was supplied at all, which is not the same as a regime measured and found
            # indeterminate. The second case reaches `admits` below and refuses, because "unknown
            # is not a pass". This case means nobody attempted the measurement, so the honest
            # report is that the check did not run. What makes that safe rather than a loophole is
            # that `unchecked` travels onto the reading, so a number produced without its envelope
            # checked says so wherever it is read.
            unchecked.append("envelope (regime not measured)")
        elif not self.envelope.admits(ctx.regime_reading):
            violations = self.envelope.violations(ctx.regime_reading)
            first = violations[0] if violations else None
            stats = (
                {
                    "condition": first.condition.name,
                    "statistic": first.statistic,
                    "threshold": first.threshold,
                }
                if first is not None
                else {}
            )
            named = ", ".join(v.condition.name for v in violations)

            # The three violation behaviours are not interchangeable, and collapsing
            # them to `refuse` throws away the two cases where an answer still exists.
            if self.envelope.on_violation == "bound":
                notes.append(
                    f"outside {named}: falling back to the bound estimator "
                    f"{self.envelope.bound_estimator!r}, which survives here. The reading is a "
                    f"bound, not a point estimate."
                )
                bound_used = self.envelope.bound_estimator
            elif self.envelope.on_violation == "downgrade":
                # The quantity stays defined and its trust drops. The worked case: a
                # before/after comparison outside STATIONARY_GRADER is still computable and is now
                # EXPLORATORY rather than REGISTERED, with the condition recorded.
                notes.append(
                    f"outside {named}: the quantity is still defined and its trust is capped at "
                    f"EXPLORATORY. The violated condition is recorded on the reading."
                )
                trust_cap = TrustLevel.EXPLORATORY
            else:
                return PreflightResult(
                    instrument=self.name,
                    ok=False,
                    regime=ctx.regime_reading,
                    refusal=Refusal(
                        instrument=self.name,
                        reason=RefusalReason.ENVELOPE_VIOLATED,
                        detail="; ".join(v.render().strip() for v in violations),
                        remedy=(
                            "restrict the window to a span where the condition holds. An "
                            "instrument that is available and invalid is worse than one that is "
                            "unavailable."
                        ),
                        statistics=stats,
                    ),
                )

        if ctx.lod is None:
            unchecked.append("limit of detection")

        return PreflightResult(
            instrument=self.name,
            ok=True,
            rung=self.rung,
            regime=ctx.regime_reading,
            unchecked=tuple(unchecked),
            notes=tuple(notes),
            bound_estimator=bound_used,
            trust_cap=trust_cap,
        )

    def _access_remedy(self, gap: dict[Component, Access]) -> str:
        """A remedy is an instruction. "Access insufficient" is not one."""
        parts = []
        for component, access in sorted(gap.items(), key=lambda kv: kv[0].name):
            for name in (a.name for a in Access if a & access and a.name):
                parts.append(f"{component.name.lower()} at {name}")
        joined = "; ".join(parts) or "more access"
        return (
            f"supply {joined}. If you cannot, ask for the same quantity at a lower rung: "
            f"`reward-lens capabilities` prints which rungs your access reaches and what each costs."
        )

    def estimate(self, ctx: Context) -> Reading:
        """Evidence or Refusal. Never a bare float, never a silent degradation.

        The default runs `preflight` and returns its refusal if there is one, then delegates to
        ``measure``. An instrument with a cheaper estimator that survives outside its envelope
        overrides this and returns the bound rather than the refusal, which is the `partial` field
        on `Refusal` and the `on_violation="bound"` branch of the envelope.

        The capability gate is translated rather than propagated, and the asymmetry is deliberate.
        `run` is the 2.0.1 ``Observable`` entry point, it is typed ``-> Evidence``, and it raises
        `CapabilityError` when the signal does not offer what the instrument declared. That is the
        right shape for a caller who asked for a measurement and made a programming error. It is the
        wrong shape here, because this method is typed as returning a `Reading` and a refusal is a
        value with a remedy and never an exception, and a missing capability is exactly an
        anticipated condition: it is the commonest thing a capability report exists to tell you
        about in advance.
        """
        pre = self.preflight(ctx)
        if not pre.ok and pre.refusal is not None:
            return pre.refusal
        try:
            return run(self, ctx)
        except CapabilityError as exc:
            needed = declared_capabilities(self)
            have = ctx.signal.caps if ctx.signal is not None else Capability.NONE
            missing = needed.missing_from(have)
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail=str(exc),
                remedy=self._capability_remedy(missing, has_signal=ctx.signal is not None),
                statistics={
                    "needed": capability_name(needed),
                    "have": capability_name(have),
                    "missing": capability_name(missing),
                },
            )

    @staticmethod
    def _capability_remedy(missing: Capability, *, has_signal: bool) -> str:
        """What to supply, as an instruction rather than as a restatement of the failure.

        The two cases want different advice and the difference is not the missing flags, which are
        the same either way. With no signal at all the likely error is a declaration rather than a
        setup: a PROGRAM-substrate instrument has no network to point at, and forcing one into
        `RewardSignal` is what previous versions got wrong.
        """
        names = ", ".join(
            sorted(c.name for c in Capability if c is not Capability.NONE and c & missing)
        )
        if not has_signal:
            return (
                f"Supply a signal offering {names}. If this instrument reads a program rather than "
                f"a network, declare Capability.NONE instead and read its subject through the "
                f"access matrix, because a program has no activations to offer."
            )
        return (
            f"Supply a signal offering {names}, or ask for the same quantity at a lower rung: "
            f"`reward-lens capabilities` prints which rungs your access reaches and what each costs."
        )


@dataclass(frozen=True)
class InstrumentLintFinding:
    """One undeclared or unenforceable declaration, with what closes it."""

    instrument: str
    field: str
    problem: str
    remedy: str

    def render(self) -> str:
        return f"{self.instrument}.{self.field}: {self.problem}  ->  {self.remedy}"


def lint_instrument(inst: Any) -> list[InstrumentLintFinding]:
    """The instrument lint rules, as findings rather than as an import-time failure.

    Four rules. An instrument whose quantity is not registered fails; an instrument with an empty
    ``baselines`` tuple fails; an instrument with no envelope fails; an instrument with no
    invariance group fails. The envelope's own two rules (an empty ``requires`` without an explicit
    justification, and a condition absent from ``measured_by``) are enforced in
    `EnvelopeSpec.__post_init__`, so an unenforceable envelope cannot be constructed at all and
    does not need checking here.

    Returning findings rather than raising is what makes the retype possible in one commit: the
    twenty-nine shipped observables are visibly unretrofitted rather than silently wrong, and a test
    asserting an empty list is what closes the gap.
    """
    from reward_lens.core.quantity import QUANTITIES

    name = getattr(inst, "name", type(inst).__name__)
    out: list[InstrumentLintFinding] = []

    quantity = getattr(inst, "quantity", "")
    if not quantity:
        out.append(
            InstrumentLintFinding(
                name,
                "quantity",
                "declares no quantity, so two rungs of its ladder cannot be compared",
                "set `quantity` to a registered id from spec/QUANTITIES.yaml",
            )
        )
    elif quantity not in QUANTITIES:
        out.append(
            InstrumentLintFinding(
                name,
                "quantity",
                f"declares {quantity!r}, which is not a registered quantity",
                "register the quantity in spec/QUANTITIES.yaml, or fix the id",
            )
        )

    if not getattr(inst, "baselines", ()):
        out.append(
            InstrumentLintFinding(
                name,
                "baselines",
                "declares no baseline, and a claim with no dumb baseline is not a claim",
                "name at least one baseline from stats/baselines/; string match and length are the "
                "usual starting pair",
            )
        )

    if getattr(inst, "envelope", None) is None:
        out.append(
            InstrumentLintFinding(
                name,
                "envelope",
                "declares no validity envelope, so nothing checks whether its assumptions hold",
                "declare an EnvelopeSpec, or pass unconditional=True with a justification if it "
                "genuinely holds in every regime",
            )
        )

    if not getattr(inst, "invariance", ""):
        out.append(
            InstrumentLintFinding(
                name,
                "invariance",
                "declares no invariance group, so it gets no generated property test",
                "name one of the seven invariance groups, or `none` if no group acts on it. "
                "`none` is an answer; a blank is not",
            )
        )

    return out


#: What makes a reading white-box: it opened the network. Reading activations, gradients, or their
#: second order are the three, and any one of them puts the incremental-validity obligation on the
#: reading.
WHITE_BOX = Capability.ACTIVATIONS | Capability.GRADIENTS | Capability.HVP


def is_white_box(inst: Any) -> bool:
    """Whether this instrument's declared capabilities mean it opens the network."""
    return bool(declared_capabilities(inst) & WHITE_BOX)


def lint_reading(reading: Any, instrument: Any) -> list[InstrumentLintFinding]:
    """Lint rule four: a white-box reading with no `IncrementalValidity` fails.

    This is the fourth of the lint rules and it is the last one to become enforceable, for a reason
    worth recording: it is a rule about a *reading*, not about an instrument, so it cannot be a
    fifth check inside `lint_instrument`, which only ever sees the declaration. And until `policy/`
    landed there was no white-box reading anywhere in the library to enforce it against, so it
    stayed implementable and unenforced.

    Why it is a rule at all: the bar for a white-box method is **decorrelation plus signal, not
    superiority**. A method ten points worse than the best black-box baseline and uncorrelated with
    it is more valuable than one two points better and redundant, and only the four numbers on an
    `IncrementalValidity` can say which you have. A white-box reading that does not carry one is
    asking to be believed because it looked inside, which is the one argument this library does not
    accept.

    A `Refusal` is exempt. A measurement that declined to happen has nothing to be incremental over,
    and demanding a baseline comparison from it would make refusing more expensive than reporting,
    which is backwards.
    """
    name = getattr(instrument, "name", type(instrument).__name__)
    if getattr(reading, "reason", None) is not None:  # a Refusal carries one; Evidence does not
        return []
    if not is_white_box(instrument):
        return []
    if getattr(reading, "incremental", None) is None:
        return [
            InstrumentLintFinding(
                name,
                "incremental",
                "is a white-box instrument and its reading carries no IncrementalValidity, so "
                "nothing records what opening the network bought over the black-box bank",
                "run stats.baselines.run_bank on the same items, hand the per-item margins to "
                "measure.meta.incremental.IncrementalValidityReading, and pass its record to "
                "ctx.emit(incremental=...)",
            )
        ]
    return []


__all__ = [
    "UNDECLARED_ENVELOPE",
    "WHITE_BOX",
    "BaseObservable",
    "CalibrationProvider",
    "Context",
    "is_white_box",
    "lint_reading",
    "capability_name",
    "declared_access",
    "declared_capabilities",
    "Instrument",
    "InstrumentLintFinding",
    "Observable",
    "PreflightResult",
    "lint_instrument",
    "lookup_calibration",
    "run",
    "set_calibration_provider",
]

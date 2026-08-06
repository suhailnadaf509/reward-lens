"""Invariance groups, because causal abstraction is vacuous without a restriction class.

Sutter et al. (2507.08802) map randomly initialised language models to the IOI circuit with perfect
accuracy. Causal abstraction, the framework under activation patching, causal scrubbing, DAS,
concept erasure and sparse dictionaries alike, says nothing at all unless the class of admissible
reparameterisations is stated. Three literatures found that independently and none of them adopted
a fix. A gauge is the fix, and this module is the gauge made executable.

What it buys, concretely: every registered instrument gets one property test it did not write, and
an instrument that never thought about the question does not merge. That is the single lint rule
that keeps the whole catalogue answering the question one way rather than one dialect per module.

Three things here are easy to get wrong, so they are typed rather than documented.

**A group does not have a status; an instrument's relation to a group does.** `repr.basis` admits
all three of invariant, covariant and raw_only, and which one applies is a property of the
instrument. Putting `status` on `InvarianceGroup` cannot express that, so the status lives on
`Relation` and the group declares which relations its assertion admits.

**A covariant instrument does not return the same value.** It scales by a stated power of the
group parameter, so the check is `v' == a**weight * v`, not `v' == v`. An implementation that only
knows how to assert equality silently forces every covariant instrument to declare itself
invariant, and then passes.

**A failure is not always a defect.** `group.permutation` failing means the instrument is sensitive
to rollout order, and for a judge that is position bias, measured. The report says so rather than
reading as a bug, because the test detecting position bias is the test working.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from random import Random
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np

InvarianceGroupID = str

#: What a deliberate answer of "no group acts on this" resolves to. Distinct from a missing
#: declaration, which is a lint failure.
TRIVIAL_GROUP: InvarianceGroupID = "trivial"

Status = Literal["invariant", "covariant", "raw_only"]


# ---------------------------------------------------------------------------
# What a group acts on
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InvariancePayload:
    """The transformable state a group acts on, in one vocabulary all seven groups share.

    A generated test has to hand every instrument the same kind of object, so the seven groups need
    a common surface to act through. This is it: the fields the seven groups name,
    each optional, so an instrument that reads only scores is not obliged to manufacture
    activations it never looks at.

    Instruments adapt their own context to this and back. That adaptation is the cost of having a
    generated test at all, and it is small: a scores-only instrument is two lines.
    """

    #: (n,) rewards or scores, one per rollout.
    scores: np.ndarray | None = None
    #: (n,) integer group label per rollout, so a within-group transform knows the partition.
    group_ids: np.ndarray | None = None
    #: (n, d) activations.
    activations: np.ndarray | None = None
    #: (k, d) readout or decoder directions, transformed with the activations under `repr.basis`.
    readouts: np.ndarray | None = None
    #: (p,) flat parameter vector, for `policy.reparam`.
    parameters: np.ndarray | None = None
    #: token ids per rollout, for `tokenization`.
    tokens: Sequence[Sequence[int]] | None = None
    #: the unit the reading is expressed in, for `units`.
    unit: Any = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def replace(self, **changes: Any) -> "InvariancePayload":
        return replace(self, **changes)

    def require(self, *fields: str) -> None:
        """Raise if the payload lacks what a group needs, rather than transforming a None."""
        missing = [f for f in fields if getattr(self, f, None) is None]
        if missing:
            raise ValueError(
                f"this group acts on {', '.join(missing)}, which the payload does not carry. "
                f"Supply it, or declare the instrument under a group that acts on what it reads."
            )


# ---------------------------------------------------------------------------
# Group elements and groups
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroupAction:
    """One element of a group, drawn, with the parameters it was drawn with.

    ``params`` is not bookkeeping. The covariant check needs the drawn scale to assert
    ``v' == a**weight * v``, and a failing report has to be able to say which draw broke it, so a
    reproducer is a dict rather than a rerun.

    ``sample`` is the generator: it draws a sibling from the same family. It lives on this type,
    so a generator is a canonical member of its own family and sampling it yields another.
    """

    name: str
    apply: Callable[[InvariancePayload], InvariancePayload]
    params: Mapping[str, float] = field(default_factory=dict)
    sample: Callable[[Random], "GroupAction"] | None = None

    def draw(self, rng: Random) -> "GroupAction":
        return self.sample(rng) if self.sample is not None else self


@dataclass(frozen=True)
class InvarianceGroup:
    """A restriction class, with samplable generators and the assertion it licenses.

    ``admits`` is the set of relations this group's assertion offers, which
    for three of the seven is more than one. An instrument declaring a relation the group does not
    admit is a lint failure, and that is a real check: declaring `raw_only` under
    `group.permutation` would be a way of opting out of a test that exists to detect position bias.
    """

    id: InvarianceGroupID
    generators: tuple[GroupAction, ...]
    acts_on: str
    admits: frozenset[str] = frozenset({"invariant"})
    assertion: str = ""
    #: Set when the group's assertion is a refusal rather than a numeric relation, as `units` is.
    refusal_only: bool = False

    def draw(self, rng: Random) -> GroupAction:
        """One element: draw from every generator and compose them in declaration order."""
        drawn = [g.draw(rng) for g in self.generators]
        if len(drawn) == 1:
            return drawn[0]
        params: dict[str, float] = {}
        for d in drawn:
            params.update(d.params)

        def _apply(p: InvariancePayload) -> InvariancePayload:
            for d in drawn:
                p = d.apply(p)
            return p

        return GroupAction(
            name="∘".join(d.name for d in drawn),
            apply=_apply,
            params=params,
        )


@dataclass(frozen=True)
class Relation:
    """How an instrument's reading transforms under a group. The instrument declares this.

    ``weight`` is the power of the group parameter a covariant reading scales by: a reading in
    reward units has weight 1 under `reward.affine`, a variance has weight 2, an invariant has
    weight 0. Getting it wrong is the failure this type exists to make visible, because a covariant
    instrument that declares itself invariant fails its generated test loudly instead of shipping a
    coordinate artifact.
    """

    status: Status = "invariant"
    weight: float = 0.0
    parameter: str = "a"

    def __post_init__(self) -> None:
        if self.status == "invariant" and self.weight:
            raise ValueError(
                "an invariant reading does not scale, so weight must be 0. A reading that scales "
                "by a power of the group parameter is covariant, and saying so is the point."
            )


INVARIANT = Relation("invariant")
COVARIANT_LINEAR = Relation("covariant", weight=1.0)
RAW_ONLY = Relation("raw_only")


# ---------------------------------------------------------------------------
# The seven groups
# ---------------------------------------------------------------------------


def _affine() -> InvarianceGroup:
    """`r → a·r + b`, with `a ~ LogUniform(0.1, 10)` and `b ~ N(0, 1)`."""

    def make(a: float, b: float) -> GroupAction:
        def apply(p: InvariancePayload) -> InvariancePayload:
            p.require("scores")
            return p.replace(scores=a * np.asarray(p.scores, dtype=np.float64) + b)

        return GroupAction(
            name=f"r → {a:.4g}·r + {b:+.4g}",
            apply=apply,
            params={"a": a, "b": b},
            sample=lambda rng: make(
                math.exp(rng.uniform(math.log(0.1), math.log(10.0))), rng.gauss(0.0, 1.0)
            ),
        )

    return InvarianceGroup(
        id="reward.affine",
        generators=(make(1.0, 0.0),),
        acts_on="scores",
        admits=frozenset({"invariant", "covariant"}),
        assertion="invariant: value unchanged to tol. covariant: value scales by a stated power of a.",
    )


def _null() -> InvarianceGroup:
    """Add any per-prompt constant within a group: advantages, and everything downstream, unchanged.

    This is the group that makes the centred advantage the object of study rather than the reward.
    A constant that is constant *within a group* cancels in every group-relative statistic, so an
    instrument that moves under it is reading a level it should not be reading.
    """

    def make(sigma: float) -> GroupAction:
        def apply(p: InvariancePayload) -> InvariancePayload:
            p.require("scores", "group_ids")
            scores = np.asarray(p.scores, dtype=np.float64)
            gids = np.asarray(p.group_ids)
            # One draw per distinct group, added to every member of that group.
            rng = np.random.default_rng(abs(hash((sigma, gids.tobytes()))) % (2**32))
            shifted = scores.copy()
            for g in np.unique(gids):
                shifted[gids == g] += rng.normal(0.0, sigma)
            return p.replace(scores=shifted)

        return GroupAction(
            name=f"r → r + g(prompt), σ={sigma:.4g}",
            apply=apply,
            params={"sigma": sigma},
            sample=lambda rng: make(math.exp(rng.uniform(math.log(0.1), math.log(10.0)))),
        )

    return InvarianceGroup(
        id="reward.null",
        generators=(make(1.0),),
        acts_on="scores",
        admits=frozenset({"invariant"}),
        assertion="advantages unchanged, and every quantity downstream of advantages unchanged",
    )


def _basis() -> InvarianceGroup:
    """`Q ~ Haar(O(d))`, activations `h → Qh`, readouts `w → Qw`.

    The readout transform is written `w → Qw` in one convention and `W → WQᵀ` in the other. Those
    are the same map under the two conventions for whether directions are rows or columns, and
    neither statement says which it means. Rows here, so activations `(n, d)` go to `h Qᵀ` and
    readouts `(k, d)` go to `w Qᵀ`, which keeps every inner product `h·w` exactly invariant.
    """

    def make(seed: int) -> GroupAction:
        def apply(p: InvariancePayload) -> InvariancePayload:
            p.require("activations")
            h = np.asarray(p.activations, dtype=np.float64)
            d = h.shape[-1]
            q, r = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, d)))
            # Sign-fix the QR so the draw is Haar rather than QR-biased.
            q = q * np.sign(np.diag(r))
            out = p.replace(activations=h @ q.T)
            if p.readouts is not None:
                out = out.replace(readouts=np.asarray(p.readouts, dtype=np.float64) @ q.T)
            return out

        return GroupAction(
            name=f"h → Qh, Q ~ Haar(O(d)) seed={seed}",
            apply=apply,
            params={"seed": float(seed)},
            sample=lambda rng: make(rng.randrange(2**31)),
        )

    return InvarianceGroup(
        id="repr.basis",
        generators=(make(0),),
        acts_on="activations",
        admits=frozenset({"invariant", "covariant", "raw_only"}),
        assertion=(
            "invariant: unchanged. covariant: requires a shared Frame or the comparison raises "
            "GaugeError. raw_only: refuses to be compared across models at all."
        ),
    )


def _reparam() -> InvarianceGroup:
    """Any smooth invertible `φ: θ → θ'` with `J_φ` sampled near identity.

    Fisher-metric quantities (KL, G, h², efficiency) are unchanged; `‖Δθ‖` and per-parameter norms
    are not, and must be declared `raw_only`. Near identity because a reparameterisation far from
    it stops being a coordinate change of the same model in any useful numerical sense.
    """

    def make(seed: int, scale: float) -> GroupAction:
        def apply(p: InvariancePayload) -> InvariancePayload:
            p.require("parameters")
            theta = np.asarray(p.parameters, dtype=np.float64)
            n = theta.shape[0]
            rng = np.random.default_rng(seed)
            j = np.eye(n) + scale * rng.standard_normal((n, n)) / max(math.sqrt(n), 1.0)
            return p.replace(parameters=j @ theta)

        return GroupAction(
            name=f"θ → φ(θ), ‖J−I‖~{scale:.3g}",
            apply=apply,
            params={"seed": float(seed), "scale": scale},
            sample=lambda rng: make(rng.randrange(2**31), rng.uniform(1e-3, 5e-2)),
        )

    return InvarianceGroup(
        id="policy.reparam",
        generators=(make(0, 1e-2),),
        acts_on="parameters",
        admits=frozenset({"invariant", "raw_only"}),
        assertion=(
            "Fisher-metric quantities (KL, G, h², efficiency) unchanged; ‖Δθ‖ and per-parameter "
            "norms are not and must be declared raw_only"
        ),
    )


def _tokenization() -> InvarianceGroup:
    """Re-tokenise the same string with a different but equivalent tokeniser.

    There is no natural sampler for this generator, because the transform is not a distribution:
    it is "use a different tokeniser that decodes to the same string". The samplable stand-in here
    is a merge-and-split perturbation of the id sequence that preserves length-weighted content,
    which is enough to catch a per-token quantity that never declared a normalisation. An
    instrument that needs the real thing supplies its own generator; this one is the default and
    says so.
    """

    def make(seed: int) -> GroupAction:
        def apply(p: InvariancePayload) -> InvariancePayload:
            p.require("tokens")
            rng = np.random.default_rng(seed)
            out = []
            for seq in p.tokens or []:
                s = list(seq)
                if len(s) > 2:
                    # Split one token into two, which is what a finer tokeniser does.
                    i = int(rng.integers(0, len(s)))
                    s = s[:i] + [s[i], s[i]] + s[i + 1 :]
                out.append(s)
            return p.replace(tokens=out)

        return GroupAction(
            name=f"re-tokenise (split), seed={seed}",
            apply=apply,
            params={"seed": float(seed)},
            sample=lambda rng: make(rng.randrange(2**31)),
        )

    return InvarianceGroup(
        id="tokenization",
        generators=(make(0),),
        acts_on="tokens",
        # The assertion offers "be invariant under it, or refuse". A refusal is a `Refusal` value
        # returned at estimate time; `raw_only` is a declaration that asserts nothing and passes
        # this test unconditionally. Reading the second as the first would let an instrument opt
        # out of the check by declaring itself raw, which is exactly what `admits` exists to stop.
        admits=frozenset({"invariant"}),
        assertion="per-token quantities must declare a normalisation and be invariant under it, or refuse",
    )


def _permutation() -> InvarianceGroup:
    """Permute rollout order within a group, `σ ~ Uniform(S_K)`.

    The group whose failure is informative rather than fatal. Any group statistic is unchanged
    under it; a position-biased judge is not, and that is the instrument detecting position bias.
    `check_invariance` marks a failure here `informative` rather than silently calling it a bug.
    """

    def make(seed: int) -> GroupAction:
        def apply(p: InvariancePayload) -> InvariancePayload:
            p.require("scores", "group_ids")
            scores = np.asarray(p.scores, dtype=np.float64)
            gids = np.asarray(p.group_ids)
            rng = np.random.default_rng(seed)
            out = scores.copy()
            order = np.arange(scores.shape[0])
            for g in np.unique(gids):
                idx = order[gids == g]
                out[idx] = scores[rng.permutation(idx)]
            return p.replace(scores=out)

        return GroupAction(
            name=f"σ ~ Uniform(S_K), seed={seed}",
            apply=apply,
            params={"seed": float(seed)},
            sample=lambda rng: make(rng.randrange(2**31)),
        )

    return InvarianceGroup(
        id="group.permutation",
        generators=(make(0),),
        acts_on="groups",
        admits=frozenset({"invariant"}),
        assertion=(
            "any group statistic unchanged. A position-biased judge fails this, which is the "
            "point: the test detects position bias rather than assuming it away."
        ),
    )


def _units() -> InvarianceGroup:
    """per-token ↔ per-sequence; nats ↔ bits; raw ↔ normalised.

    The one group whose assertion is a refusal rather than a numeric relation: a comparison across
    a unit boundary raises `UNIT_MISMATCH` rather than silently converting. So its generated check
    is a different kind of check, and `check_invariance` routes it to `check_unit_refusal` rather
    than pretending a value comparison means something here.
    """
    return InvarianceGroup(
        id="units",
        generators=(),
        acts_on="any",
        # No value relation is admitted, because the assertion is not about values. `check_invariance`
        # routes this group to `check_unit_refusal` before it consults `admits` at all.
        admits=frozenset(),
        assertion="a comparison across a unit boundary raises UNIT_MISMATCH rather than silently converting",
        refusal_only=True,
    )


def _trivial() -> InvarianceGroup:
    """What a deliberate `none` resolves to. Informative, and not the same as an omission.

    `grader.silent_zero_rate` counts grader exceptions and no affine rescaling of the reward acts
    on it usefully, so `none` is the correct answer there. It is registered, counted, and its
    generated test passes vacuously, which is honest: there is no transformation to be invariant
    under. Failing to think about the question is what the lint targets, and that is a different
    act.
    """
    return InvarianceGroup(
        id=TRIVIAL_GROUP,
        generators=(),
        acts_on="nothing",
        admits=frozenset({"invariant"}),
        assertion="no group acts on this quantity; declared deliberately rather than omitted",
    )


#: The seven registered groups, plus the trivial group a deliberate `none` resolves to.
GROUPS: dict[InvarianceGroupID, InvarianceGroup] = {
    g.id: g
    for g in (
        _affine(),
        _null(),
        _basis(),
        _reparam(),
        _tokenization(),
        _permutation(),
        _units(),
        _trivial(),
    )
}


def get_group(gid: InvarianceGroupID) -> InvarianceGroup:
    """Resolve a group id, normalising the spec's spelling of the trivial group.

    ``spec/QUANTITIES.yaml`` writes a deliberate no-group declaration as ``none`` and does so for 38
    of the 174 rows; the kernel registers it as ``trivial``. Both mean "I looked and no group acts
    on this quantity", as against an empty declaration meaning "I did not say", and the
    normalisation is already the documented behaviour of `inherited_groups` below. It was not the
    behaviour here, so twelve instruments transcribing their registry row faithfully declared
    ``none`` and would have raised the moment a generated test was written for them. One spelling
    resolves to one object rather than each caller remembering to translate.
    """
    key = TRIVIAL_GROUP if gid == "none" else gid
    try:
        return GROUPS[key]
    except KeyError:
        raise KeyError(
            f"no invariance group registered as {gid!r}. The seven groups are "
            f"{', '.join(sorted(k for k in GROUPS if k != TRIVIAL_GROUP))}, plus {TRIVIAL_GROUP!r} "
            f"(spelled `none` in spec/QUANTITIES.yaml) for a deliberate declaration of none."
        ) from None


# ---------------------------------------------------------------------------
# The generated test
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Draw:
    """One drawn group element and what the instrument did under it."""

    action: str
    params: Mapping[str, float]
    baseline: float
    transformed: float
    expected: float
    deviation: float
    within_tol: bool


@dataclass(frozen=True)
class InvarianceReport:
    """What the generated test found, including when it found something worth reporting.

    ``passed`` is the merge gate. ``informative`` says a failure is a measurement rather than a
    defect, which is true for exactly one group and is the reason that group is in the design.
    """

    instrument: str
    group: InvarianceGroupID
    relation: Relation
    n: int
    tol: float
    passed: bool
    max_deviation: float = 0.0
    draws: tuple[Draw, ...] = ()
    skipped: str = ""
    informative: bool = False
    interpretation: str = ""

    @property
    def worst(self) -> Draw | None:
        return max(self.draws, key=lambda d: d.deviation) if self.draws else None

    def render(self) -> str:
        head = f"{self.instrument}  {self.group}  ({self.relation.status})"
        if self.skipped:
            return f"{head}\n    skipped: {self.skipped}"
        state = (
            "pass" if self.passed else ("FAIL" if not self.informative else "FAIL (informative)")
        )
        lines = [
            f"{head}\n    {state}: {self.n} draws, max deviation {self.max_deviation:.4g} "
            f"(tol {self.tol:.4g})"
        ]
        w = self.worst
        if w is not None and not self.passed:
            lines.append(
                f"    worst draw: {w.action}  {w.baseline:.6g} → {w.transformed:.6g}, expected {w.expected:.6g}"
            )
        if self.interpretation:
            lines.append(f"    {self.interpretation}")
        return "\n".join(lines)


def _as_float(value: Any) -> float:
    """Reduce an instrument's reading to the scalar the relation is asserted about."""
    v = getattr(value, "value", value)
    if isinstance(v, (int, float, np.floating, np.integer)):
        return float(v)
    arr = np.asarray(v, dtype=np.float64)
    if arr.size == 1:
        return float(arr.reshape(()))
    raise TypeError(
        "check_invariance asserts a relation on a scalar reading. This instrument returned "
        f"{type(v).__name__} of size {arr.size}; wrap it in a callable that projects the reading "
        "onto the scalar the relation is declared about."
    )


def default_tol(baseline: float) -> float:
    """A tolerance scaled to the reading, because an absolute one is wrong at both ends.

    Floating-point error in the transformed pipeline grows with the magnitude of the numbers going
    through it, so a fixed 1e-9 fails spuriously on a reading of 1e4 and passes anything at all on
    a reading of 1e-12. Relative with an absolute floor is the standard answer and it is the shape
    `numpy.isclose` uses.

    The relative constant here is 1e-7, which is a hundred times *tighter* than `isclose`'s 1e-5,
    deliberately. Every transformation in this module is exact arithmetic on float64, where the
    achievable relative error is nearer 1e-15, so anything reaching 1e-7 is a property of the
    instrument rather than of the machine. Building this found one: the GRPO advantage
    `(r - mean)/(std + ε)` is **not** affine-invariant for ε > 0, because the numerator scales by
    `a` and the denominator goes to `a·std + ε` rather than `a·(std + ε)`. At ε = 1e-8 that shows
    up here as a deviation of about 1e-7, which `isclose` would have called equal. An instrument
    with a genuinely noisier pipeline passes its own `tol` and says why.
    """
    return max(1e-9, 1e-7 * abs(baseline))


def resolve_relation(instrument: Any, group_id: InvarianceGroupID) -> Relation:
    """The relation an instrument declares under one group.

    ``invariance_relation`` may be a single `Relation`, which applies to every group the instrument
    is checked under, or a mapping from group id to `Relation`. The mapping form exists because an
    instrument can genuinely transform two ways: `chi` is `Cov(f, r)`, so it is **covariant** with
    weight 1 under `reward.affine` (`Cov(f, ar+b) = a·Cov(f, r)`) and **invariant** under
    `repr.basis` (an orthogonal map acting on both the activations and the readout leaves every
    inner product alone). A single relation cannot say both, and forcing a choice loses whichever
    check is not declared.

    Absent declaration resolves to invariant, which is the correct direction to default in: a
    covariant instrument that forgot fails its generated test loudly.
    """
    declared = getattr(instrument, "invariance_relation", None)
    if isinstance(declared, Mapping):
        return declared.get(group_id) or INVARIANT
    return declared or INVARIANT


def check_invariance(
    instrument: Any,
    group: InvarianceGroup | InvarianceGroupID,
    ctx: InvariancePayload,
    n: int = 64,
    tol: float | None = None,
    *,
    relation: Relation | None = None,
    seed: int = 0,
    run: Callable[[Any, InvariancePayload], Any] | None = None,
) -> InvarianceReport:
    """Draw `n` group elements, apply each, re-run the instrument, assert the declared relation.

    This is the generated invariance property test. Every registered instrument gets one, and
    an instrument that declares no group does not merge.

    ``instrument`` is anything with a ``name`` and a way to be run. ``run`` supplies the call when
    the instrument is not simply callable, which is how an `Instrument` with ``estimate(ctx)`` and
    a bare featuriser both work here without either being special-cased.

    ``relation`` defaults to the instrument's own ``invariance_relation`` attribute, then to
    invariant. Declaring it is the instrument's job; defaulting to invariant means a covariant
    instrument that forgot fails loudly, which is the correct direction to fail in.
    """
    g = group if isinstance(group, InvarianceGroup) else get_group(group)
    rel = relation or resolve_relation(instrument, g.id)
    name = getattr(instrument, "name", getattr(instrument, "__name__", repr(instrument)))

    call = run or (lambda inst, payload: inst(payload))

    if g.refusal_only:
        return InvarianceReport(
            instrument=name,
            group=g.id,
            relation=rel,
            n=0,
            tol=0.0,
            passed=True,
            skipped=(
                "this group's assertion is a refusal, not a numeric relation. Use "
                "check_unit_refusal, which asserts the comparison raises UNIT_MISMATCH."
            ),
        )

    if not g.generators:
        return InvarianceReport(
            instrument=name,
            group=g.id,
            relation=rel,
            n=0,
            tol=0.0,
            passed=True,
            skipped="the trivial group has no generators; nothing acts on this quantity",
        )

    if rel.status not in g.admits:
        raise ValueError(
            f"{name} declares relation {rel.status!r} under {g.id!r}, which admits "
            f"{sorted(g.admits)}. This group's assertion does not offer that "
            f"relation, so the declaration is a way of opting out of a test rather than a claim."
        )

    unsamplable = [gen.name for gen in g.generators if gen.sample is None]
    if unsamplable:
        raise ValueError(
            f"group {g.id!r} has generators with no sampler ({', '.join(unsamplable)}), so drawing "
            f"n elements would return the same element n times and the report would pass on one "
            f"observation. Every generator must be samplable."
        )

    baseline = _as_float(call(instrument, ctx))
    t = tol if tol is not None else default_tol(baseline)

    if rel.status == "raw_only":
        # A raw_only instrument makes no promise about its value, so there is nothing to assert
        # about equality. What is worth recording is whether it moves at all: a raw_only reading
        # that never moves under its own group is invariant and mis-declared, which is a real (and
        # under-cautious in the other direction) finding.
        rng = Random(seed)
        deviations = []
        for _ in range(n):
            action = g.draw(rng)
            deviations.append(abs(_as_float(call(instrument, action.apply(ctx))) - baseline))
        moved = max(deviations) if deviations else 0.0
        return InvarianceReport(
            instrument=name,
            group=g.id,
            relation=rel,
            n=n,
            tol=t,
            passed=True,
            max_deviation=moved,
            skipped="raw_only: no value relation is asserted",
            interpretation=(
                f"raw coordinates, as declared; the reading moved by up to {moved:.4g} under the "
                f"group."
                if moved > t
                else (
                    f"the reading did not move (max {moved:.4g} <= tol {t:.4g}) under a group it "
                    f"declares itself raw under. It may be invariant and mis-declared."
                )
            ),
        )

    rng = Random(seed)
    draws: list[Draw] = []
    for _ in range(n):
        action = g.draw(rng)
        got = _as_float(call(instrument, action.apply(ctx)))
        if rel.status == "covariant":
            scale = float(action.params.get(rel.parameter, 1.0))
            expected = (scale**rel.weight) * baseline
        else:
            expected = baseline
        dev = abs(got - expected)
        draws.append(
            Draw(
                action=action.name,
                params=dict(action.params),
                baseline=baseline,
                transformed=got,
                expected=expected,
                deviation=dev,
                within_tol=dev <= t,
            )
        )

    worst = max((d.deviation for d in draws), default=0.0)
    passed = all(d.within_tol for d in draws)
    informative = (not passed) and g.id == "group.permutation"
    interpretation = ""
    if informative:
        interpretation = (
            "this instrument is sensitive to rollout order within a group. For a judge that is "
            "position bias, measured; the test detects it rather than assuming it away. Report it "
            "as a finding about the grader, not as a defect in the instrument."
        )
    elif not passed and g.id == "reward.affine" and rel.status == "invariant":
        interpretation = (
            "the reading moved under an affine rescaling of the reward. Either it is covariant and "
            "should declare a weight, or it is reading a level rather than a contrast."
        )

    return InvarianceReport(
        instrument=name,
        group=g.id,
        relation=rel,
        n=n,
        tol=t,
        passed=passed,
        max_deviation=worst,
        draws=tuple(draws),
        informative=informative,
        interpretation=interpretation,
    )


def check_unit_refusal(compare: Callable[[Any, Any], Any], a: Any, b: Any) -> bool:
    """The `units` group's assertion: comparing across a unit boundary refuses rather than converts.

    ``compare`` is the comparison under test and ``a``, ``b`` are two readings in incompatible
    units. Passing means the comparison produced a `UNIT_MISMATCH` refusal or raised; failing means
    it returned a number, which is the silent error this whole design exists to make impossible.
    """
    from reward_lens.core.reading import Refusal, RefusalReason

    try:
        out = compare(a, b)
    except Exception:  # noqa: BLE001 - a raise is an acceptable refusal here
        return True
    return isinstance(out, Refusal) and out.reason is RefusalReason.UNIT_MISMATCH


# ---------------------------------------------------------------------------
# The lint rule
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LintFinding:
    subject: str
    problem: str
    remedy: str


def resolve_groups(
    declared: Any,
    quantity_groups: Mapping[str, str] | None = None,
    quantities: Sequence[str] = (),
) -> tuple[frozenset[InvarianceGroupID], str]:
    """The groups an instrument is checked under, and where they came from.

    An instrument declares its own group where the catalogue carries one. Where it does not, the
    groups come from the quantities it estimates, which declare one for all 125 of them. That
    inheritance is not a convenience: a quantity's invariance is a property of the quantity, so an
    estimator of it is checked under the same group unless it says otherwise, and 46 of the 52
    instruments whose own column reads OPEN resolve this way.

    Returns the resolved set and the provenance string, which is empty when nothing resolved.
    """
    parsed = parse_group_field(declared)
    if parsed:
        return frozenset(parsed), "declared"
    if quantity_groups:
        inherited = {
            quantity_groups[q] for q in quantities if isinstance(q, str) and q in quantity_groups
        }
        inherited = {TRIVIAL_GROUP if g == "none" else g for g in inherited if g and g != "OPEN"}
        if inherited:
            return frozenset(inherited), "inherited from quantities"
    return frozenset(), ""


def parse_group_field(declared: Any) -> list[InvarianceGroupID]:
    """Read a catalogue invariance cell into group ids.

    The catalogue carries these as they were printed: backticked, comma-separated, and sometimes
    trailed by a parenthetical that explains the choice ("`policy.reparam` (Fisher-metric
    quantities are invariant; `‖Δθ‖` is not)"). Parsing rather than requiring a clean field keeps
    the provenance comments in the YAML, which is where the merge audit lives.
    """
    if declared is None:
        return []
    text = str(declared).strip()
    if not text or text == "OPEN":
        return []
    # Drop an explanatory parenthetical, then split on commas and strip backticks.
    depth = 0
    kept: list[str] = []
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif depth == 0:
            kept.append(ch)
    out = []
    for piece in "".join(kept).split(","):
        gid = piece.strip().strip("`").strip()
        if not gid:
            continue
        if gid == "none":
            gid = TRIVIAL_GROUP
        if gid in GROUPS:
            out.append(gid)
    return out


def lint_catalogue(
    instruments: Sequence[Mapping[str, Any]],
    quantity_groups: Mapping[str, str] | None = None,
) -> tuple[list[LintFinding], dict[str, frozenset[InvarianceGroupID]]]:
    """Every catalogue instrument declares a group, or inherits one, or is reported.

    The rule targets omission. A literal `OPEN` that cannot be resolved from the instrument's
    quantities is a finding; `none` is an answer and resolves to the trivial group.
    """
    findings: list[LintFinding] = []
    resolved: dict[str, frozenset[InvarianceGroupID]] = {}
    for row in instruments:
        iid = str(row.get("id", "?"))
        qs = row.get("quantities") or []
        if isinstance(qs, str):  # the catalogue stores OPEN as a bare string; do not iterate it
            qs = []
        groups, source = resolve_groups(row.get("invariance_group"), quantity_groups, qs)
        resolved[iid] = groups
        if not groups:
            findings.append(
                LintFinding(
                    subject=iid,
                    problem="declares no invariance group and none of its quantities declares one",
                    remedy=(
                        "add an invariance_group to the instrument's catalogue record, or declare "
                        "`none` if no group acts on it. `none` is an answer; a blank is not."
                    ),
                )
            )
    return findings, resolved


__all__ = [
    "COVARIANT_LINEAR",
    "GROUPS",
    "INVARIANT",
    "RAW_ONLY",
    "TRIVIAL_GROUP",
    "Draw",
    "GroupAction",
    "InvarianceGroup",
    "InvarianceGroupID",
    "InvariancePayload",
    "InvarianceReport",
    "LintFinding",
    "Relation",
    "Status",
    "check_invariance",
    "resolve_relation",
    "check_unit_refusal",
    "default_tol",
    "get_group",
    "lint_catalogue",
    "parse_group_field",
    "resolve_groups",
]

"""Held-out labels, in a type a detector's signature cannot accept.

A reward-hacking detector is scored against an answer key. The answer key has to live somewhere,
and wherever it lives it is one attribute access away from the code being scored. Every project
that has run this experiment has solved it with discipline: a convention about which dictionary
the detector reads, a code review, a note in the README. Discipline fails silently and it fails
late, because a detector that saw the answer key still produces a plausible number.

So the answer key lives in `Blind[T]`, and the barrier is the type rather than the convention:

    def detect(features: Mapping[FeatureID, float]) -> float: ...

    detect(trajectory.labels["hacked"])     # mypy: arg-type. Not a Mapping, not a float.

That is the whole idea and it is not mine. `Aarav500/flight-recorder` had it first, as two disjoint
frames with three tests enforcing the disjointness. Its repository claims Apache-2.0 and ships no
LICENSE file, so what is here is the design read from its description and written fresh, and the
three tests it names are in `tests/test_record_labels.py`.

**There is no `.unwrap()`.** There is `adjudicate`, which is a different thing: it opens the
envelope and writes a row to the evidence store saying who opened it, when, for what, and which
label it was, identified by fingerprint rather than by value. The point of this module is not that
a label can never be read. A label that can never be read cannot score anything and cannot have its
own error rate measured, and both of those are required work (`LABEL_QUALITY_UNKNOWN` exists
precisely because someone has to audit the answer key). The point is that reading one leaves a
trace a reviewer can find with a single query against the store.

**What this does not do, stated plainly.** Python has no private state. `blind._value` reads the
label, and so does `dataclasses.astuple`. Nothing here stops a determined caller and nothing here
is a security boundary. What it stops is the accident: the plausible line of code that reads the
oracle without anyone noticing, including the person who wrote it. Against a deliberate act it
offers something narrower and still useful, which is that the deliberate act is one grep away
(`_value`, `astuple`, `__dict__`) and does not look like ordinary code. The type checker is the
part of this that is actually enforced, which is why there is a mypy job in CI rather than a
docstring here.

Three runtime guards back the type up, because three leakage paths run underneath the type system:

- `RolloutFrame` and `OracleFrame` have **disjoint field sets**, so no attribute reachable from the
  visible half reaches the held-out half, whatever the annotations say.
- A **name blocklist** covers `features`, which is `Mapping[FeatureID, float]`. A label written in
  as a float named `is_hack` is invisible to any type checker and is exactly what a hurried
  converter produces.
- `check_detector` **introspects the annotations** of a detector at runtime, for the callers who do
  not run mypy, and it treats a missing annotation as a finding rather than as a pass.
"""

from __future__ import annotations

import inspect
import itertools
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Protocol,
    TypeAlias,
    TypeVar,
    get_args,
    get_type_hints,
)

from reward_lens.core.errors import RewardLensError
from reward_lens.core.evidence import Uncertainty, make_evidence, register_payload
from reward_lens.core.provenance import capture_provenance
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.types import GaugeStatus, SubjectRef, content_hash

# `FeatureID` is imported at runtime rather than under TYPE_CHECKING because `Mapping` is invariant
# in its key type, so `RolloutFrame.features` typed `Mapping[str, float]` would not satisfy a
# function annotated `Mapping[FeatureID, float]`, which is the signature a detector has to have.
# The frame's features are the trajectory's features and they should have the trajectory's
# type. No cycle: `record.schema` names this module only under TYPE_CHECKING.
from reward_lens.record.schema import FeatureID

if TYPE_CHECKING:
    from reward_lens.core.evidence import Evidence
    from reward_lens.core.store import EvidenceStore
    from reward_lens.record.schema import Trajectory

T = TypeVar("T")

#: What a held-out label is allowed to be. Deliberately narrow: every one of these is JSON-native,
#: so a label round-trips through the record and through `content_hash` with no codec of its own.
#: A label that needs a structure is a `ScoreTree` or a probe result wearing the wrong name.
LabelValue: TypeAlias = bool | int | float | str


class LabelLeak(RewardLensError):
    """The oracle reached, or could reach, somewhere it must not.

    Raised rather than refused. A refusal is a value an instrument returns when it anticipated a
    condition and cannot measure through it; this is a wiring defect in the caller's own code, and
    the correct response is to stop rather than to return a number with a note attached.

    Deliberately not a `ValueError`. Frame construction is exactly the kind of call that gets
    wrapped in `except ValueError` by a converter tidying up malformed input, and a leak swallowed
    by that handler is the failure this module exists to prevent.
    """


# ---------------------------------------------------------------------------
# What is known about the answer key itself
# ---------------------------------------------------------------------------


@register_payload
@dataclass(frozen=True)
class LabelQuality:
    """The measured error rate of the labels, and what measured it.

    Scoring an instrument against a one-third-wrong answer key measures the answer key. So this
    is not decoration on the label, it is the precondition for using the label at all, and
    `adjudicate` refuses a scoring read when it is absent.

    ``error_rate`` is the fraction of labels found wrong on the audited sample, so it lies in
    [0, 1] and `None` means nobody has looked. ``n_audited`` is how many were checked, and zero
    with a stated rate is a contradiction the constructor rejects: a rate measured on nothing is a
    guess with a number attached to it.
    """

    error_rate: float | None = None
    n_audited: int = 0
    #: How the audit was done, in a sentence. "Two raters on a stratified sample of 200, adjudicated
    #: by a third on disagreement" is a method; "manual" is not.
    method: str = ""
    #: Who or what audited. A rater pool id, a verifier version, a person.
    measured_by: str = ""

    def __post_init__(self) -> None:
        if self.error_rate is not None:
            if not 0.0 <= self.error_rate <= 1.0:
                raise ValueError(
                    f"label error_rate must be a fraction in [0, 1]; got {self.error_rate}"
                )
            if self.n_audited <= 0:
                raise ValueError(
                    f"label error_rate is {self.error_rate} but n_audited is {self.n_audited}. A "
                    f"rate measured on no samples is not a measurement. Pass the sample size, or "
                    f"leave error_rate as None and let the scoring path refuse."
                )
        if self.n_audited < 0:
            raise ValueError(f"n_audited cannot be negative; got {self.n_audited}")

    @property
    def is_measured(self) -> bool:
        """Whether these labels have an error rate that came from counting something."""
        return self.error_rate is not None and self.n_audited > 0

    def __canonical__(self) -> dict[str, Any]:
        return {
            "error_rate": self.error_rate,
            "n_audited": self.n_audited,
            "method": self.method,
            "measured_by": self.measured_by,
        }


#: The answer to "what do we know about these labels" when the answer is nothing. A distinct
#: constant so a caller can say "unmeasured" out loud instead of passing None and hoping the
#: reader knows which None it is.
UNMEASURED = LabelQuality()


# ---------------------------------------------------------------------------
# Blind[T]
# ---------------------------------------------------------------------------


@register_payload
@dataclass(frozen=True, eq=False, repr=False)
class Blind(Generic[T]):
    """A held-out value that a detector's signature cannot accept.

    Construct with `blind`. Read with `adjudicate`, which writes an evidence row. There is no
    `.unwrap()`, no `.value`, and no `.get()`, and the absence of those three names is the entire
    static contract: a call to any of them is an `attr-defined` error under mypy, which is a thing
    CI checks rather than a thing this docstring asserts. One wrinkle worth knowing about, since
    the report of a leak is part of the interface: mypy's message for `.value` is `"Blind[bool]"
    has no attribute "value"; maybe "_value"?`, so the checker that catches the accident also
    names the private field. That is the same trade the module docstring makes. The barrier is
    against the accident, and the deliberate act is left visible instead of blocked.

    Four smaller decisions, each closing a channel that leaks without one:

    ``__eq__`` is off. On a boolean label, ``blind == some_guess`` is a one-bit oracle that needs
    no adjudication row, and one bit is the whole label. Two `Blind`s therefore compare by
    identity. Where a test needs to know two blinds hold the same thing, `fingerprint` says so
    without saying what.

    ``__repr__`` never renders the payload. A record printed into a log, a traceback, or a prompt
    is a real leakage path and it is the one that does not look like an access at all.

    ``__bool__`` raises. ``if blind:`` on a `Blind[bool]` is always true regardless of the label,
    so it is a silent wrong answer rather than a leak, and this module's one unforgivable output is
    a confident wrong number.

    ``__canonical__`` hides the payload, so hashing anything that transitively contains a `Blind`
    is safe. `content_hash` is what names evidence ids, sidecar files and frames, and an id derived
    from an oracle value is a copy of the oracle in a filename. The record codec is a separate path
    and does keep the payload, which is what makes a record round-trip: `ValueCodec` reads
    ``dataclasses.fields`` and reconstructs a `Blind`, never a dict, provided this class is
    registered as a payload. It is, at the bottom of this module.
    """

    #: The payload. Private, and the leading underscore is not decoration: `canonical_bytes`
    #: skips underscore-prefixed attributes, so an object that ends up hashed by the generic path
    #: does not carry the label into the digest.
    _value: T
    #: Which label this is, in the record's `labels` mapping. Not held out: knowing that a
    #: trajectory has a `hacked` label is not knowing whether it is set.
    key: str
    #: What is known about this label's own error rate. `None` is the same answer as `UNMEASURED`
    #: and both refuse a scoring read.
    quality: LabelQuality | None = None

    @property
    def fingerprint(self) -> str:
        """A content hash of the key and the payload, for comparing two blinds without opening one.

        Not a commitment scheme. Label sets are small (`True`/`False`, a handful of verdict
        strings), so anyone who can call this can invert it by hashing the candidates. It exists so
        a round-trip test can assert a record survived storage, and so an adjudication row can name
        which label was read without the row itself becoming a second copy of the answer key.
        """
        return content_hash({"key": self.key, "value": self._value}, "lbl")

    @property
    def is_measured(self) -> bool:
        """Whether this label's own error rate has been measured. See `LabelQuality`."""
        return self.quality is not None and self.quality.is_measured

    def __repr__(self) -> str:
        state = "measured" if self.is_measured else "unmeasured"
        return f"Blind(key={self.key!r}, quality={state}, fingerprint={self.fingerprint})"

    __str__ = __repr__

    def __bool__(self) -> bool:
        raise LabelLeak(
            f"a Blind has no truth value, and `if blind:` on the {self.key!r} label would have "
            f"been true whatever the label is. Open it with adjudicate(), which writes the row "
            f"that says you did."
        )

    def __canonical__(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "fingerprint": self.fingerprint,
            "quality": None if self.quality is None else self.quality.__canonical__(),
        }


def blind(value: T, *, key: str, quality: LabelQuality | None = None) -> Blind[T]:
    """Wrap a held-out value. The one place a `Blind` is made.

    A function rather than calling `Blind(...)` directly, because the dataclass field is named
    `_value` for the hashing reason given on the class, and `Blind(_value=x, key="hacked")` is not
    a constructor anyone should have to type.
    """
    if not key.strip():
        raise ValueError("a blind label needs a key; an unnamed label cannot be adjudicated")
    return Blind(_value=value, key=key, quality=quality)


# ---------------------------------------------------------------------------
# The name blocklist: the leak the type system cannot see
# ---------------------------------------------------------------------------

#: Feature names that are labels wearing a float. `Trajectory.features` is
#: `Mapping[FeatureID, float]`, so an oracle written in as `features["is_hack"] = 1.0` type checks
#: perfectly and reaches every detector.
#:
#: Exact matches on the normalised name, not substrings. Substring matching would block
#: `target_length` and `oracle_free_baseline`, and a guard that fires on correct code gets turned
#: off. The cost of exactness is that it catches the conventional names and nothing else: a label
#: smuggled in as `f47` passes, and no list of English words will ever catch that one. This is a
#: guard against the hurried converter, not against an adversary.
BLOCKED_NAMES: frozenset[str] = frozenset(
    {
        "label",
        "labels",
        "gold",
        "gold_label",
        "oracle",
        "oracle_label",
        "ground_truth",
        "groundtruth",
        "truth",
        "y_true",
        "ytrue",
        "answer",
        "answer_key",
        "solution",
        "hack",
        "hacked",
        "is_hack",
        "is_hacked",
        "hacking",
        "exploit",
        "exploited",
        "is_exploit",
        "cheat",
        "cheated",
        "is_cheat",
        "verdict",
        "onset",
        "onset_step",
        "held_out",
        "heldout",
        "contaminated",
        "is_contaminated",
    }
)

#: Prefixes that make the intent unambiguous whatever follows them. `oracle_score`,
#: `gold_verdict` and `label_confidence` are all the answer key.
BLOCKED_PREFIXES: tuple[str, ...] = (
    "oracle_",
    "gold_",
    "label_",
    "ground_truth_",
    "held_out_",
    "heldout_",
    "answer_",
    "is_hack",
    "onset_",
)


def normalise_name(name: str) -> str:
    """Lowercase, and fold the three separators a feature name arrives with into one."""
    out = name.strip().lower()
    for ch in ("-", ".", " ", "/"):
        out = out.replace(ch, "_")
    return out


def is_blocked(name: str) -> bool:
    """Whether this name is one of the conventional spellings of the answer key."""
    norm = normalise_name(name)
    return norm in BLOCKED_NAMES or norm.startswith(BLOCKED_PREFIXES)


def blocked_names(names: Iterable[str]) -> tuple[str, ...]:
    """Every blocked name in ``names``, in the order given, as originally spelled.

    Returns the offenders rather than a bool so the caller can name all of them at once. A
    converter that put one label in the features usually put three.
    """
    return tuple(n for n in names if is_blocked(n))


# ---------------------------------------------------------------------------
# The two frames
# ---------------------------------------------------------------------------


@register_payload
@dataclass(frozen=True)
class RolloutFrame:
    """The half of a trajectory a detector is allowed to see.

    Field-disjoint from `OracleFrame`, checked by a test, which is the property that makes this
    more than a naming convention: there is no attribute path from here to the answer key, so a
    detector cannot reach it by accident even through `getattr`.

    The join key is `trajectory_id` here and `trajectory_ref` there. Two spellings on purpose. The
    frames have to be joinable or the oracle could never be lined up with the predictions it
    scores, and sharing one field name would put a common key in both field sets and turn
    "disjoint" into "disjoint apart from the one we agreed to ignore", which is the shape every
    such rule has the day before it stops holding. The asymmetry also carries the meaning: one
    frame *is* the rollout and the other one *points at* it.
    """

    trajectory_id: str
    task_id: str
    #: Turn text in order. A detector that reads the transcript is the normal case, and a frame it
    #: cannot read anything out of would just be routed around.
    turns: tuple[str, ...]
    n_tokens: int
    #: As the pipeline computed it. Visible on purpose: the advantage is an input to the training
    #: loop, so a detector that may not see it is being tested on a question nobody asks.
    advantage: float | None
    features: Mapping[FeatureID, float]

    def __post_init__(self) -> None:
        leaks = blocked_names(self.features)
        if leaks:
            raise LabelLeak(
                f"trajectory {self.trajectory_id}: {len(leaks)} feature name(s) are the answer key "
                f"in float clothing: {', '.join(repr(n) for n in leaks)}. `features` is "
                f"Mapping[FeatureID, float], so no type checker can see this one. Move them to "
                f"OracleFrame.labels as Blind values, or rename them if they are genuinely "
                f"observable at rollout time."
            )

    @property
    def n_turns(self) -> int:
        return len(self.turns)


@register_payload
@dataclass(frozen=True)
class OracleFrame:
    """The half a detector must not see, and what is known about how wrong it is.

    Every value in `labels` must be a `Blind`. The type says so and `__post_init__` checks it,
    because this frame is what a converter builds from someone else's schema and a converter is
    the one place a raw `bool` gets in.
    """

    trajectory_ref: str
    labels: Mapping[str, Blind[LabelValue]]
    #: Frame-level quality, used for any label that carries none of its own.
    quality: LabelQuality = UNMEASURED
    #: The step at which the labelled behaviour begins, where the label is an onset rather than a
    #: verdict. Held out for the same reason the label is: a detector told when to look is being
    #: scored on a different problem.
    onset_step: int | None = None
    #: Who produced these labels. A rater pool, a verifier version, a paper's release.
    labeller: str = ""

    def __post_init__(self) -> None:
        wrong = {k: type(v).__name__ for k, v in self.labels.items() if not isinstance(v, Blind)}
        if wrong:
            listed = ", ".join(f"{k!r} is a {t}" for k, t in sorted(wrong.items()))
            raise LabelLeak(
                f"trajectory {self.trajectory_ref}: every label must be a Blind and {listed}. A "
                f"raw value here is readable by anything holding the frame, and it is what the "
                f"type in `Trajectory.labels` promises is impossible. Wrap it: "
                f"blind(value, key=...)."
            )

    def quality_of(self, key: str) -> LabelQuality:
        """The quality that governs one label: its own if it has one, else the frame's."""
        item = self.labels[key]
        return item.quality if item.quality is not None else self.quality


def split_trajectory(
    traj: "Trajectory",
    *,
    quality: LabelQuality = UNMEASURED,
    onset_step: int | None = None,
    labeller: str = "",
) -> tuple[RolloutFrame, OracleFrame]:
    """Split one recorded trajectory into the visible half and the held-out half.

    This is the seam the record left open. `Trajectory` carries both halves, because the record is
    the thing that has to be complete; a detector is handed the `RolloutFrame` alone and never the
    `Trajectory`, and this function is the only place the two halves are separated so there is one
    place to read when asking what a detector could see.

    The three keyword arguments are the label metrology, which is a property of the labelling
    campaign rather than of one rollout and therefore is not on `Trajectory` to be read off. They
    default to "nobody has measured this", which is the answer that makes a scoring read refuse.
    """
    return (
        RolloutFrame(
            trajectory_id=str(traj.id),
            task_id=str(traj.task_ref),
            turns=tuple(t.text for t in traj.turns),
            n_tokens=traj.n_tokens,
            advantage=traj.advantage,
            features={FeatureID(str(k)): float(v) for k, v in traj.features.items()},
        ),
        OracleFrame(
            trajectory_ref=str(traj.id),
            labels=dict(traj.labels),
            quality=quality,
            onset_step=onset_step,
            labeller=labeller,
        ),
    )


# ---------------------------------------------------------------------------
# The detector protocol, and its runtime introspection
# ---------------------------------------------------------------------------


class Detector(Protocol):
    """What a detector's signature has to be.

    Not `runtime_checkable`, and that is the point rather than an omission. A runtime-checkable
    Protocol's `isinstance` only checks that the attributes exist, so every callable in Python
    would pass this one and the check would report a leak-free codebase without reading a single
    annotation. `check_detector` below reads the annotations instead, which is the thing anyone
    actually wanted from `isinstance` here.
    """

    def __call__(self, frame: RolloutFrame) -> float: ...


#: Types a detector may not take, and may not hand back. `Blind` covers a bare label,
#: `OracleFrame` covers the whole held-out half.
_ORACLE_TYPES: tuple[type, ...] = (Blind, OracleFrame)


def _mentions_oracle(annotation: Any) -> bool:
    """Whether an annotation reaches an oracle type, including inside a container.

    Recursive on `get_args`, because `Mapping[str, Blind[bool]]` and `list[OracleFrame]` are the
    shapes this arrives in once someone is passing a batch.
    """
    origin = getattr(annotation, "__origin__", None)
    candidate = origin if origin is not None else annotation
    if isinstance(candidate, type) and issubclass(candidate, _ORACLE_TYPES):
        return True
    return any(_mentions_oracle(arg) for arg in get_args(annotation))


#: Annotations that name no constraint at all. Every value in the language satisfies each of them,
#: so a parameter annotated with one accepts a `Blind` and a return annotated with one may be a
#: `Blind`. They are exactly as unverifiable as no annotation, and `object` is the worse of the two
#: because it looks deliberate: mypy accepts `Blind[bool]` where `object` is asked for, since every
#: type is a subtype of `object`, so the static half does not catch it either.
_UNVERIFIABLE: tuple[Any, ...] = (Any, object)


def _is_unverifiable(annotation: Any) -> bool:
    """Whether an annotation constrains nothing, directly or inside a container.

    Recursive for the same reason `_mentions_oracle` is: `Mapping[str, object]` accepts
    `Mapping[str, Blind[bool]]` because `Mapping` is covariant in its value type, so a container of
    `object` is a hole with a lid on it. `list[Any]` is the same hole.
    """
    if any(annotation is candidate for candidate in _UNVERIFIABLE):
        return True
    return any(_is_unverifiable(arg) for arg in get_args(annotation))


def _ann_name(annotation: Any) -> str:
    """An annotation as a reader would write it, rather than as `repr` prints a class object."""
    if isinstance(annotation, type):
        return f"{annotation.__module__}.{annotation.__qualname__}"
    return str(annotation)


def _callable_target(fn: Any) -> Any:
    """The function whose annotations describe ``fn``.

    A detector is usually a function and is sometimes a class instance with `__call__`, which is
    how anything stateful gets written. Both have to be readable or the check is skipped by
    whoever most needed it.
    """
    if inspect.isfunction(fn) or inspect.ismethod(fn):
        return fn
    call = getattr(type(fn), "__call__", None)
    return call if call is not None else fn


def detector_findings(fn: Any) -> tuple[str, ...]:
    """Everything wrong with a detector's signature, as sentences. Empty means nothing found.

    Three kinds of finding, and the third is the one that makes this worth running:

    A parameter annotated with an oracle type, which is the leak.

    A return annotated with one, which is the same leak facing outward: a detector that hands back
    the oracle it was given launders it into whatever consumes the score.

    A parameter or return with **no annotation at all**. That is not a pass. An unannotated
    parameter is precisely the hole a type checker cannot close, and treating it as clean would
    make this function report success on the one signature it cannot check.

    `Any` and `object` are reported on the same argument, because that argument does not turn on
    whether characters were typed. Neither constrains anything: `object` accepts a `Blind` under
    mypy as well as at runtime, since every type is a subtype of `object`, and `Any` disables
    checking by definition. A signature annotated with either reads as checked and is not, which is
    worse than one that reads as unchecked.
    """
    target = _callable_target(fn)
    name = getattr(fn, "__name__", None) or type(fn).__name__
    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError) as exc:
        return (f"{name}: signature could not be read ({exc}), so nothing about it was checked.",)
    try:
        hints = get_type_hints(target)
    except Exception as exc:  # NameError, and anything a bad annotation string raises
        # Not narrowed and not swallowed. An annotation that will not resolve is exactly as
        # unverifiable as an absent one, and it is reported as a finding rather than as a crash.
        return (
            f"{name}: annotations could not be resolved ({type(exc).__name__}: {exc}), so this "
            f"signature is unchecked. Define the types at module level, or import them under "
            f"TYPE_CHECKING so get_type_hints can see them.",
        )

    found: list[str] = []
    for pname, param in sig.parameters.items():
        if pname in ("self", "cls"):
            continue
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            found.append(
                f"{name}: **{pname} accepts anything, including an OracleFrame or a Blind. A "
                f"detector's inputs have to be named to be checkable."
            )
            continue
        if pname not in hints:
            found.append(
                f"{name}: parameter {pname!r} has no annotation, so neither mypy nor this check "
                f"can tell whether the oracle reaches it. Annotate it, with RolloutFrame if it is "
                f"the frame."
            )
            continue
        if _mentions_oracle(hints[pname]):
            found.append(
                f"{name}: parameter {pname!r} is annotated {_ann_name(hints[pname])}, which "
                f"reaches the held-out labels. A detector takes RolloutFrame; the oracle is opened "
                f"by whoever scores the detector, through adjudicate()."
            )
        elif _is_unverifiable(hints[pname]):
            found.append(
                f"{name}: parameter {pname!r} is annotated {_ann_name(hints[pname])}, which "
                f"constrains nothing and accepts a Blind. Neither mypy nor this check can tell "
                f"whether the oracle reaches it, which is the same position an unannotated "
                f"parameter leaves them in. Annotate it RolloutFrame if it is the frame."
            )
    if "return" not in hints:
        found.append(
            f"{name}: the return is not annotated. A detector returns a score; an unannotated "
            f"return could be the oracle going back out."
        )
    elif _mentions_oracle(hints["return"]):
        found.append(
            f"{name}: the return is annotated {_ann_name(hints['return'])}, which hands the oracle "
            f"back to the caller. Return the score."
        )
    elif _is_unverifiable(hints["return"]):
        found.append(
            f"{name}: the return is annotated {_ann_name(hints['return'])}, which constrains "
            f"nothing, so a Blind handed back to the caller is not a type error. A detector "
            f"returns a score; annotate it float."
        )
    return tuple(found)


def check_detector(fn: Any) -> None:
    """Raise `LabelLeak` if a detector's signature can reach the oracle. Otherwise return.

    The runtime half of the guarantee, for the callers who do not run a type checker. It is
    strictly weaker than mypy, which sees the call sites too, and it is strictly better than
    nothing, which is what a project without a types job in CI has.
    """
    found = detector_findings(fn)
    if found:
        raise LabelLeak("this detector can reach the held-out labels:\n  " + "\n  ".join(found))


# ---------------------------------------------------------------------------
# Adjudication: the audited way out
# ---------------------------------------------------------------------------


class ReadPurpose(Enum):
    """Why a label is being opened. Recorded in the row, and it decides the refusal.

    Two members, not three, because a third would become the loophole. `AUDIT` is the escape from
    the circularity: the labels need a measured error rate before anything may be
    scored against them, and measuring that rate means reading the labels. `SCORING` is everything
    else and is the one that refuses.

    A caller who writes `AUDIT` and then scores has signed a false statement into an append-only
    store. That is the enforcement level here and it is worth being honest about: this is an audit
    trail, not a sandbox.
    """

    #: Scoring an instrument, a detector or a claim against the labels. Refused when the labels
    #: have no measured error rate.
    SCORING = "scoring"
    #: Measuring or inspecting the labels themselves, which is what produces that error rate.
    AUDIT = "audit"


@register_payload
@dataclass(frozen=True)
class LabelRead:
    """One opening of the envelope, as it lands in the evidence store.

    Carries ``fingerprint`` and not the label. The evidence store is read at `RECORD` access by
    everyone including the person whose detector is being scored, so a row holding the value would
    be a second copy of the answer key sitting outside the type system, reachable by the one access
    level that every access profile has.

    ``read_at`` and ``read_index`` are in the payload rather than only in the envelope because the
    evidence id is content-derived and `created_at` is deliberately excluded from it. Without a
    varying field in the payload, two identical reads would hash to one id and the store's
    deduplication would silently collapse them, so a hundred reads would leave one row. An audit
    trail that counts wrong is worse than none. ``read_index`` counts within this process, so two
    reads of the same label for the same reason in the same microsecond of two *different*
    processes still collapse to one row. That is the one collision left and it is recorded here
    rather than papered over.
    """

    key: str
    fingerprint: str
    purpose: str
    why: str
    instrument: str
    read_at: str
    read_index: int
    n_labels: int = 1
    error_rate: float | None = None
    n_audited: int = 0


#: The observable name every adjudication row carries. One string, so the reviewer's query is
#: `label_reads(store)`, or `store.find(observable=ADJUDICATION_OBSERVABLE)` for anyone who wants
#: the envelope, rather than a guess about spelling.
ADJUDICATION_OBSERVABLE = "record.label.adjudication"
ADJUDICATION_VERSION = "1"

_read_counter = itertools.count(1)
_store_lock = threading.Lock()
_default_store: "EvidenceStore | None" = None


def _resolve_store(store: "EvidenceStore | None") -> "EvidenceStore":
    """The store an adjudication row goes to: the caller's, or the configured default.

    Cached, because building one streams the whole JSONL to index it and a scoring pass adjudicates
    once per trajectory. Re-resolved when the configured path changes, which is what a test that
    redirects the store does between sessions.
    """
    global _default_store
    if store is not None:
        return store
    from reward_lens.core.config import get_settings
    from reward_lens.core.store import EvidenceStore

    wanted = get_settings().resolved_store()
    with _store_lock:
        if _default_store is None or _default_store.path != wanted:
            _default_store = EvidenceStore(wanted)
        return _default_store


def _as_subject(subject: SubjectRef | str) -> SubjectRef:
    if isinstance(subject, SubjectRef):
        return subject
    return SubjectRef(extra={"trajectory": subject})


def _quality_refusal(*, instrument: str, keys: tuple[str, ...], quality: LabelQuality) -> Refusal:
    """A `LABEL_QUALITY_UNKNOWN` refusal, with the numbers that produced it and a remedy."""
    listed = ", ".join(repr(k) for k in keys)
    return Refusal(
        instrument=instrument,
        reason=RefusalReason.LABEL_QUALITY_UNKNOWN,
        detail=(
            f"label(s) {listed} have no measured error rate: error_rate="
            f"{quality.error_rate!r}, n_audited={quality.n_audited}. Scoring against them would "
            f"measure them rather than the instrument."
        ),
        remedy=(
            "Audit a sample of these labels, then attach the result as "
            "LabelQuality(error_rate=..., n_audited=..., method=..., measured_by=...) to the Blind "
            "or to the OracleFrame. To run that audit, adjudicate with "
            "purpose=ReadPurpose.AUDIT, which opens the labels and records that it did."
        ),
        statistics={
            "keys": list(keys),
            "error_rate": quality.error_rate,
            "n_audited": quality.n_audited,
        },
    )


def _write_row(
    *,
    payload: LabelRead,
    subject: SubjectRef,
    store: "EvidenceStore | None",
) -> "Evidence[LabelRead]":
    ev = make_evidence(
        observable=ADJUDICATION_OBSERVABLE,
        observable_version=ADJUDICATION_VERSION,
        subject=subject,
        value=payload,
        uncertainty=Uncertainty(n=payload.n_labels, method="none"),
        gauge=GaugeStatus.INVARIANT,
        provenance=capture_provenance(),
    )
    _resolve_store(store).append(ev)
    return ev


def adjudicate(
    label: Blind[T],
    *,
    instrument: str,
    purpose: ReadPurpose,
    why: str,
    subject: SubjectRef | str,
    quality: LabelQuality | None = None,
    store: "EvidenceStore | None" = None,
) -> T | Refusal:
    """Open one held-out label, and write the row that says you did.

    Returns the label, or a `Refusal` carrying `LABEL_QUALITY_UNKNOWN` when the purpose is
    `SCORING` and the label has no measured error rate. Nothing is opened and no row is written on
    the refusal path, because nothing was read.

    The return type is `T | Refusal` rather than `T`, which is deliberate: the caller has to narrow
    before using the value, and mypy will not let them skip it. That is the same shape as
    `Reading = Evidence | Refusal` with the envelope left off, since what comes back here is a
    label rather than a measurement.

    Args:
        label: the wrapped label.
        instrument: who is reading, by name. It goes in the row and it is what a reviewer sorts by.
        purpose: `SCORING` or `AUDIT`. See `ReadPurpose`.
        why: one sentence, in the reader's own words, saying what this read is for. Required and
            non-empty for the same reason `Refusal.remedy` is: a trail nobody wrote a reason into
            is a trail nobody can review.
        subject: what the label attaches to. A `SubjectRef`, or a trajectory id as a string.
        quality: overrides the label's own quality, for the frame-level case.
        store: where the row goes. Defaults to the configured evidence store.
    """
    if not why.strip():
        raise ValueError(
            "adjudicate() needs a reason. A row that says a label was opened and not why is a log "
            "line, not an audit trail."
        )
    effective = quality if quality is not None else (label.quality or UNMEASURED)
    if purpose is ReadPurpose.SCORING and not effective.is_measured:
        return _quality_refusal(instrument=instrument, keys=(label.key,), quality=effective)

    _write_row(
        payload=LabelRead(
            key=label.key,
            fingerprint=label.fingerprint,
            purpose=purpose.value,
            why=why,
            instrument=instrument,
            read_at=datetime.now(timezone.utc).isoformat(),
            read_index=next(_read_counter),
            n_labels=1,
            error_rate=effective.error_rate,
            n_audited=effective.n_audited,
        ),
        subject=_as_subject(subject),
        store=store,
    )
    return label._value


def adjudicate_frame(
    frame: OracleFrame,
    *,
    instrument: str,
    purpose: ReadPurpose,
    why: str,
    store: "EvidenceStore | None" = None,
) -> Mapping[str, LabelValue] | Refusal:
    """Open every label of one trajectory, in one row.

    The bulk path, and the one a scoring pass should use. Adjudicating a thousand trajectories one
    label at a time writes a thousand rows and the reviewer has to reconstruct the pass from them;
    this writes one row per trajectory carrying every key and fingerprint it opened, which is the
    granularity the question "what did this evaluation see" is actually asked at.

    Refuses the whole frame if any label in it lacks a measured error rate under `SCORING`, rather
    than returning the subset that qualifies. A partial answer key silently scoped to the labels
    that happened to be audited is a different experiment from the one the caller wrote.
    """
    if not why.strip():
        raise ValueError(
            "adjudicate_frame() needs a reason. A row that says labels were opened and not why is "
            "a log line, not an audit trail."
        )
    keys = tuple(sorted(frame.labels))
    if purpose is ReadPurpose.SCORING:
        unmeasured = tuple(k for k in keys if not frame.quality_of(k).is_measured)
        if unmeasured:
            return _quality_refusal(
                instrument=instrument,
                keys=unmeasured,
                quality=frame.quality_of(unmeasured[0]),
            )

    # The row reports the worst quality in the frame rather than an average, and "worst" puts an
    # unmeasured label ahead of any measured one. On the SCORING path they are all measured by the
    # time execution reaches here, so this only bites on an AUDIT read of a mixed frame, which is
    # exactly where a row saying `error_rate=0.03` would be describing the label that happens to
    # have been audited rather than the ones that have not.
    worst = min(
        (frame.quality_of(k) for k in keys),
        key=lambda q: (q.error_rate is not None, -(q.error_rate or 0.0)),
        default=UNMEASURED,
    )
    _write_row(
        payload=LabelRead(
            key=",".join(keys),
            fingerprint=content_hash({k: frame.labels[k].fingerprint for k in keys}, "lblset"),
            purpose=purpose.value,
            why=why,
            instrument=instrument,
            read_at=datetime.now(timezone.utc).isoformat(),
            read_index=next(_read_counter),
            n_labels=len(keys),
            error_rate=worst.error_rate,
            n_audited=worst.n_audited,
        ),
        subject=_as_subject(frame.trajectory_ref),
        store=store,
    )
    return {k: frame.labels[k]._value for k in keys}


def label_reads(store: "EvidenceStore") -> tuple["Evidence[LabelRead]", ...]:
    """Every adjudication row in a store, oldest first. The reviewer's query.

    This is what the whole module is for. If it comes back empty, nothing opened a label. If it
    comes back with a row whose `instrument` is the detector being scored, that is the finding.
    """
    return tuple(store.find(observable=ADJUDICATION_OBSERVABLE))


__all__ = [
    "ADJUDICATION_OBSERVABLE",
    "ADJUDICATION_VERSION",
    "BLOCKED_NAMES",
    "BLOCKED_PREFIXES",
    "UNMEASURED",
    "Blind",
    "Detector",
    "LabelLeak",
    "LabelQuality",
    "LabelRead",
    "LabelValue",
    "OracleFrame",
    "ReadPurpose",
    "RolloutFrame",
    "adjudicate",
    "adjudicate_frame",
    "blind",
    "blocked_names",
    "check_detector",
    "detector_findings",
    "is_blocked",
    "label_reads",
    "normalise_name",
    "split_trajectory",
]

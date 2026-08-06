"""The verifier series (D1 to D10): instruments that read a grader's source rather than a model.

Most frontier reward now comes from code, and a program has none of the things the rest of this
library measures. No activations, no reward head, no geometry. What it has instead is source, a
control-flow graph, and forty years of software-analysis tooling that has never once been pointed
at a reward function. Two independent literature searches return zero for "mutation testing" AND
"reward function", zero for "reward hacking" AND "software testing", and 88 for "coverage" AND
"reward function" of which every one is coverage used *as* a reward in robotics, none of them
coverage *of* a reward. Nothing in this series had to be built. It had to be aimed differently.

The aiming is the whole idea, and it comes down to one substitution: **the test suite is the
rollout corpus.** A branch is covered if some rollout took it. A mutant is killed if re-grading the
corpus with the mutated verifier changes any score. Under that substitution, `coverage.py` reports
what your grader can distinguish and `cosmic-ray` reports how it could be wrong without any rollout
you have ever collected noticing.

This module holds what the ten instruments share: the corpus and verifier protocols, the program
subject that lets a `Context` describe a piece of code rather than a network, and the sensitivity
flag that keeps a surviving-mutant list out of a rendered card unless somebody asks for it.

**On the corpus protocol.** This series does not depend on `record/`. It needs a verifier's source
and a corpus of scored rollouts, and nothing else, so `Rollout` and `RolloutCorpus` below are a
deliberately small local protocol rather than an import: an adapter from a `record.Run` is a dozen
lines, and keeping the dependency out means these ten instruments run on a corpus that never went
through the record at all. The assumption is written down rather than implied: a rollout is an id, a
mapping of keyword arguments the verifier is called with, and optionally the score the record says
it produced.
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence, cast

from reward_lens.core.extras import require_extra
from reward_lens.core.quantity import QUANTITIES, load_quantities
from reward_lens.core.types import Access, AccessMatrix, Capability, Component, content_hash

require_extra("verifier", subsystem="the verifier series (D1 to D10)")


# ---------------------------------------------------------------------------
# The quantity registry
# ---------------------------------------------------------------------------


def ensure_quantities() -> None:
    """Load `spec/QUANTITIES.yaml` into `QUANTITIES` if it is not there yet.

    Nothing in the kernel populates the registry at import, and `lint_instrument` fails an
    instrument whose quantity is unregistered. Loading is idempotent (the registry refuses only a
    *different* definition under a live name), so the guard is a cheap-path optimisation rather
    than a correctness requirement.
    """
    if "verifier.decision_coverage" not in QUANTITIES:
        load_quantities()


ensure_quantities()


# ---------------------------------------------------------------------------
# The corpus
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Rollout:
    """One graded item: what the verifier was called with, and what it returned.

    ``inputs`` is a mapping of keyword arguments, not a positional tuple, because a verifier's
    signature changes between versions and a positional corpus silently misaligns when it does.

    ``score`` is what the *record* says the verifier produced. It is optional and it is never
    trusted as the baseline for mutation testing: D2 re-runs the unmutated verifier and compares
    against that, so a stale recorded score cannot manufacture a kill. Where a recorded score is
    present it is worth something else, which is a check on replay fidelity (D10's question), and
    `RolloutCorpus.recorded_score_mismatches` is where that check lives.
    """

    id: str
    inputs: Mapping[str, Any] = field(default_factory=dict)
    score: float | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("a rollout needs an id; per-rollout attribution is keyed on it")


class RolloutCorpus(Protocol):
    """What D1, D2, D3, D5 and D10 all need: a finite, re-iterable set of graded items."""

    def __iter__(self) -> Iterator[Rollout]: ...

    def __len__(self) -> int: ...


@dataclass(frozen=True)
class ListCorpus:
    """The in-memory corpus. A `record.Run` adapter goes here when `record/` lands.

    ``checksum`` exists because `Context.subject` reaches for it: it is what puts the corpus's
    identity into the evidence id, so re-running the same instrument on a different corpus produces
    a different row rather than a silent overwrite.
    """

    rollouts: tuple[Rollout, ...]
    dataset_id: str | None = None

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for r in self.rollouts:
            if r.id in seen:
                raise ValueError(
                    f"duplicate rollout id {r.id!r}. Per-rollout branch attribution is keyed on "
                    f"the id, and two rollouts sharing one would merge their arc sets."
                )
            seen.add(r.id)

    @classmethod
    def of(cls, rollouts: Sequence[Rollout], dataset_id: str | None = None) -> "ListCorpus":
        return cls(tuple(rollouts), dataset_id)

    def __iter__(self) -> Iterator[Rollout]:
        return iter(self.rollouts)

    def __len__(self) -> int:
        return len(self.rollouts)

    def checksum(self) -> str:
        material = [
            {"id": r.id, "inputs": _stable(r.inputs), "score": r.score} for r in self.rollouts
        ]
        return content_hash({"rollouts": material}, "corpus")


def _stable(inputs: Mapping[str, Any]) -> Any:
    """A JSON-stable view of a rollout's inputs, for the corpus checksum.

    Inputs are arbitrary user objects. Anything JSON cannot hold is hashed by its repr, which is
    weaker than a structural hash and is recorded as such: two corpora differing only inside an
    unserialisable object get the same checksum if their reprs match.
    """
    try:
        return json.loads(json.dumps(inputs, sort_keys=True, default=repr))
    except (TypeError, ValueError):
        return repr(sorted(inputs.items()))


# ---------------------------------------------------------------------------
# The verifier under test
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VerifierUnderTest:
    """A grader identified by its source file and the name of its scoring function.

    Every instrument in this series starts here. ``entrypoint`` is the attribute in the module that
    gets called with a rollout's ``inputs``; it is also what D2's rung 0 restricts mutation to and
    what D9 treats as the root of the trusted-input analysis.

    ``fingerprint`` is a content hash of the source, and it is the subject identity for the whole
    series. That matters more than it looks: it is what makes the `STATIONARY_GRADER` envelope
    condition checkable at all. A coverage number aggregated over a window in which the grader was
    edited is a number about two different programs.
    """

    source_path: Path
    entrypoint: str = "grade"
    module_name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_path", Path(self.source_path).resolve())
        if not self.source_path.is_file():
            raise FileNotFoundError(
                f"no verifier source at {self.source_path}. This series reads a grader's code; "
                f"if you only have a callable, D1, D2 and D9 do not apply to it."
            )

    def source(self) -> str:
        return self.source_path.read_text(encoding="utf-8")

    @property
    def fingerprint(self) -> str:
        return content_hash({"source": self.source(), "entry": self.entrypoint}, "verifier")

    @property
    def import_name(self) -> str:
        return self.module_name or f"_rl_verifier_{self.fingerprint.split(':')[1][:12]}"

    def load(self) -> Callable[..., Any]:
        """Import the verifier from its file and return the entrypoint callable."""
        return load_entrypoint(self.source(), self.source_path, self.entrypoint, self.import_name)


def load_entrypoint(
    source: str, path: Path, entrypoint: str, module_name: str
) -> Callable[..., Any]:
    """Execute ``source`` as a fresh module and return its entrypoint.

    Deliberately not `importlib.import_module`: D2 needs to load *mutated* source that exists only
    as a string, and needs a fresh module namespace each time so a mutant cannot inherit state from
    the run before it. ``__file__`` is set to the real path so coverage attributes arcs to the file
    on disk and tracebacks point somewhere a reader can open.

    This executes the verifier's module body. That is not a new risk (importing it does the same)
    but under D2 it executes *modified* code, so a verifier with side effects at import time will
    perform them once per mutant. Verifiers with import-time side effects are worth knowing about
    for other reasons; D9's static structure reports module-level statements outside a function.
    """
    mod = ModuleType(module_name)
    mod.__file__ = str(path)
    mod.__dict__["__builtins__"] = __builtins__
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is not None:
        mod.__spec__ = spec
        mod.__loader__ = spec.loader
    code = compile(source, str(path), "exec")
    exec(code, mod.__dict__)  # noqa: S102 - executing the grader is the measurement
    try:
        fn = getattr(mod, entrypoint)
    except AttributeError:
        raise AttributeError(
            f"{path} defines no {entrypoint!r}. Set VerifierUnderTest.entrypoint to the name of "
            f"the scoring function."
        ) from None
    if not callable(fn):
        raise TypeError(f"{path}:{entrypoint} is not callable")
    return cast("Callable[..., Any]", fn)


# ---------------------------------------------------------------------------
# The program subject
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ProgramMeta:
    """The two fields `Context.subject` reads off a signal. Nothing more is claimed."""

    fingerprint: str
    lineage: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProgramSubject:
    """A grader that is a program, shaped so a `Context` can carry it.

    `measure.base.Context` types its subject as a `RewardSignal` and reads `.meta.fingerprint` and
    `.caps` off it. A program satisfies neither the protocol nor the spirit of it, and forcing one
    into `RewardSignal` is what previous versions of this library got wrong. So this is not a
    `RewardSignal` and does not pretend to be: it declares `Capability.SCORES`, which is true (a
    grader returns scores) and is the only capability a program has, and it fingerprints the
    source rather than a checkpoint.

    The friction is real and it is the kernel's, not this series': every PROGRAM-substrate
    instrument needs this shim until `Context` accepts a subject that is not a network.
    """

    verifier: VerifierUnderTest
    caps: Capability = Capability.SCORES
    intervention_fingerprints: tuple[str, ...] = ()

    @property
    def meta(self) -> _ProgramMeta:
        return _ProgramMeta(
            fingerprint=self.verifier.fingerprint,
            lineage={"source_path": str(self.verifier.source_path)},
        )


#: What "GRADER source" costs in the access vocabulary.
#:
#: D1, D2 and D9 need source access, and `Access` used to have no member that said so. The nearest
#: was `MUTATE` ("modify it: patch, ablate, edit, plant, recompile"), which overclaims: D1 and D9
#: read the code and never touch it. `Access.SOURCE` was added for exactly this and is off the
#: containment ladder, because reading a program's text neither follows from nor implies running it.
SOURCE_ACCESS: Access = Access.SOURCE

#: D1: the grader's source plus a record of rollouts.
ACCESS_SOURCE_AND_RECORD: AccessMatrix = {
    Component.GRADER: SOURCE_ACCESS,
    Component.RECORD: Access.RECORD,
}

#: D9: source alone. No rollouts, no run, no record.
ACCESS_SOURCE_ONLY: AccessMatrix = {Component.GRADER: SOURCE_ACCESS}


def program_context(
    verifier: VerifierUnderTest,
    corpus: RolloutCorpus | None = None,
    **kwargs: Any,
) -> Any:
    """A `Context` describing a program and, optionally, the corpus it graded.

    Substrate is fixed to PROGRAM; supply ``phase``, ``access`` and ``regime_reading`` through
    ``kwargs`` when you have them. A preflight that could not check something names it rather than
    counting it as a pass, so leaving them out is honest rather than lenient.
    """
    from reward_lens.core.types import Substrate
    from reward_lens.measure.base import Context

    kwargs.setdefault("substrate", Substrate.PROGRAM)
    # `Context.signal` is typed `RewardSignal`, and a program is not one. The ignore is the visible
    # form of the kernel gap this shim covers: until `Context` accepts a subject that is not a
    # network, every PROGRAM-substrate instrument carries this line.
    return Context(
        signal=ProgramSubject(verifier),  # type: ignore[arg-type]
        view=corpus,
        readout="score",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Dual use
# ---------------------------------------------------------------------------

#: Why a payload carries `sensitive`. Quoted into refusals and into the store row so the reason
#: travels with the data rather than living in a docs page nobody opens.
SENSITIVE_NOTE = (
    "this payload enumerates concrete ways to make the grader wrong. It is written to the store "
    "with a sensitive flag, excluded from rendered artifacts unless explicitly requested, and not "
    "published without a recorded decision."
)


class SensitivePayload(Protocol):
    """An evidence payload that must not be rendered or published by default.

    D2's surviving-mutant list, D5's false-positive catalogue and D8's attack surface are all
    exploit catalogues in the form the reader needs to reproduce them. The flag is a property of
    the payload rather than a convention, and `redact` is what a renderer calls to get a row it can
    show: the counts survive, the reproducers do not.
    """

    sensitive: bool

    def redacted(self) -> Any: ...


def _payload_of(obj: Any) -> Any:
    """The thing that carries the flag: the object itself, or the Evidence's value.

    Checked in that order on purpose. A payload may expose its own ``value`` property for its
    headline number, and reading the flag off a float would silently report every sensitive
    payload as safe to publish.
    """
    if getattr(obj, "sensitive", None) is not None:
        return obj
    inner = getattr(obj, "value", None)
    return obj if inner is None else inner


def is_sensitive(obj: Any) -> bool:
    """Whether a payload, or an Evidence carrying one, is sensitive by default."""
    return bool(getattr(_payload_of(obj), "sensitive", False))


def redact(obj: Any) -> Any:
    """The publishable form of a payload, or of an Evidence carrying one.

    A payload that declares itself sensitive and implements no `redacted()` is a bug, not a pass:
    this raises rather than returning the unredacted payload, because the failure mode of the
    opposite choice is publishing an exploit catalogue.
    """
    payload = _payload_of(obj)
    if not is_sensitive(payload):
        return obj
    fn = getattr(payload, "redacted", None)
    if fn is None:
        raise TypeError(
            f"{type(payload).__name__} declares sensitive=True and implements no redacted(). "
            f"Returning it unchanged would publish it, so this refuses instead."
        )
    reduced = fn()
    return reduced if payload is obj else replace(obj, value=reduced)


#: Stamped into `SubjectRef.extra` so the flag is on the evidence row itself, not only on the
#: payload. The subject is hashed into the evidence id, so a redacted row and its unredacted
#: original are distinct rows rather than one row with two histories.
SENSITIVE_SUBJECT_EXTRA: Mapping[str, str] = {"sensitive": "true"}


__all__ = [
    "ACCESS_SOURCE_AND_RECORD",
    "ACCESS_SOURCE_ONLY",
    "SENSITIVE_NOTE",
    "SENSITIVE_SUBJECT_EXTRA",
    "SOURCE_ACCESS",
    "ListCorpus",
    "ProgramSubject",
    "Rollout",
    "RolloutCorpus",
    "SensitivePayload",
    "VerifierUnderTest",
    "ensure_quantities",
    "is_sensitive",
    "load_entrypoint",
    "program_context",
    "redact",
    # -- D9
    "Arc",
    "RubricClause",
    "StaticStructure",
    "TrustedInput",
    "TrustedInputs",
    "VerifierStructure",
    "analyse_source",
    # -- D1
    "CoverageReading",
    "DecisionCoverage",
    "measure_coverage",
    # -- D2
    "Mutant",
    "MutationReading",
    "MutationScore",
    "SurvivingMutants",
    "run_mutation",
    "ATHERIS_GAP",
    "ContractSensitivity",
    "D3_ENVELOPE",
    "D4_ENVELOPE",
    "D5_ENVELOPE",
    "DisclosureDecision",
    "DisclosureRequired",
    "ExploitFamily",
    "FPCatalogue",
    "FalsePositive",
    "FalsePositiveFuzzing",
    "Grader",
    "GraderLayer",
    "LayerKind",
    "LayerRoute",
    "MetamorphicRelation",
    "MetamorphicReport",
    "MetamorphicViolations",
    "QuerySubject",
    "RelationResult",
    "Reproducer",
    "RubricInput",
    "RubricScorer",
    "SYMBOLIC_TRACTABLE",
    "SearchSpace",
    "SensitivityIndex",
    "SensitivityProfile",
    "SobolSensitivity",
    "StrictReference",
    "SymbolicFinding",
    "Transformation",
    "answer_exploit_families",
    "answer_text_relations",
    "atheris_available",
    "coverage_guided_search",
    "crosshair_available",
    "dose_response_slopes",
    "false_positive_fuzzing",
    "identity_relation",
    "looks_like_source",
    "metamorphic_violations",
    "mutation_space",
    "one_at_a_time",
    "python_source_relations",
    "resolve_grader",
    "route_symbolic",
    "run_crosshair",
    "score",
    "sobol_indices",
    "sobol_problem",
    "sobol_sample",
    "sobol_sensitivity",
    "total_effect",
    # -- D6
    "ACCESS_LOG_ONLY",
    "CrowFit",
    "D6_ENVELOPE",
    "ExploitCoverage",
    "ExploitFamilyCoverage",
    "ExploitFind",
    "ExploitLog",
    "LogSubject",
    "ReliabilityGrowth",
    "chao1_unseen",
    "chao1_unseen_bias_corrected",
    "crow_amsaa",
    "exploit_coverage",
    "exploit_family_coverage",
    "novelty_probability",
    "observations_for_a_doubleton",
    "reliability_growth",
    # -- D8
    "ACCESS_ENV_SOURCE",
    "ACCESS_ENV_SOURCE_AND_QUERY",
    "AttackCheapness",
    "AttackCheapnessRatio",
    "AttackSurface",
    "AttackSurfaceInventory",
    "BoundaryCrossing",
    "CostSample",
    "D8_CHEAPNESS_ENVELOPE",
    "D8_SURFACE_ENVELOPE",
    "ResourceAccess",
    "TaintPath",
    "TaskBoundary",
    "analyse_environment",
    "attack_cheapness",
    "attack_surface",
    # -- D10
    "ACCESS_QUERY_AND_RECORD",
    "D10_ENVELOPE",
    "DEFAULT_SCORE_FLOOR",
    "STATIONARY_GRADER_FLOOR",
    "ReplayFidelity",
    "ReplayReport",
    "TaskReplay",
    "replay_corpus",
    "replay_fidelity",
    "stationary_grader_reading",
]

# Imported last on purpose. The instrument modules import the corpus and subject types from this
# package, so they can only be imported once those names are bound. A sibling adding an instrument
# module puts its import in this block, below this line, for the same reason. Order *within* the
# block is isort's and does not matter: `attack` reaches `fuzz`'s disclosure machinery and `replay`
# reaches `metamorphic`'s grader resolution by absolute module import, which Python satisfies by
# importing that submodule rather than by reading a name off this half-built package.
from reward_lens.verifier.attack import (  # noqa: E402
    ACCESS_ENV_SOURCE,
    ACCESS_ENV_SOURCE_AND_QUERY,
    D8_CHEAPNESS_ENVELOPE,
    D8_SURFACE_ENVELOPE,
    AttackCheapness,
    AttackCheapnessRatio,
    AttackSurface,
    AttackSurfaceInventory,
    BoundaryCrossing,
    CostSample,
    ResourceAccess,
    TaintPath,
    TaskBoundary,
    analyse_environment,
    attack_cheapness,
    attack_surface,
)
from reward_lens.verifier.coverage import (  # noqa: E402
    CoverageReading,
    DecisionCoverage,
    measure_coverage,
)
from reward_lens.verifier.fuzz import (  # noqa: E402
    ATHERIS_GAP,
    D5_ENVELOPE,
    SYMBOLIC_TRACTABLE,
    DisclosureDecision,
    DisclosureRequired,
    ExploitFamily,
    FalsePositive,
    FalsePositiveFuzzing,
    FPCatalogue,
    GraderLayer,
    LayerKind,
    LayerRoute,
    SearchSpace,
    StrictReference,
    SymbolicFinding,
    answer_exploit_families,
    atheris_available,
    coverage_guided_search,
    crosshair_available,
    false_positive_fuzzing,
    mutation_space,
    route_symbolic,
    run_crosshair,
)
from reward_lens.verifier.growth import (  # noqa: E402
    ACCESS_LOG_ONLY,
    D6_ENVELOPE,
    CrowFit,
    ExploitCoverage,
    ExploitFamilyCoverage,
    ExploitFind,
    ExploitLog,
    LogSubject,
    ReliabilityGrowth,
    chao1_unseen,
    chao1_unseen_bias_corrected,
    crow_amsaa,
    exploit_coverage,
    exploit_family_coverage,
    novelty_probability,
    observations_for_a_doubleton,
    reliability_growth,
)

# D3, D4 and D5 re-exported here so the series has one import surface. Ordered after `metamorphic`
# because `sensitivity` and `fuzz` both import from it.
from reward_lens.verifier.metamorphic import (  # noqa: E402
    D3_ENVELOPE,
    Grader,
    MetamorphicRelation,
    MetamorphicReport,
    MetamorphicViolations,
    QuerySubject,
    RelationResult,
    Reproducer,
    Transformation,
    answer_text_relations,
    identity_relation,
    looks_like_source,
    metamorphic_violations,
    python_source_relations,
    resolve_grader,
    score,
)
from reward_lens.verifier.mutate import (  # noqa: E402
    Mutant,
    MutationReading,
    MutationScore,
    SurvivingMutants,
    run_mutation,
)
from reward_lens.verifier.replay import (  # noqa: E402
    ACCESS_QUERY_AND_RECORD,
    D10_ENVELOPE,
    DEFAULT_SCORE_FLOOR,
    STATIONARY_GRADER_FLOOR,
    ReplayFidelity,
    ReplayReport,
    TaskReplay,
    replay_corpus,
    replay_fidelity,
    stationary_grader_reading,
)
from reward_lens.verifier.sensitivity import (  # noqa: E402
    D4_ENVELOPE,
    ContractSensitivity,
    RubricInput,
    RubricScorer,
    SensitivityIndex,
    SensitivityProfile,
    SobolSensitivity,
    dose_response_slopes,
    one_at_a_time,
    sobol_indices,
    sobol_problem,
    sobol_sample,
    sobol_sensitivity,
    total_effect,
)
from reward_lens.verifier.static import (  # noqa: E402
    Arc,
    RubricClause,
    StaticStructure,
    TrustedInput,
    TrustedInputs,
    VerifierStructure,
    analyse_source,
)

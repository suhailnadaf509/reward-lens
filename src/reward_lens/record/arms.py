"""Arms, coupling, common random numbers, and the arm-divergence void gate (sections 6.6, 6.2).

A controlled comparison is two runs that differ in one declared thing. This module exists to make
that sentence true of the record rather than true of the intention of whoever launched it, because
the two come apart constantly: a launcher applies a stale override, a framework default moves
between the two launches, the second arm picks up a newer engine because the first pod was already
warm. None of that shows up in the numbers. All of it changes them.

**Common random numbers are the default here, and they are free.** For two arms measured on the
same axis, `Var(X_A - X_B) = Var(X_A) + Var(X_B) - 2·Cov(X_A, X_B)`. Independent seeds put that
covariance at zero. Driving both arms from one stream makes it positive, and the variance of the
difference falls by twice it, with no change to either arm's marginal distribution: coupling two
runs is not the same as biasing them. The cost is bookkeeping, which is what `CouplingSpec` is.
So `arms(...)` couples by default and a user who wants independent arms passes
`coupling=independent_seeds()` and says so in the record.

The honest limit, which belongs next to the claim rather than in a footnote. Coupling survives
only as long as the two runs stay on the same path. The prompt axis stays coupled throughout,
because the prompt order is drawn from a stream neither policy influences, so pairing over prompts
pays for the whole run. Sampling noise inside a rollout decouples as soon as the two policies
disagree on a token, so on a training run the sampling half of the coupling buys the most early
and decays. It never costs anything, and reporting it as if it held to the end would be wrong.

**The gate.** Void condition 4 is "two arms of a controlled comparison differ in configuration
beyond the declared contrast". `check_divergence` flattens both arms to dotted keys, diffs them
against the declared contrast set, and returns a report. A void that says the arms
diverged is worth nothing. This one names the key and prints both values, so the remedy is an
instruction: hold `optim.lr` at one value and re-run, or declare it in the contrast and re-freeze.
The vocabulary is the one already declared in `studies/void.py`; nothing here invents a second one.
`report.void` is a `Void(ARM_DIVERGENCE)` for a study, and `report.refusal(...)` is the same fact
as a `Refusal(VOID)` for a measurement path, which is how an unreadable run is routed.

**Staggered adoption, which this module does not solve.** There is a case where an intervention is
introduced partway through a run: verifier A at step 1000, verifier B at step 3000, `tau` annealed
at 5000. That is a staggered adoption design, and the estimator everyone reaches for
is the one shown to be broken. Two-way fixed effects is a weighted average of every 2x2 comparison
available, including comparisons of a newly treated unit against one already treated, and those
weights can be negative, so the estimate can carry the opposite sign to every underlying effect.
The corrected estimators exist (Callaway and Sant'Anna, Sun and Abraham, de Chaisemartin and
D'Haultfoeuille) and no machine-learning paper appears to use them. None of them is implemented
here. What is implemented is the part that makes the question askable: an arm carries the step at
which its intervention landed, in `Arm.adopted_at`, so a comparison of arms with different adoption
steps is visible in the record instead of being averaged away. Reading it correctly is open work.

Torch-free, like the rest of `record`. `CouplingSpec` sits on `Run.coupling`, is a registered
payload, and round-trips through the writer and reader with no change to either.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, NewType

import numpy as np

from reward_lens.core.evidence import register_payload
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.types import canonical_bytes, content_hash, hash_bytes
from reward_lens.record.schema import Run, RunID

if TYPE_CHECKING:  # the Void import is deferred; see `ArmDivergence.void` for why
    from reward_lens.studies.void import Void, VoidReason

#: An arm's identity, derived from its configuration and its coupling. Two arms with the same
#: material share an id, which is the property the divergence gate leans on: equal ids mean there
#: is nothing to diff.
ArmID = NewType("ArmID", str)

#: The base seed when nobody names one. Its value does not matter and its sharing does, which is
#: the whole content of common random numbers. Recorded either way so the comparison replays.
DEFAULT_SEED = 0

#: The stream that stays coupled for the entire run, because neither policy influences it.
PROMPT_ORDER_STREAM = "prompt_order"

#: Streams a coupling claims to drive, by default. A framework that will not accept an injected
#: seed for one of these should have it removed from the declaration rather than left in, because
#: `streams` is a claim about what is actually coupled and it is compared across arms like any
#: other key.
DEFAULT_STREAMS: tuple[str, ...] = (PROMPT_ORDER_STREAM, "sampling", "init", "dropout")

#: Derived seeds are masked to 63 bits so they survive an int64 column and every framework's
#: `--seed` argument. numpy would take the full 128, and nothing else downstream would.
_SEED_MASK = (1 << 63) - 1

#: Rendered values longer than this are truncated in a void detail. The detail is read by a person
#: holding a failed comparison, and a 4 kB nested list pasted into it hides the other keys.
_VALUE_LIMIT = 72


class ArmSpecError(ValueError):
    """An arm set that could not be built, or a comparison asked for something incoherent.

    This is an exception rather than a refusal on purpose, and the line is the one the instrument
    contract draws: a refusal is an anticipated measurement condition, and everything raised here
    is a caller mistake found before anything ran. A contrast naming a key that is not in the base
    configuration is a typo, not a hard case, and a typo that returned a value would be discovered
    later as a mysterious void.
    """


@register_payload
@dataclass(frozen=True)
class Absent:
    """A key one arm has and the other does not.

    A distinct type rather than `None`, because `None` is a legal configuration value and a run
    that sets `lora: null` in one arm and omits the key in the other has done two different things
    that a framework may resolve differently. Prints as ``<absent>``.
    """

    def __repr__(self) -> str:
        return "<absent>"


#: The sentinel. `Absent` is a frozen dataclass with no fields, so every instance compares equal.
ABSENT = Absent()


# ---------------------------------------------------------------------------
# Flattening: a configuration as dotted keys, because that is what a remedy names
# ---------------------------------------------------------------------------


def flatten_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """A nested configuration as a flat map of dotted paths to leaf values.

    Mappings are recursed into and everything else is a leaf, including lists. The reason the
    output is flat is the remedy: "the arms differ under `optim`" is not something a user can act
    on, and "`optim.lr` is 1e-06 in one arm and 2e-06 in the other" is.

    A configuration that reaches the same dotted path two ways, by carrying both a literal
    ``"a.b"`` key and a nested ``{"a": {"b": ...}}``, raises rather than letting one silently win.
    An empty sub-mapping is kept as a leaf ``{}`` so that dropping every key under a section is a
    visible difference rather than a section that quietly disappears.
    """
    out: dict[str, Any] = {}
    _flatten_into(config, "", out)
    return out


def _flatten_into(mapping: Mapping[str, Any], prefix: str, out: dict[str, Any]) -> None:
    for raw_key, value in mapping.items():
        key = str(raw_key)
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping) and value:
            _flatten_into(value, path, out)
            continue
        leaf: Any = {} if isinstance(value, Mapping) else value
        if path in out:
            raise ArmSpecError(
                f"the configuration reaches {path!r} two ways, once as a literal dotted key and "
                f"once through nesting. Rename one of them: with both present there is no way to "
                f"say which one a contrast or a divergence is talking about."
            )
        out[path] = leaf


def _rebuild(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Copy the mapping spine and leave the leaves alone.

    Not `deepcopy`. A configuration can hold objects a user handed in (a tokenizer, an open
    dataset) and copying those would be both expensive and wrong; what has to be copied is the
    dictionary structure, so that writing into one arm cannot reach into another.
    """
    return {str(k): (_rebuild(v) if isinstance(v, Mapping) else v) for k, v in mapping.items()}


def _set_path(target: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    node = target
    for depth, part in enumerate(parts[:-1]):
        nxt = node.get(part)
        if nxt is None and part not in node:
            nxt = {}
            node[part] = nxt
        elif isinstance(nxt, Mapping):
            nxt = nxt if isinstance(nxt, dict) else _rebuild(nxt)
            node[part] = nxt
        else:
            here = ".".join(parts[: depth + 1])
            raise ArmSpecError(
                f"cannot set {path!r}: {here!r} holds {type(nxt).__name__} rather than a section, "
                f"so descending into it would overwrite a value. Check the key against the base "
                f"configuration."
            )
        node = nxt
    node[parts[-1]] = value


def apply_overrides(config: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """A copy of ``config`` with dotted-path ``overrides`` applied. What a launcher does."""
    out = _rebuild(config)
    for path, value in overrides.items():
        _set_path(out, path, value)
    return out


# ---------------------------------------------------------------------------
# Coupling and common random numbers
# ---------------------------------------------------------------------------


@register_payload
@dataclass(frozen=True)
class CouplingSpec:
    """What two arms share, declared, so that "comparable" is a checked property.

    ``common_random_numbers`` and ``shared_prompt_order`` are separate flags and neither gates the
    other. That is not tidiness: the half-way house is real and common. A framework will often let
    you fix the data order and give you no way to inject the sampling seed, and a design that pairs
    the prompts and leaves the sampling noise free is still worth most of the variance reduction.
    Two flags say which of the two you got. One flag would have made the record claim both.

    ``streams`` is the declaration of which random streams this coupling actually drives. It is
    compared across arms like every other field, so an arm whose framework silently dropped the
    dropout seed shows up as a divergence rather than as noise.

    ``shared`` is for anything else the two arms must hold equal: a prompt-set hash, a dataset
    revision, a container digest. It is compared by equality, key by key.
    """

    seed: int = DEFAULT_SEED
    common_random_numbers: bool = True
    shared_prompt_order: bool = True
    engine: str = ""
    engine_revision: str = ""
    streams: tuple[str, ...] = DEFAULT_STREAMS
    shared: Mapping[str, Any] = field(default_factory=dict)
    note: str = ""

    def __post_init__(self) -> None:
        # The codec decodes a tuple as a list and a mapping as a dict, so a spec that has been
        # through the writer and back must land on the same types or equality stops holding.
        object.__setattr__(self, "streams", tuple(str(s) for s in self.streams))
        object.__setattr__(self, "shared", dict(self.shared))
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ArmSpecError(
                f"the coupling seed must be a non-negative int; got {self.seed!r}. Every arm's "
                f"stream seeds are derived from it, so it has to be something a record can store "
                f"and a second run can repeat."
            )

    # -- the declaration ------------------------------------------------------

    def covers(self, stream: str) -> bool:
        """Whether this coupling claims to drive ``stream`` at all."""
        return stream in self.streams

    def shares(self, stream: str) -> bool:
        """Whether ``stream`` is drawn identically in every arm.

        The prompt-order stream answers to `shared_prompt_order` and everything else answers to
        `common_random_numbers`.
        """
        if stream == PROMPT_ORDER_STREAM:
            return bool(self.shared_prompt_order)
        return bool(self.common_random_numbers)

    # -- the mechanism --------------------------------------------------------

    def seed_for(self, stream: str = "sampling", *, arm: str | None = None, index: int = 0) -> int:
        """The seed for one stream, in one arm, at one point in the run.

        Derived by hashing rather than by adding, so that the streams of two arms are independent
        of each other while a shared stream is bit-identical across arms. ``index`` separates
        draws that must not repeat within a run: the epoch for a prompt order, the step for a
        sampling seed.

        When the stream is shared the arm is left out of the material, which is what makes the two
        arms see the same numbers. When it is not shared, the arm is required: deriving an
        "independent" seed without it would hand both arms the same stream and silently turn the
        independence back into coupling, which is the failure this method exists to prevent.
        """
        material: dict[str, Any] = {
            "seed": int(self.seed),
            "stream": str(stream),
            "index": int(index),
        }
        if not self.shares(stream):
            if arm is None:
                raise ArmSpecError(
                    f"stream {stream!r} is not shared across arms under this coupling, so a seed "
                    f"for it needs the arm it belongs to. Pass arm=<the arm's id or label>. "
                    f"Without it both arms would draw the same stream and the arms would be "
                    f"coupled after all."
                )
            material["arm"] = str(arm)
        digest = content_hash(material, "crn").split(":", 1)[1]
        return int(digest[:16], 16) & _SEED_MASK

    def rng(
        self, stream: str = "sampling", *, arm: str | None = None, index: int = 0
    ) -> np.random.Generator:
        """A numpy generator for one stream. Identical across arms when the stream is shared."""
        return np.random.default_rng(self.seed_for(stream, arm=arm, index=index))

    def prompt_order(self, n: int, *, arm: str | None = None, epoch: int = 0) -> tuple[int, ...]:
        """A permutation of ``range(n)``, identical in every arm when the order is shared.

        This is the coupling that pays for the whole run. Prompt difficulty is usually the largest
        single variance term in a reward measurement, and it is entirely common to both arms when
        both arms see the same prompts in the same order, so it cancels in the paired difference.
        """
        if n < 0:
            raise ArmSpecError(f"prompt count cannot be negative; got {n}")
        gen = self.rng(PROMPT_ORDER_STREAM, arm=arm, index=epoch)
        return tuple(int(i) for i in gen.permutation(n))

    def __canonical__(self) -> dict[str, Any]:
        return {
            "seed": int(self.seed),
            "common_random_numbers": bool(self.common_random_numbers),
            "shared_prompt_order": bool(self.shared_prompt_order),
            "engine": self.engine,
            "engine_revision": self.engine_revision,
            "streams": list(self.streams),
            "shared": dict(self.shared),
            "note": self.note,
        }


def shared_seeds(
    seed: int = DEFAULT_SEED,
    *,
    engine: str = "",
    engine_revision: str = "",
    streams: Sequence[str] = DEFAULT_STREAMS,
    shared: Mapping[str, Any] | None = None,
    note: str = "",
) -> CouplingSpec:
    """Common random numbers across arms. The default, and the thing worth defaulting to.

    ``engine`` takes `record.tensors.Engine.id`, which is the name and revision and attention
    implementation together, because two arms served by different attention kernels are not the
    same experiment even when every configuration key matches.
    """
    return CouplingSpec(
        seed=seed,
        common_random_numbers=True,
        shared_prompt_order=True,
        engine=engine,
        engine_revision=engine_revision,
        streams=tuple(streams),
        shared=dict(shared or {}),
        note=note,
    )


def independent_seeds(
    seed: int = DEFAULT_SEED,
    *,
    engine: str = "",
    engine_revision: str = "",
    streams: Sequence[str] = DEFAULT_STREAMS,
    shared: Mapping[str, Any] | None = None,
    note: str = "",
) -> CouplingSpec:
    """Independent draws in every arm, declared rather than defaulted into.

    There are real reasons to want this: an estimator whose standard error assumes independence,
    or a check that a result is not an artifact of one shared draw. The point of making it a named
    call is that the record then says the variance reduction was declined on purpose.
    """
    return CouplingSpec(
        seed=seed,
        common_random_numbers=False,
        shared_prompt_order=False,
        engine=engine,
        engine_revision=engine_revision,
        streams=tuple(streams),
        shared=dict(shared or {}),
        note=note,
    )


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------


def _canonical_value(value: Any) -> Any:
    """A configuration value in a form the kernel's canonical serialiser will take.

    numpy leaks into configurations constantly and neither half of it survives on its own: a sweep
    library hands back `np.int64` for an integer knob, and a curriculum carries its stage
    boundaries as an array. Neither is a reason an arm cannot have an identity, so both are
    converted here rather than refused. An array is identified by its dtype, its shape and a hash
    of its bytes, which is exactly as much of it as an identity needs.
    """
    if isinstance(value, np.ndarray):
        return {
            "__ndarray__": {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "sha": hash_bytes(np.ascontiguousarray(value).tobytes(), "arr"),
            }
        }
    if isinstance(value, np.generic):
        return value.item()
    return value


def _hashable(value: Any) -> bool:
    try:
        canonical_bytes({"v": value})
    except TypeError:
        return False
    return True


def _hash_material(material: Mapping[str, Any], prefix: str) -> str:
    prepared = {k: _canonical_value(v) for k, v in material.items()}
    try:
        return content_hash(prepared, prefix)
    except TypeError as exc:
        bad = sorted(k for k, v in prepared.items() if not _hashable(v))
        shown = ", ".join(f"{k} ({type(material[k]).__name__})" for k in bad[:3])
        raise ArmSpecError(
            f"this configuration cannot be hashed, so the arms cannot be compared by content. "
            f"The keys that fail are: {shown or 'not identifiable'}. Replace them with something "
            f"a record can store, such as a name, a path or a content hash of the object."
        ) from exc


@register_payload
@dataclass(frozen=True)
class Arm:
    """One side of a controlled comparison: a configuration, its coupling, and what it varies.

    ``contrast`` is the declaration and not a derived view of the configuration. It has to be
    stored separately because the gate needs to know which keys were *allowed* to differ even in
    the case where they happen not to, and a set of keys read back off a diff cannot tell those
    apart.

    ``id`` and ``config_hash`` are computed on access rather than stored. A frozen dataclass does
    not freeze the dictionary inside it, so a stored hash would go stale the first time somebody
    wrote into `config`, and a stale content hash is worse than no content hash.

    ``adopted_at`` is the optimizer step at which this arm's intervention was introduced, for the
    staggered-adoption case in the module docstring. `None` means the arm was one thing for its
    whole length, which is the case every estimator here is safe for.
    """

    label: str
    config: Mapping[str, Any]
    contrast: Mapping[str, Any] = field(default_factory=dict)
    coupling: CouplingSpec = field(default_factory=CouplingSpec)
    index: int = 0
    run_id: RunID | None = None
    adopted_at: int | None = None

    def __post_init__(self) -> None:
        # The spine is copied so that writing into the configuration this arm was built from
        # cannot reach into the arm. The leaves are not copied: they can be anything the caller
        # handed in and copying them would be both expensive and surprising.
        object.__setattr__(self, "config", _rebuild(self.config))
        object.__setattr__(self, "contrast", dict(self.contrast))
        if not str(self.label).strip():
            raise ArmSpecError(
                "an arm needs a label. It is what the void detail names when this arm diverges "
                "from its sibling, and 'arm 0 differs from arm 1' is not a work item."
            )

    @property
    def flat_config(self) -> dict[str, Any]:
        """The configuration as dotted keys."""
        return flatten_config(self.config)

    @property
    def material(self) -> dict[str, Any]:
        """Everything two arms are compared on: the configuration and the coupling.

        The coupling is folded in under a ``coupling.`` prefix rather than compared separately, so
        a mismatched engine revision is reported in exactly the same shape as a mismatched
        learning rate. ``coupling.note`` is left out: it is prose for a human and two arms whose
        notes are worded differently are not two different experiments.
        """
        out = self.flat_config
        canonical = self.coupling.__canonical__()
        canonical.pop("note", None)
        shared = canonical.pop("shared", {})
        for key, value in canonical.items():
            out[f"coupling.{key}"] = tuple(value) if isinstance(value, list) else value
        for key, value in flatten_config(shared).items():
            out[f"coupling.shared.{key}"] = value
        return out

    @property
    def config_hash(self) -> str:
        """A content hash of the configuration alone. What goes in `RunLineage.config_hash`."""
        return _hash_material(self.flat_config, "cfg")

    @property
    def id(self) -> ArmID:
        """A content hash of the configuration and the coupling together."""
        return ArmID(_hash_material(self.material, "arm"))

    def overridden(self, changes: Mapping[str, Any], *, label: str | None = None) -> "Arm":
        """This arm with dotted-path ``changes`` applied. What a launch-time override does.

        The reason this exists in the library rather than in a test helper is that the override is
        where arms actually diverge in the field, and an override applied through this method
        lands in a new `Arm` whose material the gate can see. An override applied to a YAML file
        on the way to a trainer does not.
        """
        return Arm(
            label=self.label if label is None else label,
            config=apply_overrides(self.config, changes),
            contrast=dict(self.contrast),
            coupling=self.coupling,
            index=self.index,
            run_id=self.run_id,
            adopted_at=self.adopted_at,
        )

    def with_run_id(self, run_id: str) -> "Arm":
        """This arm, bound to the run that executed it."""
        return Arm(
            label=self.label,
            config=dict(self.config),
            contrast=dict(self.contrast),
            coupling=self.coupling,
            index=self.index,
            run_id=RunID(str(run_id)),
            adopted_at=self.adopted_at,
        )

    def seed_for(self, stream: str = "sampling", *, index: int = 0) -> int:
        """This arm's seed for a stream. Equal to its sibling's when the stream is shared.

        The label is what distinguishes the arms under an uncoupled stream, not the id. The id
        moves when the configuration moves, so an arm relaunched with a corrected flag would draw
        a different stream than the one it was supposed to replicate, and the whole point of
        recording a seed is that the second run repeats the first.
        """
        return self.coupling.seed_for(stream, arm=self.label, index=index)

    def rng(self, stream: str = "sampling", *, index: int = 0) -> np.random.Generator:
        return self.coupling.rng(stream, arm=self.label, index=index)

    def prompt_order(self, n: int, *, epoch: int = 0) -> tuple[int, ...]:
        return self.coupling.prompt_order(n, arm=self.label, epoch=epoch)


@register_payload
@dataclass(frozen=True)
class ArmSet:
    """The arms of one controlled comparison, plus the contrast they were allowed to differ in.

    Iterable and indexable, so ``control, treated = arms(...)`` reads the way the design does.
    """

    arms: tuple[Arm, ...]
    contrast_keys: tuple[str, ...]
    coupling: CouplingSpec = field(default_factory=CouplingSpec)
    design: Literal["paired", "crossed"] = "paired"

    def __post_init__(self) -> None:
        object.__setattr__(self, "arms", tuple(self.arms))
        object.__setattr__(self, "contrast_keys", tuple(str(k) for k in self.contrast_keys))
        if len(self.arms) < 2:
            raise ArmSpecError(
                f"a controlled comparison needs at least two arms; got {len(self.arms)}. One arm "
                f"is a run, not a comparison."
            )
        seen: dict[str, str] = {}
        for arm in self.arms:
            if arm.id in seen:
                raise ArmSpecError(
                    f"arms {seen[arm.id]!r} and {arm.label!r} have identical configurations, so "
                    f"the difference between them is noise and nothing else. Give the contrast "
                    f"two different values, or drop one arm."
                )
            seen[arm.id] = arm.label

    def __iter__(self) -> Iterator[Arm]:
        return iter(self.arms)

    def __len__(self) -> int:
        return len(self.arms)

    def __getitem__(self, index: int) -> Arm:
        return self.arms[index]

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(a.label for a in self.arms)

    def by_label(self, label: str) -> Arm:
        for arm in self.arms:
            if arm.label == label:
                return arm
        raise KeyError(f"no arm labelled {label!r}; this set has {list(self.labels)}")

    def sibling_run_ids(self, arm: Arm) -> tuple[RunID, ...]:
        """The other arms' run ids, for `Run.arms`.

        Arms whose run has not been bound yet are left out rather than filled with a placeholder,
        so a partially launched comparison records what exists instead of what was planned.
        """
        return tuple(a.run_id for a in self.arms if a.run_id is not None and a.id != arm.id)

    def check(self) -> tuple["ArmDivergence", ...]:
        """Every pairwise divergence report, in order. The gate for the whole set."""
        out: list[ArmDivergence] = []
        for i, left in enumerate(self.arms):
            for right in self.arms[i + 1 :]:
                out.append(check_divergence(left, right, contrast=self.contrast_keys))
        return tuple(out)

    @property
    def ok(self) -> bool:
        return all(r.ok for r in self.check())

    def voids(self) -> dict[str, "Void"]:
        """The voids this set carries, keyed by the pair that produced them.

        Empty when every pair differs only in the declared contrast. The keys are meant to be
        merged into a study's own void map, which is why they name the pair rather than an index.
        """
        out: dict[str, "Void"] = {}
        for report in self.check():
            void = report.void
            if void is not None:
                out[f"{report.left} vs {report.right}"] = void
        return out


def arms(
    base_config: Mapping[str, Any],
    *,
    contrast: Mapping[str, Sequence[Any]],
    coupling: CouplingSpec | None = None,
    labels: Sequence[str] | None = None,
    design: Literal["paired", "crossed"] = "paired",
    allow_new_keys: bool = False,
    adopted_at: Sequence[int | None] | None = None,
) -> ArmSet:
    """Build the arms of a controlled comparison from one base configuration and one contrast.

        arms(base_config, contrast={"beta_kl": [0.0, 0.02]}, coupling=shared_seeds())

    Each contrast entry maps a dotted configuration key to one value per arm, so the example above
    gives two arms that differ in `beta_kl` and in nothing else, by construction. With several
    contrast keys the default ``design="paired"`` reads them down the columns: entry `i` of every
    list belongs to arm `i`, and every list has to be the same length. ``design="crossed"`` takes
    the Cartesian product instead, which is the 2x2 anyone wants when the question is whether two
    interventions interact.

    ``coupling`` defaults to `shared_seeds()`. Common random numbers are the default because they
    cost nothing and reduce the variance of the arm difference, and because a default nobody
    changes should be the one that is right more often.

    A contrast key that is not already in the base configuration raises. It is the likeliest typo
    in the whole call and it fails quietly otherwise: a misspelled key produces two arms that both
    carry the framework's default for the key you meant plus a new key nothing reads, and the
    comparison then measures nothing. Pass ``allow_new_keys=True`` when the arms really are meant
    to introduce a key the base does not have.
    """
    if not contrast:
        raise ArmSpecError(
            "a controlled comparison needs a declared contrast: the one thing the arms are "
            "allowed to differ in. Pass contrast={'key': [value_a, value_b]}. To build "
            "configurations that are not a controlled comparison, construct Arm objects directly."
        )

    flat_base = flatten_config(base_config)
    axes: dict[str, tuple[Any, ...]] = {}
    for key, values in contrast.items():
        path = str(key)
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise ArmSpecError(
                f"the contrast for {path!r} must be a sequence with one value per arm; got "
                f"{type(values).__name__}. A single value is a change to the base configuration, "
                f"not a contrast."
            )
        column = tuple(values)
        if len(column) < 2:
            raise ArmSpecError(
                f"the contrast for {path!r} has {len(column)} value(s) and a comparison needs at "
                f"least two. A key that takes one value in every arm belongs in the base "
                f"configuration."
            )
        if path not in flat_base and not allow_new_keys:
            near = _nearest_key(path, flat_base)
            hint = f" The closest key in the base configuration is {near!r}." if near else ""
            raise ArmSpecError(
                f"the contrast names {path!r}, which the base configuration does not have.{hint} "
                f"Add it to the base configuration, or pass allow_new_keys=True if the arms are "
                f"meant to introduce it."
            )
        axes[path] = column

    combinations = _combinations(axes, design)
    coupling = shared_seeds() if coupling is None else coupling

    if labels is not None and len(labels) != len(combinations):
        raise ArmSpecError(
            f"{len(labels)} label(s) for {len(combinations)} arm(s). The {design} design over "
            f"{len(axes)} contrast key(s) gives {len(combinations)} arms."
        )
    if adopted_at is not None and len(adopted_at) != len(combinations):
        raise ArmSpecError(
            f"{len(adopted_at)} adoption step(s) for {len(combinations)} arm(s); pass one per arm "
            f"or none at all."
        )

    built: list[Arm] = []
    for i, assignment in enumerate(combinations):
        built.append(
            Arm(
                label=labels[i] if labels is not None else _default_label(assignment),
                config=apply_overrides(base_config, assignment),
                contrast=dict(assignment),
                coupling=coupling,
                index=i,
                adopted_at=None if adopted_at is None else adopted_at[i],
            )
        )
    return ArmSet(arms=tuple(built), contrast_keys=tuple(axes), coupling=coupling, design=design)


def _combinations(axes: Mapping[str, tuple[Any, ...]], design: str) -> tuple[dict[str, Any], ...]:
    if design == "paired":
        lengths = {k: len(v) for k, v in axes.items()}
        if len(set(lengths.values())) > 1:
            shown = ", ".join(f"{k} has {n}" for k, n in sorted(lengths.items()))
            raise ArmSpecError(
                f"a paired design needs one value per arm from every contrast key, so all the "
                f"lists must be the same length: {shown}. Pass design='crossed' for the factorial "
                f"design over these keys."
            )
        n = next(iter(lengths.values()))
        return tuple({k: v[i] for k, v in axes.items()} for i in range(n))
    if design == "crossed":
        out: list[dict[str, Any]] = [{}]
        for key, values in axes.items():
            out = [{**partial, key: value} for partial in out for value in values]
        return tuple(out)
    raise ArmSpecError(f"design must be 'paired' or 'crossed'; got {design!r}")


def _default_label(assignment: Mapping[str, Any]) -> str:
    return ",".join(f"{k}={_short(v, 24)}" for k, v in assignment.items())


def _nearest_key(path: str, keys: Iterable[str]) -> str | None:
    """The base-configuration key a mistyped contrast key most likely meant.

    Deliberately crude: the leaf name has to match exactly, which catches the two mistakes that
    actually happen (the wrong section, or a missing section prefix) and stays silent rather than
    guessing at anything else.
    """
    leaf = path.rsplit(".", 1)[-1]
    for candidate in keys:
        if candidate != path and candidate.rsplit(".", 1)[-1] == leaf:
            return candidate
    return None


# ---------------------------------------------------------------------------
# The arm-divergence void gate (void condition 4)
# ---------------------------------------------------------------------------


def _short(value: Any, limit: int = _VALUE_LIMIT) -> str:
    text = repr(value)
    if len(text) > limit:
        text = text[: max(limit - 3, 1)] + "..."
    return text


def _differs(left: Any, right: Any) -> bool:
    """Whether two configuration values are different, without trusting `!=` to return a bool.

    A numpy array under `!=` returns an array, and `bool()` of one with more than one element
    raises. Both branches below are reached by real configurations: an array of stage boundaries
    in a curriculum, and objects whose `__eq__` is not defined at all.

    Two nan floats count as the same declaration, because two arms that both set a threshold to
    nan have declared the same thing and `nan != nan` would otherwise report a divergence with
    identical values printed on both sides, which is the least actionable message available. The
    same fix is not applied inside an array: an array of nans still compares unequal, which is a
    limitation and not a decision.
    """
    if left is right:
        return False
    if isinstance(left, float) and isinstance(right, float):
        if left != left and right != right:
            return False
        return left != right
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not (isinstance(left, np.ndarray) and isinstance(right, np.ndarray)):
            return True
        return (
            left.shape != right.shape
            or left.dtype != right.dtype
            or not bool(np.array_equal(left, right))
        )
    try:
        return bool(left != right)
    except (TypeError, ValueError):
        pass
    try:
        return canonical_bytes(left) != canonical_bytes(right)
    except TypeError:
        return repr(left) != repr(right)


@register_payload
@dataclass(frozen=True)
class KeyDivergence:
    """One key on which two arms differ, with both values.

    The whole value of the void gate is in this object. "The arms diverged" is a complaint; "the
    arms diverged on `optim.lr`, 1e-06 against 2e-06" is a work item with an obvious next action.
    """

    key: str
    left: Any
    right: Any

    def render(self) -> str:
        return f"{self.key} {_short(self.left)} against {_short(self.right)}"


@register_payload
@dataclass(frozen=True)
class ArmDivergence:
    """The result of comparing two arms. A report either way, not a bool.

    A clean comparison carries the counts that made it clean, because "these arms were compared on
    47 keys and differ only in `beta_kl`" is the sentence a card should be able to print. A dirty
    one carries every diverging key and both values.

    ``contrast_inert`` is the mirror image and it voids too, under condition 8: contrast keys that
    were declared and did not actually differ between these two arms. On a set built by `arms()` it
    is always empty, because that constructor refuses identical arms. On two runs read back off disk
    it is the signature of an override that never reached the trainer, and it means the comparison
    has no contrast rather than too much. It is the worse of the two failures, because a divergence
    announces itself while an inert contrast produces a clean-looking null.

    ``adoption`` carries both arms' `adopted_at`, and `staggered` says they differ. Neither is a
    divergence: the timing of an intervention is not a configuration key, and in a staggered
    design the timing is the treatment. It is surfaced because the estimator changes when it is
    true, and because a difference of timing averaged into a before-and-after estimate is the
    error the module docstring describes. Reporting it is as far as this module goes.
    """

    left: str
    right: str
    left_id: str
    right_id: str
    contrast: tuple[str, ...]
    n_keys: int
    diverging: tuple[KeyDivergence, ...]
    contrast_realised: tuple[KeyDivergence, ...] = ()
    contrast_inert: tuple[str, ...] = ()
    adoption: tuple[int | None, int | None] = (None, None)

    def __post_init__(self) -> None:
        object.__setattr__(self, "contrast", tuple(self.contrast))
        object.__setattr__(self, "diverging", tuple(self.diverging))
        object.__setattr__(self, "contrast_realised", tuple(self.contrast_realised))
        object.__setattr__(self, "contrast_inert", tuple(self.contrast_inert))
        object.__setattr__(self, "adoption", tuple(self.adoption))

    @property
    def staggered(self) -> bool:
        """Whether the two arms adopted their intervention at different steps."""
        return self.adoption[0] != self.adoption[1]

    @property
    def inert(self) -> bool:
        """Whether a contrast was declared and not one of its keys actually differs.

        Deliberately not "some contrast key did not differ", which is the ordinary state of a
        crossed design: in a two-by-two, the pair that varies the learning rate holds `beta_kl`
        fixed by construction, and that pair is a legitimate comparison. What is not a comparison
        is two arms where *nothing* in the declared contrast moved, which on a constructed
        `ArmSet` cannot happen because `arms()` refuses identical arms, and on two runs read back
        off disk is the signature of an override that never reached the trainer.
        """
        return bool(self.contrast) and not self.contrast_realised

    @property
    def ok(self) -> bool:
        """Whether these two arms differ only in the declared contrast, and differ in some of it.

        Both halves are required. Arms that differ outside the contrast are not attributable, and
        arms that differ nowhere inside it were never contrasted at all.
        """
        return not self.diverging and not self.inert

    @property
    def keys(self) -> tuple[str, ...]:
        """The diverging keys, named. This is what a remedy has to hold fixed."""
        return tuple(d.key for d in self.diverging)

    @property
    def reason(self) -> "VoidReason | None":
        """Which void condition this report is, or None when the arms are comparable.

        Divergence is checked first when both are true. An arm that diverges outside the contrast
        is unreadable whatever the contrast did, so naming the inert contrast first would send the
        reader to fix the smaller of two problems.
        """
        from reward_lens.studies.void import VoidReason

        if self.diverging:
            return VoidReason.ARM_DIVERGENCE
        if self.inert:
            return VoidReason.CONTRAST_INERT
        return None

    @property
    def detail(self) -> str:
        contrast = ", ".join(self.contrast) if self.contrast else "nothing"
        if self.diverging:
            listed = "; ".join(d.render() for d in self.diverging)
            return (
                f"arms {self.left!r} and {self.right!r} were declared to differ in {contrast} and "
                f"differ in {len(self.diverging)} further key(s) out of {self.n_keys} compared: "
                f"{listed}. A difference measured between them is not attributable to the contrast."
            )
        inert = ", ".join(self.contrast_inert)
        return (
            f"arms {self.left!r} and {self.right!r} were declared to differ in {contrast}, and "
            f"{inert} hold the same value in both across {self.n_keys} keys compared. Nothing was "
            f"varied, so a difference measured between them is sampling noise."
        )

    @property
    def remedy(self) -> str:
        if self.diverging:
            named = self.keys[:3]
            rest = len(self.keys) - len(named)
            subject = ", ".join(named)
            pronoun = "it" if len(self.keys) == 1 else "them"
            tail = f" The remaining {rest} are in the detail." if rest > 0 else ""
            return (
                f"Set {subject} to the same value in both arms and re-run, or declare {pronoun} in "
                f"the contrast and re-freeze the study.{tail}"
            )
        subject = ", ".join(self.contrast_inert)
        pronoun = "it" if len(self.contrast_inert) == 1 else "them"
        return (
            f"Check that the override for {subject} reached the trainer, then re-run. If {pronoun} "
            f"was never meant to vary, drop it from the declared contrast and re-freeze the study."
        )

    @property
    def void(self) -> "Void | None":
        """This report as a `Void`, or None when the arms are comparable.

        The import is deferred rather than done at module scope. `record` is the bottom of the
        dependency graph and `studies` sits well above it, so importing the vocabulary eagerly
        would pull the study runner and the measurement base into every process that opens a
        record, and would become a genuine import cycle the moment an instrument in `measure`
        imports `record`. The vocabulary itself is not duplicated: this is `studies.void.Void` with
        condition 4 when the arms diverge and condition 8 when the declared contrast did not.
        """
        reason = self.reason
        if reason is None:
            return None
        from reward_lens.studies.void import Void

        return Void(reason=reason, detail=self.detail, remedy=self.remedy)

    def refusal(self, instrument: str) -> Refusal:
        """This report as a `Refusal(VOID)`, for a measurement path rather than a study.

        An unreadable run is routed to `VOID`, and an instrument handed two arms that diverged,
        or two arms that were never actually contrasted, has been handed an unreadable comparison
        either way. The reason, the detail and the remedy are the same facts as `void`;
        only the container differs, because a study collects `Void` objects and an instrument
        returns a `Reading`. The specific void condition is in `statistics["void_condition"]`,
        because `RefusalReason.VOID` is one member covering all eight.
        """
        reason = self.reason
        if reason is None:
            raise ArmSpecError(
                f"{instrument} asked for a refusal from a comparison that is readable: arms "
                f"{self.left!r} and {self.right!r} differ in the declared contrast and nothing "
                f"else. Check `report.ok` first."
            )
        return Refusal(
            instrument=instrument,
            reason=RefusalReason.VOID,
            detail=self.detail,
            remedy=self.remedy,
            statistics={
                "void_condition": reason.value,
                "arms": [self.left, self.right],
                "arm_ids": [self.left_id, self.right_id],
                "contrast": list(self.contrast),
                "n_keys_compared": self.n_keys,
                "diverging": {d.key: [_short(d.left), _short(d.right)] for d in self.diverging},
                "contrast_inert": list(self.contrast_inert),
            },
        )

    def render(self) -> str:
        reason = self.reason
        head = "OK" if reason is None else f"VOID({reason.value})"
        lines = [f"{head}  {self.left} vs {self.right}"]
        if reason is None:
            realised = ", ".join(d.render() for d in self.contrast_realised) or "none"
            lines.append(f"    {self.n_keys} keys compared; differ only in: {realised}")
        else:
            lines.append(f"    {self.detail}")
            lines.append(f"    Remedy: {self.remedy}")
        if self.contrast_inert and self.diverging:
            lines.append(
                f"    Also, declared contrast that did not differ: "
                f"{', '.join(self.contrast_inert)}. Both problems are real; the divergence is "
                f"reported first because it is unreadable whatever the contrast did."
            )
        if self.staggered:
            lines.append(
                f"    The intervention landed at different steps in the two arms "
                f"({self.adoption[0]} and {self.adoption[1]}). Read this with a staggered-adoption "
                f"estimator; a before-and-after difference over both arms is not one."
            )
        return "\n".join(lines)


def check_divergence(
    left: Arm, right: Arm, *, contrast: Iterable[str] | None = None
) -> ArmDivergence:
    """Void gate 4: do these two arms differ in anything outside the declared contrast?

    ``contrast`` defaults to the union of what the two arms themselves declare, so an arm set
    built by `arms()` checks itself with no argument. Passing it explicitly is for the case where
    the arms were built somewhere else and the study declares the contrast.

    Every key of either arm is compared, including keys only one of them has: a key present in one
    arm and absent in the other is a divergence, and it is reported with `<absent>` on the missing
    side rather than skipped. Skipping it is how a dropped configuration section becomes invisible.
    """
    if contrast is None:
        declared = set(left.contrast) | set(right.contrast)
    else:
        declared = {str(k) for k in contrast}

    a, b = left.material, right.material
    keys = sorted(set(a) | set(b))
    diverging: list[KeyDivergence] = []
    realised: list[KeyDivergence] = []
    inert: list[str] = []
    for key in keys:
        lv = a.get(key, ABSENT)
        rv = b.get(key, ABSENT)
        differs = _differs(lv, rv)
        if key in declared:
            if differs:
                realised.append(KeyDivergence(key=key, left=lv, right=rv))
            else:
                inert.append(key)
        elif differs:
            diverging.append(KeyDivergence(key=key, left=lv, right=rv))

    return ArmDivergence(
        left=left.label,
        right=right.label,
        left_id=str(left.id),
        right_id=str(right.id),
        contrast=tuple(sorted(declared)),
        n_keys=len(keys),
        diverging=tuple(diverging),
        contrast_realised=tuple(realised),
        contrast_inert=tuple(inert),
        adoption=(left.adopted_at, right.adopted_at),
    )


def guard_comparison(
    inner: Callable[..., Any],
    *,
    instrument: str = "",
    contrast: Sequence[str] | None = None,
) -> Callable[..., Any]:
    """Wrap a two-arm comparison so that diverged arms produce a refusal instead of a number.

    The wrapped callable takes the two arms first and anything else after. If the arms differ only
    in the declared contrast it is called unchanged; if they do not, it is never called and the
    wrapper returns `Refusal(VOID)` naming the diverging keys.

    The reason to wrap rather than to document is that the check has to happen whether or not the
    author of the comparison remembered it, which is the same argument `measure.controls.matched`
    makes for `guard_null`. A gate that depends on being called is a convention.
    """
    name = instrument or getattr(inner, "__name__", "comparison")

    @functools.wraps(inner)
    def wrapped(left: Arm, right: Arm, *args: Any, **kwargs: Any) -> Any:
        report = check_divergence(left, right, contrast=contrast)
        if not report.ok:
            return report.refusal(name)
        return inner(left, right, *args, **kwargs)

    return wrapped


# ---------------------------------------------------------------------------
# The same gate, on two records rather than two plans
# ---------------------------------------------------------------------------

#: Collected and then dropped, because it differs between two arms by construction: they were
#: launched a few seconds apart. `Run.id` and `Run.arms` differ by construction too, for the same
#: kind of reason, and they are simply never collected below.
_RUN_MATERIAL_SKIP = ("lineage.created_at",)


def run_material(run: Run, *, engines: bool = False) -> dict[str, Any]:
    """The comparable material of a recorded run, as dotted keys.

    This is what makes the gate usable after the fact. Two arms that were planned identically can
    still have run differently, and the record is the only place that shows up: a different git
    sha, a framework upgrade between the two launches, a sampling policy that fell back to a lower
    rate on the second one because the disk was filling.

    ``engines`` walks the step stream to collect the engine identities that actually served the
    rollouts, which is the "same engine and revision" half of comparability measured rather than
    declared. It is off by default because walking a record is not free, and a record can be
    hundreds of gigabytes.
    """
    out: dict[str, Any] = {"kind": run.kind}
    for component, ref in sorted(run.components.items(), key=lambda kv: kv[0].name):
        for key, value in flatten_config(ref.__canonical__()).items():
            out[f"components.{component.name}.{key}"] = value
    for component, access in sorted(run.access.items(), key=lambda kv: kv[0].name):
        out[f"access.{component.name}"] = int(access)
    for key, value in flatten_config(run.regime.__canonical__()).items():
        out[f"regime.{key}"] = value
    for key, value in flatten_config(run.lineage.__canonical__()).items():
        out[f"lineage.{key}"] = value
    for key, value in flatten_config(run.sampling_policy.__canonical__()).items():
        out[f"sampling_policy.{key}"] = value
    if isinstance(run.coupling, CouplingSpec):
        canonical = run.coupling.__canonical__()
        canonical.pop("note", None)
        for key, value in flatten_config(canonical).items():
            out[f"coupling.{key}"] = tuple(value) if isinstance(value, list) else value
    if engines:
        seen: set[str] = set()
        for step in run.steps:
            for group in step.groups:
                for traj in group.trajectories:
                    for segment in traj.provenance:
                        seen.add(segment.engine.id)
        out["engines"] = tuple(sorted(seen))
    for key in _RUN_MATERIAL_SKIP:
        out.pop(key, None)
    return out


def arm_from_run(
    run: Run, *, label: str = "", contrast: Mapping[str, Any] | None = None, engines: bool = False
) -> Arm:
    """An `Arm` whose configuration is the material of a run that already happened.

    The coupling is taken from the run when it recorded one. A run with no `CouplingSpec` gets an
    empty declaration rather than an assumed one, because a run that never said what it shared
    with its sibling did not share anything on the record.
    """
    coupling = run.coupling if isinstance(run.coupling, CouplingSpec) else CouplingSpec()
    material = run_material(run, engines=engines)
    for key in list(material):
        if key.startswith("coupling."):
            material.pop(key)
    return Arm(
        label=label or str(run.id),
        config=material,
        contrast=dict(contrast or {}),
        coupling=coupling,
        run_id=run.id,
    )


def check_runs(
    left: Run,
    right: Run,
    *,
    contrast: Iterable[str] = (),
    engines: bool = False,
    labels: tuple[str, str] | None = None,
) -> ArmDivergence:
    """Void gate 4 on two recorded runs. The same report as `check_divergence`, off disk.

    The contrast is passed in because a record does not carry the study's declaration: the arms
    were launched by something that knew what it was varying, and by the time two run directories
    are being compared that knowledge lives in the frozen study rather than in the files.
    """
    names = labels or (str(left.id), str(right.id))
    keys = tuple(str(k) for k in contrast)
    return check_divergence(
        arm_from_run(left, label=names[0], contrast=dict.fromkeys(keys), engines=engines),
        arm_from_run(right, label=names[1], contrast=dict.fromkeys(keys), engines=engines),
        contrast=keys,
    )


__all__ = [
    "ABSENT",
    "Absent",
    "Arm",
    "ArmDivergence",
    "ArmID",
    "ArmSet",
    "ArmSpecError",
    "CouplingSpec",
    "DEFAULT_SEED",
    "DEFAULT_STREAMS",
    "KeyDivergence",
    "PROMPT_ORDER_STREAM",
    "apply_overrides",
    "arm_from_run",
    "arms",
    "check_divergence",
    "check_runs",
    "flatten_config",
    "guard_comparison",
    "independent_seeds",
    "run_material",
    "shared_seeds",
]

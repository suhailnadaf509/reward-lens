"""Shared plumbing for the index library.

The indices are the vocabulary of the cards and the scoreboard, so they all reach for the same few
operations: read a reward direction ``w_r`` off a signal, read final-token activations at a site,
and turn activations into named feature values through a feature bank. Those live here so each index
module is just its own definition, and so every index reads the substrate the same way.

The feature bank is the one interface the corpus's concept layer will implement. It is deliberately
tiny: a bank names a set of properties and turns an ``(n, d)`` activation matrix into an ``(n, k)``
matrix of feature values, optionally exposing the ``(k, d)`` decoder directions. ``concepts`` (built
concurrently) will provide production banks; ``LinearFeatureBank`` is the synthetic bank of known
directions that makes an index like ``chi`` provable without waiting for it. Indices lazy-import
concepts and degrade gracefully when a bank is absent, so importing this module pulls no torch and no
concept machinery.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from reward_lens.core.envelope import RegimeCondition
from reward_lens.core.features import FeatureBank
from reward_lens.core.types import Phase, Site, Substrate

if TYPE_CHECKING:
    from reward_lens.signals.base import Readout, RewardSignal


# ---------------------------------------------------------------------------
# The observable declarations the index library shares
# ---------------------------------------------------------------------------

#: How each regime condition the index library depends on is measured, named by quantity id. Every
#: condition an index puts in ``requires`` has to appear here or `EnvelopeSpec` refuses to be
#: constructed, so this table is what decides whether a precondition is checkable at all rather
#: than decorative.
#:
#: ``STATIONARY_GRADER`` is measured by the check standard (catalogue J5): a fixed probe set
#: re-scored across the measurement window, whose drift is direct evidence that the grader moved
#: while it was being measured. ``GROUP_NONDEGENERATE`` is measured by the degenerate-group
#: fraction (E2), which counts the groups with no score spread to read a contrast from.
#: ``ABOVE_LOD`` is measured by the substrate limit of detection (M1), the grader's disagreement
#: with itself, which is the floor a reward delta has to clear before it is a measurement rather
#: than noise. ``LINEAR_RESPONSE`` is measured by the selection-explained fraction Lambda (F2).
#:
#: The same table appears in ``measure/battery/_common.py``. It is duplicated rather than shared
#: because ``measure.battery`` requires the white-box extra at import and this package is torch-free
#: by contract, so importing the battery's copy would install a dependency the indices exist
#: without.
MEASURED_BY: Mapping[RegimeCondition, str] = MappingProxyType(
    {
        RegimeCondition.STATIONARY_GRADER: "monitor.check_standard_drift",
        RegimeCondition.GROUP_NONDEGENERATE: "estimator.degenerate_fraction",
        RegimeCondition.ABOVE_LOD: "substrate.lod",
        RegimeCondition.LINEAR_RESPONSE: "selection.explained_fraction",
    }
)

#: The substrates with weights to read. An index that captures activations or projects onto a
#: readout vector applies to these two and to nothing else: a PROGRAM grader has source code where
#: these have activations, and asking it for a direction is a category error, not a hard case.
NEURAL_SUBSTRATES = frozenset({Substrate.NEURAL_SCALAR, Substrate.NEURAL_GEN})

#: Every substrate. What an index that reads only scores, or only numbers another arm measured,
#: applies to.
ANY_SUBSTRATE = frozenset(Substrate)

#: When a grader study can be read. These indices need the grader itself or a finished experiment
#: against it, so they run before optimisation starts or after it finishes. They are not on the hot
#: path, and against a DEPLOYED artifact the grader is no longer reachable at all.
GRADER_STUDY_PHASES = frozenset({Phase.PRE_RUN, Phase.POST_RUN})


# ---------------------------------------------------------------------------
# Reading the substrate: w_r and activations
# ---------------------------------------------------------------------------


def find_readout(signal: "RewardSignal", name: str = "reward") -> "Readout":
    """Look up a readout by name through the frozen ``RewardSignal`` protocol surface.

    Uses ``readouts()`` (the protocol method every adapter implements) rather than a signal-specific
    ``readout`` accessor, so an index runs against any substrate. Falls back to the first readout
    when the name is absent and there is only one, which is the common single-head case.
    """
    readouts = list(signal.readouts())
    for r in readouts:
        if r.name == name:
            return r
    accessor = getattr(signal, "readout", None)
    if callable(accessor):
        try:
            return accessor(name)
        except Exception:  # noqa: BLE001 - fall through to the single-readout convenience
            pass
    if len(readouts) == 1:
        return readouts[0]
    raise KeyError(f"unknown readout {name!r}; available: {[r.name for r in readouts]}")


def reward_vector(signal: "RewardSignal", readout: str = "reward") -> np.ndarray:
    """The reward direction ``w_r`` for a readout, as a float64 numpy vector.

    ``readouts()[0].vector`` is the head weight (a torch tensor for linear readouts); this coerces it
    to a detached fp64 numpy array so the pure index math never touches torch. Non-linear readouts
    (simplex, token-value) carry no vector and raise, which is the honest failure for an index that
    needs a linear reward direction.
    """
    read = find_readout(signal, readout)
    vec = read.vector
    if vec is None:
        raise ValueError(
            f"readout {readout!r} is a {read.kind!r} readout with no reward vector; "
            "the linear index math needs a linear or logit_diff readout"
        )
    arr = np.asarray(_to_numpy(vec), dtype=np.float64).ravel()
    return arr


def readout_site(signal: "RewardSignal", readout: str = "reward") -> Site:
    """The site the readout reads at (the final residual for a classifier head)."""
    return find_readout(signal, readout).site


def final_activations(
    signal: "RewardSignal",
    view: Any,
    site: Site | None = None,
    *,
    readout: str = "reward",
) -> np.ndarray:
    """Capture final-token activations at a site for every item, as an ``(n, d)`` float64 matrix.

    Captures in fp32 (frames and covariances refuse fp16) at the resolved final position, then coerces
    to fp64 numpy. When ``site`` is None the readout's own site is used, which is where ``w_r`` acts, so
    ``activations @ w_r`` reproduces the signal's score up to the head bias. This is the production
    path; the index math is tested directly on synthetic activation matrices.
    """
    from reward_lens.runtime.backend import CaptureSpec
    from reward_lens.signals.base import PositionSpec

    if site is None:
        site = readout_site(signal, readout)
    spec = CaptureSpec(
        sites=(site,),
        position=PositionSpec("final"),
        full_sequence=False,
        dtype="float32",
    )
    capture = next(iter(signal.capture(view, spec)))
    tensor = capture.tensors[site]
    return np.asarray(_to_numpy(tensor), dtype=np.float64)


def reward_scores(signal: "RewardSignal", view: Any, readout: str = "reward") -> np.ndarray:
    """The per-item reward scores under a readout, as a float64 vector.

    Thin wrapper over ``signal.score`` that unwraps the ``Evidence[Scores]`` to the raw values. Base
    policy samples fed here give the ``r`` that ``chi`` and ``tail`` are functionals of.
    """
    evidence = signal.score(view, readout)
    return np.asarray(_to_numpy(evidence.value.values), dtype=np.float64).ravel()


def _to_numpy(x: Any) -> np.ndarray:
    """Coerce a torch tensor or array-like to a detached CPU numpy array without importing torch."""
    if hasattr(x, "detach"):  # torch.Tensor
        return x.detach().to("cpu").numpy()
    return np.asarray(x)


# ---------------------------------------------------------------------------
# The feature bank interface (the concept layer's contract)
# ---------------------------------------------------------------------------


# The contract now lives in `reward_lens.core.features` and is re-exported here so every existing
# import keeps working. It moved because a feature bank is what every Level 1 instrument is written
# in terms of, which makes it a kernel question rather than an index-library one, and because the
# name meant two things: `loops/recorder.py` exported an unrelated container under it, now named
# `DirectionBank`.


@dataclass
class LinearFeatureBank:
    """A synthetic feature bank of known linear directions (the test-time and default bank).

    Features are linear readouts ``f = activations @ D^T`` for decoder directions ``D`` (``k, d``).
    This is exactly the object an index needs to recover a planted ``Cov(feature, reward)``: pick the
    directions, plant a reward that loads on one of them, and ``chi`` must light up that feature. A
    production concept bank implements the same ``FeatureBank`` protocol with learned features.
    """

    directions_: np.ndarray  # (k, d)
    names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.directions_ = np.asarray(self.directions_, dtype=np.float64)
        if self.directions_.ndim != 2:
            raise ValueError(f"directions must be (k, d); got shape {self.directions_.shape}")
        if not self.names:
            self.names = tuple(f"f{i}" for i in range(self.directions_.shape[0]))
        if len(self.names) != self.directions_.shape[0]:
            raise ValueError(
                f"names has {len(self.names)} entries but there are {self.directions_.shape[0]} "
                "directions"
            )

    def featurize(self, activations: np.ndarray) -> np.ndarray:
        a = np.asarray(activations, dtype=np.float64)
        return a @ self.directions_.T

    def directions(self) -> np.ndarray | None:
        return self.directions_


def load_default_bank(signal: "RewardSignal") -> FeatureBank | None:
    """Try to obtain a production feature bank from the concept layer, else None (graceful degrade).

    The concept subsystem is built concurrently, so this lazy-imports it and returns None on any
    failure. An index that gets None falls back to an injected bank or reports that no feature bank
    was available, rather than fabricating features. This is the seam that keeps the index library
    provable now and upgradeable later without a code change here.
    """
    try:  # pragma: no cover - exercised only once concepts lands
        import reward_lens.concepts as concepts  # noqa: F401

        factory = getattr(concepts, "default_feature_bank", None)
        if callable(factory):
            bank = factory(signal)
            if isinstance(bank, FeatureBank):
                return bank
    except Exception:  # noqa: BLE001 - concepts absent or incompatible: degrade to None
        return None
    return None


def percentile_within_battery(values: np.ndarray) -> np.ndarray:
    """Map a battery of raw values to their percentile ranks in ``[0, 1]`` (average-rank, ties shared).

    This is the standardization that fixes the KUI unit bug: decodability and mediation live on
    incommensurable raw scales, so both are pushed to their rank-within-battery before they are ever
    combined. With ``m`` values the ``i``-th smallest gets ``(rank + 0.5) / m`` so the percentiles are
    symmetric in ``(0, 1)`` and a singleton battery maps to ``0.5`` rather than an undefined ``0/0``.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    m = v.size
    if m == 0:
        return v
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(m, dtype=np.float64)
    ranks[order] = np.arange(m, dtype=np.float64)
    # average tied ranks so equal raw values share a percentile
    _, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
    starts = np.cumsum(counts) - counts
    avg_rank = starts[inv] + (counts[inv] - 1) / 2.0
    return (avg_rank + 0.5) / m


# ---------------------------------------------------------------------------
# The missing-injection refusal
# ---------------------------------------------------------------------------


def missing_injection(
    inst: Any, *, needs: dict[str, str], have: str, remedy: str
) -> "PreflightResult":
    """The preflight result for an index whose injected input was not supplied.

    Twelve indices used to answer this case with ``ctx.emit({"note": "... none injected"})``. That
    is a number-shaped object standing where a refusal belongs: it satisfies the letter of "returns
    Evidence or a Refusal" and tells the reader nothing they can act on, and the converter's sweep
    had to classify it as a third kind to keep its own count honest. The case is a `Refusal`
    carrying a reason and a remedy.

    It goes in `preflight` rather than in `measure` because that is where the question belongs:
    nothing has to be computed to know that a required input is absent, and `preflight` is the
    method the capability report calls with no GPU work. `BaseObservable.estimate` returns this
    refusal as a value before it ever reaches `measure`.

    `ACCESS_INSUFFICIENT` rather than `RECORD_INCOMPLETE`, on the test of whether the remedy is
    answerable where the reader is standing: the input is a constructor argument they can supply. A
    record-incomplete refusal means the field was never written and nothing the reader does
    recovers it.
    """
    from reward_lens.core.reading import refuse_access
    from reward_lens.measure.base import PreflightResult

    name = getattr(inst, "name", type(inst).__name__)
    return PreflightResult(
        instrument=name,
        ok=False,
        refusal=refuse_access(name, needs=needs, have=have, remedy=remedy),
    )


def measured_without_input(inst: Any) -> ValueError:
    """What `measure` raises when the input `preflight` refuses on is still absent.

    `measure` is typed ``-> Evidence`` and `run` is the 2.0.1 entry point that returns one. A caller
    who reached it past a refusing preflight made a programming error, which is what an exception is
    for; `estimate` is the declared entry point and returns the refusal as a value. The same split
    is in `measure/composition/composition.py:160`.
    """
    name = getattr(inst, "name", type(inst).__name__)
    return ValueError(
        f"{name}.measure was called with its injected input absent, which is a refusal rather than "
        f"a measurement. Call `estimate`, which returns the refusal as a value carrying the remedy, "
        f"or supply the input this instrument declares."
    )


# ---------------------------------------------------------------------------
# What a white-box reading owes, and what to say when it cannot pay
# ---------------------------------------------------------------------------

#: The reasons a white-box instrument's reading can carry no `IncrementalValidity`, as ids lint can
#: check rather than prose lint can only print. The record is mandatory on every white-box reading,
#: and the honest answer for most of this library's white-box instruments is not "here is one" but
#: "here is exactly why there cannot be one", which is a different statement from silence and has
#: to be distinguishable from it.
#:
#: Three reasons, and they are not interchangeable. The first is a property of the quantity and will
#: not change: a reading that is a cosine matrix or a set overlap has no per-item verdict, so there
#: is no error vector, so there is nothing to correlate. The second is a property of the inputs: an
#: instrument fed bare arrays with no text, no logged series and no judge gives the black-box bank
#: nothing to read on its items, and every one of the six refuses. The third is a property of the
#: subject and it is the one that can be fixed by pointing the instrument at something else, which
#: is why it names the subject that would work.
#:
#: The same table appears in ``measure/battery/_common.py``, duplicated for the reason `MEASURED_BY`
#: above is duplicated: this package is torch-free by contract and the battery is not.
INCREMENTAL_EXEMPTIONS: frozenset[str] = frozenset(
    {
        # The reading is not a per-item verdict, so no per-item error vector exists.
        "NO_PER_ITEM_VERDICT",
        # No black-box method can read this instrument's items, so the bank has no competitor to be
        # incremental over.
        "NO_BLACK_BOX_ON_THESE_ITEMS",
        # Both methods can run and the available subject carries no signal for either, so the four
        # numbers would describe the fixture rather than the instrument.
        "NO_SUBJECT_WITH_SIGNAL",
    }
)

#: How short a reason is allowed to be before it is not a reason. A remedy string is a user
#: interface and "not applicable" is not one; the number is a floor, not a target.
_MIN_REASON_CHARS = 120


def incremental_exemption_findings(inst: Any) -> list[str]:
    """What is wrong with an instrument's `incremental_exemption`, as strings, or an empty list.

    Checked here rather than trusted, because an exemption nobody validates is a comment that
    silences a lint rule. The id has to be one of `INCREMENTAL_EXEMPTIONS` and the prose beside it
    has to be long enough to be an argument. An instrument that declares no exemption at all returns
    nothing from this function: whether that is a failure is `lint_reading`'s question and it needs
    the reading, not the declaration.
    """
    declared = getattr(inst, "incremental_exemption", None)
    name = getattr(inst, "name", type(inst).__name__)
    if declared is None:
        return []
    if not (isinstance(declared, tuple) and len(declared) == 2):
        return [
            f"{name}.incremental_exemption is {declared!r}; it has to be a "
            f"(reason_id, prose) pair so the id is checkable and the prose is readable"
        ]
    reason, prose = declared
    out: list[str] = []
    if reason not in INCREMENTAL_EXEMPTIONS:
        out.append(
            f"{name}.incremental_exemption names {reason!r}, which is not one of "
            f"{sorted(INCREMENTAL_EXEMPTIONS)}. A new reason is a library-wide decision, not a "
            f"string an instrument can invent"
        )
    if not isinstance(prose, str) or len(prose.strip()) < _MIN_REASON_CHARS:
        out.append(
            f"{name}.incremental_exemption carries {len(str(prose).strip())} characters of reason "
            f"and the floor is {_MIN_REASON_CHARS}. Say what the reading is, why that shape has no "
            f"per-item error vector or no competitor, and what would change the answer"
        )
    return out


def measure_incremental_validity(
    own_id: str,
    own_scores: Any,
    labels: Any,
    *,
    texts: tuple[str, ...] = (),
    seed_labels: tuple[Any, ...] = (),
    series: Any = None,
    markers: tuple[str, ...] = (),
    n_resamples: int = 2000,
    seed: int = 0,
) -> tuple[Any, dict[str, Any]]:
    """The incremental-validity record for a white-box reading, against the black-box bank.

    Returns ``(IncrementalValidity | None, notes)``. The record is None when no baseline in the bank
    could run or could separate anything, and ``notes`` always says which and why, because a reading
    that quietly dropped its comparison is the failure the bank exists to prevent.

    Nothing new is computed here. `stats.baselines.run_bank` is the six dumb baselines and
    `measure.meta.incremental.IncrementalValidityReading` is M9; this function is the wiring between
    an instrument's per-item scores and those two, so that no instrument grows its own bank and no
    instrument grows its own increment.

    **The ceiling check is the part that is easy to leave out.** The library's first incremental
    record was measured against a baseline whose accuracy was 1.0, because the reference run's
    grader is a length function and the length baseline therefore solves its task outright. Every
    number in that record was correct and the record established that the four-number shape works,
    not that opening the network bought anything. So when the best baseline is at ceiling, or when
    neither method separates the labels, that is reported at the reading rather than left for a
    reader to infer from an ensemble gain of zero.
    """
    import numpy as _np

    from reward_lens.measure.meta.incremental import Detector, IncrementalValidityReading
    from reward_lens.stats.baselines import run_bank
    from reward_lens.stats.baselines.base import DetectionTask

    scores = _np.asarray(own_scores, dtype=_np.float64).ravel()
    y = _np.asarray(labels).ravel().astype(int)
    notes: dict[str, Any] = {"n_items": int(y.size)}
    if scores.size != y.size:
        raise ValueError(
            f"{own_id} supplied {scores.size} scores for {y.size} labels. The increment is a paired "
            f"quantity and a misalignment here is a sampling difference nothing downstream can "
            f"separate from the instrument's contribution."
        )

    task = DetectionTask(
        labels=y,
        texts=texts,
        seed_labels=seed_labels,
        series=series,
        markers=markers,
        name=own_id,
    )
    bank = run_bank(task)
    notes["baseline_auroc"] = {b: float(r.auroc) for b, r in sorted(bank.scored().items())}
    notes["baseline_refusals"] = {b: r.detail for b, r in sorted(bank.refusals().items())}

    usable = [
        Detector.from_scores(
            b,
            r.scores,
            y,
            threshold=_midpoint_threshold(_np.asarray(r.scores, dtype=_np.float64), y),
            note=r.detail,
        )
        for b, r in sorted(bank.scored().items())
        if float(_np.std(r.scores)) > 0.0
    ]
    if not usable:
        notes["refused"] = (
            "no baseline in the bank produced a varying score on these items, so there is nothing "
            "to be incremental over. Supply the inputs the bank names in `baseline_refusals`."
        )
        return None, notes

    own = Detector.from_scores(
        own_id, scores, y, threshold=_midpoint_threshold(scores, y), note="the white-box reading"
    )
    increment = IncrementalValidityReading(
        own=own, baselines_run=usable, n_resamples=n_resamples, seed=seed
    ).compute()
    if not hasattr(increment, "record"):  # a Refusal from M9, carried out rather than swallowed
        notes["refused"] = getattr(increment, "detail", "M9 refused")
        return None, notes

    record = increment.record
    notes["increment"] = increment.increment
    notes["ci"] = [increment.ci_low, increment.ci_high, increment.ci_level]
    notes["adds_nothing"] = increment.adds_nothing
    notes["says"] = increment.says()

    # The two ways the four numbers can be true and mean nothing about the instrument.
    if record.baseline_score >= 1.0 - 1e-12:
        notes["ceiling"] = (
            f"{record.baseline_id} scores {record.baseline_score:.4f} on these items, which is "
            f"ceiling. An ensemble cannot improve on a baseline that is already right about "
            f"everything, so the gain of {record.ensemble_gain:+.4f} is a fact about this item set "
            f"and not about what opening the network bought. Read this record as evidence that the "
            f"comparison ran, not as evidence about the instrument."
        )
    # Chance is the majority-class rate, not 0.5. On a task that is 75% negative, a method scoring
    # 0.62 is worse than a constant answer and calling that "above chance" is how a floor gets
    # reported as a finding. `notes["chance"]` is carried so a reader can check the comparison.
    chance = max(float((y == 1).mean()), float((y == 0).mean()))
    notes["chance"] = chance
    if max(record.own_score, record.baseline_score) <= chance + 1e-12:
        notes["floor"] = (
            f"neither method beats answering with the majority class: the reading scores "
            f"{record.own_score:.4f} and the best baseline {record.baseline_score:.4f} against a "
            f"chance rate of {chance:.4f}. The error correlation of "
            f"{record.error_correlation:+.4f} is between two noise processes and says nothing "
            f"about complementarity. Point this instrument at a subject where at least one of the "
            f"two methods works before reading the increment."
        )
    return record, notes


def _midpoint_threshold(scores: Any, labels: Any) -> float:
    """The threshold halfway between the two class-mean scores, which is what the bank uses.

    A fixed rule rather than the best threshold on the evaluation set, for the reason
    `stats.baselines.base.accuracy_at_midpoint` gives: a threshold maximising test accuracy is a
    free parameter fitted on the test data, and a comparator that takes one is no longer dumb.
    """
    import numpy as _np

    s = _np.asarray(scores, dtype=_np.float64).ravel()
    y = _np.asarray(labels).ravel().astype(int)
    if _np.unique(y).size < 2:
        return float(s.mean()) if s.size else 0.0
    return 0.5 * (float(s[y == 1].mean()) + float(s[y == 0].mean()))


if TYPE_CHECKING:
    from reward_lens.measure.base import PreflightResult


__all__ = [
    "INCREMENTAL_EXEMPTIONS",
    "incremental_exemption_findings",
    "measure_incremental_validity",
    "ANY_SUBSTRATE",
    "GRADER_STUDY_PHASES",
    "MEASURED_BY",
    "NEURAL_SUBSTRATES",
    "find_readout",
    "reward_vector",
    "readout_site",
    "final_activations",
    "reward_scores",
    "FeatureBank",
    "LinearFeatureBank",
    "load_default_bank",
    "measured_without_input",
    "missing_injection",
    "percentile_within_battery",
]
